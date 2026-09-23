"""
semantic.py

Deterministic semantic column inference for the multilingual CVD ontology.

Current mode: deterministic ontology inference only, using aliases, dtype,
observed values, expected ranges, and optional ontology metadata.

NLP / SapBERT inference is intentionally disabled for now. Unresolved
columns are returned as UNKNOWN.

Typical usage:
    from semantic import SemanticInferencer, SemanticDataset

    inferencer = SemanticInferencer(
        ontology_path="cvd_ontology.json",
    )

    schema = inferencer.infer(df)
    dataset = SemanticDataset(df, schema)
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# NLP / SapBERT module -- DISABLED FOR NOW
# ---------------------------------------------------------------------------
# from generate_embedding import generate_embeddings
# DEFAULT_EMBEDDING_INDEX_PATH = "all_embed_ontology.json"
# DEFAULT_EMBEDDING_THRESHOLD = 0.88
# DEFAULT_EMBEDDING_MARGIN = 0.05
# No Hugging Face model or embedding index is loaded in this version.
# ---------------------------------------------------------------------------


DEFAULT_ONTOLOGY_PATH = "cvd_ontology.json"
UNKNOWN = "UNKNOWN"

# Conservative defaults. Increase rather than decrease these when false
# inferences are more costly than missed inferences.
DEFAULT_DETERMINISTIC_THRESHOLD = 0.80
DEFAULT_DETERMINISTIC_MARGIN = 0.10


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return obj


def normalize_text(value: Any) -> str:
    """Conservative Unicode normalization for multilingual column names."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value).strip())
    replacements = {
        "ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ۀ": "ه",
        "ة": "ه", "ؤ": "و", "ـ": "", "\u200c": " ",
        "\u200d": " ", "\ufeff": "",
    }
    s = "".join(replacements.get(c, c) for c in s)
    s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
                                  "01234567890123456789"))
    s = s.casefold()
    s = re.sub(r"[_\-.\\/]+", " ", s)
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def compact_normalize(value: Any) -> str:
    return normalize_text(value).replace(" ", "")


def string_similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if compact_normalize(a) == compact_normalize(b):
        return 0.985
    char = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return char
    jaccard = len(ta & tb) / len(ta | tb)
    containment = len(ta & tb) / min(len(ta), len(tb))
    token = 0.65 * jaccard + 0.35 * containment
    return 0.65 * char + 0.35 * token


@dataclass(frozen=True)
class Alias:
    text: str
    language: str
    concept_id: str


@dataclass
class Concept:
    concept_id: str
    category: Optional[str]
    data_type: Optional[str]
    unit: Any
    expected_range: Optional[Tuple[float, float]]
    aliases: List[Alias]
    raw: Dict[str, Any]


class Ontology:
    """Loads the user's JSON ontology and indexes every alias."""

    def __init__(self, data: Mapping[str, Any]):
        self.raw = dict(data)
        self.name = data.get("ontology_name")
        self.version = data.get("version")
        self.languages = dict(data.get("languages", {}))
        raw_concepts = data.get("concepts")
        if not isinstance(raw_concepts, Mapping):
            raise ValueError("Ontology must contain a 'concepts' object.")

        self.concepts: Dict[str, Concept] = {}
        self.aliases: Dict[str, List[Alias]] = {}
        self.compact_aliases: Dict[str, List[Alias]] = {}

        for concept_id, raw in raw_concepts.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"Concept '{concept_id}' must be an object.")
            alias_map = raw.get("aliases", {}) or {}
            if not isinstance(alias_map, Mapping):
                raise ValueError(f"Aliases of '{concept_id}' must be an object.")

            aliases: List[Alias] = []
            for language, values in alias_map.items():
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    raise ValueError(
                        f"Aliases of '{concept_id}'/{language} must be a list."
                    )
                for value in values:
                    if not isinstance(value, str) or not value.strip():
                        continue
                    alias = Alias(value.strip(), str(language), str(concept_id))
                    aliases.append(alias)
                    key = normalize_text(alias.text)
                    self.aliases.setdefault(key, []).append(alias)
                    self.compact_aliases.setdefault(
                        compact_normalize(alias.text), []
                    ).append(alias)

            expected_range = raw.get("expected_range", raw.get("range"))
            parsed_range = None
            if isinstance(expected_range, (list, tuple)) and len(expected_range) == 2:
                try:
                    parsed_range = (float(expected_range[0]), float(expected_range[1]))
                except (TypeError, ValueError):
                    pass

            self.concepts[str(concept_id)] = Concept(
                concept_id=str(concept_id),
                category=str(raw["category"]) if raw.get("category") is not None else None,
                data_type=str(raw["data_type"]) if raw.get("data_type") is not None else None,
                unit=raw.get("unit"),
                expected_range=parsed_range,
                aliases=aliases,
                raw=dict(raw),
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "Ontology":
        return cls(load_json(path))


@dataclass
class ColumnProfile:
    name: str
    normalized_name: str
    dtype: str
    numeric: bool
    boolean: bool
    datetime: bool
    non_null: int
    unique: int
    missing: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    samples: List[Any] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)

    @classmethod
    def from_series(cls, name: str, s: pd.Series, sample_size: int = 20):
        nonnull = s.dropna()
        numeric = bool(pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s))
        boolean = bool(pd.api.types.is_bool_dtype(s))
        datetime = bool(pd.api.types.is_datetime64_any_dtype(s))
        samples = nonnull.head(sample_size).tolist()
        min_value = max_value = None
        if numeric and len(nonnull):
            n = pd.to_numeric(nonnull, errors="coerce").dropna()
            if len(n):
                min_value, max_value = float(n.min()), float(n.max())
        patterns = []
        strings = [str(x).strip() for x in samples]
        if strings and all(re.fullmatch(r"[01]", x) for x in strings):
            patterns.append("binary_numeric")
        if strings and all(re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", x) for x in strings):
            patterns.append("date_like")
        return cls(
            name=str(name), normalized_name=normalize_text(name), dtype=str(s.dtype),
            numeric=numeric, boolean=boolean, datetime=datetime,
            non_null=int(nonnull.size), unique=int(nonnull.nunique()),
            missing=int(s.isna().sum()), min_value=min_value, max_value=max_value,
            samples=samples, patterns=patterns,
        )


def profile_dataset(data: pd.DataFrame, sample_size: int = 20) -> List[ColumnProfile]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    return [ColumnProfile.from_series(str(c), data[c], sample_size) for c in data.columns]


@dataclass
class Candidate:
    concept_id: str
    score: float
    matched_text: Optional[str] = None
    language: Optional[str] = None
    method: Optional[str] = None


@dataclass
class SemanticMatch:
    column: str
    concept_id: str
    confidence: float
    method: str
    status: str
    matched_text: Optional[str] = None
    language: Optional[str] = None
    category: Optional[str] = None
    data_type: Optional[str] = None
    unit: Any = None
    reason: Optional[str] = None
    second_best_concept: Optional[str] = None
    second_best_score: Optional[float] = None
    margin: Optional[float] = None
    candidates: List[Candidate] = field(default_factory=list)

    @property
    def is_known(self) -> bool:
        return self.status == "KNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticSchema:
    matches: Dict[str, SemanticMatch]
    ontology_name: Optional[str]
    ontology_version: Optional[str]
    embedding_model: Optional[str]

    def __getitem__(self, column: str) -> SemanticMatch:
        return self.matches[column]

    def known_columns(self) -> Dict[str, str]:
        return {c: m.concept_id for c, m in self.matches.items() if m.is_known}

    def unknown_columns(self) -> List[str]:
        return [c for c, m in self.matches.items() if not m.is_known]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ontology_name": self.ontology_name,
            "ontology_version": self.ontology_version,
            "embedding_model": self.embedding_model,
            "matches": {c: m.to_dict() for c, m in self.matches.items()},
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for match in self.matches.values():
            d = match.to_dict()
            d.pop("candidates", None)
            rows.append(d)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NLP / SapBERT embedding index -- DISABLED FOR NOW
# ---------------------------------------------------------------------------
# class EmbeddingIndex:
#     ...
#
# The embedding index is deliberately not loaded while deterministic-only
# mode is active.
# ---------------------------------------------------------------------------


class DeterministicMatcher:
    """First-stage, explainable ontology inference."""

    def __init__(self, ontology: Ontology, threshold: float, margin: float):
        self.ontology = ontology
        self.threshold = threshold
        self.margin = margin

    def match(self, p: ColumnProfile) -> Optional[SemanticMatch]:
        # Exact normalized alias. If one alias maps to multiple concepts,
        # refuse to choose. The NLP/SapBERT fallback is disabled for now.
        aliases = self.ontology.aliases.get(p.normalized_name, [])
        if not aliases:
            aliases = self.ontology.compact_aliases.get(compact_normalize(p.name), [])
        concept_ids = {a.concept_id for a in aliases}
        if len(concept_ids) == 1:
            a = aliases[0]
            c = self.ontology.concepts[a.concept_id]
            return SemanticMatch(
                p.name, a.concept_id, 1.0, "deterministic", "KNOWN",
                a.text, a.language, c.category, c.data_type, c.unit,
                "Exact ontology alias match.", None, None, 1.0,
                [Candidate(a.concept_id, 1.0, a.text, a.language, "deterministic")],
            )

        # Weighted deterministic evidence for aliases. Name evidence is
        # dominant, while observed values/dtype/range can support it.
        ranked = []
        for cid, concept in self.ontology.concepts.items():
            best = None
            for alias in concept.aliases:
                name_score = string_similarity(p.name, alias.text)
                value_score = self._value_score(p, concept)
                dtype_score = self._dtype_score(p, concept)
                range_score = self._range_score(p, concept)
                pattern_score = self._pattern_score(p, concept)
                score = (
                    0.60 * name_score + 0.20 * value_score +
                    0.10 * dtype_score + 0.05 * range_score +
                    0.05 * pattern_score
                )
                item = (score, alias, name_score)
                if best is None or score > best[0]:
                    best = item
            if best:
                ranked.append((cid, *best))

        ranked.sort(key=lambda x: x[1], reverse=True)
        if not ranked:
            return None
        cid, score, alias, name_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else None
        margin = score - second_score if second_score is not None else score

        # Fuzzy deterministic matching is intentionally strict. This prevents
        # a short abbreviation from accidentally becoming an unrelated concept.
        if score < self.threshold or name_score < 0.92:
            return None
        if second_score is not None and margin < self.margin:
            return None

        c = self.ontology.concepts[cid]
        candidates = [
            Candidate(x[0], float(x[1]), x[2].text, x[2].language, "deterministic")
            for x in ranked[:5]
        ]
        return SemanticMatch(
            p.name, cid, float(score), "deterministic", "KNOWN",
            alias.text, alias.language, c.category, c.data_type, c.unit,
            "High-confidence deterministic ontology match.",
            ranked[1][0] if len(ranked) > 1 else None,
            float(second_score) if second_score is not None else None,
            float(margin), candidates,
        )

    @staticmethod
    def _dtype_score(p: ColumnProfile, c: Concept) -> float:
        dtype = normalize_text(c.data_type or "")
        if dtype in {"numeric", "continuous", "float", "integer", "number"}:
            return 1.0 if p.numeric else 0.0
        if dtype in {"categorical", "category", "nominal", "ordinal", "binary", "boolean"}:
            return 1.0 if p.boolean else (0.85 if p.unique <= 10 else 0.0)
        if dtype in {"identifier", "id", "string", "text"}:
            return 0.8 if not p.numeric else 0.2
        if dtype in {"date", "datetime", "timestamp"}:
            return 1.0 if p.datetime else 0.0
        return 0.0

    @staticmethod
    def _range_score(p: ColumnProfile, c: Concept) -> float:
        if c.expected_range is None or p.min_value is None or p.max_value is None:
            return 0.0
        lo, hi = c.expected_range
        if p.min_value >= lo and p.max_value <= hi:
            return 1.0
        width = max(hi - lo, 1.0)
        violation = max(0.0, lo - p.min_value) + max(0.0, p.max_value - hi)
        return 0.5 if violation / width <= 0.05 else 0.0

    @staticmethod
    def _value_score(p: ColumnProfile, c: Concept) -> float:
        definitions = []
        raw = c.raw
        for key in ("value_aliases", "allowed_values", "values", "categories"):
            value = raw.get(key)
            if isinstance(value, Mapping):
                for vals in value.values():
                    definitions.extend(vals if isinstance(vals, list) else [vals])
            elif isinstance(value, list):
                definitions.extend(value)
            elif isinstance(value, str):
                definitions.append(value)
        if not definitions or not p.samples:
            return 0.0
        allowed = {normalize_text(x) for x in definitions if normalize_text(x)}
        if not allowed:
            return 0.0
        return sum(normalize_text(x) in allowed for x in p.samples) / len(p.samples)

    @staticmethod
    def _pattern_score(p: ColumnProfile, c: Concept) -> float:
        expected = c.raw.get("patterns", [])
        if isinstance(expected, str):
            expected = [expected]
        if isinstance(expected, list) and expected:
            return 1.0 if set(p.patterns) & {normalize_text(x) for x in expected} else 0.0
        return 0.0


class SemanticInferencer:
    """
    Deterministic-only semantic inference pipeline.

    NLP / SapBERT inference is intentionally disabled.

    Any column that cannot be resolved with the deterministic ontology
    matcher is returned as UNKNOWN. No external model is loaded, no
    Hugging Face files are accessed, and no embedding index is required.
    """

    def __init__(
        self,
        ontology: Optional[Ontology] = None,
        ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
        deterministic_threshold: float = DEFAULT_DETERMINISTIC_THRESHOLD,
        deterministic_margin: float = DEFAULT_DETERMINISTIC_MARGIN,
        profile_sample_size: int = 20,
        # -----------------------------------------------------------------
        # NLP / SapBERT parameters are intentionally disabled for now.
        # They are kept only as comments to document the future extension:
        # embedding_index_path, embedding_threshold, embedding_margin,
        # embedding_top_k, embedding_device, embedding_batch_size.
        # -----------------------------------------------------------------
    ):
        self.ontology = ontology or Ontology.from_json(ontology_path)
        self.deterministic = DeterministicMatcher(
            self.ontology,
            deterministic_threshold,
            deterministic_margin,
        )
        self.profile_sample_size = profile_sample_size

    def _unknown_match(self, column: str) -> SemanticMatch:
        """Return an UNKNOWN result when deterministic matching is inconclusive."""
        return SemanticMatch(
            column=column,
            concept_id=UNKNOWN,
            confidence=0.0,
            method="deterministic",
            status="UNKNOWN",
            reason=(
                "No sufficiently confident deterministic ontology match. "
                "NLP/SapBERT inference is currently disabled."
            ),
        )

    def infer(self, data: pd.DataFrame) -> SemanticSchema:
        profiles = profile_dataset(data, self.profile_sample_size)
        matches: Dict[str, SemanticMatch] = {}

        # -----------------------------------------------------------------
        # Stage 1: deterministic ontology inference.
        # -----------------------------------------------------------------
        for p in profiles:
            match = self.deterministic.match(p)

            if match is None:
                # ---------------------------------------------------------
                # Stage 2: NLP / SapBERT fallback -- DISABLED FOR NOW.
                # Previously, unresolved columns were encoded with
                # generate_embeddings() and compared with
                # all_embed_ontology.json. For now they remain UNKNOWN.
                # ---------------------------------------------------------
                matches[p.name] = self._unknown_match(p.name)
            else:
                matches[p.name] = match

        return SemanticSchema(
            matches=matches,
            ontology_name=self.ontology.name,
            ontology_version=self.ontology.version,
            # NLP / SapBERT disabled.
            embedding_model=None,
        )

    def infer_columns(
        self,
        column_names: Sequence[str],
    ) -> SemanticSchema:
        """Infer from column names only using deterministic matching."""
        matches: Dict[str, SemanticMatch] = {}

        for name in column_names:
            name = str(name)
            p = ColumnProfile(
                name=name,
                normalized_name=normalize_text(name),
                dtype="unknown",
                numeric=False,
                boolean=False,
                datetime=False,
                non_null=0,
                unique=0,
                missing=0,
            )

            match = self.deterministic.match(p)

            if match is None:
                # NLP / SapBERT fallback intentionally disabled.
                matches[name] = self._unknown_match(name)
            else:
                matches[name] = match

        return SemanticSchema(
            matches=matches,
            ontology_name=self.ontology.name,
            ontology_version=self.ontology.version,
            embedding_model=None,
        )


class SemanticDataset:
    """Container connecting a dataframe with its inferred semantic schema."""

    def __init__(self, data: pd.DataFrame, schema: SemanticSchema):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data
        self.schema = schema
        self.column_to_concept = schema.known_columns()
        self.concept_to_columns: Dict[str, List[str]] = {}
        for col, concept in self.column_to_concept.items():
            self.concept_to_columns.setdefault(concept, []).append(col)

    def columns_for(self, concept_id: str) -> List[str]:
        return list(self.concept_to_columns.get(concept_id, []))

    def unknown_columns(self) -> List[str]:
        return self.schema.unknown_columns()


def infer_semantics(data: pd.DataFrame, **kwargs: Any) -> SemanticSchema:
    """Convenience function for one-shot inference."""
    return SemanticInferencer(**kwargs).infer(data)


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> Ontology:
    return Ontology.from_json(path)


# ---------------------------------------------------------------------------
# NLP / SapBERT helper -- DISABLED FOR NOW
# ---------------------------------------------------------------------------
# def load_embedding_index(...):
#     ...
# ---------------------------------------------------------------------------



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Infer CVD dataset column semantics using deterministic "
            "ontology matching only."
        )
    )
    parser.add_argument("dataset", help="CSV or XLSX dataset")
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument(
        "--deterministic-threshold",
        type=float,
        default=DEFAULT_DETERMINISTIC_THRESHOLD,
    )
    parser.add_argument(
        "--deterministic-margin",
        type=float,
        default=DEFAULT_DETERMINISTIC_MARGIN,
    )

    # ---------------------------------------------------------------------
    # NLP / SapBERT CLI options are intentionally disabled for now.
    # ---------------------------------------------------------------------
    # parser.add_argument("--embeddings", ...)
    # parser.add_argument("--embedding-threshold", ...)
    # parser.add_argument("--embedding-margin", ...)
    # parser.add_argument("--device", ...)
    # parser.add_argument("--batch-size", ...)

    args = parser.parse_args()
    path = Path(args.dataset)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Dataset must be CSV, XLSX, or XLS.")

    engine = SemanticInferencer(
        ontology_path=args.ontology,
        deterministic_threshold=args.deterministic_threshold,
        deterministic_margin=args.deterministic_margin,
    )

    schema = engine.infer(df)

    print(schema.to_dataframe().to_string(index=False))
    print(f"\nKnown: {len(schema.known_columns())}")
    print(f"Unknown: {len(schema.unknown_columns())}")
