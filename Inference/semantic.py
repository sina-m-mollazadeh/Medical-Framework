from __future__ import annotations

import json

import re

import unicodedata

from dataclasses import asdict, dataclass, field

from difflib import SequenceMatcher

from pathlib import Path

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# NLP / SapBERT second-stage inference
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_INDEX_PATH = (
    "/home/sinam/Python/AI/qenv/Medical-Framework/Inference/"
    "all_embed_ontology.json"
)
DEFAULT_EMBEDDING_MODEL = (
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large"
)
DEFAULT_EMBEDDING_THRESHOLD = 0.88
DEFAULT_EMBEDDING_MARGIN = 0.05
DEFAULT_EMBEDDING_TOP_K = 5
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_EMBEDDING_MAX_LENGTH = 64


# The Hugging Face imports are intentionally lazy. The deterministic stage
# remains usable even if the NLP dependencies are unavailable.


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

    s = re.sub(r"[_\-./\\]+", " ", s)

    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)

    s = re.sub(r"\s+", " ", s).strip()

    # Remove only trailing numeric suffixes such as age_1, age-2, age 3.
    # Internal numbers such as type2 or t1 are preserved.
    s = re.sub(r"\s+\d+$", "", s)

    return s



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



class EmbeddingIndex:
    """
    Loads precomputed ontology alias embeddings from JSON.

    The loader accepts the common formats produced by embedding extraction:
      * a list of records containing text/concept_id/language/embedding
      * a dict containing an ``embeddings`` list
      * a mapping from alias text to embedding vectors

    The index is never re-encoded. Only unresolved dataset column names are
    encoded at inference time.
    """

    def __init__(self, path: str | Path, ontology: Ontology):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Embedding index not found: {self.path}"
            )

        raw = load_json(self.path)
        self.records = self._parse_records(raw, ontology)

        if not self.records:
            raise ValueError(
                f"No embedding records could be read from {self.path}. "
                "Check the structure of all_embed_ontology.json."
            )

        matrix = np.asarray(
            [r["embedding"] for r in self.records],
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise ValueError(
                f"Embedding matrix must be 2-D, got shape {matrix.shape}."
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    @staticmethod
    def _looks_like_vector(value: Any) -> bool:
        if not isinstance(value, list) or not value:
            return False
        return all(isinstance(x, (int, float)) for x in value[:20])

    @classmethod
    def _parse_records(
        cls,
        raw: Any,
        ontology: Ontology,
    ) -> List[Dict[str, Any]]:
        """Normalize common JSON embedding layouts into flat records."""

        records: List[Dict[str, Any]] = []
        seen = set()

        def add_record(
            text: Optional[str],
            vector: Any,
            concept_id: Optional[str] = None,
            language: Optional[str] = None,
        ) -> None:
            if text is None or not cls._looks_like_vector(vector):
                return

            text = str(text)

            if concept_id is None:
                aliases = ontology.aliases.get(normalize_text(text), [])
                concept_ids = {a.concept_id for a in aliases}
                if len(concept_ids) == 1:
                    concept_id = next(iter(concept_ids))
                    language = language or aliases[0].language

            if concept_id is None:
                return

            # Prevent accidental duplicate records when a wrapper key and
            # an inner record expose the same embedding.
            key = (
                str(concept_id),
                text,
                language,
                tuple(float(x) for x in vector),
            )
            if key in seen:
                return
            seen.add(key)

            records.append({
                "text": text,
                "concept_id": str(concept_id),
                "language": language,
                "embedding": vector,
            })

        def walk(
            node: Any,
            hint_text: Optional[str] = None,
            hint_concept: Optional[str] = None,
            hint_language: Optional[str] = None,
        ) -> None:
            if isinstance(node, list):
                # A raw numeric vector can be associated with a parent alias.
                if cls._looks_like_vector(node):
                    add_record(
                        hint_text, node, hint_concept, hint_language
                    )
                    return

                for item in node:
                    walk(item, hint_text, hint_concept, hint_language)
                return

            if not isinstance(node, dict):
                return

            text = (
                node.get("text")
                or node.get("alias")
                or node.get("string")
                or node.get("term")
                or node.get("sentence")
                or hint_text
            )
            concept_id = (
                node.get("concept_id")
                or node.get("concept")
                or node.get("id")
                or hint_concept
            )
            language = (
                node.get("language")
                or node.get("lang")
                or hint_language
            )

            vector = (
                node.get("embedding")
                or node.get("vector")
                or node.get("values")
            )
            if vector is not None:
                add_record(text, vector, concept_id, language)
                return

            # Otherwise recursively inspect wrappers such as:
            # {"metadata": ..., "embeddings": [...]} or
            # {"concept_id": ..., "aliases": {"en": {...}}}.
            for key, value in node.items():
                if key in {"metadata", "config", "model", "ontology"}:
                    continue

                child_text = text
                child_concept = concept_id
                child_language = language

                # In mapping-style JSON, a key may itself be the alias or
                # concept identifier. We use it as a hint and let ontology
                # aliases resolve the concept when possible.
                if isinstance(value, (dict, list)):
                    if key in ontology.concepts:
                        child_concept = key
                    elif key in ontology.languages:
                        child_language = key
                    elif not child_text:
                        child_text = key

                walk(value, child_text, child_concept, child_language)

        walk(raw)
        return records

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = DEFAULT_EMBEDDING_TOP_K,
    ) -> List[Dict[str, Any]]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        if query.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match ontology embedding "
                f"dimension: {query.shape[0]} != {self.matrix.shape[1]}"
            )

        scores = self.matrix @ query
        indices = np.argsort(-scores)[:top_k]

        return [
            {**self.records[int(i)], "score": float(scores[int(i)])}
            for i in indices
        ]


class SapBERTEncoder:
    """Lazy SapBERT-XLMR encoder used only for unresolved columns."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = DEFAULT_EMBEDDING_DEVICE,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_length: int = DEFAULT_EMBEDDING_MAX_LENGTH,
        local_files_only: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "SapBERT inference requires torch and transformers. "
                "Install them in the active environment."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        outputs = []

        for start in range(0, len(texts), self.batch_size):
            batch = [str(x) for x in texts[start:start + self.batch_size]]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                model_output = self.model(**inputs)

            token_embeddings = model_output.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts

            embeddings = self.torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )

            outputs.append(embeddings.cpu().numpy())

        if not outputs:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(outputs).astype(np.float32)


class SemanticInferencer:
    """
    Two-stage semantic inference pipeline.

    Stage 1: deterministic ontology matching.
    Stage 2: SapBERT similarity against precomputed ontology embeddings,
             only for columns unresolved by Stage 1.
    """

    def __init__(
        self,
        ontology: Optional[Ontology] = None,
        ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
        deterministic_threshold: float = DEFAULT_DETERMINISTIC_THRESHOLD,
        deterministic_margin: float = DEFAULT_DETERMINISTIC_MARGIN,
        profile_sample_size: int = 20,
        embedding_index_path: str | Path = DEFAULT_EMBEDDING_INDEX_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
        embedding_margin: float = DEFAULT_EMBEDDING_MARGIN,
        embedding_top_k: int = DEFAULT_EMBEDDING_TOP_K,
        embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        embedding_max_length: int = DEFAULT_EMBEDDING_MAX_LENGTH,
        embedding_local_files_only: bool = True,
    ):
        self.ontology = ontology or Ontology.from_json(ontology_path)
        self.deterministic = DeterministicMatcher(
            self.ontology,
            deterministic_threshold,
            deterministic_margin,
        )
        self.profile_sample_size = profile_sample_size

        self.embedding_threshold = embedding_threshold
        self.embedding_margin = embedding_margin
        self.embedding_top_k = embedding_top_k

        # Load the precomputed ontology vectors. This does NOT load the
        # Hugging Face model. The model is loaded lazily only if a column
        # reaches Stage 2.
        self.embedding_index = EmbeddingIndex(
            embedding_index_path,
            self.ontology,
        )

        self._embedding_model_name = embedding_model
        self._embedding_device = embedding_device
        self._embedding_batch_size = embedding_batch_size
        self._embedding_max_length = embedding_max_length
        self._embedding_local_files_only = embedding_local_files_only
        self._encoder: Optional[SapBERTEncoder] = None

    @property
    def embedding_encoder(self) -> SapBERTEncoder:
        if self._encoder is None:
            self._encoder = SapBERTEncoder(
                model_name=self._embedding_model_name,
                device=self._embedding_device,
                batch_size=self._embedding_batch_size,
                max_length=self._embedding_max_length,
                local_files_only=self._embedding_local_files_only,
            )
        return self._encoder

    def _unknown_match(
        self,
        column: str,
        reason: str,
        candidates: Optional[List[Candidate]] = None,
    ) -> SemanticMatch:
        return SemanticMatch(
            column=column,
            concept_id=UNKNOWN,
            confidence=0.0,
            method="deterministic+sapbert",
            status="UNKNOWN",
            reason=reason,
            candidates=candidates or [],
        )

    def _embedding_match(
        self,
        column: str,
        query_embedding: np.ndarray,
    ) -> SemanticMatch:
        results = self.embedding_index.search(
            query_embedding,
            top_k=self.embedding_top_k,
        )

        if not results:
            return self._unknown_match(
                column,
                "No usable SapBERT ontology embedding candidates were found.",
            )

        # Multiple aliases can represent the same concept. For concept-level
        # inference, keep the highest-scoring alias for each concept.
        best_by_concept: Dict[str, Dict[str, Any]] = {}
        for item in results:
            cid = item["concept_id"]
            if cid not in best_by_concept or item["score"] > best_by_concept[cid]["score"]:
                best_by_concept[cid] = item

        ranked = sorted(
            best_by_concept.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        best_score = float(best["score"])
        second_score = float(second["score"]) if second else None
        margin = best_score - second_score if second else best_score

        candidates = [
            Candidate(
                str(x["concept_id"]),
                float(x["score"]),
                str(x.get("text", "")),
                x.get("language"),
                "sapbert",
            )
            for x in ranked[:5]
        ]

        if best_score < self.embedding_threshold:
            return self._unknown_match(
                column,
                (
                    f"Best SapBERT similarity {best_score:.4f} is below "
                    f"the threshold {self.embedding_threshold:.4f}."
                ),
                candidates,
            )

        if second is not None and margin < self.embedding_margin:
            return self._unknown_match(
                column,
                (
                    f"SapBERT margin {margin:.4f} is below the required "
                    f"margin {self.embedding_margin:.4f}."
                ),
                candidates,
            )

        concept = self.ontology.concepts.get(str(best["concept_id"]))
        if concept is None:
            return self._unknown_match(
                column,
                f"Embedding matched unknown ontology concept '{best['concept_id']}'.",
                candidates,
            )

        return SemanticMatch(
            column=column,
            concept_id=str(best["concept_id"]),
            confidence=best_score,
            method="sapbert",
            status="KNOWN",
            matched_text=best.get("text"),
            language=best.get("language"),
            category=concept.category,
            data_type=concept.data_type,
            unit=concept.unit,
            reason=(
                "High-confidence SapBERT similarity match against "
                "precomputed ontology embeddings."
            ),
            second_best_concept=(
                str(second["concept_id"]) if second else None
            ),
            second_best_score=second_score,
            margin=float(margin),
            candidates=candidates,
        )

    def infer(self, data: pd.DataFrame) -> SemanticSchema:
        profiles = profile_dataset(data, self.profile_sample_size)
        matches: Dict[str, SemanticMatch] = {}

        # Stage 1: deterministic inference.
        unresolved: List[ColumnProfile] = []

        for p in profiles:
            match = self.deterministic.match(p)
            if match is None:
                unresolved.append(p)
            else:
                matches[p.name] = match

        # Stage 2: encode ONLY unresolved column names.
        if unresolved:
            embeddings = self.embedding_encoder.encode(
                [p.name for p in unresolved]
            )

            for p, embedding in zip(unresolved, embeddings):
                matches[p.name] = self._embedding_match(
                    p.name,
                    embedding,
                )

        return SemanticSchema(
            matches=matches,
            ontology_name=self.ontology.name,
            ontology_version=self.ontology.version,
            embedding_model=self._embedding_model_name,
        )

    def infer_columns(
        self,
        column_names: Sequence[str],
    ) -> SemanticSchema:
        """Infer from column names using deterministic then SapBERT matching."""
        profiles = []
        for name in column_names:
            name = str(name)
            profiles.append(
                ColumnProfile(
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
            )

        matches: Dict[str, SemanticMatch] = {}
        unresolved: List[ColumnProfile] = []

        for p in profiles:
            match = self.deterministic.match(p)
            if match is None:
                unresolved.append(p)
            else:
                matches[p.name] = match

        if unresolved:
            embeddings = self.embedding_encoder.encode(
                [p.name for p in unresolved]
            )
            for p, embedding in zip(unresolved, embeddings):
                matches[p.name] = self._embedding_match(
                    p.name,
                    embedding,
                )

        return SemanticSchema(
            matches=matches,
            ontology_name=self.ontology.name,
            ontology_version=self.ontology.version,
            embedding_model=self._embedding_model_name,
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
            "ontology matching followed by SapBERT fallback."
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

    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDING_INDEX_PATH)
    parser.add_argument("--embedding-threshold", type=float, default=DEFAULT_EMBEDDING_THRESHOLD)
    parser.add_argument("--embedding-margin", type=float, default=DEFAULT_EMBEDDING_MARGIN)
    parser.add_argument("--embedding-top-k", type=int, default=DEFAULT_EMBEDDING_TOP_K)
    parser.add_argument("--device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face to download missing model files.",
    )

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
        embedding_index_path=args.embeddings,
        embedding_threshold=args.embedding_threshold,
        embedding_margin=args.embedding_margin,
        embedding_top_k=args.embedding_top_k,
        embedding_device=args.device,
        embedding_batch_size=args.batch_size,
        embedding_local_files_only=not args.allow_download,
    )

    schema = engine.infer(df)

    print(schema.to_dataframe().to_string(index=False))

    print(f"\nKnown: {len(schema.known_columns())}")

    print(f"Unknown: {len(schema.unknown_columns())}")