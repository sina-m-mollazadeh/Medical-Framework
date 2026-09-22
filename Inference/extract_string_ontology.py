"""
extract_string_ontology.py

Build a persistent embedding index for all ontology aliases using
SapBERT-XLMR-large from generate_embedding.py.

Usage:
    python extract_string_ontology.py
    python extract_string_ontology.py --ontology cvd_ontology.json --output all_embed_ontology.json

The generated JSON is intended to be loaded by semantic.py later. The
ontology is encoded once; future datasets only encode unresolved column names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from generate_embedding import (
    MODEL_NAME,
    DEFAULT_BATCH_SIZE,
    SapBERTEmbedder,
)


DEFAULT_ONTOLOGY_PATH = "cvd_ontology.json"
DEFAULT_OUTPUT_PATH = "all_embed_ontology.json"


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected '{path}' to contain a JSON object.")
    return data


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dedup_key(text: str) -> str:
    """Stable key for exact semantic-string deduplication."""
    return " ".join(text.strip().split()).casefold()


def clean_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip().split())
    return value or None


def extract_alias_records(ontology: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract concept aliases while preserving concept/language provenance.

    Only aliases are embedded. Structural fields such as category, data_type,
    unit, and concept IDs are not themselves column-name strings and therefore
    are not sent to the biomedical encoder.
    """
    concepts = ontology.get("concepts")
    if not isinstance(concepts, dict):
        raise ValueError("Ontology must contain a 'concepts' object.")

    records: List[Dict[str, str]] = []

    for concept_id, concept in concepts.items():
        if not isinstance(concept, dict):
            raise ValueError(f"Concept '{concept_id}' must be an object.")

        aliases = concept.get("aliases", {})
        if aliases is None:
            continue
        if not isinstance(aliases, dict):
            raise ValueError(f"Aliases for '{concept_id}' must be an object.")

        for language, values in aliases.items():
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                raise ValueError(
                    f"Aliases for '{concept_id}' / '{language}' must be a list."
                )

            for value in values:
                value = clean_string(value)
                if value is not None:
                    records.append({
                        "concept_id": str(concept_id),
                        "language": str(language),
                        "text": value,
                        "source": "alias",
                    })

    return records


def deduplicate_records(records: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Keep one embedding per unique text while retaining every concept and
    language associated with that text.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for record in records:
        key = dedup_key(record["text"])
        if key not in grouped:
            grouped[key] = {
                "text": record["text"],
                "concepts": [],
                "languages": [],
                "sources": [],
            }
            order.append(key)

        item = grouped[key]
        if record["concept_id"] not in item["concepts"]:
            item["concepts"].append(record["concept_id"])
        if record["language"] not in item["languages"]:
            item["languages"].append(record["language"])
        if record["source"] not in item["sources"]:
            item["sources"].append(record["source"])

    return [grouped[key] for key in order]


def build_embedding_index(
    ontology: Dict[str, Any],
    embedder: SapBERTEmbedder,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Extract aliases and encode each unique ontology string once."""
    raw_records = extract_alias_records(ontology)
    unique_records = deduplicate_records(raw_records)

    if not unique_records:
        raise ValueError("No ontology aliases were found to embed.")

    strings = [item["text"] for item in unique_records]

    print(f"Raw alias occurrences : {len(raw_records):,}")
    print(f"Unique strings        : {len(strings):,}")
    print(f"Encoder               : {embedder.model_name}")
    print(f"Device                : {embedder.device}")

    embeddings = embedder.encode(
        strings,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress,
    )

    if len(embeddings) != len(unique_records):
        raise RuntimeError(
            "Embedding count does not match ontology-string count."
        )
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2-D embeddings, got {embeddings.shape}.")

    dimension = int(embeddings.shape[1])
    embedding_records = []

    for record, vector in zip(unique_records, embeddings):
        embedding_records.append({
            "text": record["text"],
            "concepts": record["concepts"],
            "languages": record["languages"],
            "sources": record["sources"],
            "embedding": vector.tolist(),
        })

    return {
        "embedding_index_name": "multilingual_cardiovascular_medical_ontology_embeddings",
        "embedding_index_version": "1.0.0",
        "ontology": {
            "ontology_name": ontology.get("ontology_name"),
            "version": ontology.get("version"),
            "languages": ontology.get("languages", {}),
        },
        "encoder": {
            "model_name": embedder.model_name,
            "model_type": "SapBERT-XLMR-large",
            "embedding_dimension": dimension,
            "normalized": bool(normalize),
            "similarity_recommended": "cosine",
        },
        "statistics": {
            "raw_alias_count": len(raw_records),
            "unique_string_count": len(unique_records),
            "embedding_count": len(embedding_records),
            "embedding_dimension": dimension,
        },
        "embeddings": embedding_records,
    }


def validate_index(index: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate the generated JSON before it is written to disk."""
    errors: List[str] = []

    for key in (
        "embedding_index_name",
        "embedding_index_version",
        "ontology",
        "encoder",
        "statistics",
        "embeddings",
    ):
        if key not in index:
            errors.append(f"Missing top-level key: {key}")

    records = index.get("embeddings")
    if not isinstance(records, list):
        errors.append("'embeddings' must be a list.")
        return False, errors

    dimension = index.get("encoder", {}).get("embedding_dimension")
    seen = set()

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"Record {i} is not an object.")
            continue

        for key in ("text", "concepts", "languages", "sources", "embedding"):
            if key not in record:
                errors.append(f"Record {i} is missing '{key}'.")

        text = record.get("text")
        if isinstance(text, str):
            key = dedup_key(text)
            if key in seen:
                errors.append(f"Duplicate text in record {i}: {text!r}")
            seen.add(key)

        vector = record.get("embedding")
        if not isinstance(vector, list):
            errors.append(f"Record {i} embedding is not a list.")
        elif isinstance(dimension, int) and len(vector) != dimension:
            errors.append(
                f"Record {i}: expected dimension {dimension}, got {len(vector)}."
            )

    declared = index.get("statistics", {}).get("embedding_count")
    if declared != len(records):
        errors.append("statistics.embedding_count does not match embeddings.")

    return not errors, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the persistent SapBERT-XLMR-large ontology embedding index."
    )
    parser.add_argument(
        "--ontology",
        default=DEFAULT_ONTOLOGY_PATH,
        help=f"Ontology JSON path (default: {DEFAULT_ONTOLOGY_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Encoding batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Device; default is CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize vectors. Normalization is recommended for cosine similarity.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the encoder progress bar.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    print("=" * 72)
    print("CVD Ontology Embedding Extraction")
    print("=" * 72)
    print(f"Ontology: {args.ontology}")
    print(f"Output  : {args.output}")
    print(f"Model   : {MODEL_NAME}")
    print()

    ontology = load_json(args.ontology)

    concepts = ontology.get("concepts", {})
    if not isinstance(concepts, dict):
        raise ValueError("Ontology 'concepts' must be an object.")

    print(f"Ontology name    : {ontology.get('ontology_name', '<unknown>')}")
    print(f"Ontology version : {ontology.get('version', '<unknown>')}")
    print(f"Concept count    : {len(concepts):,}")
    print()

    print("Loading SapBERT-XLMR-large...")
    embedder = SapBERTEmbedder(
        model_name=MODEL_NAME,
        device=args.device,
    )

    index = build_embedding_index(
        ontology=ontology,
        embedder=embedder,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        show_progress=not args.no_progress,
    )

    valid, errors = validate_index(index)
    if not valid:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError("Invalid embedding index; output was not written.")

    save_json(index, args.output)

    print()
    print("=" * 72)
    print("Completed successfully")
    print("=" * 72)
    print(f"Unique strings : {index['statistics']['unique_string_count']:,}")
    print(f"Dimension      : {index['statistics']['embedding_dimension']:,}")
    print(f"Output         : {args.output}")


if __name__ == "__main__":
    main()
