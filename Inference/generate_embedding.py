"""
generate_embedding.py

Generate biomedical text embeddings using SapBERT-XLMR-large.

Model:
    cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large

The model is the multilingual SapBERT-XLMR-large model trained on UMLS.
Its model card recommends using the [CLS] representation as the embedding.

This module is intentionally independent of the CVD ontology. It only:
    1. accepts a list of strings,
    2. encodes them with SapBERT-XLMR-large,
    3. returns the embeddings in the same order.

Required packages:
    pip install torch transformers numpy tqdm
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


MODEL_NAME = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large"
DEFAULT_MAX_LENGTH = 25
DEFAULT_BATCH_SIZE = 128


class SapBERTEmbedder:
    """Reusable SapBERT-XLMR-large encoder."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible GPU is available."
            )

        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        strings: Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode strings and return an array in the same order as the input."""
        if strings is None:
            raise ValueError("strings cannot be None.")

        strings = list(strings)

        if len(strings) == 0:
            return np.empty((0, 1024), dtype=np.float32)

        cleaned_strings: List[str] = []
        for i, value in enumerate(strings):
            if not isinstance(value, str):
                raise TypeError(
                    f"Item at index {i} is {type(value).__name__}; "
                    "every item must be a string."
                )
            value = value.strip()
            if not value:
                raise ValueError(f"Item at index {i} is empty.")
            cleaned_strings.append(value)

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than zero.")

        embeddings = []
        starts = range(0, len(cleaned_strings), batch_size)

        if show_progress:
            starts = tqdm(
                starts,
                desc="Generating SapBERT embeddings",
                total=(len(cleaned_strings) + batch_size - 1) // batch_size,
            )

        for start in starts:
            batch = cleaned_strings[start : start + batch_size]

            tokens = self.tokenizer(
                batch,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}

            outputs = self.model(**tokens)

            # SapBERT-XLMR-large's model card recommends the [CLS] representation.
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

            if normalize:
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, p=2, dim=1
                )

            embeddings.append(
                batch_embeddings.detach().cpu().numpy().astype(np.float32)
            )

        return np.concatenate(embeddings, axis=0)


_default_embedder: Optional[SapBERTEmbedder] = None


def generate_embeddings(
    strings: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = False,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Generate embeddings using one cached SapBERT-XLMR-large model instance."""
    global _default_embedder

    if _default_embedder is None:
        _default_embedder = SapBERTEmbedder(device=device)

    return _default_embedder.encode(
        strings,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress,
    )


def reset_model() -> None:
    """Release the cached model from this module."""
    global _default_embedder

    if _default_embedder is not None:
        del _default_embedder
        _default_embedder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    """Simple command-line test interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SapBERT-XLMR-large embeddings for strings."
    )
    parser.add_argument("strings", nargs="+", help="Strings to encode.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    embedder = SapBERTEmbedder(device=args.device, max_length=args.max_length)
    embeddings = embedder.encode(
        args.strings,
        batch_size=args.batch_size,
        normalize=args.normalize,
        show_progress=True,
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Number of strings: {len(args.strings)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(embeddings)


if __name__ == "__main__":
    main()
