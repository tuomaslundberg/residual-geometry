"""Sentence embedding with frozen multilingual models."""

from __future__ import annotations

import numpy as np


class Embedder:
    """Encodes text with a frozen SentenceTransformer model.

    LaBSE (``sentence-transformers/LaBSE``) is the default.
    XLM-R variants and any other ST-compatible model work too.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

        if device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Return L2-normalised embeddings, shape (n, d)."""
        self._load()
        vecs = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


def load_encoder(model_name: str, device: str = "auto", batch_size: int = 64) -> Embedder:
    """Convenience constructor — returns a ready-to-use Embedder."""
    return Embedder(model_name=model_name, device=device, batch_size=batch_size)


def embed_batch(
    embedder: Embedder, texts: list[str], show_progress: bool = False
) -> np.ndarray:
    """Embed a list of texts, returning (n, d) float32 array."""
    return embedder.embed(texts, show_progress=show_progress)


def embed_pairs(
    embedder: Embedder,
    src_texts: list[str],
    tgt_texts: list[str],
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed parallel sentence pairs, returning (src_embs, tgt_embs)."""
    src = embedder.embed(src_texts, show_progress=show_progress)
    tgt = embedder.embed(tgt_texts, show_progress=show_progress)
    return src, tgt
