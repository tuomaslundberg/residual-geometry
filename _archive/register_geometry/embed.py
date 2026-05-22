"""Stage 1: document embedding with a frozen sentence-transformer model."""

from __future__ import annotations

import numpy as np


class Embedder:
    """Encodes text documents with a frozen SentenceTransformer model.

    LaBSE is the baseline (``sentence-transformers/LaBSE``).
    Any model accepted by the ``sentence-transformers`` library works.
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
