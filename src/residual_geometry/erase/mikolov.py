"""Mikolov linear projection — least-squares map between parallel embedding spaces."""

from __future__ import annotations

import numpy as np


class MikolovProjection:
    """Learn a linear translation matrix W such that fi_embs @ W ≈ en_embs.

    Fit via least-squares (Mikolov et al., 2013). After fitting, ``transform``
    maps Finnish embeddings into the English embedding space.

    Reference: https://arxiv.org/abs/1309.4168
    """

    def __init__(self) -> None:
        self._W: np.ndarray | None = None

    @property
    def translation_matrix(self) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("Call fit() before accessing translation_matrix.")
        return self._W

    def fit(self, fi_embs: np.ndarray, en_embs: np.ndarray) -> "MikolovProjection":
        """Fit W = argmin ||fi_embs @ W - en_embs||_F via least squares."""
        X = fi_embs.astype(np.float64)
        Y = en_embs.astype(np.float64)
        # lstsq solution: W = (X^T X)^{-1} X^T Y
        self._W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return self

    def transform(self, fi_embs: np.ndarray) -> np.ndarray:
        """Map FI embeddings into EN embedding space."""
        if self._W is None:
            raise RuntimeError("Call fit() before transform().")
        return (fi_embs.astype(np.float64) @ self._W).astype(np.float32)
