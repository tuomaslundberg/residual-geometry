"""LEACE — Linear Concept Erasure (Belrose et al., NeurIPS 2023).

Requires: pip install concept-erasure
Reference: https://arxiv.org/abs/2306.03819
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ._base import LanguageProjector


class LEACEEraser(LanguageProjector):
    """Closed-form oblique projection that removes a linear concept."""

    def __init__(self, rank: int | None = None) -> None:
        self.rank = rank
        self._eraser = None

    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "LEACEEraser":
        try:
            from concept_erasure import LeaceEraser  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "Install concept-erasure: pip install concept-erasure"
            ) from e

        import torch

        le = LabelEncoder()
        y = le.fit_transform(language_labels)
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y)
        self._eraser = LeaceEraser.fit(X_t, y_t)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._eraser is None:
            raise RuntimeError("Call fit() before transform().")
        import torch

        X_t = torch.from_numpy(X.astype(np.float32))
        return self._eraser(X_t).numpy()
