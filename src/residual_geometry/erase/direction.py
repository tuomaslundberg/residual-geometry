"""Control and translation-based erasers.

GlobalCenteringEraser moves the origin without removing anything: it subtracts the
pooled mean of all documents, so the Finnish-English difference stays. Set beside
mean centering (which does both) it separates "move the origin" from "remove the
difference"; LEACE does the second without the first.

DirectionEraser projects out one fixed direction. Built from translation pairs, the
direction is the average Finnish-minus-English difference of sentences that say the
same thing, i.e. a pure-language direction; it ignores the labels it is fitted on.
"""

from __future__ import annotations

import numpy as np

from ._base import LanguageProjector


class GlobalCenteringEraser(LanguageProjector):
    """Subtract the pooled mean of all documents (a control: removes no difference)."""

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None

    def fit(self, X: np.ndarray, language_labels: np.ndarray | None = None) -> "GlobalCenteringEraser":
        self._mean = X.astype(np.float64).mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Call fit() before transform().")
        return (X.astype(np.float64) - self._mean).astype(np.float32)


class DirectionEraser(LanguageProjector):
    """Orthogonally project out one fixed direction: x - (x . d) d."""

    def __init__(self, direction: np.ndarray) -> None:
        d = np.asarray(direction, dtype=np.float64)
        self.direction = d / np.linalg.norm(d)

    @classmethod
    def from_pairs(cls, fi_embs: np.ndarray, en_embs: np.ndarray) -> "DirectionEraser":
        """Direction = mean of (Finnish - English) over aligned translation pairs."""
        return cls((fi_embs.astype(np.float64) - en_embs.astype(np.float64)).mean(axis=0))

    def fit(self, X: np.ndarray, language_labels: np.ndarray | None = None) -> "DirectionEraser":
        return self  # the direction is fixed at construction

    def transform(self, X: np.ndarray) -> np.ndarray:
        X64 = X.astype(np.float64)
        return (X64 - np.outer(X64 @ self.direction, self.direction)).astype(np.float32)
