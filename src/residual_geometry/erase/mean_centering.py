"""Mean-centering baseline — subtract per-language mean embedding."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ._base import LanguageProjector


class MeanCenteringEraser(LanguageProjector):
    """Subtract each language's mean vector, mapping all languages to a shared origin.

    A minimal, parameter-free baseline for language-signal removal.
    """

    def __init__(self) -> None:
        self._means: dict[str, np.ndarray] = {}
        self._global_mean: np.ndarray | None = None

    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "MeanCenteringEraser":
        self._means = {}
        for lang in np.unique(language_labels):
            mask = language_labels == lang
            self._means[str(lang)] = X[mask].mean(axis=0).astype(np.float64)
        self._global_mean = X.mean(axis=0).astype(np.float64)
        return self

    def transform(self, X: np.ndarray, language_labels: np.ndarray | None = None) -> np.ndarray:
        """Subtract per-language mean. Falls back to global mean if labels not given."""
        if not self._means:
            raise RuntimeError("Call fit() before transform().")
        X_out = X.astype(np.float64)
        if language_labels is not None:
            for lang, mean in self._means.items():
                mask = language_labels == lang
                if mask.any():
                    X_out[mask] -= mean
        else:
            assert self._global_mean is not None
            X_out -= self._global_mean
        return X_out.astype(np.float32)
