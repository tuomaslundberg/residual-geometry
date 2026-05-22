"""Abstract base and identity projector."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LanguageProjector(ABC):
    """Sklearn-style interface for language-signal removal."""

    @abstractmethod
    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "LanguageProjector":
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray, language_labels: np.ndarray) -> np.ndarray:
        return self.fit(X, language_labels).transform(X)


class IdentityProjector(LanguageProjector):
    """No-op baseline — returns a copy of the input unchanged."""

    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "IdentityProjector":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.copy()


_REGISTRY: dict[str, type[LanguageProjector]] = {}


def _register(name: str, cls: type[LanguageProjector]) -> None:
    _REGISTRY[name] = cls


def build_projector(cfg: dict) -> LanguageProjector:
    """Instantiate an eraser from a config dict (``projection`` subtree)."""
    from .inlp import INLPEraser
    from .leace import LEACEEraser
    from .mean_centering import MeanCenteringEraser

    registry: dict[str, type[LanguageProjector]] = {
        "none": IdentityProjector,
        "inlp": INLPEraser,
        "leace": LEACEEraser,
        "mean": MeanCenteringEraser,
    }
    method = cfg.get("method", "none")
    if method not in registry:
        raise ValueError(
            f"Unknown projection method '{method}'. Choose from {list(registry)}."
        )
    klass = registry[method]
    method_cfg = cfg.get(method, {}) or {}
    return klass(**method_cfg)
