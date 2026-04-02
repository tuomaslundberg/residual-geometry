"""Stage 2: language-signal removal via linear projection.

All projectors share the ``LanguageProjector`` interface so methods are
swappable via config (``projection.method``).

Implemented:
  - IdentityProjector  — no-op baseline
  - INLPProjector      — Ravfogel et al. (2020), iterative nullspace projection

Stubbed (requires optional dep ``concept-erasure``):
  - LEACEProjector     — Belrose et al. (2023), minimal-norm concept erasure
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class LanguageProjector(ABC):
    """Abstract interface for language-signal removal."""

    @abstractmethod
    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "LanguageProjector":
        """Fit projector on embeddings and string language labels."""
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project embeddings into language-neutral subspace."""
        ...

    def fit_transform(
        self, X: np.ndarray, language_labels: np.ndarray
    ) -> np.ndarray:
        return self.fit(X, language_labels).transform(X)


class IdentityProjector(LanguageProjector):
    """No-op projector. Used as the pre-projection baseline."""

    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "IdentityProjector":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.copy()


class INLPProjector(LanguageProjector):
    """Iterative Nullspace Projection (Ravfogel et al., ACL 2020).

    Algorithm:
      1. Fit a linear language-ID classifier; extract its weight direction w.
      2. Compute the orthogonal projection P = I - ww^T / ||w||^2.
      3. Apply P; repeat for ``n_iterations`` rounds.
      4. ``transform`` applies the composed projection matrix.

    Reference: https://aclanthology.org/2020.acl-main.647/
    """

    def __init__(self, n_iterations: int = 20, classifier: str = "logistic") -> None:
        if classifier != "logistic":
            raise NotImplementedError("Only 'logistic' is currently supported.")
        self.n_iterations = n_iterations
        self._P: np.ndarray | None = None  # composed projection matrix

    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "INLPProjector":
        le = LabelEncoder()
        y = le.fit_transform(language_labels)

        dim = X.shape[1]
        P_composed = np.eye(dim, dtype=np.float64)
        X_iter = X.astype(np.float64)

        for _ in range(self.n_iterations):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X_iter, y)

            # Weight matrix shape: (n_classes, dim). For binary, one row suffices;
            # for multiclass we project out all directions.
            W = clf.coef_  # (n_classes, dim)
            for w in W:
                norm_sq = float(np.dot(w, w))
                if norm_sq < 1e-10:
                    continue
                P_row = np.eye(dim) - np.outer(w, w) / norm_sq
                P_composed = P_row @ P_composed
                X_iter = X_iter @ P_row.T

        self._P = P_composed
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._P is None:
            raise RuntimeError("Call fit() before transform().")
        return (X.astype(np.float64) @ self._P.T).astype(np.float32)


class LEACEProjector(LanguageProjector):
    """Concept Erasure via LEACE (Belrose et al., 2023).

    Requires: ``pip install concept-erasure``
    Reference: https://arxiv.org/abs/2306.03819
    """

    def __init__(self, rank: int | None = None) -> None:
        self.rank = rank
        self._eraser = None

    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "LEACEProjector":
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


_REGISTRY: dict[str, type[LanguageProjector]] = {
    "none": IdentityProjector,
    "inlp": INLPProjector,
    "leace": LEACEProjector,
}


def build_projector(cfg: dict) -> LanguageProjector:
    """Instantiate a projector from a config dict (``projection`` subtree)."""
    method = cfg.get("method", "none")
    if method not in _REGISTRY:
        raise ValueError(f"Unknown projection method '{method}'. Choose from {list(_REGISTRY)}.")
    klass = _REGISTRY[method]
    method_cfg = cfg.get(method, {}) or {}
    return klass(**method_cfg)
