"""INLP — Iterative Nullspace Projection (Ravfogel et al., ACL 2020)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ._base import LanguageProjector


class INLPEraser(LanguageProjector):
    """Remove language signal by iteratively projecting out classifier directions.

    Reference: https://aclanthology.org/2020.acl-main.647/
    """

    def __init__(
        self,
        n_iterations: int = 20,
        min_accuracy: float | None = None,
        classifier: str = "logistic",
    ) -> None:
        if classifier != "logistic":
            raise NotImplementedError("Only 'logistic' is currently supported.")
        self.n_iterations = n_iterations
        self.min_accuracy = min_accuracy
        self._P: np.ndarray | None = None
        self._n_directions_removed: int = 0

    @property
    def projection_matrix(self) -> np.ndarray:
        if self._P is None:
            raise RuntimeError("Call fit() before accessing projection_matrix.")
        return self._P

    @property
    def n_directions_removed(self) -> int:
        return self._n_directions_removed

    def fit(self, X: np.ndarray, language_labels: np.ndarray) -> "INLPEraser":
        le = LabelEncoder()
        y = le.fit_transform(language_labels)

        dim = X.shape[1]
        P_composed = np.eye(dim, dtype=np.float64)
        X_iter = X.astype(np.float64)
        n_dirs = 0

        for iteration in range(self.n_iterations):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X_iter, y)

            if self.min_accuracy is not None:
                acc = clf.score(X_iter, y)
                if acc <= self.min_accuracy:
                    break

            W = clf.coef_  # (n_classes, dim)
            for w in W:
                norm_sq = float(np.dot(w, w))
                if norm_sq < 1e-10:
                    continue
                P_row = np.eye(dim) - np.outer(w, w) / norm_sq
                P_composed = P_row @ P_composed
                X_iter = X_iter @ P_row.T
                n_dirs += 1

        self._P = P_composed
        self._n_directions_removed = n_dirs
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._P is None:
            raise RuntimeError("Call fit() before transform().")
        return (X.astype(np.float64) @ self._P.T).astype(np.float32)
