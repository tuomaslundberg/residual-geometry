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
        n_iterations: int = 50,
        min_accuracy: float | None = None,
        classifier: str = "logistic",
        stop_at_chance: bool = True,
        dev_fraction: float = 0.2,
        tolerance: float = 0.01,
        seed: int = 0,
    ) -> None:
        """Stopping rule (Ravfogel et al. 2020): hold out ``dev_fraction`` of the
        documents and stop before removing a direction whose classifier scores no
        better on them than guessing the majority class plus ``tolerance``
        (``min_accuracy`` replaces that threshold if given). ``n_iterations`` is
        only a cap. ``stop_at_chance=False`` restores the old behaviour: every
        iteration removes a direction, trained and judged on all documents.
        """
        if classifier != "logistic":
            raise NotImplementedError("Only 'logistic' is currently supported.")
        self.n_iterations = n_iterations
        self.min_accuracy = min_accuracy
        self.stop_at_chance = stop_at_chance
        self.dev_fraction = dev_fraction
        self.tolerance = tolerance
        self.seed = seed
        self._P: np.ndarray | None = None
        self._n_directions_removed: int = 0
        self._directions: list[np.ndarray] = []
        self.dev_accuracies: list[float] = []
        self.stop_threshold: float | None = None

    @property
    def projection_matrix(self) -> np.ndarray:
        if self._P is None:
            raise RuntimeError("Call fit() before accessing projection_matrix.")
        return self._P

    @property
    def directions(self) -> np.ndarray:
        """Unit vectors of the removed classifier directions, in removal order, (k, d)."""
        if self._P is None:
            raise RuntimeError("Call fit() before accessing directions.")
        return np.array(self._directions).reshape(-1, self._P.shape[0])

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
        self._directions = []
        self.dev_accuracies = []

        if self.stop_at_chance:
            from sklearn.model_selection import train_test_split

            train, dev = train_test_split(
                np.arange(len(y)), test_size=self.dev_fraction, stratify=y,
                random_state=self.seed)
            majority = np.bincount(y[dev]).max() / len(dev)
            self.stop_threshold = (self.min_accuracy if self.min_accuracy is not None
                                   else majority + self.tolerance)
        else:
            train = dev = np.arange(len(y))
            self.stop_threshold = self.min_accuracy

        for iteration in range(self.n_iterations):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X_iter[train], y[train])

            if self.stop_threshold is not None:
                acc = clf.score(X_iter[dev], y[dev])
                self.dev_accuracies.append(float(acc))
                if acc <= self.stop_threshold:
                    break

            W = clf.coef_  # (n_classes, dim)
            for w in W:
                norm_sq = float(np.dot(w, w))
                if norm_sq < 1e-10:
                    continue
                P_row = np.eye(dim) - np.outer(w, w) / norm_sq
                self._directions.append(w / np.sqrt(norm_sq))
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
