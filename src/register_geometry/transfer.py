"""Stage 4: zero-shot diagnostic classifier transfer.

Train a linear register classifier on projected embeddings from language A,
evaluate on language B with no further training. Reports macro-F1 and a
confusion matrix.

RQ3: functional isomorphism. If register subspaces are isomorphic post-projection,
a classifier trained in one language should transfer near-perfectly to the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder


@dataclass
class TransferResult:
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: np.ndarray
    classes: list[str]

    def summary(self) -> dict:
        return {
            "macro_f1": round(self.macro_f1, 4),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
        }


class LinearRegisterClassifier:
    """Logistic regression classifier over register-class embeddings."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self._clf: LogisticRegression | None = None
        self._le = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegisterClassifier":
        y_enc = self._le.fit_transform(y)
        self._clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs", multi_class="auto"
        )
        self._clf.fit(X, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit() first.")
        y_enc = self._clf.predict(X)
        return self._le.inverse_transform(y_enc)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> TransferResult:
        y_pred = self.predict(X)
        classes = list(self._le.classes_)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        report = classification_report(
            y_true, y_pred, labels=classes, output_dict=True, zero_division=0
        )
        per_class = {cls: report[cls]["f1-score"] for cls in classes if cls in report}
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        return TransferResult(
            macro_f1=macro_f1,
            per_class_f1=per_class,
            confusion_matrix=cm,
            classes=classes,
        )


def cross_lingual_transfer(
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_tgt: np.ndarray,
    y_tgt: np.ndarray,
    C: float = 1.0,
) -> TransferResult:
    """Train on source language embeddings, evaluate on target language."""
    clf = LinearRegisterClassifier(C=C)
    clf.fit(X_src, y_src)
    return clf.evaluate(X_tgt, y_tgt)
