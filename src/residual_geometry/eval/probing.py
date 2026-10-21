"""Linear probing for language and register identity, plus cross-lingual transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


@dataclass
class ProbeResult:
    """Held-out accuracy of a probe against the guessing level.

    ``chance_accuracy`` is the share of the largest class (always guessing it).
    ``above_chance`` = accuracy - chance_accuracy. This is *not* Hewitt & Liang's
    (2019) selectivity, which needs a control task over word types.
    """

    accuracy: float
    chance_accuracy: float
    above_chance: float

    @property
    def selectivity(self) -> float:
        """Deprecated name for ``above_chance`` (kept for the Phase 1 scripts)."""
        return self.above_chance


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


def make_probe(C: float = 1.0, scale: bool = False, Cs=None):
    """Logistic-regression probe, optionally standardised and with C chosen by inner CV.

    ``scale``: standardise each feature on the training data first. Needed for anisotropic
    encoders: unscaled, an L2-penalised probe underfits XLM-R badly (Finnish register 0.53
    vs 0.77 standardised; topic 0.16 vs 0.64), while LaBSE is barely affected.
    ``Cs``: a grid of C values chosen by 3-fold inner CV (LogisticRegressionCV), so each
    encoder and condition gets a fitting penalty instead of one fixed C.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = (LogisticRegressionCV(Cs=list(Cs), cv=3, max_iter=2000) if Cs
           else LogisticRegression(C=C, max_iter=1000, solver="lbfgs"))
    return make_pipeline(StandardScaler(), clf) if scale else clf


def linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    C: float = 1.0,
    seed: int = 42,
    scale: bool = False,
    Cs=None,
) -> float:
    """Cross-validated accuracy of a logistic regression probe."""
    scores = cross_val_score(make_probe(C, scale, Cs), X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def apply_eraser(eraser, X: np.ndarray, language_labels: np.ndarray) -> np.ndarray:
    """Transform with an eraser; mean centering needs each document's language."""
    from residual_geometry.erase import MeanCenteringEraser

    if isinstance(eraser, MeanCenteringEraser):
        return eraser.transform(X, language_labels)
    return eraser.transform(X)


def majority_share(y: np.ndarray) -> float:
    """Accuracy of always guessing the largest class."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def language_probe(
    X: np.ndarray,
    language_labels: np.ndarray,
    cv: int = 5,
    C: float = 1.0,
    seed: int = 42,
    eraser_factory=None,
) -> ProbeResult:
    """Cross-validated language-ID probe, optionally after erasure.

    With ``eraser_factory`` (a no-argument callable returning an unfitted eraser),
    the eraser is fitted on each training fold only and applied to that fold and
    the held-out fold. Fitting it on all documents first leaks the held-out fold
    into the erasure and pushes accuracy *below* guessing (0.365 on LaBSE).
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(language_labels)
    accs = []
    for train, test in StratifiedKFold(cv, shuffle=True, random_state=seed).split(X, y):
        X_tr, X_te = X[train], X[test]
        if eraser_factory is not None:
            eraser = eraser_factory().fit(X_tr, y[train])
            X_tr = apply_eraser(eraser, X_tr, y[train])
            X_te = apply_eraser(eraser, X_te, y[test])
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs").fit(X_tr, y[train])
        accs.append(clf.score(X_te, y[test]))
    acc = float(np.mean(accs))
    chance = majority_share(y)
    return ProbeResult(accuracy=acc, chance_accuracy=chance, above_chance=acc - chance)


def guessing_macro_f1(train_labels: np.ndarray, test_labels: np.ndarray) -> dict[str, float]:
    """Expected macro-F1 of label-blind guessing on the test set.

    A guesser predicting class k with probability p_k scores, for class k with test
    share q_k, precision q_k and recall p_k, so F1_k = 2 p_k q_k / (p_k + q_k).
    ``proportional``: p = training class shares; ``uniform``: p_k = 1/K.
    """
    classes = sorted(set(np.unique(train_labels)) | set(np.unique(test_labels)))
    q = np.array([np.mean(test_labels == c) for c in classes])
    out = {}
    for name, p in (("proportional", np.array([np.mean(train_labels == c) for c in classes])),
                    ("uniform", np.full(len(classes), 1 / len(classes)))):
        f1 = np.where(p + q > 0, 2 * p * q / np.maximum(p + q, 1e-12), 0.0)
        out[name] = float(f1.mean())
    return out


# ---------------------------------------------------------------------------
# Migrated from transfer.py
# ---------------------------------------------------------------------------

class LinearRegisterClassifier:
    """Logistic regression classifier over register-class embeddings."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, scale: bool = False, Cs=None) -> None:
        self.C = C
        self.max_iter = max_iter
        self.scale = scale
        self.Cs = Cs
        self._clf = None
        self._le = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegisterClassifier":
        y_enc = self._le.fit_transform(y)
        self._clf = make_probe(self.C, self.scale, self.Cs)
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
    scale: bool = False,
    Cs=None,
) -> TransferResult:
    """Train on source language embeddings, evaluate on target language.

    With ``scale``, the standardiser is fitted on the source language only.
    """
    clf = LinearRegisterClassifier(C=C, scale=scale, Cs=Cs)
    clf.fit(X_src, y_src)
    return clf.evaluate(X_tgt, y_tgt)
