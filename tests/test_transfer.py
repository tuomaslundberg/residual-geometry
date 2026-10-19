"""Tests for transfer.py."""

import numpy as np
import pytest

from residual_geometry.eval.probing import LinearRegisterClassifier, TransferResult, cross_lingual_transfer

REGISTERS = ["HI", "IN", "NA", "OP"]


def _make_data(rng, n=20, dim=16):
    """Linearly separable synthetic register data."""
    X, y = [], []
    for i, reg in enumerate(REGISTERS):
        mean = np.zeros(dim)
        mean[i] = 3.0
        X.append(rng.standard_normal((n, dim)) + mean)
        y.extend([reg] * n)
    return np.vstack(X).astype(np.float32), np.array(y)


def test_classifier_fit_predict_shape(rng):
    X, y = _make_data(rng)
    clf = LinearRegisterClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape


def test_classifier_predict_before_fit_raises():
    clf = LinearRegisterClassifier()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        clf.predict(np.zeros((5, 16)))


def test_classifier_high_accuracy_separable(rng):
    X, y = _make_data(rng)
    clf = LinearRegisterClassifier()
    result = clf.fit(X, y).evaluate(X, y)
    assert result.macro_f1 > 0.9


def test_transfer_result_fields(rng):
    X, y = _make_data(rng)
    result = cross_lingual_transfer(X, y, X, y)
    assert isinstance(result, TransferResult)
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.confusion_matrix.shape == (len(REGISTERS), len(REGISTERS))
    assert set(result.classes) == set(REGISTERS)


def test_summary_keys(rng):
    X, y = _make_data(rng)
    result = cross_lingual_transfer(X, y, X, y)
    s = result.summary()
    assert "macro_f1" in s
    assert "per_class_f1" in s


def test_guessing_macro_f1_by_hand():
    from residual_geometry.eval.probing import guessing_macro_f1

    train = np.array(["A"] * 3 + ["B"])          # p = (0.75, 0.25)
    test = np.array(["A", "B"])                   # q = (0.5, 0.5)
    g = guessing_macro_f1(train, test)
    f1_a = 2 * 0.75 * 0.5 / 1.25
    f1_b = 2 * 0.25 * 0.5 / 0.75
    assert g["proportional"] == pytest.approx((f1_a + f1_b) / 2)
    assert g["uniform"] == pytest.approx(0.5)
