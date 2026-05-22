"""Tests for evaluate.py."""

import numpy as np
import pytest

from residual_geometry.analyse.geometry import (
    GeometryResult,
    evaluate_geometry,
    procrustes_disparity,
    register_centroids,
    rsa,
)

REGISTERS = ["HI", "IN", "NA", "OP"]


def _make_labeled(rng, n=10, dim=16, registers=REGISTERS):
    X, labels = [], []
    for i, reg in enumerate(registers):
        mean = np.zeros(dim)
        mean[i] = 2.0
        X.append(rng.standard_normal((n, dim)) + mean)
        labels.extend([reg] * n)
    return np.vstack(X).astype(np.float32), np.array(labels)


def test_register_centroids_shape(rng):
    X, labels = _make_labeled(rng)
    centroids = register_centroids(X, labels, REGISTERS)
    assert centroids.shape == (len(REGISTERS), X.shape[1])


def test_register_centroids_missing_class(rng):
    X, labels = _make_labeled(rng)
    with pytest.raises(ValueError, match="No documents"):
        register_centroids(X, labels, REGISTERS + ["MISSING"])


def test_procrustes_identical():
    A = np.eye(4, dtype=np.float64)
    assert procrustes_disparity(A, A) == pytest.approx(0.0, abs=1e-9)


def test_procrustes_permuted_rows(rng):
    A = rng.standard_normal((6, 8))
    B = A + rng.standard_normal((6, 8)) * 0.01
    d = procrustes_disparity(A, B)
    assert 0.0 <= d <= 1.0


def test_rsa_identical(rng):
    A = rng.standard_normal((6, 8))
    corr, pval = rsa(A, A)
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_rsa_range(rng):
    A = rng.standard_normal((6, 8))
    B = rng.standard_normal((6, 8))
    corr, pval = rsa(A, B)
    assert -1.0 <= corr <= 1.0
    assert 0.0 <= pval <= 1.0


def test_evaluate_geometry_runs(rng):
    X_fi, labels_fi = _make_labeled(rng)
    X_en, labels_en = _make_labeled(rng)
    result = evaluate_geometry(
        X_fi, labels_fi, X_en, labels_en,
        registers=REGISTERS,
        n_permutations=20,
        seed=0,
    )
    assert isinstance(result, GeometryResult)
    assert 0.0 <= result.procrustes_disparity <= 1.0
    assert -1.0 <= result.rsa_correlation <= 1.0
    assert len(result.null_procrustes) == 20
    assert len(result.null_rsa) == 20
