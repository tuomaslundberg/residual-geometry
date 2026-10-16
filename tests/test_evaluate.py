"""Tests for evaluate.py."""

import numpy as np
import pytest

from residual_geometry.analyse.geometry import (
    GeometryResult,
    evaluate_geometry,
    permutation_summary,
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


def test_permutation_summary_direction():
    null = np.array([0.1, 0.2, 0.3, 0.9])
    hi = permutation_summary(0.85, null, higher_is_better=True)
    assert hi["permutation_pvalue"] == pytest.approx(2 / 5)  # 0.9 is >= 0.85
    lo = permutation_summary(0.15, null, higher_is_better=False)
    assert lo["permutation_pvalue"] == pytest.approx(2 / 5)  # 0.1 is <= 0.15
    assert hi["n_permutations"] == 4
    assert permutation_summary(0.5, np.array([])) == {}


def test_probe_axis_keeps_nulls(rng):
    """The Phase 2 pilot must report the permutation nulls, not drop them."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "experiments" / "pilot_phase2.py"
    spec = importlib.util.spec_from_file_location("pilot_phase2", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)

    X_fi, labels_fi = _make_labeled(rng)
    X_en, labels_en = _make_labeled(rng)
    out = pilot._probe_axis(
        X_fi, X_en, labels_fi, labels_en, REGISTERS, n_permutations=20
    )
    assert "rsa_spearman_pvalue" in out
    for key in ("rsa_null", "procrustes_null"):
        nl = out[key]
        assert nl["n_permutations"] == 20
        assert nl["null_p05"] <= nl["null_mean"] <= nl["null_p95"]
        assert 1 / 21 <= nl["permutation_pvalue"] <= 1.0
