"""Pairwise distance and divergence statistics for comparing embedding populations."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def pairwise_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """All pairwise L2 distances between rows of A and rows of B, shape (n_A, n_B)."""
    return cdist(A.astype(np.float64), B.astype(np.float64), metric="euclidean")


def divergence_stats(
    A: np.ndarray,
    B: np.ndarray,
    subsample: int | None = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Summary statistics for the cross-population L2 distance distribution.

    Returns mean, median, std, p5, and p95 of pairwise distances.
    Subsample controls the maximum number of rows drawn from each matrix.
    """
    rng = np.random.default_rng(seed)
    if subsample is not None:
        if len(A) > subsample:
            A = A[rng.choice(len(A), size=subsample, replace=False)]
        if len(B) > subsample:
            B = B[rng.choice(len(B), size=subsample, replace=False)]

    dists = pairwise_l2(A, B).ravel()
    return {
        "mean": float(dists.mean()),
        "median": float(np.median(dists)),
        "std": float(dists.std()),
        "p5": float(np.percentile(dists, 5)),
        "p95": float(np.percentile(dists, 95)),
    }
