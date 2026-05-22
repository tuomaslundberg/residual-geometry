"""Stage 3: geometric evaluation — Procrustes disparity and RSA.

Both metrics operate on register-class centroids, enabling comparison
of the register subspace geometry pre- and post-projection.

Procrustes disparity: after optimal rotation/scaling alignment of two
    centroid matrices, the residual distance. Lower = more isomorphic.

RSA (Representational Similarity Analysis): Spearman correlation between
    the pairwise distance matrices (RDMs) of the two centroid sets.
    Higher = more isomorphic. Assumption-light; from Kriegeskorte et al. (2008).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import procrustes as scipy_procrustes
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


@dataclass
class GeometryResult:
    procrustes_disparity: float
    rsa_correlation: float
    rsa_pvalue: float
    # Null distribution (optional; populated when n_permutations > 0)
    null_procrustes: np.ndarray = field(default_factory=lambda: np.array([]))
    null_rsa: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> dict:
        d = {
            "procrustes_disparity": round(self.procrustes_disparity, 6),
            "rsa_correlation": round(self.rsa_correlation, 6),
            "rsa_pvalue": round(self.rsa_pvalue, 6),
        }
        if self.null_procrustes.size:
            d["null_procrustes_mean"] = round(float(self.null_procrustes.mean()), 6)
        if self.null_rsa.size:
            d["null_rsa_mean"] = round(float(self.null_rsa.mean()), 6)
        return d


def register_centroids(
    X: np.ndarray,
    register_labels: np.ndarray,
    registers: list[str],
) -> np.ndarray:
    """Compute per-register mean embedding, shape (n_registers, d).

    Registers are returned in the order given by ``registers`` so that
    row indices correspond across languages.
    """
    d = X.shape[1]
    centroids = np.zeros((len(registers), d), dtype=np.float64)
    for i, reg in enumerate(registers):
        mask = register_labels == reg
        if mask.sum() == 0:
            raise ValueError(f"No documents found for register '{reg}'.")
        centroids[i] = X[mask].mean(axis=0)
    return centroids


def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    """Procrustes disparity between centroid matrices A and B (same row order).

    Returns the normalised residual sum-of-squares after optimal
    rotation + uniform scaling alignment. Range [0, 1]; 0 = identical.
    """
    _, _, disparity = scipy_procrustes(A.astype(np.float64), B.astype(np.float64))
    return float(disparity)


def rsa(A: np.ndarray, B: np.ndarray, metric: str = "cosine") -> tuple[float, float]:
    """RSA: Spearman correlation of lower-triangle RDM entries.

    Returns (correlation, p-value).
    """
    rdm_a = cdist(A, A, metric=metric)
    rdm_b = cdist(B, B, metric=metric)
    # Lower triangle, excluding diagonal
    idx = np.tril_indices(len(A), k=-1)
    corr, pval = spearmanr(rdm_a[idx], rdm_b[idx])
    return float(corr), float(pval)


def evaluate_geometry(
    X_fi: np.ndarray,
    labels_fi: np.ndarray,
    X_en: np.ndarray,
    labels_en: np.ndarray,
    registers: list[str],
    n_permutations: int = 1000,
    seed: int = 42,
) -> GeometryResult:
    """Full geometry evaluation between FI and EN centroid spaces.

    Computes Procrustes disparity and RSA, plus a permutation null distribution
    obtained by shuffling register labels before computing centroids.
    """
    A = register_centroids(X_fi, labels_fi, registers)
    B = register_centroids(X_en, labels_en, registers)

    disp = procrustes_disparity(A, B)
    corr, pval = rsa(A, B)

    null_disp = np.zeros(n_permutations)
    null_corr = np.zeros(n_permutations)
    rng = np.random.default_rng(seed)

    for i in range(n_permutations):
        A_perm = register_centroids(X_fi, rng.permutation(labels_fi), registers)
        B_perm = register_centroids(X_en, rng.permutation(labels_en), registers)
        null_disp[i] = procrustes_disparity(A_perm, B_perm)
        null_corr[i], _ = rsa(A_perm, B_perm)

    return GeometryResult(
        procrustes_disparity=disp,
        rsa_correlation=corr,
        rsa_pvalue=pval,
        null_procrustes=null_disp,
        null_rsa=null_corr,
    )
