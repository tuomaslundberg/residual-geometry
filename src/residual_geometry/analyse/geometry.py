"""Geometric diagnostics for residual embedding spaces.

Includes:
  - Participation ratio (PR) — effective PCA dimensionality
  - Anisotropy — mean cosine similarity between random pairs (Ethayarajh 2019)
  - TwoNN intrinsic dimensionality (Facco et al. 2017)
  - Procrustes disparity and RSA (Kriegeskorte 2008) — migrated from evaluate.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import procrustes as scipy_procrustes
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# New geometry diagnostics
# ---------------------------------------------------------------------------

def participation_ratio(X: np.ndarray) -> float:
    """Participation ratio: (sum λ_i)^2 / sum λ_i^2, where λ_i are PCA eigenvalues.

    Measures effective PCA dimensionality. Range: [1, d].
    Higher = variance is spread across more dimensions (more isotropic).
    """
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T.astype(np.float64))
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sum_sq = float(np.sum(eigenvalues) ** 2)
    sq_sum = float(np.sum(eigenvalues ** 2))
    if sq_sum < 1e-12:
        return 1.0
    return sum_sq / sq_sum


def anisotropy(X: np.ndarray, n_samples: int = 10_000, seed: int = 42) -> float:
    """Mean cosine similarity between randomly sampled embedding pairs.

    Near 0 = isotropic; near 1 = all vectors collapse into one direction.
    Uses sampling for large matrices.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n < 2:
        return float("nan")
    n_pairs = min(n_samples, n * (n - 1) // 2)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % n

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_norm = X / norms

    cosines = np.sum(X_norm[idx_a] * X_norm[idx_b], axis=1)
    return float(cosines.mean())


def twonn_intrinsic_dim(X: np.ndarray) -> float:
    """TwoNN intrinsic dimensionality estimator (Facco et al., Sci. Rep. 2017).

    Uses the ratio of 2nd-NN to 1st-NN distances.
    Reference: https://doi.org/10.1038/s41598-017-11873-y
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    if n < 3:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # distances[:, 0] is self (0), so 1st and 2nd NN are [:, 1] and [:, 2]
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Avoid division by zero
    valid = r1 > 1e-10
    mu = r2[valid] / r1[valid]
    if len(mu) < 2:
        return float("nan")

    # MLE: d = n / sum(log(mu_i))
    log_mu = np.log(mu)
    if log_mu.sum() < 1e-10:
        return float("nan")
    return float(len(mu) / log_mu.sum())


# ---------------------------------------------------------------------------
# Migrated from evaluate.py
# ---------------------------------------------------------------------------

@dataclass
class GeometryResult:
    procrustes_disparity: float
    rsa_correlation: float
    rsa_pvalue: float
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
    """Per-register mean embedding, shape (n_registers, d)."""
    d = X.shape[1]
    centroids = np.zeros((len(registers), d), dtype=np.float64)
    for i, reg in enumerate(registers):
        mask = register_labels == reg
        if mask.sum() == 0:
            raise ValueError(f"No documents found for register '{reg}'.")
        centroids[i] = X[mask].mean(axis=0)
    return centroids


def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    """Normalised residual after optimal rotation/scaling alignment. Range [0, 1]."""
    _, _, disparity = scipy_procrustes(A.astype(np.float64), B.astype(np.float64))
    return float(disparity)


def rsa(A: np.ndarray, B: np.ndarray, metric: str = "cosine") -> tuple[float, float]:
    """Spearman correlation of lower-triangle RDMs. Returns (correlation, p-value)."""
    rdm_a = cdist(A, A, metric=metric)
    rdm_b = cdist(B, B, metric=metric)
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
    """Procrustes disparity and RSA between FI and EN centroid spaces."""
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
