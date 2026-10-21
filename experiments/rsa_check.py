"""Why is register-centroid RSA unreliable on XLM-R? (thesis sample, 2026-10-21)

In thesis_analyses D, XLM-R's cosine RSA sits at the chance edge (0.45) under raw, mean,
LEACE and translation-direction removal, and jumps to 0.76 after INLP's two directions.
Hypothesis: cosine between XLM-R vectors is dominated by a few high-variance directions
(its anisotropy), which drown the register differences; INLP's second direction removes one.

Checks, each with the 1,000-shuffle chance level:
- distance: cosine vs Euclidean between centroids;
- space: raw · pooled-centred · standardised per feature · top-k principal components
  removed after centring (k = 1, 2, 3, 5, 10; "all-but-the-top") · INLP;
- what INLP's directions are: their overlap with the top principal components, and how
  much of the pooled variance lies along them.

  python experiments/rsa_check.py --model FacebookAI/xlm-roberta-base --out outputs/thesis/rsa_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from thesis_analyses import REGISTERS, load_lang  # noqa: E402


def centroids(X, y):
    return np.stack([X[y == r].mean(0) for r in REGISTERS])


def rsa(A, B, metric):
    i = np.tril_indices(len(A), -1)
    return float(spearmanr(cdist(A, A, metric)[i], cdist(B, B, metric)[i]).statistic)


def rsa_with_null(Xf, yf, Xe, ye, metric, rng, n_perm=1000):
    obs = rsa(centroids(Xf, yf), centroids(Xe, ye), metric)
    null = np.array([rsa(centroids(Xf, rng.permutation(yf)), centroids(Xe, rng.permutation(ye)), metric)
                     for _ in range(n_perm)])
    return {"rsa": round(obs, 3), "null_p95": round(float(np.percentile(null, 95)), 3),
            "p": round((1 + int(np.sum(null >= obs))) / (1 + n_perm), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--per-class", type=int, default=6000,
                    help="documents per register per language (centroids; keeps the nulls fast)")
    args = ap.parse_args()

    from residual_geometry.embed.cache import EmbeddingCache
    from residual_geometry.embed.encoder import Embedder
    from residual_geometry.erase import INLPEraser

    rng = np.random.default_rng(args.seed)
    cache = EmbeddingCache(args.model)
    emb = Embedder(model_name=args.model, device="cpu")
    X, y = {}, {}
    for lang in ("fi", "en"):
        rows, Z = load_lang(lang, cache, emb)
        lab = np.array([r["label"] for r in rows])
        idx = np.concatenate([rng.choice(np.flatnonzero(lab == r), args.per_class, replace=False)
                              for r in REGISTERS])
        X[lang], y[lang] = Z[idx], lab[idx]

    pooled = np.vstack([X["fi"], X["en"]])
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-12
    _, S, Vt = np.linalg.svd(pooled - mu, full_matrices=False)
    var_share = S ** 2 / np.sum(S ** 2)

    def anisotropy(Z):
        a, b = rng.integers(0, len(Z), 20000), rng.integers(0, len(Z), 20000)
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return round(float(np.mean(np.sum(Zn[a] * Zn[b], axis=1))), 3)

    spaces = {"raw": lambda Z: Z, "centred": lambda Z: Z - mu, "standardised": lambda Z: (Z - mu) / sd}
    for k in (1, 2, 3, 5, 10):
        Vk = Vt[:k]
        spaces[f"centred_minus_top{k}pc"] = (lambda Vk: lambda Z: (Z - mu) - ((Z - mu) @ Vk.T) @ Vk)(Vk)
    n_fit = 20000
    fi_i, en_i = rng.choice(len(X["fi"]), n_fit, False), rng.choice(len(X["en"]), n_fit, False)
    inlp = INLPEraser().fit(np.vstack([X["fi"][fi_i], X["en"][en_i]]), np.array(["fi"] * n_fit + ["en"] * n_fit))
    spaces["inlp"] = lambda Z: Z @ inlp.projection_matrix.T

    res = {"model": args.model, "per_class": args.per_class,
           "top_pc_variance_share": [round(float(v), 4) for v in var_share[:10]], "spaces": {}}
    for name, f in spaces.items():
        Zf, Ze = f(X["fi"]), f(X["en"])
        res["spaces"][name] = {
            "anisotropy": anisotropy(np.vstack([Zf, Ze])),
            "cosine": rsa_with_null(Zf, y["fi"], Ze, y["en"], "cosine", rng),
            "euclidean": rsa_with_null(Zf, y["fi"], Ze, y["en"], "euclidean", rng),
        }
        print(name, res["spaces"][name], flush=True)

    D = inlp.directions
    res["inlp"] = {
        "n_directions": int(len(D)),
        "overlap_with_top10_pcs": [[round(float(abs(d @ v)), 3) for v in Vt[:10]] for d in D],
        "pooled_variance_share_along": [round(float(np.var((pooled - mu) @ d) / np.sum(S ** 2 / len(pooled))), 4)
                                        for d in D],
        "language_gap_along": [round(float((X["fi"].mean(0) - X["en"].mean(0)) @ d), 4) for d in D],
    }
    print("inlp", res["inlp"], flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(args.model).name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
