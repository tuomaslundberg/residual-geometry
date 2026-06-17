"""Phase 1 pilot: language-signal removal completeness check.

Gateway experiment:
  - Model: LaBSE (sentence-transformers/LaBSE)
  - Data: Europarl FI-EN, 5 000 parallel pairs
  - Erasers: IdentityProjector, MeanCenteringEraser, INLPEraser
  - Metrics: language probe accuracy + selectivity, bitext retrieval P@1/MRR,
             divergence stats, geometry diagnostics (PR, anisotropy, TwoNN ID)

Run:
  python experiments/pilot_phase1.py --fi data/europarl.fi \
                                     --en data/europarl.en \
                                     --out outputs/pilot_phase1.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def run_pilot(
    fi_path: str,
    en_path: str,
    out_path: str,
    n_pairs: int = 5000,
    model_name: str = "sentence-transformers/LaBSE",
) -> None:
    from residual_geometry.data.loaders import load_europarl
    from residual_geometry.embed.encoder import Embedder
    from residual_geometry.erase import IdentityProjector, INLPEraser, MeanCenteringEraser
    from residual_geometry.analyse import (
        anisotropy, divergence_stats, participation_ratio, twonn_intrinsic_dim,
    )
    from residual_geometry.eval.probing import language_probe
    from residual_geometry.eval.retrieval import bitext_retrieval

    print(f"Loading {n_pairs} Europarl pairs …")
    fi_sents, en_sents = load_europarl(fi_path, en_path, max_pairs=n_pairs)
    all_texts = fi_sents + en_sents
    lang_labels = np.array(["fi"] * len(fi_sents) + ["en"] * len(en_sents))

    print(f"Embedding with {model_name} …")
    t0 = time.time()
    embedder = Embedder(model_name=model_name)
    X = embedder.embed(all_texts, show_progress=True)
    print(f"Embedding done in {time.time() - t0:.1f}s")

    X_fi = X[: len(fi_sents)]
    X_en = X[len(fi_sents) :]

    erasers = {
        "identity": IdentityProjector(),
        "mean": MeanCenteringEraser(),
        "inlp": INLPEraser(n_iterations=20),
    }

    results: dict = {}

    for name, eraser in erasers.items():
        print(f"\n--- {name} ---")
        eraser.fit(X, lang_labels)

        if isinstance(eraser, MeanCenteringEraser):
            X_proj = eraser.transform(X, lang_labels)
        else:
            X_proj = eraser.transform(X)

        X_fi_proj = X_proj[: len(fi_sents)]
        X_en_proj = X_proj[len(fi_sents) :]

        probe = language_probe(X_proj, lang_labels, cv=5)
        retrieval = bitext_retrieval(X_fi_proj, X_en_proj)
        div = divergence_stats(X_fi_proj, X_en_proj)
        pr = participation_ratio(X_proj)
        aniso = anisotropy(X_proj)
        twonn = twonn_intrinsic_dim(X_proj)

        if hasattr(eraser, "n_directions_removed"):
            n_dirs = eraser.n_directions_removed
        else:
            n_dirs = None

        results[name] = {
            "language_probe_accuracy": round(probe.accuracy, 4),
            "language_probe_selectivity": round(probe.selectivity, 4),
            "bitext_p_at_1": round(retrieval["p_at_1"], 4),
            "bitext_mrr": round(retrieval["mrr"], 4),
            "divergence_mean": round(div["mean"], 4),
            "participation_ratio": round(pr, 4),
            "anisotropy": round(aniso, 4),
            "twonn_id": round(twonn, 2) if not (twonn != twonn) else None,
            "n_directions_removed": n_dirs,
        }
        print(json.dumps(results[name], indent=2))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": model_name, "n_pairs": n_pairs, "results": results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fi", required=True)
    parser.add_argument("--en", required=True)
    parser.add_argument("--out", default="outputs/pilot_phase1.json")
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--model", default="sentence-transformers/LaBSE")
    args = parser.parse_args()
    run_pilot(args.fi, args.en, args.out, n_pairs=args.n_pairs, model_name=args.model)
