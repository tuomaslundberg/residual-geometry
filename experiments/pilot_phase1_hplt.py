"""Phase 1 pilot (HPLT monolingual): language-signal removal on web text.

Runs identity, mean centering, and INLP on HPLT v3 FI+EN.
Bitext retrieval is omitted — monolingual data has no parallel pairs.
All other Phase 1 metrics apply: language probe, geometry diagnostics.

Run:
  python experiments/pilot_phase1_hplt.py \\
      --fi data/hplt-samples/v3/fin_Latn_sample_exploded.jsonl \\
      --en data/hplt-samples/v3/eng_Latn_sample_exploded.jsonl \\
      --out outputs/pilot_phase1_hplt.json \\
      --n-docs 5000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def run_pilot(fi_path: str, en_path: str, out_path: str, n_docs: int = 5000) -> None:
    from residual_geometry.data.loaders import load_hplt
    from residual_geometry.embed.encoder import Embedder
    from residual_geometry.erase import IdentityProjector, INLPEraser, MeanCenteringEraser
    from residual_geometry.analyse import (
        anisotropy, divergence_stats, participation_ratio, twonn_intrinsic_dim,
    )
    from residual_geometry.eval.probing import language_probe

    print(f"Loading {n_docs} docs per language from HPLT v3 …")
    fi_texts = load_hplt(fi_path, max_docs=n_docs)
    en_texts = load_hplt(en_path, max_docs=n_docs)
    all_texts = fi_texts + en_texts
    lang_labels = np.array(["fi"] * len(fi_texts) + ["en"] * len(en_texts))
    print(f"  FI: {len(fi_texts)}  EN: {len(en_texts)}")

    print("Embedding with LaBSE …")
    t0 = time.time()
    embedder = Embedder(model_name="sentence-transformers/LaBSE")
    X = embedder.embed(all_texts, show_progress=True)
    print(f"Embedding done in {time.time() - t0:.1f}s")

    X_fi = X[: len(fi_texts)]
    X_en = X[len(fi_texts) :]

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

        X_fi_proj = X_proj[: len(fi_texts)]
        X_en_proj = X_proj[len(fi_texts) :]

        probe = language_probe(X_proj, lang_labels, cv=5)
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
            "divergence_mean": round(div["mean"], 4),
            "participation_ratio": round(pr, 4),
            "anisotropy": round(aniso, 4),
            "twonn_id": round(twonn, 2) if not (twonn != twonn) else None,
            "n_directions_removed": n_dirs,
        }
        print(json.dumps(results[name], indent=2))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"model": "LaBSE", "corpus": "hplt-v3", "n_docs": n_docs, "results": results},
            f,
            indent=2,
        )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fi", required=True)
    parser.add_argument("--en", required=True)
    parser.add_argument("--out", default="outputs/pilot_phase1_hplt.json")
    parser.add_argument("--n-docs", type=int, default=5000)
    args = parser.parse_args()
    run_pilot(args.fi, args.en, args.out, n_docs=args.n_docs)
