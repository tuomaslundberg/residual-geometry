"""Phase 2 pilot: supervised probing of residual structure.

Tests whether candidate feature axes survive language erasure and whether
a linear classifier trained on FI residual embeddings transfers to EN.

Current axes: register (HPLT v3 web-register field, 8 main CORE classes).
To add a new axis: load or compute labels independently, then call _probe_axis().

Run:
  python experiments/pilot_phase2.py \\
      --fi data/hplt-samples/v3/fin_Latn_sample_exploded.jsonl \\
      --en data/hplt-samples/v3/eng_Latn_sample_exploded.jsonl \\
      --out outputs/pilot_phase2.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Probe axis runner — call once per (eraser × axis) combination
# ---------------------------------------------------------------------------

def _probe_axis(
    X_fi: np.ndarray,
    X_en: np.ndarray,
    fi_labels: np.ndarray,
    en_labels: np.ndarray,
    shared_classes: list[str],
    n_permutations: int = 200,
) -> dict:
    """Run the full probing suite for one feature axis.

    Metrics:
    - fi_within_acc / en_within_acc: 5-fold CV accuracy within each language
      (upper bound; shows the axis is linearly decodable before transfer)
    - crosslingual_macro_f1: classifier trained on FI, evaluated on EN
    - per_class_f1: per-class breakdown for cross-lingual transfer
    - procrustes_disparity: geometric distance between FI and EN centroid spaces
    - rsa_correlation / rsa_pvalue: RSA between FI and EN centroid RDMs
    """
    from residual_geometry.eval.probing import cross_lingual_transfer, linear_probe
    from residual_geometry.analyse.geometry import evaluate_geometry

    fi_acc = linear_probe(X_fi, fi_labels, cv=5)
    en_acc = linear_probe(X_en, en_labels, cv=5)
    transfer = cross_lingual_transfer(X_fi, fi_labels, X_en, en_labels)
    geometry = evaluate_geometry(
        X_fi, fi_labels, X_en, en_labels,
        registers=shared_classes,
        n_permutations=n_permutations,
    )

    return {
        "fi_within_acc": round(fi_acc, 4),
        "en_within_acc": round(en_acc, 4),
        "crosslingual_macro_f1": round(transfer.macro_f1, 4),
        "per_class_f1": {k: round(v, 4) for k, v in transfer.per_class_f1.items()},
        "procrustes_disparity": round(geometry.procrustes_disparity, 4),
        "rsa_correlation": round(geometry.rsa_correlation, 4),
        "rsa_pvalue": round(geometry.rsa_pvalue, 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pilot(
    fi_path: str,
    en_path: str,
    out_path: str,
    n_docs: int = 5000,
    model_name: str = "sentence-transformers/LaBSE",
    min_confidence: float = 0.4,
) -> None:
    from residual_geometry.data.loaders import load_hplt
    from residual_geometry.embed.encoder import Embedder
    from residual_geometry.erase import IdentityProjector, INLPEraser, MeanCenteringEraser
    from residual_geometry.eval.probing import language_probe

    # --- Load ---
    print(f"Loading HPLT v3 (min_confidence={min_confidence}, max_docs={n_docs}) …")
    fi_texts_all, fi_labels_all = load_hplt(
        fi_path, max_docs=n_docs, return_labels=True, min_confidence=min_confidence
    )
    en_texts_all, en_labels_all = load_hplt(
        en_path, max_docs=n_docs, return_labels=True, min_confidence=min_confidence
    )

    fi_labels_arr = np.array(fi_labels_all)
    en_labels_arr = np.array(en_labels_all)

    print(f"  FI: {len(fi_texts_all)} docs  EN: {len(en_texts_all)} docs")
    print(f"  FI register distribution: {dict(sorted(Counter(fi_labels_all).items()))}")
    print(f"  EN register distribution: {dict(sorted(Counter(en_labels_all).items()))}")

    # Filter to registers present in both languages
    shared_registers = sorted(set(fi_labels_arr) & set(en_labels_arr))
    print(f"  Shared registers: {shared_registers}")

    fi_mask = np.isin(fi_labels_arr, shared_registers)
    en_mask = np.isin(en_labels_arr, shared_registers)
    fi_texts = [t for t, m in zip(fi_texts_all, fi_mask) if m]
    en_texts = [t for t, m in zip(en_texts_all, en_mask) if m]
    fi_labels = fi_labels_arr[fi_mask]
    en_labels = en_labels_arr[en_mask]
    print(f"  After filtering: FI={len(fi_texts)}  EN={len(en_texts)}")

    # --- Embed ---
    all_texts = fi_texts + en_texts
    lang_labels = np.array(["fi"] * len(fi_texts) + ["en"] * len(en_texts))

    print(f"\nEmbedding {len(all_texts)} texts with {model_name} …")
    t0 = time.time()
    embedder = Embedder(model_name=model_name)
    X = embedder.embed(all_texts, show_progress=True)
    print(f"Embedding done in {time.time() - t0:.1f}s  shape={X.shape}")

    # --- Erasers ---
    erasers: dict = {
        "identity": IdentityProjector(),
        "mean": MeanCenteringEraser(),
        "inlp": INLPEraser(n_iterations=20),
    }

    results: dict = {}

    for name, eraser in erasers.items():
        print(f"\n=== {name} ===")
        eraser.fit(X, lang_labels)

        if isinstance(eraser, MeanCenteringEraser):
            X_proj = eraser.transform(X, lang_labels)
        else:
            X_proj = eraser.transform(X)

        X_fi_proj = X_proj[: len(fi_texts)]
        X_en_proj = X_proj[len(fi_texts) :]

        # Language sanity check (should be near chance after erasure)
        lang_result = language_probe(X_proj, lang_labels, cv=5)
        print(f"  Language probe selectivity: {lang_result.selectivity:.4f}")

        n_dirs = getattr(eraser, "n_directions_removed", None)

        # --- Register axis ---
        print("  Probing: register …")
        register_metrics = _probe_axis(
            X_fi_proj, X_en_proj, fi_labels, en_labels, shared_registers
        )
        print(
            f"    FI within={register_metrics['fi_within_acc']:.3f}  "
            f"EN within={register_metrics['en_within_acc']:.3f}  "
            f"transfer macro-F1={register_metrics['crosslingual_macro_f1']:.3f}  "
            f"Procrustes={register_metrics['procrustes_disparity']:.3f}"
        )

        # -----------------------------------------------------------------
        # To add a further probe axis, load labels externally and call:
        #   dep_metrics = _probe_axis(X_fi_proj, X_en_proj, fi_dep_bins, en_dep_bins, dep_classes)
        #   results[name]["dep_distance"] = dep_metrics
        # -----------------------------------------------------------------

        results[name] = {
            "language_selectivity": round(lang_result.selectivity, 4),
            "language_accuracy": round(lang_result.accuracy, 4),
            "n_directions_removed": n_dirs,
            "register": register_metrics,
        }

    # --- Save ---
    output = {
        "model": model_name,
        "corpus": "hplt-v3",
        "n_docs_fi": len(fi_texts),
        "n_docs_en": len(en_texts),
        "shared_registers": shared_registers,
        "min_confidence": min_confidence,
        "results": results,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fi", required=True)
    parser.add_argument("--en", required=True)
    parser.add_argument("--out", default="outputs/pilot_phase2.json")
    parser.add_argument("--n-docs", type=int, default=5000)
    parser.add_argument("--model", default="sentence-transformers/LaBSE")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    args = parser.parse_args()
    run_pilot(
        args.fi, args.en, args.out,
        n_docs=args.n_docs,
        model_name=args.model,
        min_confidence=args.min_confidence,
    )
