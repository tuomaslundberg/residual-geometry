"""Summarise the class-balanced Phase 2 draws (outputs/phase2_balanced/<rule>/draw_*.json).

Per register rule and eraser: the observed score and its chance level, as median and
range across draws, and in how many draws the observed score beats chance (p < 0.05).

  python scripts/summarise_balanced.py [outputs/phase2_balanced]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/phase2_balanced")


def fmt(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{np.median(v):.3f} [{v.min():.3f}, {v.max():.3f}]"


for rule_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    draws = [json.loads(f.read_text()) for f in sorted(rule_dir.glob("draw_*.json"))]
    if not draws:
        continue
    d0 = draws[0]
    print(f"\n## {rule_dir.name}: {len(draws)} draws, {d0['per_class']} per register per language, "
          f"registers {', '.join(d0['shared_registers'])}")
    print("| Eraser | Metric | Observed | Chance mean | Chance 95th pct (Procrustes: 5th) | Draws p<0.05 |")
    print("|---|---|---|---|---|---|")
    for eraser in d0["results"]:
        regs = [d["results"][eraser]["register"] for d in draws]
        for metric, null_key in (("rsa_correlation", "rsa_null"),
                                 ("procrustes_disparity", "procrustes_null")):
            pct = "p05" if metric.startswith("procrustes") else "p95"
            sig = sum(r[null_key]["permutation_pvalue"] < 0.05 for r in regs)
            print(f"| {eraser} | {metric.split('_')[0]} | {fmt([r[metric] for r in regs])} | "
                  f"{fmt([r[null_key]['null_mean'] for r in regs])} | "
                  f"{fmt([r[null_key]['null_' + pct] for r in regs])} | {sig}/{len(regs)} |")
        print(f"| {eraser} | transfer F1 | {fmt([r['crosslingual_macro_f1'] for r in regs])} | "
              f"{1 / len(d0['shared_registers']):.3f} (uniform) | | |")
