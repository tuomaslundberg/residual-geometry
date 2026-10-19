"""Build the thesis sample from HPLT 4 sampler output (Tuomas's scheme, 2026-10-19).

Labelling, per document, at the language's model-card threshold (fin 0.45, eng 0.40):
- exactly one main register among HI ID IN IP LY NA OP SP: keep it as the label;
  subregisters of that main register are coalesced into it (NA + ne -> NA);
- two or more main registers, or a subregister whose main register is not the
  document's (it crosses the main-label boundary, or stands without its main): drop;
- no main register and no subregister: drop (no label), except
- MT (machine translated) is kept as a flag, not a register: MT + one register keeps
  that register with ``mt: true``; MT alone becomes the class ``MT``. Excluding MT
  later is one filter on ``mt``.

  filter:  zstdcat samples.jsonl.zst | python scripts/thesis_sample.py filter --lang fin \
               --source fin-9-10 --out eligible.jsonl
           Writes slim records (id, url, text, wds, probs, label, mt, source) for eligible
           documents and prints per-label counts as JSON on stderr at the end.
  draw:    python scripts/thesis_sample.py draw --fi FI.jsonl --en EN.jsonl --per-class N \
               --seed S --exclude LY --out DIR
           Seeded reservoir draw of N documents per label per language (all of a label if
           fewer), streaming.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

MAIN = ("LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP")
# Subregister -> main register: same mapping as webmeta (src/webmeta/register.py) and
# the TurkuNLP annotation guidelines.
PARENT = {
    "it": "SP",
    "ne": "NA", "sr": "NA", "nb": "NA",
    "re": "HI",
    "en": "IN", "ra": "IN", "dtp": "IN", "fi": "IN", "lt": "IN",
    "rv": "OP", "ob": "OP", "rs": "OP", "av": "OP",
    "ds": "IP", "ed": "IP",
}
THRESHOLD = {"fin": 0.45, "eng": 0.40}


def label(probs: dict, t: float) -> tuple[str | None, bool]:
    """(label or None to drop, mt flag) under the scheme above."""
    mains = [m for m in MAIN if probs.get(m, 0.0) >= t]
    subs = [s for s in PARENT if probs.get(s, 0.0) >= t]
    mt = probs.get("MT", 0.0) >= t
    if len(mains) >= 2:
        return None, mt
    if any(PARENT[s] not in mains for s in subs):
        return None, mt
    if len(mains) == 1:
        return mains[0], mt
    return ("MT" if mt else None), mt


def cmd_filter(args) -> None:
    t = THRESHOLD[args.lang]
    counts, seen = Counter(), 0
    with open(args.out, "w", encoding="utf-8") as out:
        for line in sys.stdin:
            if args.limit and seen >= args.limit:
                break
            seen += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                counts["_unparseable"] += 1
                continue
            text = (d.get("text") or "").strip()
            if not text:
                counts["_empty"] += 1
                continue
            probs = d.get("web-register") or {}
            lab, mt = label(probs, t)
            if lab is None:
                counts["_dropped"] += 1
                continue
            counts[lab] += 1
            counts["_mt_flagged"] += int(mt)
            scores = d.get("doc_scores") or [None]
            out.write(json.dumps({
                "id": d.get("id"), "url": d.get("u"), "text": text, "wds": scores[0],
                "probs": probs, "label": lab, "mt": mt, "source": args.source,
                "lang": args.lang,
            }, ensure_ascii=False) + "\n")
    counts["_read"] = seen
    print(json.dumps(dict(sorted(counts.items()))), file=sys.stderr)


def cmd_draw(args) -> None:
    """One streaming pass per language: a seeded reservoir of N lines per label.

    Memory holds only the reservoirs (N x labels lines), so the 62 GB Finnish file is
    fine. Labels in --exclude are skipped; a label with fewer than N documents is kept
    whole and reported.
    """
    import random

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = set(args.exclude)
    summary = {}
    for lang, paths in (("fi", args.fi), ("en", args.en)):
        rng = random.Random(f"{args.seed}:{lang}")
        res: dict[str, list[str]] = {}
        seen: Counter = Counter()
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    lab = json.loads(line)["label"]
                    if lab in exclude:
                        continue
                    k = seen[lab]
                    seen[lab] += 1
                    bucket = res.setdefault(lab, [])
                    if k < args.per_class:
                        bucket.append(line)
                    else:
                        j = rng.randint(0, k)
                        if j < args.per_class:
                            bucket[j] = line
        chosen = [line for lab in sorted(res) for line in res[lab]]
        rng.shuffle(chosen)
        with open(out / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(chosen)
        summary[lang] = {lab: {"available": seen[lab], "drawn": len(res[lab])} for lab in sorted(res)}
    (out / "summary.json").write_text(json.dumps(
        {"per_class": args.per_class, "seed": args.seed, "excluded": sorted(exclude),
         "fi": [str(p) for p in args.fi], "en": [str(p) for p in args.en],
         "cells": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--lang", required=True, choices=sorted(THRESHOLD))
    f.add_argument("--source", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--limit", type=int, default=0, help="stop after this many input lines")
    d = sub.add_parser("draw")
    d.add_argument("--fi", nargs="+", required=True)
    d.add_argument("--en", nargs="+", required=True)
    d.add_argument("--per-class", type=int, required=True)
    d.add_argument("--seed", type=int, default=20261019)
    d.add_argument("--exclude", nargs="*", default=[], help="labels to leave out, e.g. LY")
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    {"filter": cmd_filter, "draw": cmd_draw}[a.cmd](a)
