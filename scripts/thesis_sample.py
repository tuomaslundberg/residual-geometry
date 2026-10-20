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
           [--host-cap 500 --no-mt-hybrids]
           Seeded draw of N documents per label per language (all of a label if fewer),
           optionally capped per site and without MT + register documents.
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
    """Seeded draw of N documents per label per language, two passes per file.

    Pass 1 indexes (label, site, MT flag, byte offset) for every line; pass 2 reads
    back only the chosen lines, so memory stays small for the 62 GB Finnish pool.
    Within a label, documents are taken in a seeded random order, skipping any whose
    site (URL host) already has ``--host-cap`` documents in that label: a uniform draw
    subject to the cap. ``--no-mt-hybrids`` drops documents that carry MT together
    with a register (MT + HI counts as a hybrid); MT-only documents stay a class.
    A label with fewer eligible documents than N is kept whole and reported.
    """
    import random
    from urllib.parse import urlsplit

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = set(args.exclude)
    summary = {}
    for lang, paths in (("fi", args.fi), ("en", args.en)):
        index: dict[str, list[tuple[int, int, str]]] = {}
        skipped = Counter()
        for fi_, p in enumerate(paths):
            with open(p, "rb") as f:
                pos = 0
                for raw in f:
                    d = json.loads(raw)
                    lab = d["label"]
                    if lab in exclude:
                        skipped["excluded_label"] += 1
                    elif args.no_mt_hybrids and d["mt"] and lab != "MT":
                        skipped["mt_hybrid"] += 1
                    else:
                        host = (urlsplit(d.get("url") or "").hostname or "").lower()
                        index.setdefault(lab, []).append((fi_, pos, host))
                    pos += len(raw)
        rng = random.Random(f"{args.seed}:{lang}")
        chosen: list[tuple[int, int]] = []
        summary[lang] = {"skipped": dict(skipped), "cells": {}}
        for lab in sorted(index):
            docs = index[lab]
            order = list(range(len(docs)))
            rng.shuffle(order)
            per_host: Counter = Counter()
            take = []
            for k in order:
                fi_, pos, host = docs[k]
                if args.host_cap and per_host[host] >= args.host_cap:
                    continue
                per_host[host] += 1
                take.append((fi_, pos))
                if len(take) == args.per_class:
                    break
            chosen += take
            top = per_host.most_common(1)[0] if per_host else ("", 0)
            summary[lang]["cells"][lab] = {
                "available": len(docs), "drawn": len(take), "sites": len(per_host),
                "largest_site": top[0], "largest_site_docs": top[1]}
        rng.shuffle(chosen)
        handles = [open(p, "rb") for p in paths]
        with open(out / f"{lang}.jsonl", "wb") as f:
            for fi_, pos in chosen:
                handles[fi_].seek(pos)
                f.write(handles[fi_].readline())
        for h in handles:
            h.close()
    (out / "summary.json").write_text(json.dumps(
        {"per_class": args.per_class, "seed": args.seed, "excluded": sorted(exclude),
         "host_cap": args.host_cap, "no_mt_hybrids": args.no_mt_hybrids,
         "fi": [str(p) for p in args.fi], "en": [str(p) for p in args.en],
         "result": summary}, indent=2))
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
    d.add_argument("--host-cap", type=int, default=0,
                   help="at most this many documents from one site (URL host) per label; 0 = no cap")
    d.add_argument("--no-mt-hybrids", action="store_true",
                   help="drop documents labelled MT together with a register")
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    {"filter": cmd_filter, "draw": cmd_draw}[a.cmd](a)
