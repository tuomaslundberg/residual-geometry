"""Light topic modelling of the thesis sample, one model per language (analysis B).

Follows the KKLT0040 notebook's approach (Sep 2026, scikit-learn only): function words
removed so topics capture aboutness, the number of topics K chosen by windowed NPMI
coherence (UMass rejected there: monotone in K). Additions for Finnish: simplemma
lemmatisation, so inflected forms (talossa, taloon -> talo) count as one word, and
NLTK stopword lists for both languages.

Each document contributes its first ``--max-words`` words (default 500; the encoders
read 256 or 512 tokens, the register labels came from 512). NMF on TF-IDF.

  python experiments/topics.py --sample DIR --out outputs/topics
Writes, per language: topics-<lang>.json (K curve, chosen K, top words per topic)
and doc-topics-<lang>.tsv (id, topic, weight, second topic) for every document.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
LANG = {"fi": ("fi", "finnish"), "en": ("en", "english")}
_STATE: dict = {}


def _init(lang: str) -> None:
    import nltk
    import simplemma

    nltk.data.path.insert(0, os.path.join(os.environ.get("WORK", ""), "nltk_data"))
    from nltk.corpus import stopwords

    code, name = LANG[lang]
    _STATE["code"] = code
    _STATE["stop"] = set(stopwords.words(name))
    _STATE["lemma"] = simplemma.lemmatize


def _tokens(args: tuple[str, int]) -> list[str]:
    text, max_words = args
    stop, lemma, code = _STATE["stop"], _STATE["lemma"], _STATE["code"]
    out = []
    for w in WORD.findall(text)[:max_words]:
        w = w.lower()
        if w in stop:
            continue
        lw = lemma(w, lang=code).lower()
        if len(lw) >= 3 and lw not in stop:
            out.append(lw)
    return out


def npmi_coherence(top_words: list[list[str]], docs: list[list[str]], window: int = 10) -> list[float]:
    """Mean pairwise NPMI of each topic's top words over sliding windows."""
    vocab = {w for ws in top_words for w in ws}
    single, pair, n_win = Counter(), Counter(), 0
    for toks in docs:
        toks = [t if t in vocab else None for t in toks]
        for s in range(0, max(1, len(toks) - window + 1)):
            present = {t for t in toks[s:s + window] if t is not None}
            n_win += 1
            single.update(present)
            pair.update(frozenset(p) for p in combinations(sorted(present), 2))
    scores = []
    for ws in top_words:
        vals = []
        for a, b in combinations(ws, 2):
            pab = pair[frozenset((a, b))] / n_win
            pa, pb = single[a] / n_win, single[b] / n_win
            if pab == 0 or pa == 0 or pb == 0:
                vals.append(-1.0)
            else:
                vals.append(math.log(pab / (pa * pb)) / -math.log(pab))
        scores.append(float(np.mean(vals)))
    return scores


def main() -> None:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-grid", default="8,12,16,20,25,30")
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--coherence-docs", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = [int(k) for k in args.k_grid.split(",")]

    for lang in ("fi", "en"):
        t0 = time.time()
        rows = [json.loads(l) for l in open(Path(args.sample) / f"{lang}.jsonl", encoding="utf-8")]
        with Pool(args.workers, initializer=_init, initargs=(lang,)) as pool:
            toks = pool.map(_tokens, [(r["text"], args.max_words) for r in rows], chunksize=500)
        print(f"{lang}: tokenised {len(toks)} docs in {time.time() - t0:.0f}s", flush=True)

        vec = TfidfVectorizer(analyzer=lambda x: x, min_df=20, max_df=0.3, max_features=30000)
        X = vec.fit_transform(toks)
        vocab = np.array(vec.get_feature_names_out())
        rng = np.random.default_rng(args.seed)
        coh_idx = rng.choice(len(toks), size=min(args.coherence_docs, len(toks)), replace=False)
        coh_docs = [toks[i] for i in coh_idx]

        curve, models = [], {}
        for k in grid:
            nmf = NMF(n_components=k, init="nndsvda", random_state=args.seed, max_iter=400)
            W = nmf.fit_transform(X)
            top = [list(vocab[np.argsort(-row)[:10]]) for row in nmf.components_]
            coh = npmi_coherence(top, coh_docs)
            curve.append({"k": k, "npmi_mean": round(float(np.mean(coh)), 4)})
            models[k] = (W, top, coh)
            print(f"{lang}: K={k} NPMI={np.mean(coh):.4f}", flush=True)
        best = max(curve, key=lambda c: c["npmi_mean"])["k"]
        W, top, coh = models[best]
        order = np.argsort(-W, axis=1)
        sums = W.sum(axis=1)
        with open(out / f"doc-topics-{lang}.tsv", "w", encoding="utf-8") as f:
            f.write("id\ttopic\tweight\tsecond\n")
            for r, w, o, s in zip(rows, W, order, sums):
                f.write(f"{r['id']}\t{o[0]}\t{w[o[0]] / s if s else 0:.3f}\t{o[1]}\n")
        counts = Counter(order[:, 0].tolist())
        (out / f"topics-{lang}.json").write_text(json.dumps({
            "lang": lang, "k_curve": curve, "k": best, "n_docs": len(rows),
            "vocab_size": len(vocab), "max_words": args.max_words,
            "topics": [{"topic": i, "top_words": top[i], "npmi": round(coh[i], 4),
                        "n_docs_argmax": counts.get(i, 0)} for i in range(best)],
        }, ensure_ascii=False, indent=2))
        print(f"{lang}: chose K={best} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
