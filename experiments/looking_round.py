"""Looking round: what does language removal take away, and what remains?

Three lenses, one encoder per run, all on cached embeddings:

1. Read the removed part. Score every web document along the language direction
   (Finnish mean minus English mean) and read the documents at both ends, within
   each language. Decile profiles (domain, register, quality, length) back the reading.
2. Retrieval. (a) Europarl held-out pairs: does a Finnish sentence still find its
   English translation after erasure? (b) Web: for Finnish documents, their nearest
   English documents before and after erasure, with what the neighbours share.
3. Language vs culture. Translation pairs differ only in language, so their mean
   difference is a "pure language" direction. The web direction also carries what
   Finnish and English pages are about. The part of the web direction the parallel
   direction does not explain is a candidate cultural direction; read its extremes.

Erasers: global centring (moves the origin only), mean centering (moves each
language's origin; changes no within-language distance), LEACE (removes the mean
difference, keeps the origin), INLP (removes directions, so it strips content from
every document) and the translation-pair direction projected out.

  python experiments/looking_round.py --model sentence-transformers/LaBSE \
      --out outputs/looking_round/labse
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

N_READ = 12          # documents shown at each end of a direction
SNIPPET = 320        # characters of text shown per document
N_QUERIES = 2000     # Finnish web documents used as retrieval queries
N_SHOWCASE = 14      # Finnish documents shown with their neighbours
K = 10               # neighbours per query
# raw · global centring (origin only) · mean centring · LEACE · INLP ·
# translation-pair direction projected out
ERASERS = ("raw", "global", "mean", "leace", "inlp", "transdir")


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def snippet(text: str, n: int = SNIPPET) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def doc_line(r: dict, score: float | None = None) -> str:
    s = f"`{score:+.3f}` " if score is not None else ""
    return (f"- {s}**{r['register']}** · WDS {r['wds']} · {len(r['text'].split())} words · "
            f"`{r['host']}`\n  > {snippet(r['text'])}")


def decile_profile(scores: np.ndarray, recs: list[dict], top_tlds: list[str]) -> list[dict]:
    order = np.argsort(scores)
    rows = []
    for i, chunk in enumerate(np.array_split(order, 10)):
        rs = [recs[j] for j in chunk]
        regs = Counter(r["register"] for r in rs)
        tlds = Counter(r["tld"] for r in rs)
        rows.append({
            "decile": i + 1,
            "score_range": [round(float(scores[chunk].min()), 3), round(float(scores[chunk].max()), 3)],
            "median_words": int(np.median([len(r["text"].split()) for r in rs])),
            "median_wds": round(float(np.median([r["wds"] for r in rs if r["wds"] is not None])), 2),
            "tld_share": {t: round(tlds[t] / len(rs), 3) for t in top_tlds},
            "top_registers": {k: round(v / len(rs), 3) for k, v in regs.most_common(4)},
        })
    return rows


def profile_table(rows: list[dict], top_tlds: list[str]) -> str:
    head = "| Decile | Score | Median words | Median WDS | " + " | ".join(f".{t}" for t in top_tlds) + " | Top registers |"
    sep = "|" + "---|" * (5 + len(top_tlds))
    lines = [head, sep]
    for r in rows:
        regs = ", ".join(f"{k} {v:.0%}" for k, v in r["top_registers"].items())
        tl = " | ".join(f"{r['tld_share'][t]:.0%}" for t in top_tlds)
        lines.append(f"| {r['decile']} | {r['score_range'][0]:+.3f}…{r['score_range'][1]:+.3f} | "
                     f"{r['median_words']} | {r['median_wds']} | {tl} | {regs} |")
    return "\n".join(lines)


def read_direction(name: str, d: np.ndarray, X: dict, R: dict, md: list, metrics: dict,
                   lang_note: dict) -> None:
    """Lens 1 for one direction: correlates, decile profiles and documents at both ends."""
    md.append(f"\n### Along the {name}\n")
    metrics[name] = {}
    for lang, top_tlds in (("fi", ["fi", "com", "net", "eu", "org"]),
                           ("en", ["com", "org", "uk", "edu", "au"])):
        s = X[lang] @ d
        words = np.array([len(r["text"].split()) for r in R[lang]])
        wds = np.array([r["wds"] if r["wds"] is not None else np.nan for r in R[lang]])
        ok = ~np.isnan(wds)
        rho_len = spearmanr(s, np.log(words)).statistic
        rho_wds = spearmanr(s[ok], wds[ok]).statistic
        prof = decile_profile(s, R[lang], top_tlds)
        metrics[name][lang] = {"spearman_log_words": round(float(rho_len), 3),
                               "spearman_wds": round(float(rho_wds), 3), "deciles": prof}
        order = np.argsort(s)
        md.append(f"\n#### {lang_note[lang]} — {len(s):,} documents\n")
        md.append(f"Correlation of the score with length (log words): ρ = {rho_len:+.2f}; "
                  f"with quality (WDS): ρ = {rho_wds:+.2f}. Decile 10 = most Finnish-leaning.\n")
        md.append(profile_table(prof, top_tlds) + "\n")
        md.append(f"\n**Most Finnish-leaning {N_READ}**\n")
        md += [doc_line(R[lang][i], s[i]) for i in order[::-1][:N_READ]]
        md.append(f"\n**Most English-leaning {N_READ}**\n")
        md += [doc_line(R[lang][i], s[i]) for i in order[:N_READ]]


def neighbours(Q: np.ndarray, D: np.ndarray, k: int = K, chunk: int = 500) -> np.ndarray:
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
    out = []
    for i in range(0, len(Qn), chunk):
        sims = Qn[i:i + chunk] @ Dn.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        out.append(np.take_along_axis(top, order, axis=1))
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fi", default="/scratch/project_462001491/tlundber/data/hplt-samples/v3/fin_Latn_sample_exploded.jsonl")
    ap.add_argument("--en", default="/scratch/project_462001491/tlundber/data/hplt-samples/v3/eng_Latn_sample_exploded.jsonl")
    ap.add_argument("--par", default="/scratch/project_462001491/tlundber/data/europarl/sample-s20261018-n20000")
    ap.add_argument("--par2", default="/scratch/project_462001491/tlundber/data/infopankki/sample-s20261019-n14419",
                    help="Second parallel corpus (infopankki: life in Finland, both languages); '' to skip")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    from residual_geometry.data.loaders import load_hplt_records
    from residual_geometry.embed.cache import EmbeddingCache
    from residual_geometry.embed.encoder import Embedder
    from residual_geometry.erase import INLPEraser

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    cache = EmbeddingCache(args.model)
    embedder = Embedder(model_name=args.model, device="cpu")  # only used on a cache miss

    # --- Data: non-MT web documents, Europarl pairs ---
    R, X = {}, {}
    for lang, path in (("fi", args.fi), ("en", args.en)):
        recs = [r for r in load_hplt_records(path) if not r["mt"]]
        R[lang] = recs
        X[lang] = cache.embed(embedder, [r["text"] for r in recs]).astype(np.float64)
        print(f"{lang}: {len(recs)} non-MT documents", flush=True)
    par = {}
    for lang in ("fi", "en"):
        with open(f"{args.par}.{lang}", encoding="utf-8") as f:
            par[lang] = [line.strip() for line in f if line.strip()]
    P = {lang: cache.embed(embedder, par[lang]).astype(np.float64) for lang in par}
    n_par = len(par["fi"])
    held = rng.choice(n_par, size=2000, replace=False)
    fit_mask = np.ones(n_par, bool)
    fit_mask[held] = False

    # --- Erasers, fitted on web text (a balanced subset for INLP) ---
    mu = {lang: X[lang].mean(0) for lang in X}
    n_fit = min(10_000, len(X["fi"]), len(X["en"]))
    fi_fit = rng.choice(len(X["fi"]), n_fit, replace=False)
    en_fit = rng.choice(len(X["en"]), n_fit, replace=False)
    inlp = INLPEraser().fit(
        np.vstack([X["fi"][fi_fit], X["en"][en_fit]]),
        np.array(["fi"] * n_fit + ["en"] * n_fit))
    Pm = inlp.projection_matrix
    Qb, _ = np.linalg.qr(inlp.directions.T)  # orthonormal basis of the removed subspace
    print(f"INLP removed {inlp.n_directions_removed} directions", flush=True)

    from residual_geometry.erase import LEACEEraser

    web_X = np.vstack([X["fi"][fi_fit], X["en"][en_fit]])
    web_y = np.array(["fi"] * n_fit + ["en"] * n_fit)
    leace = LEACEEraser().fit(web_X, web_y)
    mu_all = np.vstack([X["fi"], X["en"]]).mean(0)

    def erase(Z: np.ndarray, lang: str, how: str) -> np.ndarray:
        if how == "raw":
            return Z
        if how == "global":
            return Z - mu_all
        if how == "mean":
            return Z - mu[lang]
        if how == "leace":
            return leace.transform(Z).astype(np.float64)
        if how == "transdir":
            return transdir.transform(Z).astype(np.float64)
        return Z @ Pm.T

    # --- Directions ---
    d_web = unit(mu["fi"] - mu["en"])
    d_par = unit((P["fi"][fit_mask] - P["en"][fit_mask]).mean(0))
    cos_wp = float(d_web @ d_par)
    d_cult = unit(d_web - cos_wp * d_par)
    from residual_geometry.erase import DirectionEraser as _DE
    transdir = _DE(d_par)
    in_inlp = lambda v: float(np.linalg.norm(Qb.T @ v))  # share of a unit vector inside it
    gap_web = float(np.linalg.norm(mu["fi"] - mu["en"]))
    gap_par = float(np.linalg.norm((P["fi"][fit_mask] - P["en"][fit_mask]).mean(0)))
    mean_norm = float(np.mean(np.linalg.norm(np.vstack([X["fi"], X["en"]]), axis=1)))

    # A second parallel corpus: does the pure-language direction depend on what the
    # translations are about? (Europarl: EU parliament; infopankki: living in Finland.)
    par2 = None
    if args.par2:
        P2 = {}
        for lang in ("fi", "en"):
            with open(f"{args.par2}.{lang}", encoding="utf-8") as f:
                P2[lang] = cache.embed(embedder, [l.strip() for l in f if l.strip()]).astype(np.float64)
        d_par2 = unit((P2["fi"] - P2["en"]).mean(0))
        par2 = {"name": Path(args.par2).parent.name, "n_pairs": len(P2["fi"]),
                "cos_with_parallel": round(float(d_par2 @ d_par), 3),
                "cos_with_web": round(float(d_par2 @ d_web), 3),
                "cos_with_cultural": round(float(d_par2 @ d_cult), 3),
                "inside_inlp_subspace": round(in_inlp(d_par2), 3)}

    metrics: dict = {
        "model": args.model,
        "n_docs": {lang: len(R[lang]) for lang in R},
        "n_parallel_pairs": n_par,
        "inlp_directions_removed": inlp.n_directions_removed,
        "directions": {
            "cos_web_parallel": round(cos_wp, 3),
            "web_gap_norm": round(gap_web, 4),
            "parallel_gap_norm": round(gap_par, 4),
            "mean_doc_norm": round(mean_norm, 4),
            "web_direction_inside_inlp_subspace": round(in_inlp(d_web), 3),
            "parallel_direction_inside_inlp_subspace": round(in_inlp(d_par), 3),
            "cultural_direction_inside_inlp_subspace": round(in_inlp(d_cult), 3),
        },
        "second_parallel": par2,
    }

    md: list[str] = [f"# Looking round — {args.model}\n",
                     f"*{len(R['fi']):,} Finnish + {len(R['en']):,} English HPLT v3 documents "
                     f"(machine-translated dropped); {n_par:,} Europarl pairs (2,000 held out for "
                     f"retrieval). INLP fitted on {n_fit:,} + {n_fit:,} web documents, "
                     f"{inlp.n_directions_removed} directions removed. Generated by "
                     f"`experiments/looking_round.py`.*\n"]

    md.append("\n## The directions\n")
    dm = metrics["directions"]
    md.append(
        f"- **Web language direction** (Finnish web mean − English web mean): length {gap_web:.3f} "
        f"(a typical document vector has length {mean_norm:.3f}).\n"
        f"- **Parallel language direction** (mean of Finnish − English over translation pairs): "
        f"length {gap_par:.3f}.\n"
        f"- **Agreement between the two:** cosine {cos_wp:+.3f}. "
        f"The share of the web direction the parallel one does not explain: "
        f"{np.sqrt(max(0.0, 1 - cos_wp ** 2)):.2f} (sine of the angle). That remainder is the "
        f"candidate cultural direction.\n"
        f"- **How much of each lies inside what INLP removes** (1 = entirely): web {dm['web_direction_inside_inlp_subspace']:.2f} · "
        f"parallel {dm['parallel_direction_inside_inlp_subspace']:.2f} · cultural {dm['cultural_direction_inside_inlp_subspace']:.2f}.\n")

    if par2:
        md.append(
            f"- **Second translation corpus ({par2['name']}, {par2['n_pairs']:,} pairs, about living in "
            f"Finland):** cosine {par2['cos_with_parallel']:+.3f} with the Europarl direction, "
            f"{par2['cos_with_web']:+.3f} with the web direction, {par2['cos_with_cultural']:+.3f} with the "
            f"cultural direction; {par2['inside_inlp_subspace']:.2f} inside what INLP removes. If the two "
            f"translation directions agree, the pure-language direction does not depend on subject matter.\n")

    # --- Lens 1 and 3: read along the directions ---
    md.append("\n## Lens 1 — what the language direction separates\n")
    lang_note = {"fi": "Finnish documents", "en": "English documents"}
    read_direction("web language direction", d_web, X, R, md, metrics, lang_note)
    md.append("\n## Lens 3 — language versus culture\n")
    md.append("The cultural direction is the web direction with the parallel (pure-language) "
              "direction projected out. High = what Finnish web pages carry that translation does not.\n")
    read_direction("cultural direction", d_cult, X, R, md, metrics, lang_note)

    # --- Lens 2a: translation retrieval on held-out Europarl pairs ---
    md.append("\n## Lens 2 — retrieval\n\n### Do translations still find each other? (2,000 held-out Europarl pairs)\n")
    md.append("| Eraser | FI→EN P@1 | EN→FI P@1 |\n|---|---|---|")
    metrics["europarl_p_at_1"] = {}
    for how in ERASERS:
        A = erase(P["fi"][held], "fi", how)
        B = erase(P["en"][held], "en", how)
        f2e = float(np.mean(neighbours(A, B, k=1)[:, 0] == np.arange(len(held))))
        e2f = float(np.mean(neighbours(B, A, k=1)[:, 0] == np.arange(len(held))))
        metrics["europarl_p_at_1"][how] = {"fi_en": round(f2e, 4), "en_fi": round(e2f, 4)}
        md.append(f"| {how} | {f2e:.3f} | {e2f:.3f} |")

    # --- Lens 2b: web neighbours, Finnish queries over all English documents ---
    single_fi = np.array([i for i, r in enumerate(R["fi"]) if r["register"] not in ("hybrid", "none")])
    queries = rng.choice(single_fi, size=min(N_QUERIES, len(single_fi)), replace=False)
    en_reg = np.array([r["register"] for r in R["en"]])
    en_tld = np.array([r["tld"] for r in R["en"]])
    q_reg = np.array([R["fi"][i]["register"] for i in queries])
    q_fi_domain = np.array([R["fi"][i]["tld"] == "fi" for i in queries])
    nb = {how: neighbours(erase(X["fi"][queries], "fi", how), erase(X["en"], "en", how))
          for how in ERASERS}
    base_fi_tld = float(np.mean(en_tld == "fi"))
    md.append(f"\n### Nearest English documents of {len(queries):,} Finnish documents (top {K})\n")
    md.append(f"Register agreement counts only neighbours that have a single register. "
              f"`.fi` share: English pages on Finnish domains (base rate {base_fi_tld:.2%} of all English documents), "
              f"split by whether the Finnish query itself is on a `.fi` domain.\n")
    md.append("| Eraser | Same register | `.fi` neighbours (query on .fi) | `.fi` neighbours (query elsewhere) | Overlap with raw top-10 |\n|---|---|---|---|---|")
    metrics["web_neighbours"] = {"base_rate_fi_tld_en": round(base_fi_tld, 4)}
    for how in ERASERS:
        N = nb[how]
        reg = en_reg[N]
        single = ~np.isin(reg, ["hybrid", "none"])
        same = float(np.sum((reg == q_reg[:, None]) & single) / max(1, np.sum(single)))
        fi_dom = (en_tld[N] == "fi")
        on, off = float(fi_dom[q_fi_domain].mean()), float(fi_dom[~q_fi_domain].mean())
        overlap = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(N, nb["raw"])]))
        metrics["web_neighbours"][how] = {"same_register": round(same, 3), "fi_tld_query_fi": round(on, 4),
                                          "fi_tld_query_other": round(off, 4), "overlap_with_raw": round(overlap, 3)}
        md.append(f"| {how} | {same:.1%} | {on:.2%} | {off:.2%} | {overlap:.0%} |")

    md.append(f"\n### {N_SHOWCASE} Finnish documents and their nearest English documents\n")
    md.append("Top 3 neighbours before erasure (raw) and after INLP. Queries drawn across registers.\n")
    by_reg: dict = {}
    for qi, i in enumerate(queries):
        by_reg.setdefault(q_reg[qi], []).append(qi)
    picks = []
    while len(picks) < N_SHOWCASE and any(by_reg.values()):
        for reg in sorted(by_reg):
            if by_reg[reg] and len(picks) < N_SHOWCASE:
                picks.append(by_reg[reg].pop(0))
    for qi in picks:
        r = R["fi"][queries[qi]]
        md.append(f"\n#### Query: {r['register']} · `{r['host']}`\n> {snippet(r['text'], 400)}\n")
        for how in ("raw", "inlp"):
            md.append(f"\n*{how}:*\n")
            md += [doc_line(R["en"][j]) for j in nb[how][qi][:3]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved → {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
