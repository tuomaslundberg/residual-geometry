"""Thesis analyses A-D on the HPLT 4 thesis sample, one encoder per run.

Sample: `$WORK/data/thesis-sample-v4/sample-N18000-cap500-s20261020/` — 18,000 documents per
class per language (7 registers + MT-only), WDS 9-10, one main register, <= 500 per site.
Register analyses use the 7 registers; the MT-only class is left out of them.

Removal conditions (fitted on web text; INLP and LEACE on a seeded 20k + 20k subset):
  raw · global (pooled mean subtracted: moves the origin only) · mean (each language's
  mean subtracted) · leace (equal language means, origin kept) · inlp (directions removed
  until a held-out probe is at chance) · transdir_ep / transdir_ip (the Europarl /
  infopankki translation-pair direction projected out).

A. Decompose the language direction: web direction (FI mean - EN mean; registers are
   balanced, so register mix does not enter it) against the pure-language directions from
   translation pairs; per-register web directions; bootstrap 95 % intervals.
B. Systematic reading: documents at both ends of the web and cultural directions as a
   coding sheet, decile profiles (sites, registers, topics, place-mention proxies).
C. Retrieval: translations (held-out pairs) and web neighbours, before/after removal.
D. What organises the residual: register probes, transfer, centroid RSA with permutation
   chance levels, topic probes, and the language probe (eraser fitted inside each fold).

  python experiments/thesis_analyses.py --model sentence-transformers/LaBSE --out outputs/thesis/LaBSE
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

WORK = "/scratch/project_462001491/tlundber"
SAMPLE = f"{WORK}/data/thesis-sample-v4/sample-N18000-cap500-s20261020"
PARALLEL = {
    "ep": f"{WORK}/data/europarl/sample-s20261018-n20000",
    "ip": f"{WORK}/data/infopankki/sample-s20261019-n14419",
}
REGISTERS = ["HI", "ID", "IN", "IP", "NA", "OP", "SP"]
CONDITIONS = ["raw", "global", "mean", "leace", "inlp", "transdir_ep", "transdir_ip"]

# Crude, countable place-mention proxies (B, C): word-initial matches in either language.
PLACE = {
    "finland": r"\b(suomi|suome[nsl]|suomalai|finland|finnish|finn\b|helsin|tampere|turku|turu[ns]|"
               r"oulu|espoo|vantaa|jyväskylä|kuopio|lahde[ns]|lahti|lappi|lapin|lapland|kela\b|eduskun|yle\b)",
    "us": r"\b(usa\b|yhdysval|amerik|america|united states|washington|new york|california|kaliforni|"
          r"texas|teksas|trump|obama|biden|congress|kongress|senate\b|senaat)",
}
PLACE_RE = {k: re.compile(v, re.IGNORECASE) for k, v in PLACE.items()}


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def snippet(text: str, n: int = 400) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def load_lang(lang: str, cache, embedder) -> tuple[list[dict], np.ndarray]:
    rows = []
    with open(Path(SAMPLE) / f"{lang}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            host = (urlsplit(d.get("url") or "").hostname or "").lower()
            words = d["text"].split()
            head = " ".join(words[:500])
            rows.append({
                "id": d["id"], "text": d["text"], "label": d["label"], "url": d.get("url"),
                "host": host, "tld": host.rsplit(".", 1)[-1] if "." in host else "",
                "wds": d.get("wds"), "words": len(words),
                "finland": bool(PLACE_RE["finland"].search(head)),
                "us": bool(PLACE_RE["us"].search(head)),
            })
    X = cache.embed(embedder, [r["text"] for r in rows]).astype(np.float64)
    return rows, X


def load_topics(topic_dir: Path | None, lang: str) -> tuple[dict, dict] | tuple[None, None]:
    if topic_dir is None or not (topic_dir / f"doc-topics-{lang}.tsv").exists():
        return None, None
    meta = json.loads((topic_dir / f"topics-{lang}.json").read_text())
    names = {t["topic"]: " ".join(t["top_words"][:4]) for t in meta["topics"]}
    by_id = {}
    with open(topic_dir / f"doc-topics-{lang}.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            i, t, w, _ = line.rstrip("\n").split("\t")
            by_id[i] = int(t)
    return by_id, names


def neighbours(Q: np.ndarray, D: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
    out = []
    for i in range(0, len(Qn), chunk):
        sims = Qn[i:i + chunk] @ Dn.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        out.append(np.take_along_axis(top, order, axis=1))
    return np.vstack(out)


def p_at_1(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.mean(neighbours(A, B, 1)[:, 0] == np.arange(len(A))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topics", default="outputs/thesis/topics")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--n-read", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--probe-per-class", type=int, default=3000)
    ap.add_argument("--n-queries", type=int, default=3000)
    args = ap.parse_args()

    from residual_geometry.analyse.geometry import evaluate_geometry, permutation_summary
    from residual_geometry.embed.cache import EmbeddingCache
    from residual_geometry.embed.encoder import Embedder
    from residual_geometry.erase import (DirectionEraser, GlobalCenteringEraser, INLPEraser,
                                         LEACEEraser, MeanCenteringEraser)
    from residual_geometry.eval.probing import (apply_eraser, cross_lingual_transfer,
                                                language_probe, linear_probe, majority_share)

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cache = EmbeddingCache(args.model)
    embedder = Embedder(model_name=args.model, device="cpu")  # only on a cache miss
    lap = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    R, X = {}, {}
    for lang in ("fi", "en"):
        R[lang], X[lang] = load_lang(lang, cache, embedder)
    lap("sample loaded")
    reg = {l: np.array([r["label"] for r in R[l]]) for l in R}
    is_reg = {l: np.isin(reg[l], REGISTERS) for l in R}
    topic_dir = Path(args.topics) if args.topics else None
    topics = {l: load_topics(topic_dir, l) for l in R}

    P, held = {}, {}
    for name, prefix in PARALLEL.items():
        side = {}
        for lang in ("fi", "en"):
            with open(f"{prefix}.{lang}", encoding="utf-8") as f:
                side[lang] = cache.embed(embedder, [s.strip() for s in f if s.strip()]).astype(np.float64)
        P[name] = side
        h = rng.choice(len(side["fi"]), size=2000, replace=False)
        m = np.ones(len(side["fi"]), bool)
        m[h] = False
        held[name] = (h, m)
    lap("parallel loaded")

    # --- Erasers, fitted on register documents ---
    Xr = {l: X[l][is_reg[l]] for l in X}
    n_fit = 20000
    fit_idx = {l: rng.choice(len(Xr[l]), n_fit, replace=False) for l in Xr}
    fit_X = np.vstack([Xr["fi"][fit_idx["fi"]], Xr["en"][fit_idx["en"]]])
    fit_y = np.array(["fi"] * n_fit + ["en"] * n_fit)
    mu = {l: Xr[l].mean(0) for l in Xr}
    d_par = {n: unit((P[n]["fi"][held[n][1]] - P[n]["en"][held[n][1]]).mean(0)) for n in P}
    erasers = {
        "raw": None,
        "global": GlobalCenteringEraser().fit(np.vstack([Xr["fi"], Xr["en"]])),
        "mean": MeanCenteringEraser().fit(np.vstack([Xr["fi"], Xr["en"]]),
                                          np.array(["fi"] * len(Xr["fi"]) + ["en"] * len(Xr["en"]))),
        "leace": LEACEEraser().fit(fit_X, fit_y),
        "inlp": INLPEraser().fit(fit_X, fit_y),
        "transdir_ep": DirectionEraser(d_par["ep"]),
        "transdir_ip": DirectionEraser(d_par["ip"]),
    }
    factories = {
        "raw": None, "global": GlobalCenteringEraser, "mean": MeanCenteringEraser,
        "leace": LEACEEraser, "inlp": INLPEraser,
        "transdir_ep": lambda: erasers["transdir_ep"], "transdir_ip": lambda: erasers["transdir_ip"],
    }

    def erase(Z: np.ndarray, lang: str, how: str) -> np.ndarray:
        if how == "raw":
            return Z
        return apply_eraser(erasers[how], Z, np.array([lang] * len(Z))).astype(np.float64)

    inlp = erasers["inlp"]
    Qb, _ = np.linalg.qr(inlp.directions.T)
    lap(f"erasers fitted (INLP removed {inlp.n_directions_removed})")
    metrics: dict = {"model": args.model, "sample": SAMPLE, "seed": args.seed,
                     "inlp_directions_removed": inlp.n_directions_removed}

    # === A. Decompose the language direction ===
    d_web = unit(mu["fi"] - mu["en"])
    span, _ = np.linalg.qr(np.stack([d_par["ep"], d_par["ip"]], axis=1))
    d_cult = unit(d_web - span @ (span.T @ d_web))

    def boot_cos(name: str) -> list[float]:
        vals = []
        nf, ne = len(Xr["fi"]), len(Xr["en"])
        fit_pairs = np.flatnonzero(held[name][1])
        for _ in range(args.n_boot):
            a = Xr["fi"][rng.integers(0, nf, nf)].mean(0) - Xr["en"][rng.integers(0, ne, ne)].mean(0)
            pp = rng.choice(fit_pairs, len(fit_pairs))
            b = (P[name]["fi"][pp] - P[name]["en"][pp]).mean(0)
            vals.append(float(unit(a) @ unit(b)))
        return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]

    A = {
        "web_gap_norm": round(float(np.linalg.norm(mu["fi"] - mu["en"])), 4),
        "mean_doc_norm": round(float(np.mean(np.linalg.norm(np.vstack([Xr["fi"], Xr["en"]]), axis=1))), 4),
        "parallel": {},
        "cos_ep_ip": round(float(d_par["ep"] @ d_par["ip"]), 3),
        "web_inside_parallel_span": round(float(np.linalg.norm(span.T @ d_web)), 3),
        "web_inside_inlp": round(float(np.linalg.norm(Qb.T @ d_web)), 3),
        "cultural_inside_inlp": round(float(np.linalg.norm(Qb.T @ d_cult)), 3),
        "per_register": {},
    }
    for n in P:
        A["parallel"][n] = {
            "gap_norm": round(float(np.linalg.norm((P[n]["fi"][held[n][1]] - P[n]["en"][held[n][1]]).mean(0))), 4),
            "cos_with_web": round(float(d_web @ d_par[n]), 3),
            "cos_with_web_ci95": boot_cos(n),
            "inside_inlp": round(float(np.linalg.norm(Qb.T @ d_par[n])), 3),
        }
    for r in REGISTERS:
        dr = unit(X["fi"][reg["fi"] == r].mean(0) - X["en"][reg["en"] == r].mean(0))
        A["per_register"][r] = {"cos_with_web": round(float(dr @ d_web), 3),
                                "cos_with_ep": round(float(dr @ d_par["ep"]), 3),
                                "cos_with_ip": round(float(dr @ d_par["ip"]), 3)}
    metrics["A"] = A
    lap("A done")

    # === B. Systematic reading: coding sheet and decile profiles ===
    B: dict = {}
    sheet = []
    for dname, d in (("web", d_web), ("cultural", d_cult)):
        B[dname] = {}
        for lang in ("fi", "en"):
            idx = np.flatnonzero(is_reg[lang])
            s = Xr[lang] @ d
            order = np.argsort(s)
            rows = [R[lang][i] for i in idx]
            tmap, tnames = topics[lang]
            prof = []
            for k, chunk in enumerate(np.array_split(order, 10)):
                rs = [rows[j] for j in chunk]
                tl = Counter(r["tld"] for r in rs)
                lb = Counter(r["label"] for r in rs)
                entry = {
                    "decile": k + 1,
                    "finland_mention": round(float(np.mean([r["finland"] for r in rs])), 3),
                    "us_mention": round(float(np.mean([r["us"] for r in rs])), 3),
                    "median_words": int(np.median([r["words"] for r in rs])),
                    "top_tlds": {t: round(c / len(rs), 3) for t, c in tl.most_common(4)},
                    "registers": {t: round(c / len(rs), 3) for t, c in sorted(lb.items())},
                }
                if tmap:
                    tc = Counter(tmap.get(r["id"]) for r in rs)
                    entry["top_topics"] = {tnames[t]: round(c / len(rs), 3) for t, c in tc.most_common(3) if t is not None}
                prof.append(entry)
            B[dname][lang] = prof
            for end, sel in (("finnish_end", order[::-1][:args.n_read]), ("english_end", order[:args.n_read])):
                for rank, j in enumerate(sel, 1):
                    r = rows[j]
                    sheet.append({
                        "model": args.model, "direction": dname, "lang": lang, "end": end, "rank": rank,
                        "score": round(float(s[j]), 4), "id": r["id"], "register": r["label"],
                        "host": r["host"], "wds": r["wds"], "words": r["words"],
                        "topic": (tnames.get(tmap.get(r["id"])) if tmap else ""),
                        "auto_finland": int(r["finland"]), "auto_us": int(r["us"]),
                        "code_place": "", "code_note": "", "text": snippet(r["text"]),
                    })
    with open(out / "coding_sheet.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]), delimiter="\t")
        w.writeheader()
        w.writerows(sheet)
    metrics["B"] = B
    lap("B done")

    # === C. Retrieval ===
    C: dict = {"translation_p_at_1": {}, "web": {}}
    for how in CONDITIONS:
        C["translation_p_at_1"][how] = {}
        for n in P:
            h = held[n][0]
            a, b = erase(P[n]["fi"][h], "fi", how), erase(P[n]["en"][h], "en", how)
            C["translation_p_at_1"][how][n] = {"fi_en": round(p_at_1(a, b), 4), "en_fi": round(p_at_1(b, a), 4)}
    q = rng.choice(np.flatnonzero(is_reg["fi"]), size=args.n_queries, replace=False)
    en_idx = np.flatnonzero(is_reg["en"])
    en_lab = reg["en"][en_idx]
    en_fin = np.array([R["en"][i]["finland"] for i in en_idx])
    q_lab = reg["fi"][q]
    q_fin = np.array([R["fi"][i]["finland"] for i in q])
    raw_nb = None
    for how in CONDITIONS:
        nb = neighbours(erase(X["fi"][q], "fi", how), erase(X["en"][en_idx], "en", how), 10)
        if how == "raw":
            raw_nb = nb
        C["web"][how] = {
            "same_register": round(float(np.mean(en_lab[nb] == q_lab[:, None])), 3),
            "neighbour_mentions_finland_if_query_does": round(float(en_fin[nb][q_fin].mean()), 3),
            "neighbour_mentions_finland_if_query_does_not": round(float(en_fin[nb][~q_fin].mean()), 3),
            "overlap_with_raw": round(float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nb, raw_nb)])), 3),
        }
    C["web_base_rate_finland_en"] = round(float(en_fin.mean()), 3)
    metrics["C"] = C
    lap("C done")

    # === D. What organises the residual ===
    D: dict = {}
    sub = {}
    for lang in ("fi", "en"):
        pick = np.concatenate([rng.choice(np.flatnonzero(reg[lang] == r), args.probe_per_class, replace=False)
                               for r in REGISTERS])
        sub[lang] = pick
    lang_sub_X = np.vstack([X["fi"][sub["fi"]], X["en"][sub["en"]]])
    lang_sub_y = np.array(["fi"] * len(sub["fi"]) + ["en"] * len(sub["en"]))
    for how in CONDITIONS:
        Zf, Ze = erase(X["fi"], "fi", how), erase(X["en"], "en", how)
        d = {}
        if how == "raw":
            lp = language_probe(lang_sub_X, lang_sub_y)
        else:
            lp = language_probe(lang_sub_X, lang_sub_y, eraser_factory=factories[how])
        d["language_accuracy"] = round(lp.accuracy, 4)
        d["fi_register_acc"] = round(linear_probe(Zf[sub["fi"]], reg["fi"][sub["fi"]]), 4)
        d["en_register_acc"] = round(linear_probe(Ze[sub["en"]], reg["en"][sub["en"]]), 4)
        d["register_chance"] = round(1 / len(REGISTERS), 4)
        d["transfer_fi_en_f1"] = round(cross_lingual_transfer(Zf[sub["fi"]], reg["fi"][sub["fi"]],
                                                              Ze[is_reg["en"]], reg["en"][is_reg["en"]]).macro_f1, 4)
        d["transfer_en_fi_f1"] = round(cross_lingual_transfer(Ze[sub["en"]], reg["en"][sub["en"]],
                                                              Zf[is_reg["fi"]], reg["fi"][is_reg["fi"]]).macro_f1, 4)
        g = evaluate_geometry(Zf[is_reg["fi"]], reg["fi"][is_reg["fi"]], Ze[is_reg["en"]], reg["en"][is_reg["en"]],
                              registers=REGISTERS, n_permutations=1000, seed=args.seed)
        d["register_rsa"] = round(g.rsa_correlation, 4)
        d["register_rsa_null"] = permutation_summary(g.rsa_correlation, g.null_rsa, True)
        for lang, Z in (("fi", Zf), ("en", Ze)):
            tmap, _ = topics[lang]
            if tmap:
                ids = [R[lang][i]["id"] for i in sub[lang]]
                ty = np.array([tmap.get(i, -1) for i in ids])
                d[f"{lang}_topic_acc"] = round(linear_probe(Z[sub[lang]], ty), 4)
                d[f"{lang}_topic_majority"] = round(majority_share(ty), 4)
        D[how] = d
        lap(f"D {how}")
    metrics["D"] = D

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    lap(f"saved -> {out}")


if __name__ == "__main__":
    main()
