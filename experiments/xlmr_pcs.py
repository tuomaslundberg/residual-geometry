"""What XLM-R's top principal components are (thesis sample, 2026-10-21): correlations
with language, length, digit and upper-case share; mean score per class; documents at each end."""
import sys, re, numpy as np
sys.path.insert(0, "experiments")
from thesis_analyses import load_lang, REGISTERS
from residual_geometry.embed.cache import EmbeddingCache
from scipy.stats import spearmanr
c = EmbeddingCache("FacebookAI/xlm-roberta-base")
R, X = {}, {}
for l in ("fi", "en"):
    R[l], X[l] = load_lang(l, c, None)
P = np.vstack([X["fi"], X["en"]]); mu = P.mean(0)
rng = np.random.default_rng(0); sub = rng.choice(len(P), 40000, replace=False)
_, S, Vt = np.linalg.svd(P[sub] - mu, full_matrices=False)
rows = R["fi"] + R["en"]
isfi = np.r_[np.ones(len(R["fi"])), np.zeros(len(R["en"]))]
w = np.log([r["words"] for r in rows])
def share(f): return np.array([sum(f(ch) for ch in r["text"][:2000]) / max(1, len(r["text"][:2000])) for r in rows])
digits, upper = share(str.isdigit), share(str.isupper)
regs = np.array([r["label"] for r in rows])
for k in (0, 1, 2):
    s = (P - mu) @ Vt[k]
    print(f"PC{k+1}: rho log-words {spearmanr(s, w).statistic:+.2f}  language {spearmanr(s, isfi).statistic:+.2f}  "
          f"digits {spearmanr(s, digits).statistic:+.2f}  uppercase {spearmanr(s, upper).statistic:+.2f}")
    print("   mean by class:", {r: round(float(s[regs == r].mean()), 3) for r in REGISTERS + ["MT"]})
    o = np.argsort(s)
    for name, sel in (("low", o[:4]), ("high", o[-4:])):
        for i in sel:
            t = re.sub(r"\s+", " ", rows[i]["text"])[:110]
            print(f"   {name} {rows[i]['label']} {rows[i]['host'][:25]}: {t}")
