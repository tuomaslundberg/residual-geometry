# Sprint 2 results — 2026-06-18

**Models:** LaBSE (sentence-transformers/LaBSE), XLM-R base (FacebookAI/xlm-roberta-base)
**Corpus:** HPLT v3 FI + EN, 5 000 docs/lang · **Erasers:** Identity, MeanCentering, INLP (20 iter)

---

## Encoder comparison: LaBSE vs XLM-R base (Phase 1 replicated)

| | LaBSE identity | XLM-R identity |
|---|---|---|
| Language probe selectivity | +0.49 | +0.50 |
| Anisotropy | 0.35 | **0.998** |
| Participation ratio | 76.4 | **5.8** |
| TwoNN ID | 19.0 | 22.2 |

XLM-R base has near-perfect anisotropy — the representation space is almost entirely
collapsed into a single dominant direction. PR = 5.8 means only ~6 effective dimensions
are in use (vs LaBSE's 76). This is the well-known degenerate geometry of raw transformer
LMs without semantic fine-tuning.

**After erasure:**

| Condition | XLM-R selectivity | XLM-R anisotropy | XLM-R PR | XLM-R TwoNN ID | n_dirs |
|---|---|---|---|---|---|
| Identity | +0.50 | 0.998 | 5.8 | 22.2 | — |
| Mean | −0.065 | 0.008 | 15.2 | 16.3 | — |
| INLP | −0.074 | 0.999 | 15.1 | 22.5 | **1** |

Key findings:

- **XLM-R's language signal is rank-1.** INLP removes only 1 direction (vs 9–10 for LaBSE)
  and that single direction accounts for essentially all language separability AND all
  anisotropy. Mean centering confirms this: removing the mean shift collapses anisotropy
  from 0.998 to 0.008.
- **TwoNN ID is stable under INLP (22.2 → 22.5) but drops under mean centering (22.2 → 16.3).**
  INLP preserves the local manifold structure; mean centering changes the scale relationships
  (Euclidean-distance-based ID is not translation-invariant).
- **Despite degenerate global geometry, local manifold complexity is higher in XLM-R (22) than
  LaBSE (19).** PR/anisotropy measure global variance distribution; TwoNN ID measures local
  neighbourhood structure — these can diverge.
- **The language direction is a general property of multilingual encoders, not LaBSE-specific.**
  But its character differs: in XLM-R it IS the dominant direction; in LaBSE it is one of
  several.

---

## Phase 2 pilot: register probing on LaBSE + HPLT v3

**Setup:** 7 shared CORE classes (HI, ID, IN, IP, NA, OP, SP); min_confidence = 0.4.
Class distribution is imbalanced (IN ≈ 1600, NA ≈ 1800, SP = 44). SP F1 = 0.0 throughout
(too few FI examples); expected and acceptable for pilot.

### Within-language register accuracy (erased embeddings)

| Condition | FI | EN |
|---|---|---|
| Identity | 0.735 | 0.705 |
| Mean | 0.734 | 0.705 |
| INLP | 0.703 | 0.693 |

Register is strongly decodable (~73% on 7-class) and barely changes with erasure. Confirms
register lives in the language-neutral component.

### Cross-lingual transfer: FI classifier → EN (macro-F1)

| Condition | Transfer F1 | vs. chance (0.143) |
|---|---|---|
| Identity | 0.359 | +0.22 |
| Mean | 0.354 | +0.21 |
| INLP | 0.343 | +0.20 |

**Register survives language erasure and remains cross-lingually transferable.** F1 is
stable across all three conditions — erasure changes almost nothing. Register is not
co-encoded with the language direction.

Per-class (identity / mean / INLP): IN 0.56/0.66/0.62 · NA 0.61/0.57/0.51 ·
OP 0.52/0.42/0.48 · IP 0.37/0.43/0.45 · HI 0.34/0.21/0.13 · ID 0.12/0.20/0.22 · SP 0.0

### Centroid geometry (Procrustes + RSA)

| Condition | Procrustes disparity | RSA | p-value |
|---|---|---|---|
| Identity | 0.109 | 0.355 | 0.115 (n.s.) |
| Mean | 0.109 | **0.851** | <0.000001 |
| INLP | 0.136 | 0.547 | 0.010 |

**Headline finding:** RSA jumps from 0.355 (n.s.) to 0.851 (p < 10⁻⁶) after mean centering.
The FI/EN register centroid geometry is not significantly aligned in the full space, but
becomes very strongly aligned after removing the language direction.

Why: RSA uses cosine distance, which is sensitive to vector orientation. Before centering,
all FI centroids lean toward the FI language mean and EN centroids toward the EN language
mean — different reference frames. The language direction "masks" the cross-lingual register
geometry. After centering, both sets are around the origin and the angular structure reflects
register alone.

Procrustes is unchanged (0.109 → 0.109) because it optimally rotates before measuring
disparity — it is invariant to the mean shift. INLP gives lower RSA than mean centering
(0.547 vs 0.851) and slightly higher Procrustes disparity (0.136) — the extra directions
INLP removes apparently carry some cross-lingual register alignment.

---

## Methodological notes (this sprint)

- **Residuals are not re-normalized.** Erasers return raw projected/centered vectors. Scale
  effects are tiny for our settings (10/768 dimensions projected = <1% norm reduction) but
  should be acknowledged in the methodology writeup. TwoNN ID and PR are Euclidean-based
  and scale-sensitive; RSA/anisotropy/Procrustes are scale-invariant. Consider
  `--normalize-residual` flag for future robustness checks.
- **XLM-R loading:** `Embedder._load()` now falls back to `ST(modules=[Transformer, Pooling])`
  when the model has no `sentence_bert_config.json`, enabling any HF model as encoder.
- **`load_hplt` extended:** `return_labels=True` returns hard register labels via argmax over
  8 main CORE classes at `min_confidence=0.4` (CORE-optimised threshold for EN/FI).

---

## Next (post-summer)

Design an extensive grid over:
- Encoders: LaBSE, XLM-R base, + at least one more (e5-multilingual, mpnet-multilingual)
- Erasure methods: Identity, Mean, INLP, LEACE
- Feature axes: register (current), morphological complexity (negative control),
  mean dependency distance (UD parse), NE density, sentence length (covariate)
- Corpus expansion: infopankki (parallel, high quality), Tatoeba (informal register contrast)
- Scale: increase to 20k+ docs for better class coverage (especially SP, ID)
- Consider `--normalize-residual` comparison for TwoNN ID robustness
