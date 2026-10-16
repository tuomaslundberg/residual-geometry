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

*p-values above are Spearman's, which treat the 21 pairs as independent. Superseded by the permutation nulls below (2026-10-16).*

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

## Permutation nulls for the centroid geometry — 2026-10-16

*Re-run of the Phase 2 pilot (LaBSE + HPLT v3, same 5 000 + 4 999 docs, same seed) with the
label-permutation nulls reported instead of discarded. Output: `outputs/pilot_phase2_nulls.json`
(job 22919409). The observed scores reproduce the June run exactly.*

**What the null is.** Within each language, register labels are shuffled among the documents,
centroids are recomputed and RSA / Procrustes rescored; 1 000 times. That gives the score a
*meaningless* grouping of the same documents, with the same class sizes, reaches under the same
eraser — the chance level. The permutation p-value is the share of shuffles scoring at least as
well as the real labels (+1 correction; floor 1/1001 ≈ 0.001).

### RSA (higher = more aligned)

| Condition | Observed | Chance mean | Chance 5–95 % | Perm. p |
|---|---|---|---|---|
| Identity | 0.354 | 0.696 | 0.578–0.796 | 1.000 |
| Mean | **0.851** | 0.413 | 0.130–0.662 | **0.001** |
| INLP | 0.547 | 0.695 | 0.578–0.795 | 0.988 |

### Procrustes disparity (lower = more aligned)

| Condition | Observed | Chance mean | Chance 5–95 % | Perm. p |
|---|---|---|---|---|
| Identity | 0.109 | 0.185 | 0.126–0.254 | 0.017 |
| Mean | 0.109 | 0.185 | 0.126–0.254 | 0.017 |
| INLP | 0.136 | 0.185 | 0.124–0.253 | 0.106 |

**Reading.**

- **After mean centering, 0.851 is well above chance.** The best of 1 000 shuffles stays below it
  (95th percentile 0.66). So mean centering does *not* raise the score for any grouping
  whatsoever: register lines up across FI and EN better than meaningless groupings do.
- **But the chance level is not low, and it is not the same under each eraser.** Shuffled
  labels score ~0.70 with the language signal in place, ~0.41 after mean centering. A raw RSA
  number therefore has no fixed meaning here; it must be read against its own null.
- **Without erasure, real registers score *below* chance (0.354 vs 0.696, p = 1.0).** The
  "0.355 → 0.851 jump" is not a move from "no alignment" to "strong alignment": the identity
  baseline is depressed below what random groupings get. Compare each score to its own chance
  level, not the two observed scores to each other.
- **INLP's 0.547 is below chance (p = 0.99).** The Sprint 2 table's p = 0.010 for INLP came from
  the Spearman test, which treats the 21 centroid pairs as independent; it does not hold up.
- **Procrustes** is modestly above chance under identity and mean (identical, since it is
  translation-invariant; p = 0.017), not under INLP (p = 0.11).

**Why random groupings score so high: class sizes.** A centroid of a small class is noisier,
so it lies farther from the others. Both languages have the same small classes (SP 44/118,
ID 138/46, HI 145/142) and the same large ones (IN, NA), so the two distance matrices agree on
"which centroids are far off" even when the labels mean nothing. A simulation confirms this:
pure Gaussian noise plus a language mean, no register signal, real class sizes → RSA ≈ 0.74
(identity) / 0.67 (mean); same simulation with equal class sizes → ≈ 0.00 / 0.00. Part of what
centroid RSA measures on this data is therefore class imbalance, not register.

**Caveats (stated, not fixed).** 7 centroids → 21 pairwise distances, effectively fewer;
SP is nearly empty (n = 44 FI, F1 0.0). p = 0.001 is the floor at 1 000 permutations. The
class-size effect would disappear with class-balanced subsampling (equal n per register per
language) — a design choice for the next sprint, together with the 20k+ scale-up.


---

## Class-balanced re-run — 2026-10-17

*Follows from the section above: unequal class sizes alone produce a high chance level. Same
HPLT v3 files, now read in full (100k docs/language), machine-translated documents dropped
(MT ≥ 0.4), LY left out (too rare), **385 documents per register per language** (the Finnish
SP count under the stricter rule), **20 seeded draws**, 1 000 permutations each. Two register
rules: *argmax* (strongest main register ≥ 0.4, as before) and *single* (exactly one main
register ≥ 0.4). Embeddings from the cache. Outputs: `outputs/phase2_balanced/<rule>/draw_NN.json`
(job 22922027), table from `scripts/summarise_balanced.py`. Values are median [min, max] over
the 20 draws.*

### RSA

| Rule | Eraser | Observed | Chance mean | Chance 95th pct | Draws above chance (p < 0.05) |
|---|---|---|---|---|---|
| argmax | Identity | 0.716 [0.625, 0.804] | 0.00 | 0.424 | 20/20 |
| argmax | Mean | 0.785 [0.733, 0.870] | 0.00 | 0.416 | 20/20 |
| argmax | INLP | 0.614 [0.373, 0.770] | 0.00 | 0.424 | 19/20 |
| single | Identity | 0.719 [0.639, 0.795] | 0.00 | 0.418 | 20/20 |
| single | Mean | **0.819 [0.782, 0.888]** | 0.00 | 0.418 | 20/20 |
| single | INLP | 0.667 [0.543, 0.770] | 0.00 | 0.420 | 20/20 |

### Cross-lingual transfer (FI classifier → EN, macro-F1; chance 0.143)

| Rule | Identity | Mean | INLP |
|---|---|---|---|
| argmax | 0.595 | 0.630 | 0.585 |
| single | 0.632 | **0.670** | 0.660 |

### Paired differences (same documents within a draw)

| Rule | Comparison | RSA difference | Draws positive | F1 difference | Draws positive |
|---|---|---|---|---|---|
| argmax | Mean − Identity | +0.068 | 19/20 | +0.032 | 20/20 |
| single | Mean − Identity | +0.117 | 20/20 | +0.037 | 20/20 |
| argmax | INLP − Identity | −0.100 | 2/20 | −0.004 | 6/20 |
| single | INLP − Identity | −0.056 | 4/20 | +0.025 | 18/20 |

Language probe selectivity: identity +0.50, mean −0.16, INLP −0.15 (INLP removes 7–9 directions).

**Reading.**

- **With balanced classes the chance level is ≈ 0 under every eraser**, as the simulation
  predicted. RSA scores can now be read directly.
- **Register geometry is already aligned across FI and EN without any erasure: RSA ≈ 0.72,
  far above chance in every draw.** The Sprint 2 identity score of 0.355 was depressed by the
  class-size effect, not a sign of missing alignment.
- **Mean centering raises alignment modestly and consistently:** +0.07 (argmax) / +0.12
  (single) in RSA, +0.03–0.04 in transfer F1, positive in 19–20 of 20 draws. So the headline
  becomes "language removal sharpens an alignment that is already there", not "removal
  reveals a hidden alignment" (0.355 → 0.851).
- **INLP does not help RSA** (lower than identity in 16–18 of 20 draws) and is mixed on
  transfer. Mean centering beats INLP on RSA in every draw. The Sprint 2 conclusion "mean
  centering > INLP" holds.
- **Single-register documents give cleaner centroids:** higher RSA after mean centering
  (0.82 vs 0.79) and higher transfer F1 throughout. Supports using prototypical documents in
  the final study.
- **Transfer F1 roughly doubles versus Sprint 2 (0.35 → 0.60–0.67)**: macro-F1 there was
  dragged down by tiny classes (SP F1 0.0). Balanced train and test sets fix that.

**Procrustes: the shuffle null is not informative here.** Observed disparity (≈ 0.05) is
*higher* than chance (≈ 0.03) in every draw. With balanced shuffles, each language's 7
centroids are near-equal noise around the language mean; in 768 dimensions such points form
an almost regular shape, and two regular shapes align almost perfectly after rotation. Real
register centroids have uneven distances, so they align less well than noise does. This says
nothing about register; it means rotation-based alignment of 7 points in 768-d needs a
different reference (e.g. dimensionality reduction first, or a null that keeps the shape).
The imbalanced Procrustes p-values in the previous section are subject to the same doubt.

**Caveats.** Still 7 centroids / 21 distances per draw; one corpus, one encoder (LaBSE).
The draws share documents (385 of 385 SP every time in FI under *single*), so their spread
understates true sampling variability for the rarest class.

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
