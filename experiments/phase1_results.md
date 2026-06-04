# Phase 1 pilot results — 2026-06-03

**Model:** LaBSE · **Erasers:** IdentityProjector, MeanCentering, INLP (20 iterations)

## Europarl FI-EN (5 000 parallel pairs)

| Condition | Probe acc | Selectivity | Bitext P@1 | TwoNN ID |
|-----------|-----------|-------------|------------|----------|
| Identity  | 0.957     | +0.46       | 0.977      | 1.70     |
| Mean      | 0.429     | −0.07       | 0.978      | 1.54     |
| INLP (10 dirs) | 0.420 | −0.08      | 0.978      | 1.53     |

## HPLT v3 FI + EN (5 000 docs/lang, monolingual)

| Condition | Probe acc | Selectivity | TwoNN ID |
|-----------|-----------|-------------|----------|
| Identity  | 0.998     | +0.49       | 19.03    |
| Mean      | 0.355     | −0.15       | 19.23    |
| INLP (9 dirs) | 0.362 | −0.15      | 19.36    |

## Gateway verdict: passed

**Language removal** — both erasers drive probe selectivity to ≈ 0 (slightly negative, i.e. just below chance). Works on both corpora.

**Semantic preservation** (Europarl) — bitext P@1 ≈ 0.978 across all conditions, essentially unchanged by erasure. Cross-lingual alignment intact.

**Residual structure** — TwoNN intrinsic dimensionality is stable across erasure conditions. The eraser removes the language signal without collapsing the manifold geometry. However:

- **Europarl:** TwoNN ID ≈ 1.5 (near-degenerate). Europarl is too homogeneous — a single register, narrow domain. The residual has almost no geometric structure to probe.
- **HPLT:** TwoNN ID ≈ 19. A rich, roughly 19-dimensional manifold survives erasure. This is where Phase 2 happens.

## Implications for Phase 2

Use HPLT v3 as the primary corpus. The residual is geometrically rich (ID ≈ 19), meaning multiple feature axes can in principle be recovered. The question is what labels organise it. Europarl remains useful only for bitext retrieval validation (it has parallel pairs); it is not viable as a Phase 2 probing corpus.

Next: XLM-R comparison run, then Phase 2 pilot design (candidate labels: register, formality, named-entity density, dependency distance, morphological complexity as negative control).
