# TODO — residual-geometry
*Project task list. Updated in-session. Research decisions belong in context-card-turkunlp.md.*
*Format: `- [ ]` open · `- [x]` done · `[BLOCKED]` cannot proceed without external input*

---

## State snapshot (2026-05-21)

**Pipeline restructure: done.** Package renamed to `residual-geometry`; new layout
`src/residual_geometry/{embed,erase,analyse,eval,data}`; 23 tests pass; pilot stubs in
`experiments/`. Research direction locked at 2026-05-19 sync (see context card).

**What's next: LUMI setup + Phase 1 pilot.**

---

## Immediate — LUMI setup

- [x] **Clone repo on LUMI scratch** — `/scratch/project_462000999/tlundber/residual-geometry/`
- [x] **Install extras on LUMI** — `pip install --user --no-build-isolation -e ".[dev]"`; no venv needed
- [x] **Run pytest on LUMI** — 23/23 pass (Python 3.11.0rc1, pytorch/2.7 container)
- [ ] **Fetch Europarl FI-EN data to scratch** — OPUS download or copy from existing LUMI storage;
      target: `/scratch/project_462000999/tlundber/data/europarl/europarl.fi` + `.en`

---

## Immediate — Phase 1 pilot (gateway check)

Entrypoint: `experiments/pilot_phase1.py`

- [ ] **Confirm evaluation framework** — Phase 1 measures:
  - Language probe accuracy + selectivity (Hewitt & Liang 2019) → quantifies removal completeness
  - Bitext retrieval P@1/MRR → checks semantic preservation post-removal
  - Divergence stats (pairwise L2 mean/std/p95) → characterises cross-lingual distribution shift
  - Geometry diagnostics: participation ratio, anisotropy, TwoNN intrinsic dim → residual structure
  These are already wired in `pilot_phase1.py`. Confirm with supervisor if any are unexpected.
- [ ] **Run Phase 1 pilot** (LaBSE, Europarl 5k pairs, identity + mean + INLP)
  ```bash
  python experiments/pilot_phase1.py \
      --fi data/europarl/europarl.fi \
      --en data/europarl/europarl.en \
      --out outputs/pilot_phase1.json
  ```
- [ ] **Inspect results** — key gateway questions:
  - Does INLP reduce language probe accuracy to near-chance (selectivity ≈ 0)?
  - Does bitext retrieval stay high (P@1 > 0.8) after removal?
  - Does the residual have non-trivial TwoNN ID (not 1–2, not equal to original)?

---

## Short-term (after Phase 1 passes)

- [ ] **XLM-R baseline** — run Phase 1 pilot with `FacebookAI/xlm-roberta-base` for encoder comparison
- [ ] **Phase 2 pilot design** — decide candidate labels for residual probing:
      domain (Europarl committee topic), source type, formality, or combinations;
      implement `experiments/pilot_phase2.py`
- [ ] **SLURM script** — update `experiments/slurm/run_baseline.sh` for Phase 1
      (currently wired to old `run_pipeline.py` entry point)
- [ ] **Visualisation** — UMAP of residual coloured by encoder, language, and candidate labels;
      add to `src/residual_geometry/visualise.py` (optional deps already declared)
- [ ] **`scripts/prepare_data.py`** — convert CORE corpus to JSONL if register labels needed for Phase 2

---

## LEACE (requires optional dep)

- [ ] Confirm `concept-erasure` available on LUMI (`pip install concept-erasure` or from wheel)
- [ ] Add LEACE to Phase 1 eraser sweep once dep confirmed
- [ ] Add a test fixture that exercises `LEACEEraser` end-to-end

---

## Infrastructure

- [ ] Update `.gitignore` — add `outputs/`, `data/`, `*.npz`, `*.json` result files if not already
- [ ] LUMI module documentation — record exact `module load` string used after first successful run;
      add to `experiments/lumi-onboarding.md`

---

## Done

- [x] Repo rename: `register-geometry` → `residual-geometry`; local folder + GitHub remote updated
- [x] Package restructure: `src/residual_geometry/{embed,erase,analyse,eval,data}` created
- [x] New erasers: `INLPEraser` (with `projection_matrix` + `n_directions_removed`),
      `MeanCenteringEraser`, `MikolovProjection`, `LEACEEraser`
- [x] New analysis: `participation_ratio`, `anisotropy`, `twonn_intrinsic_dim`, `divergence_stats`
- [x] New eval: `language_probe` (selectivity), `linear_probe`, `bitext_retrieval`
- [x] Data: `load_europarl`, `load_infopankki` loaders added
- [x] Old `register_geometry` archived to `_archive/`
- [x] 23 tests migrated (import paths only); all pass locally
- [x] `experiments/pilot_phase1.py` runnable; `experiments/pilot_phase2.py` stub
- [x] `pyproject.toml`: name → `residual-geometry`, added `scikit-dimension>=0.3`
- [x] `README.md` rewritten
- [x] Overleaf thesis scaffold: all 8 chapters written, bibliography complete (22 entries)
