# TODO — residual-geometry
*Project task list. Updated in-session. Research decisions belong in context-card-turkunlp.md.*
*Format: `- [ ]` open · `- [x]` done · `[BLOCKED]` cannot proceed without external input*

---

## State snapshot (2026-06-04)

**Phase 1 pilot: done.** Gateway passed on both Europarl (parallel) and HPLT v3 (monolingual).
Key finding: HPLT residual has TwoNN ID ≈ 19 (rich structure); Europarl ≈ 1.5 (near-degenerate, too homogeneous).
HPLT is the right corpus for Phase 2. See `experiments/phase1_results.md`.

**What's next: XLM-R comparison + Phase 2 pilot design.**

---

## Immediate — LUMI setup

- [x] **Clone repo on LUMI scratch** — `/scratch/project_462000999/tlundber/projects/residual-geometry/`
- [x] **Install extras on LUMI** — `pip install --user --no-build-isolation -e ".[dev]"`; no venv needed
- [x] **Run pytest on LUMI** — 23/23 pass (Python 3.11.0rc1, pytorch/2.7 container)
- [x] **Fetch Europarl FI-EN data to scratch** — `data/europarl/Europarl.en-fi.{fi,en}`

---

## Immediate — Phase 1 pilot (gateway check) — DONE

- [x] **Run Phase 1 pilot — Europarl** (LaBSE, 5k pairs, identity + mean + INLP): passed
- [x] **Run Phase 1 pilot — HPLT v3** (LaBSE, 5k docs/lang, identity + mean + INLP): passed
- [x] **SLURM scripts** — `sl-run-baseline` and `sl-run-hplt` with `SLURM_SUBMIT_DIR` fix

---

## Short-term (Phase 2)

- [ ] **XLM-R comparison** — run Phase 1 pilot with `FacebookAI/xlm-roberta-base` (or `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`); compare TwoNN ID and probe selectivity
- [ ] **Phase 2 pilot design** — decide candidate feature labels; implement `experiments/pilot_phase2.py`
      Primary corpus: HPLT v3. Candidates: register (HPLT web-register field), formality, NE density,
      dependency distance, morphological complexity (negative control)
- [ ] **Visualisation** — UMAP of residual coloured by encoder, language, and candidate labels

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
