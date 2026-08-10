# TODO — residual-geometry
*Project task list. Updated in-session. Research decisions belong in context-card-turkunlp.md.*
*Format: `- [ ]` open · `- [x]` done · `[BLOCKED]` cannot proceed without external input*

---

## State snapshot (2026-06-18)

**Sprint 2 complete.** XLM-R comparison and Phase 2 register pilot both done.
Key findings in `experiments/sprint2_results.md`.

**Summer break.** Next sprint: read related work, design extensive probe grid.
See "Next sprint" section below.

---

## Next sprint (post-summer)

- [ ] **Extensive probe grid** — encoders × erasers × feature axes × corpora:
  - Encoders: LaBSE, XLM-R base, + at least one more (e5-multilingual or mpnet-multilingual)
  - Erasure: Identity, Mean, INLP, LEACE (confirm LEACE dep first)
  - Feature axes: register (done), morphological complexity (negative control, needs UD parse),
    mean dependency distance (UD parse), NE density, sentence length (covariate)
  - Corpora: HPLT v3 (current), infopankki (parallel, high quality), Tatoeba (informal contrast)
  - Scale: 20k+ docs for better class coverage (SP=44 is too sparse at 5k)
- [ ] **LEACE** — confirm `pip install concept-erasure` on LUMI; add to eraser sweep
- [ ] **UD parsing** — run Turku Neural Parser on FI side; spaCy/Stanza on EN side;
      extract morphological complexity and mean dependency distance as Phase 2 labels
- [ ] **NE density** — run NER pipeline on FI+EN HPLT samples; compute NE token ratio per doc
- [ ] **`--normalize-residual` flag** — add to pilot scripts for robustness check on TwoNN ID;
      compare normalized vs raw residual results
- [ ] **Visualisation** — UMAP of residual coloured by encoder, language, register label
- [ ] **Cross-dataset validation** — fit INLP on Europarl → apply to HPLT → check language probe
      (lightweight RQ1b operationalisation of cultural confound question)

---

## Infrastructure

- [x] `.gitignore` — `outputs/` and `/data/` correctly anchored
- [x] LUMI module documentation — `experiments/lumi-onboarding.md` and `platform.md`
- [ ] **Execute bit** — `chmod +x sl-run-xlmr sl-run-phase2` committed (pending user push)

---

## Done

- [x] Repo rename: `register-geometry` → `residual-geometry`; local folder + GitHub remote updated
- [x] Package restructure: `src/residual_geometry/{embed,erase,analyse,eval,data}` created
- [x] New erasers: `INLPEraser`, `MeanCenteringEraser`, `MikolovProjection`, `LEACEEraser`
- [x] New analysis: `participation_ratio`, `anisotropy`, `twonn_intrinsic_dim`, `divergence_stats`
- [x] New eval: `language_probe` (selectivity), `linear_probe`, `bitext_retrieval`
- [x] Data: `load_europarl`, `load_infopankki`, `load_hplt` (+ `return_labels`, `min_confidence`)
- [x] Old `register_geometry` artifact removed from repo
- [x] 23 tests pass; `pyproject.toml` and `README.md` updated
- [x] Phase 1 pilot — Europarl (LaBSE, 5k pairs): gateway passed
- [x] Phase 1 pilot — HPLT v3 (LaBSE, 5k docs/lang): gateway passed; TwoNN ID ≈ 19
- [x] Encoder comparison — XLM-R base on HPLT v3: language signal rank-1, TwoNN ID ≈ 22
- [x] Phase 2 pilot — register probing on LaBSE + HPLT v3: register survives erasure,
      RSA 0.35 (n.s.) → 0.85 (p<10⁻⁶) after mean centering
- [x] `Embedder` fallback loading for non-ST HF models
- [x] SLURM scripts: `sl-run-baseline`, `sl-run-hplt`, `sl-run-xlmr`, `sl-run-phase2`
- [x] Overleaf thesis scaffold: all 8 chapters written, bibliography complete (22 entries)
