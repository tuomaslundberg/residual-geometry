# TODO — register-geometry

Tasks that are genuinely open; research decisions that aren't yet locked are
marked explicitly. Roughly ordered by dependency.

---

## Immediate — first LUMI run

- [ ] Clone repo to LUMI scratch: `/scratch/project_462000999/tlundber/register-geometry`
- [ ] Create venv on LUMI and install deps: `pip install -e ".[dev]"`
- [ ] Verify `--toy` run on a CPU node (no data or GPU needed — pure smoke test)
- [ ] Write `scripts/prepare_data.py` — convert CORE corpus to pipeline JSONL format
      (`{"text": "...", "language": "fi"|"en", "register": "HI"|...|"SP"}`)
- [ ] Update `experiments/configs/exp_baseline.yaml` `data.path` to converted JSONL
- [ ] Submit first real run: `sbatch experiments/slurm/run_baseline.sh`
- [ ] Inspect `outputs/results.json`; sanity-check geometry metrics

---

## Short-term

- [ ] Implement `save_embeddings` in pipeline — config flag exists but saving is not yet wired;
      persist pre- and post-projection arrays as `.npz` for reuse across runs
- [ ] Visualisation module (`src/register_geometry/visualise.py`):
      UMAP plots coloured by language and by register, pre and post projection
      (optional deps `umap-learn` + `matplotlib` are already declared in `[vis]`)
- [ ] Human-readable results summary: pretty-print key metrics to stdout alongside
      `results.json` (avoids having to read raw JSON after every run)

---

## Medium-term

*(Some items below depend on the Filip sync — do not commit to a method until then.)*

- [ ] Implement and test `LEACEProjector` end-to-end on LUMI
      (requires `pip install -e ".[leace]"`; add a test fixture once the dep is confirmed available)
- [ ] Ablation: projection depth sweep — vary INLP `n_iterations` and LEACE rank;
      add a SLURM array script and a corresponding experiment config
- [ ] Second encoder baseline — open; not yet decided (candidate: mBERT or XLM-R)

---

## Research decisions (open — do not resolve without supervisor input)

- [ ] **Projection method**: INLP vs LEACE vs Filip's translation-pair projection, or combination —
      depends on Filip sync; boundary with his work not yet confirmed
- [ ] **RQ wording**: finalise after method is locked
- [ ] **Metric set**: Procrustes disparity + RSA are in; CKA is optional/deferred —
      confirm before writing the Methods chapter
- [ ] **Diagnostic transfer methodology**: exact train/test split strategy,
      whether to use held-out data or cross-validation

---

## Implementation course (TKO-5330)

- [ ] Document exact LUMI module versions used after first successful run
      (add to `experiments/` or README)
- [ ] Final README pass before course submission — confirm install instructions,
      example run, and repo structure description are up to date
