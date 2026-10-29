# Running residual-geometry on LUMI
*Environment setup, paths and entrypoints. Written 2026-05-21; the package layout is in the README.*

---

## Environment setup (one-time)

```bash
# 1. Load modules — CSC stack (NOT LUMI/24.03 EasyBuild — unsupported, ROCm mismatch).
# pytorch/2.7 is a Singularity container; python3/pip3 call into it transparently.
# Python inside the container: 3.11.0rc1.
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

# 2. Clone repo (no venv needed — runtime deps pre-installed in container)
cd /scratch/project_462001491/tlundber/projects/
git clone git@github.com:tuomaslundberg/residual-geometry.git
cd residual-geometry

# 3. Install extras (scikit-dimension, dev deps) — runtime deps already in container
export PYTHONUSERBASE=/scratch/project_462001491/tlundber/pythonuserbase
python3 -m pip install --user --upgrade pip
python3 -m pip install --user --no-build-isolation -e ".[dev]"

# 4. Smoke test — no GPU, no data, no model download
PYTHONPATH=src python3 -m pytest tests/ -v
```

**Expected:** all tests pass (23 as of 2026-05-21). If any fail, debug imports before running experiments.

Afterwards, each time: load the module (step 1), `cd` into the repo, `git pull`, and rerun the tests.

---

## LUMI paths

| Resource | Path |
|----------|------|
| Scratch root | `/scratch/project_462001491/tlundber/` |
| Repo | `/scratch/project_462001491/tlundber/projects/residual-geometry/` |
| User packages | `/scratch/project_462001491/tlundber/pythonuserbase/` |
| Data (target) | `/scratch/project_462001491/tlundber/data/europarl/` |
| Outputs | `residual-geometry/outputs/` |
| Slurm logs | `residual-geometry/experiments/slurm/logs/` |
| LUMI project | `project_462001491` |
| GPU partition | `small-g` |

---

## Experiment entrypoints

### Phase 1 pilot (runnable now)

**Interactive:**
```bash
PYTHONPATH=src python3 experiments/pilot_phase1.py \
    --fi /scratch/project_462001491/tlundber/data/europarl/europarl.fi \
    --en /scratch/project_462001491/tlundber/data/europarl/europarl.en \
    --out outputs/pilot_phase1.json \
    --n-pairs 5000
```

**Slurm batch:**
```bash
bash experiments/slurm/sl-run-baseline  # self-submits; archives previous logs
```

Runs LaBSE on 5k Europarl FI-EN pairs with three erasers (identity, mean, INLP).
Outputs: language probe accuracy/selectivity, bitext P@1/MRR, divergence stats,
participation ratio, anisotropy, TwoNN intrinsic dimensionality.

**Gateway pass criteria:**
- INLP language probe selectivity ≈ 0 (signal removed)
- Bitext P@1 stays high (> 0.8) after removal (semantics preserved)
- Residual TwoNN ID is non-trivial (not 1–2)

### Phase 2 pilot

Stub only — `experiments/pilot_phase2.py` raises `NotImplementedError`.
Design pending Phase 1 results.

---

## Data

**Europarl FI-EN:** download from OPUS if not already on scratch.
```bash
# One-liner using opus-dl (if available) or wget:
# https://opus.nlpl.eu/Europarl/fi&en/v8/Europarl
# Target files: europarl.fi, europarl.en (one sentence per line, aligned)
```

**CORE corpus** (for Phase 2 register probing, if needed):
scratch path to be confirmed.
Conversion script: `scripts/prepare_data.py` — to be written.

---

## Notes

- No venv needed — runtime deps are pre-installed in the `pytorch/2.7` container.
  Old venv `venvs/register-geometry` is stale — do not use.
- Old repo dir `register-geometry/` archived to `projects/_archive/register-geometry/` (2026-06-01).
  Outputs were toy-data only (near-random F1); safe to delete if space needed.
- Slurm script: `experiments/slurm/sl-run-baseline` (self-submitting, archives logs).
- `concept-erasure` for LEACE is listed as optional dep (`pip install -e ".[leace]"`).
  Confirm it installs cleanly on LUMI before adding to eraser sweep.
