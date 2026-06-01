# LUMI session handoff — residual-geometry
*Feed at the start of a remote VSCode session on LUMI. Updated 2026-05-21.*

---

## Research direction (locked 2026-05-19)

**What:** Characterise language signal and its residual in multilingual sentence encoders.
Two-phase structure:
- **Phase 1 (no labels):** How much language-identifying signal is there, and what does
  the residual look like geometrically after linear removal?
- **Phase 2 (labels):** What systematic structure organises the residual? (open empirical
  question — register is one candidate, not a decided answer)

**Encoders:** LaBSE (bitext-trained, `sentence-transformers/LaBSE`) and XLM-R
(`FacebookAI/xlm-roberta-base`) for comparison.
**Language pair:** Finnish–English.
**Methods:** mean centring (baseline), INLP (completeness guarantee), LEACE (optional dep).

**Thesis title (working):** "What Remains After Language Removal? Characterising Residual
Structure in Multilingual Sentence Encoders"
**Supervisors:** Jenna Kanerva (primary), Amanda Myntti (secondary), Veronika Laippala (PI).

---

## LUMI environment setup (one-time)

```bash
# 1. Load modules — CSC stack (NOT LUMI/24.03 EasyBuild — unsupported, ROCm mismatch).
# pytorch/2.7 is a Singularity container; python3/pip3 call into it transparently.
# Python inside the container: 3.11.0rc1.
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

# 2. Clone repo (no venv needed — runtime deps pre-installed in container)
cd /scratch/project_462000999/tlundber/projects/
git clone git@github.com:tuomaslundberg/residual-geometry.git
cd residual-geometry

# 3. Install extras (scikit-dimension, dev deps) — runtime deps already in container
export PYTHONUSERBASE=/scratch/project_462000999/tlundber/pythonuserbase
python3 -m pip install --user --upgrade pip
python3 -m pip install --user --no-build-isolation -e ".[dev]"

# 4. Smoke test — no GPU, no data, no model download
PYTHONPATH=src python3 -m pytest tests/ -v
```

**Expected:** 23 tests pass. If any fail, stop and debug imports before running experiments.

---

## LUMI paths

| Resource | Path |
|----------|------|
| Scratch root | `/scratch/project_462000999/tlundber/` |
| Repo | `/scratch/project_462000999/tlundber/projects/residual-geometry/` |
| User packages | `/scratch/project_462000999/tlundber/pythonuserbase/` |
| Data (target) | `/scratch/project_462000999/tlundber/data/europarl/` |
| Outputs | `residual-geometry/outputs/` |
| SLURM logs | `residual-geometry/experiments/slurm/logs/` |
| LUMI project | `project_462000999` |
| GPU partition | `small-g` |

---

## Package layout

```
src/residual_geometry/
├── embed/encoder.py       # Embedder, load_encoder, embed_pairs
├── erase/
│   ├── __init__.py        # LanguageProjector, IdentityProjector, INLPProjector alias, build_projector
│   ├── inlp.py            # INLPEraser — projection_matrix, n_directions_removed properties
│   ├── leace.py           # LEACEEraser — requires concept-erasure
│   ├── mean_centering.py  # MeanCenteringEraser
│   └── mikolov.py         # MikolovProjection
├── analyse/
│   ├── geometry.py        # participation_ratio, anisotropy, twonn_intrinsic_dim,
│   │                      # GeometryResult, procrustes_disparity, rsa, evaluate_geometry
│   └── divergence.py      # pairwise_l2, divergence_stats
├── eval/
│   ├── probing.py         # language_probe (selectivity), linear_probe,
│   │                      # LinearRegisterClassifier, cross_lingual_transfer
│   └── retrieval.py       # bitext_retrieval (P@1, MRR)
└── data/
    ├── _core.py           # Document, load_jsonl, make_toy_data, split_by_language
    └── loaders.py         # load_europarl, load_infopankki
```

---

## Experiment entrypoints

### Phase 1 pilot (runnable now)

**Interactive:**
```bash
PYTHONPATH=src python3 experiments/pilot_phase1.py \
    --fi /scratch/project_462000999/tlundber/data/europarl/europarl.fi \
    --en /scratch/project_462000999/tlundber/data/europarl/europarl.en \
    --out outputs/pilot_phase1.json \
    --n-pairs 5000
```

**SLURM batch:**
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
check with Veronika for current scratch path.
Conversion script: `scripts/prepare_data.py` — to be written.

---

## Session start checklist

1. `module use /appl/local/csc/modulefiles && module load pytorch/2.7`
2. `cd /scratch/project_462000999/tlundber/projects/residual-geometry`
3. `git pull` (sync with local changes)
4. `PYTHONPATH=src python3 -m pytest tests/ -q` — confirm 23 pass
5. Open `TODO.md` for current open tasks

---

## Key open tasks (summary — see `TODO.md` for full list)

- Fetch Europarl data to scratch
- Run Phase 1 pilot; inspect gateway criteria
- After gateway passes: design Phase 2 (candidate labels for residual probing)
- XLM-R comparison run
- LEACE: confirm `concept-erasure` available on LUMI; add to eraser sweep

---

## Notes

- No venv needed — runtime deps are pre-installed in the `pytorch/2.7` container.
  Old venv `venvs/register-geometry` is stale — do not use.
- Old repo dir `register-geometry/` archived to `projects/_archive/register-geometry/` (2026-06-01).
  Outputs were toy-data only (near-random F1); safe to delete if space needed.
- SLURM script: `experiments/slurm/sl-run-baseline` (self-submitting, archives logs).
- `concept-erasure` for LEACE is listed as optional dep (`pip install -e ".[leace]"`).
  Confirm it installs cleanly on LUMI before adding to eraser sweep.
