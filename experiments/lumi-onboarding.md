# LUMI onboarding — register-geometry
*Feed this at the start of a remote Claude Code session on LUMI.*

---

## Research context

**Thesis title (working):** Register Geometry in Multilingual Sentence Encoders:
Cross-Lingual Isomorphism after Language-Signal Removal

**Core question:** After removing language-identifying signal from LaBSE embeddings
via linear projection, are the register subspaces for Finnish and English
geometrically isomorphic? "Geometrically isomorphic" means: do register categories
cluster in the same shape, with the same pairwise structure, across languages?

**This is not a probing paper.** The question is not "is register encoded?" —
strong classifiers exist. The question is about the *geometry* of the subspace,
and whether that geometry is consistent cross-lingually.

**Closest prior work to differentiate from:** Libovický, Rosa & Fraser (2020) —
they remove language signal and evaluate on task-transfer accuracy. This thesis
removes language signal and asks a geometric question about register subspace structure.

**Language pair:** Finnish (FI) and English (EN).
**Baseline encoder:** LaBSE (`sentence-transformers/LaBSE`). Frozen throughout —
no fine-tuning.
**Dual output:** thesis (MSc, UTU) + publishable article.
**Supervisors:** Veronika Laippala and Filip Ginter (TurkuNLP, UTU).

---

## What's decided vs open

| | Status |
|---|---|
| Encoder: LaBSE | Decided |
| Language pair: FI-EN | Decided |
| Projection scaffold: INLP implemented | Decided |
| Geometry evaluation: Procrustes disparity + RSA | Decided |
| Projection method: INLP vs LEACE vs Filip's approach | **Open — pending supervisor sync** |
| Exact RQ wording | **Open — after method is locked** |
| CKA as additional metric | Optional / deferred |
| Diagnostic transfer train/test methodology | **Open** |
| Second encoder baseline (mBERT / XLM-R) | Open — not yet decided |

---

## Repo layout

```
register-geometry/
├── config/
│   ├── default.yaml          # base config (model, projection, evaluation)
│   └── toy.yaml              # --toy overrides (CPU, synthetic data, fast settings)
├── src/register_geometry/
│   ├── data.py               # Document dataclass, load_jsonl, make_toy_data
│   ├── embed.py              # Embedder wrapping frozen SentenceTransformer
│   ├── project.py            # LanguageProjector ABC; INLPProjector (done); LEACEProjector (stub)
│   ├── evaluate.py           # register_centroids, procrustes_disparity, rsa, evaluate_geometry
│   └── transfer.py           # LinearRegisterClassifier, cross_lingual_transfer
├── scripts/
│   └── run_pipeline.py       # CLI entry point: --toy for smoke test, --config for real runs
├── experiments/
│   ├── configs/
│   │   └── exp_baseline.yaml # LUMI run config (update data.path before submitting)
│   └── slurm/
│       └── run_baseline.sh   # SLURM batch script
├── tests/                    # pytest; all pass on CPU, no model download needed
└── TODO.md                   # Current task list
```

**Key entry point:** `python scripts/run_pipeline.py --config <yaml>`

---

## LUMI setup checklist

```bash
# 1. Load modules (verify versions are still available)
module purge
module load LUMI/24.03 partition/G
module load PyTorch/2.2.2-rocm-5.6.1-python-3.10-singularity-20240315

# 2. Create venv (one-time)
python -m venv /scratch/project_462000999/tlundber/venvs/register-geometry
source /scratch/project_462000999/tlundber/venvs/register-geometry/bin/activate

# 3. Clone repo (one-time)
git clone <repo-url> /scratch/project_462000999/tlundber/register-geometry
cd /scratch/project_462000999/tlundber/register-geometry

# 4. Install
pip install -e ".[dev]"

# 5. Smoke test (CPU, no data, no GPU allocation needed)
python scripts/run_pipeline.py --toy
```

---

## Data

CORE corpus (FI + EN, register-labelled) exists on LUMI scratch. It needs converting
to the pipeline's JSONL format before the first real run.

**Expected format** — one document per line:
```json
{"text": "...", "language": "fi", "register": "NA"}
```

**Conversion script to write:** `scripts/prepare_data.py`
This is the first code task for this session if not yet done.

**CORE registers:** `HI`, `ID`, `IN`, `IP`, `LY`, `NA`, `OP`, `SP`

CORE corpus local path (raw): check with Veronika / lab storage if path is not
already known. Converted output should go to:
`/scratch/project_462000999/tlundber/data/register-geometry/fi_en_docs.jsonl`

---

## First real run

1. Ensure JSONL data exists at the path above
2. Update `experiments/configs/exp_baseline.yaml` → `data.path` if needed
3. Create SLURM log dir: `mkdir -p experiments/slurm/logs`
4. Submit: `sbatch experiments/slurm/run_baseline.sh`
5. Monitor: `squeue --me`
6. Results: `outputs/results.json` (or the scratch output dir configured in the yaml)

---

## Key open tasks

See `TODO.md` at repo root. The immediate priority is:
1. Write `scripts/prepare_data.py`
2. Submit first SLURM run
3. Inspect geometry metrics (pre vs post projection)
