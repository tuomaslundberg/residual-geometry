# register-geometry

Cross-lingual register subspace analysis in multilingual sentence encoders.

Master's thesis implementation — TurkuNLP, University of Turku.

## Research question

Are register subspaces in LaBSE (and related encoders) geometrically isomorphic
across Finnish and English after language-identifying signal is linearly projected out?

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Embed | `embed.py` | Encode documents with a frozen sentence-transformer (LaBSE baseline) |
| 2. Project | `project.py` | Remove language-ID signal (INLP / LEACE / none) |
| 3. Evaluate geometry | `evaluate.py` | Procrustes disparity and RSA on register-class centroids |
| 4. Diagnostic transfer | `transfer.py` | Zero-shot cross-lingual register classifier transfer |

## Quick start

```bash
# Smoke test — no GPU, no corpus, synthetic data
# No installation needed: scripts add src/ to sys.path automatically,
# and all runtime deps (sentence-transformers, scikit-learn, scipy, numpy)
# are pre-installed in the pytorch/2.7 container on LUMI.
python3 scripts/run_pipeline.py --toy

# Full run with custom config
python3 scripts/run_pipeline.py --config experiments/configs/exp_baseline.yaml
```

## Run tests

```bash
# Tests mock the model — no GPU or corpus needed.
# Set PYTHONPATH so pytest finds the package without an editable install:
PYTHONPATH=src python3 -m pytest
```

## Installing additional packages (LUMI)

Runtime deps are pre-installed in the `pytorch/2.7` container. To add packages:

```bash
# PYTHONUSERBASE is exported in ~/.zshrc and the SSH RemoteCommand.
# Use python3 -m pip (not pip3 — resolves to system Python 3.6).
# Use --no-build-isolation for editable installs (build subprocess falls
# back to host Python 3.6 otherwise).
python3 -m pip install --user --upgrade pip
python3 -m pip install --user --no-build-isolation -e ".[dev]"
```

## Project layout

```
register-geometry/
├── config/
│   ├── default.yaml        # base configuration
│   └── toy.yaml            # --toy overrides
├── src/register_geometry/
│   ├── data.py             # loading, toy data, Document dataclass
│   ├── embed.py            # Embedder
│   ├── project.py          # LanguageProjector ABC, INLP, LEACE stubs
│   ├── evaluate.py         # Procrustes, RSA, GeometryResult
│   └── transfer.py         # LinearRegisterClassifier, cross_lingual_transfer
├── tests/                  # pytest unit tests (no GPU or model download needed)
├── scripts/
│   └── run_pipeline.py     # CLI entry point
├── experiments/
│   ├── configs/            # per-experiment YAML configs
│   └── slurm/              # LUMI batch scripts
└── notebooks/              # scratch / EDA (not version-controlled)
```

## LUMI

See `experiments/slurm/sl-run-baseline`. Uses the CSC `pytorch/2.7` module
(Singularity container with transparent Python wrappers). The script
self-submits when invoked directly and archives previous logs on each run.

```bash
bash experiments/slurm/sl-run-baseline
```

## Data format

Input JSONL, one document per line:

```json
{"text": "...", "language": "fi", "register": "NA"}
```

## Projection methods

| Key | Class | Status |
|-----|-------|--------|
| `none` | `IdentityProjector` | Baseline (no projection) |
| `inlp` | `INLPProjector` | Implemented — Ravfogel et al. (ACL 2020) |
| `leace` | `LEACEProjector` | Stub — requires `pip install concept-erasure` |

## Dependencies

Core: `sentence-transformers`, `scikit-learn`, `scipy`, `numpy`, `pyyaml`
Visualisation: `umap-learn`, `matplotlib` (optional, `pip install -e ".[vis]"`)
LEACE: `concept-erasure` (optional, `pip install -e ".[leace]"`)
