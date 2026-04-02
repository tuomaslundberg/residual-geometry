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
# Install (editable)
pip install -e ".[dev,vis]"

# Smoke test — no GPU, no corpus, synthetic data
python scripts/run_pipeline.py --toy

# Full run with custom config
python scripts/run_pipeline.py --config experiments/configs/exp_baseline.yaml
```

## Run tests

```bash
pytest
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

See `experiments/slurm/run_baseline.sh`. Adjust `--account`, paths, and module
versions to match your allocation.

```bash
mkdir -p experiments/slurm/logs
sbatch experiments/slurm/run_baseline.sh
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
