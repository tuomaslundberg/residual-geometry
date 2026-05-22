# residual-geometry

Language-signal removal and residual structure characterisation in multilingual sentence encoders.

Master's thesis implementation — TurkuNLP, University of Turku.

## Research question

What remains in LaBSE and XLM-R sentence embeddings after language-identifying signal is linearly removed? Is the residual linguistically meaningful, and does it differ across encoder families?

## Package layout

```
src/residual_geometry/
├── embed/
│   └── encoder.py        # Embedder, load_encoder, embed_pairs
├── erase/
│   ├── inlp.py           # INLPEraser (Ravfogel et al. ACL 2020)
│   ├── leace.py          # LEACEEraser (Belrose et al. NeurIPS 2023)
│   ├── mean_centering.py # MeanCenteringEraser — per-language mean subtraction
│   └── mikolov.py        # MikolovProjection — least-squares linear map
├── analyse/
│   ├── geometry.py       # PR, anisotropy, TwoNN ID, Procrustes, RSA
│   └── divergence.py     # pairwise_l2, divergence_stats
├── eval/
│   ├── probing.py        # language_probe (with selectivity), linear probe, transfer
│   └── retrieval.py      # bitext_retrieval (P@1, MRR)
└── data/
    ├── _core.py          # Document, load_jsonl, make_toy_data, split_by_language
    └── loaders.py        # load_europarl, load_infopankki
```

## Quick start (local)

```bash
pip install -e ".[dev]"
pytest

# Phase 1 pilot (requires Europarl data)
python experiments/pilot_phase1.py \
    --fi data/europarl.fi \
    --en data/europarl.en \
    --out outputs/pilot_phase1.json
```

## Quick start (LUMI)

Runtime deps (`sentence-transformers`, `scikit-learn`, `scipy`, `numpy`) are
pre-installed in the `pytorch/2.7` container. No editable install needed to run experiments.

```bash
# Tests — set PYTHONPATH so pytest finds the package without an editable install:
PYTHONPATH=src python3 -m pytest

# Installing extras (e.g. scikit-dimension, concept-erasure):
# PYTHONUSERBASE is exported in ~/.zshrc and the SSH RemoteCommand.
# Use python3 -m pip (not pip3 — resolves to system Python 3.6 outside container).
# Use --no-build-isolation for editable installs.
python3 -m pip install --user --upgrade pip
python3 -m pip install --user --no-build-isolation -e ".[dev]"
```

## SLURM

See `experiments/slurm/sl-run-baseline`. Uses the CSC `pytorch/2.7` module
(Singularity container with transparent Python wrappers). The script
self-submits when invoked directly and archives previous logs on each run.

```bash
bash experiments/slurm/sl-run-baseline
```

## Erasure methods

| Key | Class | Description |
|-----|-------|-------------|
| `none` | `IdentityProjector` | No-op baseline |
| `mean` | `MeanCenteringEraser` | Subtract per-language mean |
| `inlp` | `INLPEraser` | Iterative nullspace projection |
| `leace` | `LEACEEraser` | Closed-form oblique projection (requires `pip install concept-erasure`) |

## Geometry diagnostics

| Function | Description |
|----------|-------------|
| `participation_ratio` | Effective PCA dimensionality |
| `anisotropy` | Mean cosine between random pairs |
| `twonn_intrinsic_dim` | TwoNN intrinsic dimensionality (Facco et al. 2017) |
| `procrustes_disparity` | Normalised residual after optimal rotation |
| `rsa` | Spearman correlation of representational dissimilarity matrices |

## Dependencies

Core: `sentence-transformers`, `scikit-learn`, `scikit-dimension`, `scipy`, `numpy`, `pyyaml`
LEACE: `concept-erasure` (optional — `pip install -e ".[leace]"`)
Vis: `umap-learn`, `matplotlib` (optional — `pip install -e ".[vis]"`)
