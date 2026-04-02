"""Shared test fixtures."""

import numpy as np
import pytest

RNG_SEED = 0
N_DOCS = 40
N_REGISTERS = 4
N_DIM = 32
REGISTERS = ["HI", "IN", "NA", "OP"]
LANGUAGES = ["fi", "en"]


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def synthetic_embeddings(rng):
    """Return (X_fi, labels_fi, X_en, labels_en) with mild register structure."""
    X, labels, langs = [], [], []
    for lang_idx, lang in enumerate(LANGUAGES):
        for reg_idx, reg in enumerate(REGISTERS):
            # Each class has a distinct mean; register structure is weak but present
            mean = np.zeros(N_DIM)
            mean[reg_idx] = 1.0
            mean[N_DIM // 2 + lang_idx] = 2.0  # language signal
            X_class = rng.standard_normal((N_DOCS // N_REGISTERS, N_DIM)) + mean
            X.append(X_class)
            labels.extend([reg] * (N_DOCS // N_REGISTERS))
            langs.extend([lang] * (N_DOCS // N_REGISTERS))

    X_all = np.vstack(X).astype(np.float32)
    labels_all = np.array(labels)
    langs_all = np.array(langs)

    n = N_DOCS
    fi_mask = langs_all == "fi"
    en_mask = langs_all == "en"
    return (
        X_all[fi_mask], labels_all[fi_mask],
        X_all[en_mask], labels_all[en_mask],
        langs_all,
        X_all,
    )
