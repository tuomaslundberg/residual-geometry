"""Tests for embed.py. Model is mocked — no download required."""

from unittest.mock import MagicMock

import numpy as np

from register_geometry.embed import Embedder


def test_embed_shape():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.default_rng(0).standard_normal(
        (5, 768)
    ).astype(np.float32)

    embedder = Embedder(model_name="mock/model", device="cpu", batch_size=8)
    embedder._model = mock_model  # bypass _load(); model is never downloaded

    result = embedder.embed(["a", "b", "c", "d", "e"])

    assert result.shape == (5, 768)
    assert result.dtype == np.float32


def test_device_auto_resolves():
    embedder = Embedder(device="auto")
    assert embedder.device in ("cpu", "cuda")


def test_device_explicit():
    embedder = Embedder(device="cpu")
    assert embedder.device == "cpu"
