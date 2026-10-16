"""Tests for embed.py. Model is mocked — no download required."""

from unittest.mock import MagicMock

import numpy as np

from residual_geometry.embed.encoder import Embedder


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


def test_embedding_cache_embeds_only_missing(tmp_path):
    from residual_geometry.embed.cache import EmbeddingCache

    calls = []

    class FakeEmbedder:
        def embed(self, texts, show_progress=False):
            calls.append(list(texts))
            return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    cache = EmbeddingCache("org/model", cache_dir=tmp_path)
    X1 = cache.embed(FakeEmbedder(), ["aa", "b", "aa"])
    assert calls == [["aa", "b"]]
    assert X1[:, 0].tolist() == [2, 1, 2]

    reloaded = EmbeddingCache("org/model", cache_dir=tmp_path)
    assert len(reloaded) == 2
    X2 = reloaded.embed(FakeEmbedder(), ["b", "ccc"])
    assert calls[-1] == ["ccc"]
    assert X2[:, 0].tolist() == [1, 3]
