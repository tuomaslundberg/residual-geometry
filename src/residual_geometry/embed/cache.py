"""On-disk embedding cache, keyed by model and by the exact text.

One ``.npz`` per model holds every text that model has embedded, under the
SHA-1 of the text. Any subset of documents (a different filter, a balanced
draw, a larger sample from the same files) is served from the cache and only
unseen texts are embedded.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def default_cache_dir() -> Path:
    """``$RG_EMBED_CACHE``, else ``$WORK/cache/residual-geometry/embeddings``."""
    if os.environ.get("RG_EMBED_CACHE"):
        return Path(os.environ["RG_EMBED_CACHE"])
    base = Path(os.environ.get("WORK", Path.home()))
    return base / "cache" / "residual-geometry" / "embeddings"


def text_key(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).hexdigest().encode("ascii")


class EmbeddingCache:
    def __init__(self, model_name: str, cache_dir: str | Path | None = None) -> None:
        cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.path = cache_dir / (model_name.replace("/", "__") + ".npz")
        self._index: dict[bytes, int] = {}
        self._X: np.ndarray | None = None
        if self.path.exists():
            with np.load(self.path) as f:
                keys, self._X = f["keys"], f["X"]
            self._index = {k: i for i, k in enumerate(keys.tolist())}

    def __len__(self) -> int:
        return len(self._index)

    def embed(self, embedder, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embeddings for ``texts`` in order; embeds and saves only the missing ones."""
        keys = [text_key(t) for t in texts]
        missing: dict[bytes, str] = {}
        for k, t in zip(keys, texts):
            if k not in self._index and k not in missing:
                missing[k] = t
        print(f"Embedding cache {self.path.name}: {len(texts) - len(missing)} hit, "
              f"{len(missing)} to embed")
        if missing:
            X_new = embedder.embed(list(missing.values()), show_progress=show_progress)
            start = len(self._index)
            self._X = X_new if self._X is None else np.vstack([self._X, X_new])
            for i, k in enumerate(missing):
                self._index[k] = start + i
            self._save()
        return self._X[[self._index[k] for k in keys]]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._index), dtype="S40")
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, X=self._X)
        os.replace(tmp, self.path)
