"""Embed every document of one or more HPLT JSONL files into the embedding cache.

Files not ending in .jsonl are read as plain text, one segment per line
(e.g. a sampled Europarl side).

Filter-independent: later runs choose any subset (register rule, MT handling,
balanced draws) and read their vectors from the cache without re-embedding.

  python scripts/embed_corpus.py FILE [FILE ...] [--model sentence-transformers/LaBSE]
"""

from __future__ import annotations

import argparse
import time

from residual_geometry.data.loaders import load_hplt
from residual_geometry.embed.cache import EmbeddingCache
from residual_geometry.embed.encoder import Embedder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--model", default="sentence-transformers/LaBSE")
    args = parser.parse_args()

    embedder = Embedder(model_name=args.model)
    print(f"model={args.model} device={embedder.device}")
    cache = EmbeddingCache(args.model)
    for path in args.files:
        t0 = time.time()
        if path.endswith(".jsonl"):
            texts = load_hplt(path)
        else:
            with open(path, encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        X = cache.embed(embedder, texts, show_progress=True)
        print(f"{path}: {X.shape} in {time.time() - t0:.0f}s; cache now {len(cache)} texts")
