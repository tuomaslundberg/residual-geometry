"""Bitext retrieval evaluation — P@1 and MRR.

Used as semantic preservation check: after language-signal removal,
parallel sentences should still be nearest neighbours.
"""

from __future__ import annotations

import numpy as np


def bitext_retrieval(
    src_embs: np.ndarray,
    tgt_embs: np.ndarray,
    metric: str = "cosine",
) -> dict[str, float]:
    """Compute P@1 and MRR for src→tgt bitext retrieval.

    For each source embedding, finds the nearest neighbour in the target set
    and checks whether it is the gold parallel sentence (same row index).

    Args:
        src_embs: (n, d) source embeddings (already normalised for cosine).
        tgt_embs: (n, d) target embeddings.
        metric: distance metric for nearest-neighbour search.

    Returns:
        dict with keys "p_at_1" and "mrr".
    """
    from scipy.spatial.distance import cdist

    n = len(src_embs)
    assert len(tgt_embs) == n, "src and tgt must have the same number of embeddings."

    dists = cdist(src_embs.astype(np.float64), tgt_embs.astype(np.float64), metric=metric)
    # Rank by ascending distance (nearest first)
    ranks = np.argsort(dists, axis=1)  # (n, n)

    gold_ranks = np.where(ranks == np.arange(n)[:, None])[1] + 1  # 1-indexed

    p_at_1 = float((gold_ranks == 1).mean())
    mrr = float((1.0 / gold_ranks).mean())

    return {"p_at_1": p_at_1, "mrr": mrr}
