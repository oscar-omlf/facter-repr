from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

import numpy as np


def recall_at_k(ranked_items: Sequence[int], relevant_items: Set[int], k: int = 10) -> float:
    """
    Recall@K: |TopK ∩ Relevant| / |Relevant|
    For MovieLens next-item prediction, |Relevant| is usually 1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(relevant_items) == 0:
        return 0.0
    topk = ranked_items[:k]
    hits = sum(1 for x in topk if int(x) in relevant_items)
    return float(hits) / float(len(relevant_items))


def ndcg_at_k(ranked_items: Sequence[int], relevant_items: Set[int], k: int = 10) -> float:
    """
    NDCG@K with binary relevance:
      DCG = sum_{i=1..K} rel_i / log2(i+1)
      IDCG = best possible DCG (all relevant at top)
    If |Relevant|=1, IDCG = 1/log2(1+1) = 1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(relevant_items) == 0:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        rel = 1.0 if int(item) in relevant_items else 0.0
        if rel > 0:
            dcg += rel / np.log2(rank + 1.0)

    # Ideal DCG: place all relevant items at the top
    # binary relevance -> IDCG depends on number of relevant items
    ideal_rels = [1.0] * min(len(relevant_items), k)
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        idcg += rel / np.log2(rank + 1.0)

    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_recall_ndcg(
    ranked_lists: Sequence[Sequence[int]],
    targets: Sequence[int],
    k: int = 10,
) -> dict:
    """
    Convenience: assumes one relevant target per example.
    """
    if len(ranked_lists) != len(targets):
        raise ValueError("ranked_lists and targets must have the same length")

    recalls = []
    ndcgs = []
    for ranked, tgt in zip(ranked_lists, targets):
        rel = {int(tgt)}
        recalls.append(recall_at_k(ranked, rel, k=k))
        ndcgs.append(ndcg_at_k(ranked, rel, k=k))

    return {"Recall@%d" % k: float(np.mean(recalls)), "NDCG@%d" % k: float(np.mean(ndcgs))}


def count_violations(scores: Sequence[float], q_alpha: float) -> int:
    """
    Violations count per definition: S_new > Q_alpha.
    """
    return int(np.sum(np.array(scores, dtype=np.float32) > float(q_alpha)))
