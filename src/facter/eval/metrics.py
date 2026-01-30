"""Compute evaluation metrics for ranking and fairness-style proxies.

This module provides:

- Utility metrics for ranked recommendation lists (Recall@k and NDCG@k).
- Helper utilities for counting threshold violations.
- Embedding-based proxy metrics (SNSR/SNSV) computed from recommendation lists
    and protected-group keys.

Where metric definitions differ between papers and implementations, the
docstrings describe the behavior implemented here.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Optional, Tuple, Any, Union

import numpy as np


Relevant = Union[int, Sequence[int], Set[int]]

def _to_relevant_set(rel: Relevant) -> Set[int]:
    """Normalize a "relevant items" specifier to a set of ints.

    Args:
        rel (Relevant): Relevant item specification. May be a single item id, a
            sequence of item ids, or a set of item ids.

    Returns:
        Set[int]: Set of relevant item ids.
    """
    if isinstance(rel, (set, frozenset)):
        return {int(x) for x in rel}
    if isinstance(rel, (list, tuple, np.ndarray)):
        return {int(x) for x in rel}
    return {int(rel)}


def mean_recall_ndcg_multi(
    ranked_lists: Sequence[Sequence[int]],
    relevants: Sequence[Relevant],
    k: int = 10,
) -> Dict[str, float]:
    """Compute mean Recall@k and NDCG@k for multi-relevance targets.

    Each entry in ``relevants`` may be a single relevant id or a collection of
    relevant ids.

    Args:
        ranked_lists (Sequence[Sequence[int]]): Ranked recommendation lists.
        relevants (Sequence[Relevant]): Relevant item ids per example.
        k (int): Cutoff for the top-k list.

    Returns:
        Dict[str, float]: Mapping with keys ``"Recall@{k}"`` and ``"NDCG@{k}"``.

    Raises:
        ValueError: If ``ranked_lists`` and ``relevants`` have different
            lengths.
        ValueError: If ``k`` is non-positive (via called metric functions).
    """
    if len(ranked_lists) != len(relevants):
        raise ValueError("ranked_lists and relevants must have the same length")

    recalls, ndcgs = [], []
    for ranked, rel in zip(ranked_lists, relevants):
        rel_set = _to_relevant_set(rel)
        recalls.append(recall_at_k(ranked, rel_set, k=k))
        ndcgs.append(ndcg_at_k(ranked, rel_set, k=k))

    return {f"Recall@{k}": float(np.mean(recalls)), f"NDCG@{k}": float(np.mean(ndcgs))}


def recall_at_k(ranked_items: Sequence[int], relevant_items: Set[int], k: int = 10) -> float:
    """Compute Recall@k for a ranked list with a set of relevant items.

    The score is defined as:

    $$\\text{Recall@k} = \\frac{|\\text{Top-k}\\cap\\text{Relevant}|}{|\\text{Relevant}|}.$$

    Args:
        ranked_items (Sequence[int]): Ranked item ids.
        relevant_items (Set[int]): Relevant item ids.
        k (int): Cutoff for the top-k list.

    Returns:
        float: Recall@k.

    Raises:
        ValueError: If ``k`` is non-positive.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(relevant_items) == 0:
        return 0.0
    topk = ranked_items[:k]
    hits = sum(1 for x in topk if int(x) in relevant_items)
    return float(hits) / float(len(relevant_items))


def ndcg_at_k(ranked_items: Sequence[int], relevant_items: Set[int], k: int = 10) -> float:
    """Compute NDCG@k for binary relevance.

    This implementation uses binary relevance and computes DCG with a log
    discount, then normalizes by the ideal DCG for the given number of relevant
    items.

    Args:
        ranked_items (Sequence[int]): Ranked item ids.
        relevant_items (Set[int]): Relevant item ids.
        k (int): Cutoff for the top-k list.

    Returns:
        float: NDCG@k.

    Raises:
        ValueError: If ``k`` is non-positive.
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
    """Compute mean Recall@k and NDCG@k for single-target relevance.

    Args:
        ranked_lists (Sequence[Sequence[int]]): Ranked recommendation lists.
        targets (Sequence[int]): Single relevant target id per example.
        k (int): Cutoff for the top-k list.

    Returns:
        dict: Mapping with keys ``"Recall@{k}"`` and ``"NDCG@{k}"``.

    Raises:
        ValueError: If ``ranked_lists`` and ``targets`` have different lengths.
        ValueError: If ``k`` is non-positive (via called metric functions).
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
    """Count threshold violations for a list of scores.

    A score is counted as a violation when ``score > q_alpha``.

    Args:
        scores (Sequence[float]): Score values.
        q_alpha (float): Threshold value.

    Returns:
        int: Number of violations.
    """
    return int(np.sum(np.array(scores, dtype=np.float32) > float(q_alpha)))

# SNSR/SNSV proxy functions.
# NOTE: I implemented them as per FACTER's code: https://github.com/AryaFayyazi/FACTER
# These formulae compeltely differ when compared to the ones reported in the paper, and the ones in the paper the authors cite.
# However, the formulae used in the authors implementation makes sense because it actually uses the embeddings.
@dataclass(frozen=True)
class SNSMetrics:
    """Store computed SNSR/SNSV proxy metric values.

    Attributes:
        SNSR (float): Maximum pairwise cosine distance between group mean
            vectors.
        SNSV (float): Mean pairwise cosine distance between group mean vectors.
        n_groups_used (int): Number of groups included after filtering by
            ``min_group_size``.
    """
    SNSR: float
    SNSV: float
    n_groups_used: int


def _encode_texts_np(embedder: Any, texts: List[str]) -> np.ndarray:
    """Encode a list of strings with an embedding model into a NumPy array.

    The function supports two embedder APIs:

    - ``encode_texts(list[str]) -> array-like``
    - ``encode(list[str]) -> array-like``

    Args:
        embedder (Any): Embedder object.
        texts (List[str]): Input texts.

    Returns:
        np.ndarray: Float32 array of shape ``(n, d)``.

    Raises:
        TypeError: If the embedder exposes neither ``encode_texts`` nor
            ``encode``.
    """
    if hasattr(embedder, "encode_texts"):
        E = embedder.encode_texts(texts)
    elif hasattr(embedder, "encode"):
        E = embedder.encode(texts)
    else:
        raise TypeError("embedder must expose encode_texts(...) or encode(...)")

    E = np.asarray(E, dtype=np.float32)
    if E.ndim == 1:
        E = E[None, :]
    return E


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector if its norm exceeds a small epsilon.

    Args:
        v (np.ndarray): Input vector.
        eps (float): Lower bound for the norm to avoid division by zero.

    Returns:
        np.ndarray: Normalized vector (or the original vector if norm < eps).
    """
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def _pool_topk_mean_embedding(embedder: Any, titles: Sequence[str], k: int) -> np.ndarray:
    """Pool embeddings for a top-k title list by mean then L2-normalize.

    The function filters the input list to the first ``k`` entries, removes
    empty/whitespace-only strings, computes a mean embedding, and normalizes the
    result.

    For an empty filtered list, the function returns a zero vector. The
    dimension is inferred by encoding a single dummy text.

    Args:
        embedder (Any): Embedder object accepted by :func:`_encode_texts_np`.
        titles (Sequence[str]): Title strings.
        k (int): Number of items to consider from the front of the list.

    Returns:
        np.ndarray: Normalized pooled embedding vector.
    """
    # Filter to Top-K non-empty strings
    ts = [str(t).strip() for t in list(titles)[:k] if str(t).strip()]
    if len(ts) == 0:
        # infer dimension cheaply
        d = _encode_texts_np(embedder, ["__dummy__"]).shape[1]
        return np.zeros((d,), dtype=np.float32)

    E = _encode_texts_np(embedder, ts)  # [m,d]
    v = np.mean(E, axis=0).astype(np.float32)
    return _l2_normalize(v)


def snsr_snsv_proxy_from_title_lists(
    rec_title_lists: Sequence[Sequence[str]],
    group_keys: Sequence[str],
    embedder: Any,
    *,
    k: int = 10,
    min_group_size: int = 30,
) -> SNSMetrics:
    """Compute SNSR/SNSV proxy metrics from recommendation title lists.

    The computation is performed as:

    1. Pool each example's top-k title embeddings by mean and L2-normalize.
    2. Compute a mean pooled vector per group key and L2-normalize.
    3. Compute pairwise cosine distances between group vectors.
    4. Return the maximum distance (SNSR) and mean distance (SNSV).

    Args:
        rec_title_lists (Sequence[Sequence[str]]): Recommended title lists per
            example.
        group_keys (Sequence[str]): Group key per example.
        embedder (Any): Embedder object accepted by :func:`_encode_texts_np`.
        k (int): Cutoff for the top-k list.
        min_group_size (int): Minimum number of examples required for a group
            to be included.

    Returns:
        SNSMetrics: Computed proxy metrics.

    Raises:
        ValueError: If ``rec_title_lists`` and ``group_keys`` have different
            lengths.
    """
    if len(rec_title_lists) != len(group_keys):
        raise ValueError("rec_title_lists and group_keys must have same length")

    # pooled vector per example
    pooled = np.stack(
        [_pool_topk_mean_embedding(embedder, titles, k=k) for titles in rec_title_lists],
        axis=0,
    )  # [n,d]

    # aggregate by group (mean pooled vector)
    groups: Dict[str, List[int]] = {}
    for i, g in enumerate(group_keys):
        groups.setdefault(str(g), []).append(i)

    group_vecs: List[np.ndarray] = []
    for g, idxs in groups.items():
        if len(idxs) < min_group_size:
            continue
        v = np.mean(pooled[idxs], axis=0).astype(np.float32)
        group_vecs.append(_l2_normalize(v))

    n_groups = len(group_vecs)
    if n_groups < 2:
        return SNSMetrics(SNSR=0.0, SNSV=0.0, n_groups_used=n_groups)

    G = np.stack(group_vecs, axis=0)  # [G,d], normalized
    # cosine similarity = dot (since normalized), distance = 1 - dot
    sim = G @ G.T
    dist = 1.0 - sim
    # take upper triangle i<j
    dists = dist[np.triu_indices(n_groups, k=1)]
    SNSR = float(np.max(dists)) if dists.size else 0.0
    SNSV = float(np.mean(dists)) if dists.size else 0.0
    return SNSMetrics(SNSR=SNSR, SNSV=SNSV, n_groups_used=n_groups)


def snsr_snsv_proxy_from_mid_lists(
    rec_mid_lists: Sequence[Sequence[int]],
    group_keys: Sequence[str],
    embedder: Any,
    item_db: Dict[int, Dict[str, str]],
    *,
    k: int = 10,
    min_group_size: int = 30,
) -> SNSMetrics:
    """Compute SNSR/SNSV proxy metrics from recommendation item-id lists.

    This helper builds title lists by looking up each recommended mid in
    ``item_db`` (using the ``"title"`` entry when present), then delegates to
    :func:`snsr_snsv_proxy_from_title_lists`.

    Args:
        rec_mid_lists (Sequence[Sequence[int]]): Recommended item ids per
            example.
        group_keys (Sequence[str]): Group key per example.
        embedder (Any): Embedder object accepted by :func:`_encode_texts_np`.
        item_db (Dict[int, Dict[str, str]]): Item metadata mapping.
        k (int): Cutoff for the top-k list.
        min_group_size (int): Minimum number of examples required for a group
            to be included.

    Returns:
        SNSMetrics: Computed proxy metrics.
    """
    title_lists: List[List[str]] = []
    for mids in rec_mid_lists:
        titles = []
        for mid in list(mids)[:k]:
            info = item_db.get(int(mid), {})
            t = str(info.get("title", "")).strip()
            if t:
                titles.append(t)
        title_lists.append(titles)

    return snsr_snsv_proxy_from_title_lists(
        rec_title_lists=title_lists,
        group_keys=group_keys,
        embedder=embedder,
        k=k,
        min_group_size=min_group_size,
    )
