from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import pandas as pd


def recall_at_k(
    ranked_items: Sequence[int], relevant_items: Set[int], k: int = 10
) -> float:
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


def ndcg_at_k(
    ranked_items: Sequence[int], relevant_items: Set[int], k: int = 10
) -> float:
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

    return {
        "Recall@%d" % k: float(np.mean(recalls)),
        "NDCG@%d" % k: float(np.mean(ndcgs)),
    }


def count_violations(scores: Sequence[float], q_alpha: float) -> int:
    """
    Violations count per definition: S_new > Q_alpha.
    """
    return int(np.sum(np.array(scores, dtype=np.float32) > float(q_alpha)))


# SNSR/SNSV proxy functions.
# NOTE: I implemented them as per FACTER's code: https://github.com/AryaFayyazi/FACTER
# These formulae compeltely differ when compared to the ones reported in the paper, and the ones in the paper the authors cite.
# However, the formulae used in the authors implementation makes sense because it actually uses the embeddings.
@dataclass(frozen=True)
class SNSMetrics:
    SNSR: float
    SNSV: float
    n_groups_used: int


def _encode_texts_np(embedder: Any, texts: List[str]) -> np.ndarray:
    """
    Supports:
      - TextEmbedder: encode_texts(list[str]) -> np.ndarray
      - SentenceTransformer: encode(list[str]) -> np.ndarray
    Returns float32 numpy array [n,d].
    """
    if hasattr(embedder, "encode_texts"):
        E = embedder.encode_texts(texts)

    elif hasattr(embedder, "encode"):
        E = embedder.encode(texts)

    else:
        raise TypeError("embedder must expose encode_texts(...) or encode(...)")

    if isinstance(E, torch.Tensor):
        E = E.detach().cpu().numpy()

    E = np.asarray(E, dtype=np.float32)
    if E.ndim == 1:
        E = E[None, :]
    return E


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def _pool_topk_mean_embedding(
    embedder: Any, titles: Sequence[str], k: int
) -> np.ndarray:
    """
    Authors' proxy:
      pooled embedding per example = mean of title embeddings over Top-K list.

    Empty/invalid list -> zero vector (dim inferred once via dummy embed).
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
    """
    Compute SNSR/SNSV proxies exactly per authors:
      - pooled embedding per example = mean(title embeddings of Top-K list)
      - group mean = mean pooled vector within group
      - pairwise cosine distances across group means
      - SNSR = max distance, SNSV = mean distance

    group_keys: one string per example, e.g. "gender=F|age=25-34|occupation=3"
    """
    if len(rec_title_lists) != len(group_keys):
        raise ValueError("rec_title_lists and group_keys must have same length")

    # 1. Collect all unique titles
    all_titles = set()
    for titles in rec_title_lists:
        # Take top-k valid strings
        ts = [str(t).strip() for t in list(titles)[:k] if str(t).strip()]
        all_titles.update(ts)

    # 2. Encode all unique titles
    unique_titles_list = sorted(list(all_titles))

    # Edge case: no valid titles at all
    if not unique_titles_list:
        return SNSMetrics(SNSR=0.0, SNSV=0.0, n_groups_used=0)

    # Add a dummy token for padding/invalid entries
    unique_titles_list = ["<PAD>"] + unique_titles_list

    # [U+1, D]
    all_embs = _encode_texts_np(embedder, unique_titles_list)

    # Ensure PAD vector is zero
    all_embs[0] = 0.0

    # 3. Map titles to integer indices for vectorization
    title_to_idx = {t: i for i, t in enumerate(unique_titles_list)}
    # title_to_vec = {t: all_embs[i] for i, t in enumerate(unique_titles_list)}

    # Construct indices matrix [N, K]
    N = len(rec_title_lists)
    indices = np.zeros((N, k), dtype=np.int32)  # defaults to 0 (PAD)

    for i, titles in enumerate(rec_title_lists):
        ts = [str(t).strip() for t in list(titles)[:k] if str(t).strip()]
        for j, t in enumerate(ts):
            if j < k:
                indices[i, j] = title_to_idx.get(t, 0)

    # 4. Gather embeddings [N, K, D] (replaces loop over N users)
    gathered_embs = all_embs[indices]

    # 5. Average pooling per user [N, D]
    # We want to ignore PAD vectors in the mean.
    # Count non-zero vectors along axis 1
    # Check if index is not 0
    valid_mask = indices != 0  # [N, K]
    counts = valid_mask.sum(axis=1, keepdims=True)  # [N, 1]

    sum_embs = gathered_embs.sum(axis=1)  # [N, D]

    # Avoid divide by zero
    counts = np.maximum(counts, 1.0)
    pooled = sum_embs / counts  # [N, D]

    # Normalize (vectorized L2 normalization)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    pooled = pooled / norms

    # 6. Aggregate by Group
    # Use Pandas for fast groupby mean
    df_pool = pd.DataFrame(pooled)
    df_pool["group"] = list(group_keys)

    # Filter small groups
    g_counts = df_pool["group"].value_counts()
    valid_groups = g_counts[g_counts >= min_group_size].index

    n_valid_groups = len(valid_groups)
    if n_valid_groups < 2:
        return SNSMetrics(SNSR=0.0, SNSV=0.0, n_groups_used=n_valid_groups)

    # Compute group means [G, D]
    group_means = df_pool[df_pool["group"].isin(valid_groups)].groupby("group").mean()
    G = group_means.values.astype(np.float32)

    # Re-normalize group vectors
    g_norms = np.linalg.norm(G, axis=1, keepdims=True)
    g_norms = np.maximum(g_norms, 1e-12)
    G = G / g_norms

    # 7. Pairwise Distances
    sim = G @ G.T
    dist = 1.0 - sim
    n_groups = G.shape[0]

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
    """
    Same SNSR/SNSV proxy, but recs are item IDs.
    We convert mids -> canonical titles from item_db, then use title embedding pooling.
    """
    title_lists: List[List[str]] = [
        [
            str(item_db.get(int(mid), {}).get("title", "")).strip()
            for mid in list(mids)[:k]
        ]
        for mids in rec_mid_lists
    ]

    return snsr_snsv_proxy_from_title_lists(
        rec_title_lists=title_lists,
        group_keys=group_keys,
        embedder=embedder,
        k=k,
        min_group_size=min_group_size,
    )
