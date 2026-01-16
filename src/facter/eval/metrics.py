from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Optional, Tuple, Any

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

    E = np.asarray(E, dtype=np.float32)
    if E.ndim == 1:
        E = E[None, :]
    return E


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def _pool_topk_mean_embedding(embedder: Any, titles: Sequence[str], k: int) -> np.ndarray:
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
    """
    Same SNSR/SNSV proxy, but recs are item IDs.
    We convert mids -> canonical titles from item_db, then use title embedding pooling.
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
