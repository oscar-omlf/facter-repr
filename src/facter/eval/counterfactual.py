"""Compute counterfactual recommendation sensitivity metrics.

This module implements a counterfactual evaluation routine that perturbs
protected-attribute fields in MovieLens-style user context (e.g., gender, age,
occupation), reruns a ranking or open-generation model, and measures the
distance between the resulting recommendation lists in an embedding space.

TODO(doc): Clarify how the returned metric relates to any counterfactual metric
definition in the paper; this file documents only implemented behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from facter.data.prompts import AGE_ID2LABEL, OCC_ID2LABEL
from facter.data.prompts import PromptConfig, build_ranking_prompt, build_open_prompt
from facter.eval.catalogue_map import CatalogueMapper
from facter.eval.prediction import build_title_to_mid_dict
from facter.fairness.scoring import item_text
from facter.models.embedder import TextEmbedder
from facter.models.generator import Generator
from facter.models.ranker import Ranker

if TYPE_CHECKING:
    from facter.models.item_embedder import ItemEmbedder

_ML_AGE_BUCKETS = sorted(AGE_ID2LABEL.keys())   # [1, 18, 25, 35, 45, 50, 56]
_ML_OCC_IDS = sorted(OCC_ID2LABEL.keys())       # [0..20]

@dataclass(frozen=True)
class CFRConfig:
    """Configure counterfactual recommendation evaluation.

    The configuration specifies which protected attributes to flip, how to flip
    them, how many examples to sample from a dataset, and how many
    recommendations to consider.

    Note:
        ``__post_init__`` normalizes ``flip_attr`` so it is always a sequence.

    Attributes:
        protected_cols (Tuple[str, ...]): Column names considered protected
            attributes in the input DataFrame.
        k (int): Number of items to consider for each recommendation list.
        flip_attr (Optional[Sequence[str]]): Protected attributes to flip.
            If a string is provided, it is converted to a single-element list.
            If None, defaults to ``["gender"]``.
        flip_strategy (str): Strategy name used by :func:`get_flipped_value`.
            Supported values are ``"random"`` and ``"minimal"``.
        n_samples (int): Number of rows to sample from the input DataFrame.
        seed (int): Random seed used for DataFrame sampling.

    Raises:
        ValueError: If ``flip_strategy`` is not one of ``"random"`` or
            ``"minimal"``.
    """
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    k: int = 10
    # which attribute(s) to flip for CFR. Can be a single str or list of str.
    # If list, all attributes are flipped simultaneously.
    flip_attr: Optional[Sequence[str]] = None
    # flip strategy: "random" (uniform random from valid values) or "minimal" (adjacent/opposite values)
    flip_strategy: str = "random"
    n_samples: int = 200
    seed: int = 42
    
    def __post_init__(self):
        """Normalize and validate configuration fields."""
        # Ensure flip_attr is always a sequence, convert str to list if needed
        if isinstance(self.flip_attr, str):
            object.__setattr__(self, 'flip_attr', [self.flip_attr])
        elif self.flip_attr is None:
            object.__setattr__(self, 'flip_attr', ["gender"])
        
        # Validate flip_strategy
        if self.flip_strategy not in ["random", "minimal"]:
            raise ValueError("flip_strategy must be 'random' or 'minimal'")


def get_flipped_value(attr: str, value, strategy: str = "random") -> str:
    """Return a flipped value for a MovieLens-style protected attribute.

    The flip behavior depends on the ``strategy``:

    - ``"random"``: sample a random valid value for the attribute (may be the
      same as the input value).
    - ``"minimal"``: apply a deterministic in-domain flip as implemented by
      :func:`flip_protected_value`.

    Args:
        attr (str): Attribute name (e.g., ``"gender"``, ``"age"``,
            ``"occupation"``).
        value (Any): Current value for the attribute.
        strategy (str): Flip strategy name.

    Returns:
        str: Flipped value.

    Raises:
        ValueError: If ``strategy`` is not recognized.
    """
    if strategy == "random":
        return get_random_protected_value(attr)
    elif strategy == "minimal":
        return flip_protected_value(attr, value)
    else:
        raise ValueError(f"Unknown flip strategy: {strategy}")


def get_random_protected_value(attr: str):
    """Sample a random valid value for a MovieLens-style protected attribute.

    The return types follow the per-attribute branches in this function.

    Args:
        attr (str): Attribute name.

    Returns:
        Any: Randomly sampled value. For ``"gender"`` this is a string; for
        ``"age"`` and ``"occupation"`` this is an ``int``.
    """
    if attr == "gender":
        return str(np.random.choice(["M", "F"]))

    if attr == "age":
        return int(np.random.choice(_ML_AGE_BUCKETS))

    if attr == "occupation":
        return int(np.random.choice(_ML_OCC_IDS))
    # generic fallback
    return np.random.choice([True, False])


def flip_protected_value(attr: str, value):
    """Apply a deterministic in-domain flip for MovieLens-style attributes.

    Minimal in-domain flips implemented by this function:
    - ``gender``: swap ``"M"`` and ``"F"`` (case-insensitive).
    - ``age``: move to an adjacent valid bucket in ``_ML_AGE_BUCKETS`` (after
      snapping to the closest bucket if needed).
    - ``occupation``: compute ``(o + 1) % (max(_ML_OCC_IDS) + 1)`` after
      converting to int and snapping invalid values to 0.

    Args:
        attr (str): Attribute name.
        value (Any): Current value.

    Returns:
        Any: Flipped value. Type depends on the attribute branch.
    """
    if attr == "gender":
        v = str(value)
        if v.upper() == "M":
            return "F"
        if v.upper() == "F":
            return "M"
        return v

    if attr == "age":
        try:
            a = int(value)
        except Exception:
            return value

        if a not in _ML_AGE_BUCKETS:
            a = min(_ML_AGE_BUCKETS, key=lambda x: abs(x - a))

        idx = _ML_AGE_BUCKETS.index(a)
        if idx < len(_ML_AGE_BUCKETS) - 1:
            return int(_ML_AGE_BUCKETS[idx + 1])
        else:
            return int(_ML_AGE_BUCKETS[max(idx - 1, 0)])

    if attr == "occupation":
        try:
            o = int(value)
        except Exception:
            return value
        if o not in _ML_OCC_IDS:
            o = 0
        return int((o + 1) % (max(_ML_OCC_IDS) + 1))

    return value



def _embed_list_mean(
    embedder: TextEmbedder, 
    mids: Sequence[int], 
    item_db: Dict[int, Dict[str, str]],
    item_embedder: Optional["ItemEmbedder"] = None
) -> np.ndarray:
    """Compute a normalized mean embedding for a list of item ids.

    If an ``item_embedder`` is provided, this function uses precomputed item
    embeddings via ``item_embedder.get_embeddings``. Otherwise it falls back to
    embedding item text constructed by :func:`facter.fairness.scoring.item_text`.

    Args:
        embedder (TextEmbedder): Text embedder used when ``item_embedder`` is
            not provided.
        mids (Sequence[int]): Item ids to embed.
        item_db (Dict[int, Dict[str, str]]): Item metadata used by
            :func:`item_text` when embedding from text.
        item_embedder (Optional[ItemEmbedder]): Optional item-embedding cache.

    Returns:
        np.ndarray: A 1D normalized mean embedding vector.
    """
    if item_embedder is not None:
        # Use pre-computed item embeddings
        embs = item_embedder.get_embeddings(mids)  # [K,D], normalized
    else:
        # Fall back to text-based embedding
        texts = [item_text(int(m), item_db) for m in mids]
        embs = embedder.encode_texts(texts)  # [K,D], normalized
    
    v = np.mean(embs, axis=0).astype(np.float32)
    # normalize mean vector to compare with cosine/dot
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def _l2_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the Euclidean distance between two vectors.

    Args:
        u (np.ndarray): First vector.
        v (np.ndarray): Second vector.

    Returns:
        float: Euclidean (L2) distance.
    """
    return float(np.linalg.norm(u - v))


def compute_cfr(
    df: pd.DataFrame,
    embedder: TextEmbedder,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    cfg: CFRConfig,
    predict_mode: str = "rank",  # "rank" | "open"
    ranker: Optional[Ranker] = None,
    generator: Optional[Generator] = None,
    catalogue_mapper: Optional[CatalogueMapper] = None,
    title_to_mid: Optional[Dict[str, int]] = None,
    min_sim: float = 0.65,
    iter: Optional[int] = None,
    progress: bool = False,
    item_embedder: Optional["ItemEmbedder"] = None,
) -> float:
    """Compute a counterfactual recommendation distance score.

    The function samples up to ``cfg.n_samples`` rows from ``df`` (without
    replacement), constructs an "original" and a "counterfactual" prompt per
    sampled row by flipping each attribute in ``cfg.flip_attr``, runs the model
    in either rank or open-generation mode, and measures the L2 distance between
    the mean embeddings of the two resulting recommendation lists.

    - In ``predict_mode == 'rank'``, candidates are taken from
      ``row['candidate_titles']``/``row['candidate_mids']`` and prompts from
      ``row['prompt_rank']``.
    - In ``predict_mode == 'open'``, prompts are taken from ``row['prompt_open']``
      (or ``row['prompt_gen']``) if present, otherwise built with
      :func:`build_open_prompt`.

    Only rows where both the original and counterfactual recommendation lists
    can be mapped to at least one item id contribute to the mean.

    Args:
        df (pd.DataFrame): Source examples.
        embedder (TextEmbedder): Text embedder used to embed recommended items
            (unless ``item_embedder`` is provided).
        item_db (Dict[int, Dict[str, str]]): Item metadata mapping used for
            embedding and (in open mode) for title-to-id fallback building.
        prompt_cfg (PromptConfig): Prompt configuration used when building open
            prompts.
        cfg (CFRConfig): Counterfactual evaluation configuration.
        predict_mode (str): Either ``'rank'`` or ``'open'``.
        ranker (Optional[Ranker]): Required when ``predict_mode == 'rank'``.
        generator (Optional[Generator]): Required when ``predict_mode == 'open'``.
        catalogue_mapper (Optional[CatalogueMapper]): Optional embedding-based
            mapper used in open-generation mode.
        title_to_mid (Optional[Dict[str, int]]): Optional fallback mapping used
            in open-generation mode.
        min_sim (float): Minimum similarity threshold passed to the catalogue
            mapper.
        iter (Optional[int]): Optional iteration index used to select
            ``system_prompt_iter{iter}`` from each row.
        progress (bool): Whether to show progress in model batch calls.
        item_embedder (Optional[ItemEmbedder]): Optional item-embedding cache.

    Returns:
        float: Mean L2 distance between original and counterfactual mean-list
        embeddings. Returns 0.0 if no valid pairs contribute.

    Raises:
        ValueError: If any attribute in ``cfg.flip_attr`` is not contained in
            ``cfg.protected_cols``.
        ValueError: If ``predict_mode == 'rank'`` and ``ranker`` is not
            provided.
        ValueError: If ``predict_mode == 'open'`` and ``generator`` is not
            provided.
        ValueError: If ``predict_mode`` is not ``'rank'`` or ``'open'``.
    """
    for attr in cfg.flip_attr:
        if attr not in cfg.protected_cols:
            raise ValueError(f"flip_attr must contain only attributes from {cfg.protected_cols}, got {attr}")

    if predict_mode == "rank" and ranker is None:
        raise ValueError("rank mode requires ranker")
    if predict_mode == "open" and generator is None:
        raise ValueError("open mode requires generator")

    rows = df.sample(
        n=min(cfg.n_samples, len(df)), replace=False, random_state=cfg.seed
    )

    dists: List[float] = []

    if predict_mode == "rank":
        batched_prompts: List[str] = []
        batched_candidates: List[Sequence[str]] = []
        batched_systems: List[Optional[str]] = []
        row_refs: List[Tuple[int, str]] = []  # (row_index, "orig"|"cf")

        for ridx, (_, row) in enumerate(rows.iterrows()):
            cand_titles = row["candidate_titles"]
            prompt_orig = row["prompt_rank"]
            system_prompt = row[f"system_prompt_iter{iter}" if iter is not None else "system_prompt"]

            row_cf = row.to_dict()
            for attr in cfg.flip_attr:
                row_cf[attr] = get_flipped_value(attr, row_cf[attr], strategy=cfg.flip_strategy)
            prompt_cf = build_ranking_prompt(row_cf, cand_titles, prompt_cfg)

            batched_prompts.extend([prompt_orig, prompt_cf])
            batched_candidates.extend([cand_titles, cand_titles])
            batched_systems.extend([system_prompt, system_prompt])
            row_refs.extend([(ridx, "orig"), (ridx, "cf")])

        ranked_all = ranker.rank_batch(batched_prompts, batched_candidates, batched_systems, progress=progress)

        row_to_indices: Dict[int, Dict[str, List[int]]] = {}
        for (ridx, kind), (idxs, _) in zip(row_refs, ranked_all):
            row_to_indices.setdefault(ridx, {})[kind] = idxs

        for ridx, (_, row) in enumerate(rows.iterrows()):
            idx_orig = row_to_indices.get(ridx, {}).get("orig", [])[: cfg.k]
            idx_cf = row_to_indices.get(ridx, {}).get("cf", [])[: cfg.k]

            mids_orig = [int(row["candidate_mids"][i]) for i in idx_orig] if idx_orig else []
            mids_cf = [int(row["candidate_mids"][i]) for i in idx_cf] if idx_cf else []

            if not mids_orig or not mids_cf:
                continue

            v_orig = _embed_list_mean(embedder, mids_orig, item_db, item_embedder)
            v_cf = _embed_list_mean(embedder, mids_cf, item_db, item_embedder)
            dists.append(_l2_distance(v_orig, v_cf))

    elif predict_mode == "open":
        if title_to_mid is None and catalogue_mapper is None:
            title_to_mid = build_title_to_mid_dict(item_db)

        prompts_all: List[str] = []
        system_prompts_all: List[str] = []
        
        for _, row in rows.iterrows():
            system_prompt = row[f"system_prompt_iter{iter}" if iter is not None else "system_prompt"]

            prompt_orig = row.get("prompt_open", row.get("prompt_gen", None))
            if prompt_orig is None:
                prompt_orig = build_open_prompt(row.to_dict(), prompt_cfg)

            row_cf = row.to_dict()
            for attr in cfg.flip_attr:
                row_cf[attr] = get_flipped_value(attr, row_cf[attr], strategy=cfg.flip_strategy)
            prompt_cf = build_open_prompt(row_cf, prompt_cfg)

            prompts_all.append(prompt_orig)
            prompts_all.append(prompt_cf)
            system_prompts_all.append(system_prompt)
            system_prompts_all.append(system_prompt)

        all_titles = generator.generate_topk(prompts_all, system_prompts_all, k=cfg.k, progress=progress)

        for i in range(0, len(all_titles), 2):
            titles_orig = all_titles[i]
            titles_cf = all_titles[i + 1]

            mids_orig: List[int] = []
            mids_cf: List[int] = []

            if catalogue_mapper is not None:
                map_res_orig = catalogue_mapper.map_list(titles_orig, k=cfg.k, min_sim=min_sim)
                mids_orig = [int(m) for m in getattr(map_res_orig, "mapped_mids", []) if m is not None]
                map_res_cf = catalogue_mapper.map_list(titles_cf, k=cfg.k, min_sim=min_sim)
                mids_cf = [int(m) for m in getattr(map_res_cf, "mapped_mids", []) if m is not None]
            elif title_to_mid is not None:
                for tt in titles_orig:
                    key = str(tt).strip().lower()
                    mid = title_to_mid.get(key, -1)
                    if mid != -1 and int(mid) not in mids_orig:
                        mids_orig.append(int(mid))
                for tt in titles_cf:
                    key = str(tt).strip().lower()
                    mid = title_to_mid.get(key, -1)
                    if mid != -1 and int(mid) not in mids_cf:
                        mids_cf.append(int(mid))

            if mids_orig and mids_cf:
                v_orig = _embed_list_mean(embedder, mids_orig, item_db, item_embedder)
                v_cf = _embed_list_mean(embedder, mids_cf, item_db, item_embedder)
                dists.append(_l2_distance(v_orig, v_cf))

    else:
        raise ValueError("predict_mode must be 'rank' or 'open'")

    return float(np.mean(dists)) if dists else 0.0
