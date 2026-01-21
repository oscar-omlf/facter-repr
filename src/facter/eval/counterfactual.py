from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


_ML_AGE_BUCKETS = sorted(AGE_ID2LABEL.keys())   # [1, 18, 25, 35, 45, 50, 56]
_ML_OCC_IDS = sorted(OCC_ID2LABEL.keys())       # [0..20]

@dataclass(frozen=True)
class CFRConfig:
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
        # Ensure flip_attr is always a sequence, convert str to list if needed
        if isinstance(self.flip_attr, str):
            object.__setattr__(self, 'flip_attr', [self.flip_attr])
        elif self.flip_attr is None:
            object.__setattr__(self, 'flip_attr', ["gender"])
        
        # Validate flip_strategy
        if self.flip_strategy not in ["random", "minimal"]:
            raise ValueError("flip_strategy must be 'random' or 'minimal'")


def get_flipped_value(attr: str, value, strategy: str = "random") -> str:
    """
    Get a flipped value for a protected attribute (MovieLens).
    
    Strategy options:
    - "random": uniformly random valid value (may be the same as original)
    - "minimal": minimal in-domain flip (gender: opposite, age: adjacent, occupation: next)
    """
    if strategy == "random":
        return get_random_protected_value(attr)
    elif strategy == "minimal":
        return flip_protected_value(attr, value)
    else:
        raise ValueError(f"Unknown flip strategy: {strategy}")


def get_random_protected_value(attr: str) -> str:
    """
    Get a random valid value for a protected attribute (MovieLens):
    - gender: random choice from [M, F]
    - age: random choice from valid age buckets
    - occupation: random choice from valid occupation IDs
    """
    if attr == "gender":
        return np.random.choice(["M", "F"])
    
    if attr == "age":
        return str(np.random.choice(_ML_AGE_BUCKETS))
    
    if attr == "occupation":
        return str(np.random.choice(_ML_OCC_IDS))
    
    return str(np.random.choice([True, False]))


def flip_protected_value(attr: str, value) -> str:
    """
    Minimal in-domain flips (MovieLens):
    - gender: M <-> F
    - age: move to adjacent valid age bucket (never produce invalid codes)
    - occupation: (o + 1) mod 21
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
            return str(value)

        if a not in _ML_AGE_BUCKETS:
            # snap to nearest bucket
            a = min(_ML_AGE_BUCKETS, key=lambda x: abs(x - a))

        idx = _ML_AGE_BUCKETS.index(a)
        if idx < len(_ML_AGE_BUCKETS) - 1:
            return str(_ML_AGE_BUCKETS[idx + 1])
        else:
            return str(_ML_AGE_BUCKETS[max(idx - 1, 0)])

    if attr == "occupation":
        try:
            o = int(value)
        except Exception:
            return str(value)
        if o not in _ML_OCC_IDS:
            o = 0
        return str((o + 1) % (max(_ML_OCC_IDS) + 1))

    return str(value)


def _embed_list_mean(embedder: TextEmbedder, mids: Sequence[int], item_db: Dict[int, Dict[str, str]]) -> np.ndarray:
    texts = [item_text(int(m), item_db) for m in mids]
    embs = embedder.encode_texts(texts)  # [K,D], normalized
    v = np.mean(embs, axis=0).astype(np.float32)
    # normalize mean vector to compare with cosine/dot
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def _l2_distance(u: np.ndarray, v: np.ndarray) -> float:
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
) -> float:
    """
    CFR proxy via counterfactual flips:
      CFR = mean_{examples} || f(x,a) - f(x,a') ||_2
    where f(.) is the mean embedding of top-k recommended items.

    Supports both rank and open-generation modes:
    - Rank mode: Uses ranker to rank candidates for original and counterfactual prompts
    - Open mode: Uses generator to generate titles for both prompts, maps to mids via catalogue_mapper/title_to_mid
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
        # Rank mode: batch all prompts (original + counterfactual) and score once
        batched_prompts: List[str] = []
        batched_candidates: List[Sequence[str]] = []
        batched_systems: List[Optional[str]] = []
        row_refs: List[Tuple[int, str]] = []  # (row_index, "orig"|"cf")

        for ridx, (_, row) in enumerate(rows.iterrows()):
            cand_titles = row["candidate_titles"]
            prompt_orig = row["prompt_rank"]
            system_prompt = row[f"system_prompt_iter{iter}" if iter is not None else "system_prompt"]

            # counterfactual prompt: flip all specified attributes
            row_cf = row.to_dict()
            for attr in cfg.flip_attr:
                row_cf[attr] = get_flipped_value(attr, row_cf[attr], strategy=cfg.flip_strategy)
            prompt_cf = build_ranking_prompt(row_cf, cand_titles, prompt_cfg)

            batched_prompts.extend([prompt_orig, prompt_cf])
            batched_candidates.extend([cand_titles, cand_titles])
            batched_systems.extend([system_prompt, system_prompt])
            row_refs.extend([(ridx, "orig"), (ridx, "cf")])

        ranked_all = ranker.rank_batch(batched_prompts, batched_candidates, batched_systems, progress=progress)

        # Collect results per row
        row_to_indices: Dict[int, Dict[str, List[int]]] = {}
        for (ridx, kind), (idxs, _) in zip(row_refs, ranked_all):
            row_to_indices.setdefault(ridx, {})[kind] = idxs

        for ridx, (_, row) in enumerate(rows.iterrows()):
            idx_orig = row_to_indices.get(ridx, {}).get("orig", [])[: cfg.k]
            idx_cf = row_to_indices.get(ridx, {}).get("cf", [])[: cfg.k]

            mids_orig = [int(row["candidate_mids"][i]) for i in idx_orig]
            mids_cf = [int(row["candidate_mids"][i]) for i in idx_cf]

            if not mids_orig or not mids_cf:
                continue

            v_orig = _embed_list_mean(embedder, mids_orig, item_db)
            v_cf = _embed_list_mean(embedder, mids_cf, item_db)
            dists.append(_l2_distance(v_orig, v_cf))

    elif predict_mode == "open":
        # Open mode: use generator to produce titles, map to mids
        if title_to_mid is None and catalogue_mapper is None:
            title_to_mid = build_title_to_mid_dict(item_db)

        # Collect all prompts (original and counterfactual) for batched generation
        prompts_all: List[str] = []
        system_prompts_all: List[str] = []
        
        for _, row in rows.iterrows():
            system_prompt = row[f"system_prompt_iter{iter}" if iter is not None else "system_prompt"]

            # Original prompt
            prompt_orig = row.get("prompt_open", row.get("prompt_gen", None))
            if prompt_orig is None:
                prompt_orig = build_open_prompt(row.to_dict(), prompt_cfg)

            # Counterfactual prompt: flip all specified attributes
            row_cf = row.to_dict()
            for attr in cfg.flip_attr:
                row_cf[attr] = get_flipped_value(attr, row_cf[attr], strategy=cfg.flip_strategy)
            prompt_cf = build_open_prompt(row_cf, prompt_cfg)

            prompts_all.append(prompt_orig)
            prompts_all.append(prompt_cf)
            system_prompts_all.append(system_prompt)
            system_prompts_all.append(system_prompt)

        # Single batched generation call for all prompts
        all_titles = generator.generate_topk(prompts_all, system_prompts_all, k=cfg.k, progress=progress)

        # Process results in pairs (original, counterfactual)
        for i in range(0, len(all_titles), 2):
            titles_orig = all_titles[i]
            titles_cf = all_titles[i + 1]

            # Map to mids
            mids_orig: List[int] = []
            mids_cf: List[int] = []

            if catalogue_mapper is not None:
                # Use embedding-based mapper
                map_res_orig = catalogue_mapper.map_list(titles_orig, k=cfg.k, min_sim=min_sim)
                mids_orig = [int(m) for m in getattr(map_res_orig, "mapped_mids", []) if m is not None]
                map_res_cf = catalogue_mapper.map_list(titles_cf, k=cfg.k, min_sim=min_sim)
                mids_cf = [int(m) for m in getattr(map_res_cf, "mapped_mids", []) if m is not None]
            elif title_to_mid is not None:
                # Use normalized dict mapping
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

            # Compute distance if we have valid mids
            if mids_orig and mids_cf:
                v_orig = _embed_list_mean(embedder, mids_orig, item_db)
                v_cf = _embed_list_mean(embedder, mids_cf, item_db)
                dists.append(_l2_distance(v_orig, v_cf))

    else:
        raise ValueError("predict_mode must be 'rank' or 'open'")

    return float(np.mean(dists)) if dists else 0.0

