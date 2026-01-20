from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from facter.data.prompts import AGE_ID2LABEL, OCC_ID2LABEL
from facter.data.prompts import PromptConfig, build_ranking_prompt, build_open_prompt
from facter.eval.catalogue_map import CatalogueMapper
from facter.eval.prediction import build_title_to_mid_dict
from facter.fairness.scoring import item_text
from facter.models.embedder import TextEmbedder
from facter.models.generator import Generator
from facter.models.ranker import Ranker


_ML_AGE_BUCKETS = sorted(AGE_ID2LABEL.keys())  # [1, 18, 25, 35, 45, 50, 56]
_ML_OCC_IDS = sorted(OCC_ID2LABEL.keys())  # [0, ..., 20]


@dataclass(frozen=True)
class CFRConfig:
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    k: int = 10
    # which attribute(s) to flip for CFR. Can be a single str or list of str.
    # If list, all attributes are flipped simultaneously.
    flip_attr: Optional[Sequence[str]] = None
    # flip strategy: "random" (uniform random from valid values) or "minimal" (adjacent/opposite values)
    flip_strategy: str = "random"

    def __post_init__(self):
        # Ensure flip_attr is always a sequence, convert str to list if needed
        if isinstance(self.flip_attr, str):
            object.__setattr__(self, "flip_attr", [self.flip_attr])
        elif self.flip_attr is None:
            object.__setattr__(self, "flip_attr", ["gender"])

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


def _embed_list_mean(
    embedder: TextEmbedder, mids: Sequence[int], item_db: Dict[int, Dict[str, str]]
) -> np.ndarray:
    texts = [item_text(int(m), item_db) for m in mids]
    embs = embedder.encode_texts(texts)  # [K,D], normalized
    if isinstance(embs, torch.Tensor):
        embs = embs.detach().cpu().numpy()

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
            raise ValueError(
                f"flip_attr must contain only attributes from {cfg.protected_cols}, got {attr}"
            )

    if predict_mode == "rank" and ranker is None:
        raise ValueError("rank mode requires ranker")

    if predict_mode == "open" and generator is None:
        raise ValueError("open mode requires generator")

    # Determine system prompt column once
    sys_col = f"system_prompt_iter{iter}" if iter is not None else "system_prompt"

    # Helpers for constructing prompt strings via apply (faster than iterrows)
    def build_cf_row_dict(row: pd.Series):
        d = row.to_dict()
        for attr in cfg.flip_attr:
            d[attr] = get_flipped_value(attr, d[attr], strategy=cfg.flip_strategy)

        return d

    def make_cf_prompt(row: pd.Series):
        cf_data = build_cf_row_dict(row)
        return build_ranking_prompt(cf_data, row["candidate_titles"], prompt_cfg)

    dists: List[float] = []
    if predict_mode == "rank":
        # OPTIMIZATION: Prepare batches for Ranker
        prompts_orig = df["prompt_rank"].tolist()
        candidates_list = df["candidate_titles"].tolist()
        systems = df[sys_col].tolist()

        # Build counterfactual prompts
        prompts_cf = df.apply(make_cf_prompt, axis=1).tolist()

        # Batch Inference - Original
        # We process in one go (Ranker handles batching internally or we pass all list)
        # HFChatRanker.rank_batch handles list inputs
        # Batch Inference - Counterfactual
        idx_orig_list, _ = ranker.rank_batch(prompts_orig, candidates_list, systems)
        idx_cf_list, _ = ranker.rank_batch(prompts_cf, candidates_list, systems)

        # Compute distances
        cand_mids_list = df["candidate_mids"].tolist()
        for i in range(len(df)):
            cand_mids = cand_mids = cand_mids_list[i]

            # Slice top-k
            top_orig = idx_orig_list[i][: cfg.k]
            top_cf = idx_cf_list[i][: cfg.k]

            mids_orig = [int(cand_mids[x]) for x in top_orig]
            mids_cf = [int(cand_mids[x]) for x in top_cf]

            v_orig = _embed_list_mean(embedder, mids_orig, item_db)
            v_cf = _embed_list_mean(embedder, mids_cf, item_db)
            dists.append(_l2_distance(v_orig, v_cf))

    elif predict_mode == "open":
        if title_to_mid is None and catalogue_mapper is None:
            title_to_mid = build_title_to_mid_dict(item_db)

        # Prefer existing prompt columns if available
        if "prompt_open" in df.columns:
            prompts_orig = df["prompt_open"].fillna("").tolist()
            # fallback if empty strings
            if any(not p for p in prompts_orig):
                # Only re-build missing
                prompts_orig = df.apply(
                    lambda r: r.get("prompt_open")
                    or r.get("prompt_gen")
                    or build_open_prompt(r.to_dict(), prompt_cfg),
                    axis=1,
                ).tolist()

        elif "prompt_gen" in df.columns:
            prompts_orig = df["prompt_gen"].tolist()

        else:
            prompts_orig = df.apply(
                lambda r: build_open_prompt(r.to_dict(), prompt_cfg), axis=1
            ).tolist()

        # Counterfactual prompts
        prompts_cf = df.apply(
            lambda r: build_open_prompt(build_cf_row_dict(r), prompt_cfg), axis=1
        ).tolist()

        systems = df[sys_col].tolist()

        # Interleave for generation? Or just concat. Concat is easier.
        prompts_all = prompts_orig + prompts_cf
        systems_all = systems + systems

        # Single batched generation call for all prompts
        all_titles = generator.generate_topk(prompts_all, systems_all, k=cfg.k)

        n = len(df)
        titles_orig_list = all_titles[:n]
        titles_cf_list = all_titles[n:]

        # Map and Compute
        for i in range(n):
            titles_orig = titles_orig_list[i]
            titles_cf = titles_cf_list[i]

            # Map to mids
            mids_orig: List[int] = []
            mids_cf: List[int] = []

            if catalogue_mapper is not None:
                # Use embedding-based mapper
                map_res_orig = catalogue_mapper.map_list(
                    titles_orig, k=cfg.k, min_sim=min_sim
                )
                mids_orig = [
                    int(m)
                    for m in getattr(map_res_orig, "mapped_mids", [])
                    if m is not None
                ]

                map_res_cf = catalogue_mapper.map_list(
                    titles_cf, k=cfg.k, min_sim=min_sim
                )
                mids_cf = [
                    int(m)
                    for m in getattr(map_res_cf, "mapped_mids", [])
                    if m is not None
                ]

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
