from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from facter.models.ranker import Ranker
from facter.models.generator import Generator
from facter.eval.metrics import mean_recall_ndcg
from facter.eval.prediction import predict_batch_rank, predict_batch_open, build_title_to_mid_dict
from facter.eval.catalogue_map import CatalogMapper
from facter.data.prompts import PromptConfig


@dataclass(frozen=True)
class BaselineConfig:
    k: int = 10


def run_zero_shot(
    df: pd.DataFrame,
    ranker: Optional[Ranker] = None,
    generator: Optional[Generator] = None,
    item_db: Optional[Dict[int, Dict[str, str]]] = None,
    predict_mode: str = "rank",
    k: int = 10,
    system_prompt: str | None = None,
    progress: bool = False,
    catalogue_mapper: Optional[CatalogMapper] = None,
    title_to_mid: Optional[Dict[str, int]] = None,
    min_sim: float = 0.65,
) -> pd.DataFrame:
    """
    Unified zero-shot prediction: rank or open-generation based on predict_mode.
    
    Rank mode:
    - Requires ranker
    - Returns ranked_mids from candidate selection
    
    Open mode:
    - Requires generator and item_db
    - Generates top-k titles and maps to mids via catalogue_mapper or title_to_mid fallback
    """
    if predict_mode == "rank":
        if ranker is None:
            raise ValueError("rank mode requires ranker")
        res = predict_batch_rank(df, ranker, item_db or {}, system_prompt=system_prompt, progress=progress)
        df["pred_mid"] = res.pred_mids
        df["ranked_mids"] = res.ranked_mids_list
        df["system_prompt"] = [system_prompt] * len(df)
        df["ranker_response"] = res.model_responses
        
    elif predict_mode == "open":
        if generator is None or item_db is None:
            raise ValueError("open mode requires generator and item_db")
        
        # Build title_to_mid if not provided and no catalog mapper
        if title_to_mid is None and catalogue_mapper is None:
            title_to_mid = build_title_to_mid_dict(item_db)
        
        # Create prompt config with k_recs set to k
        prompt_cfg = PromptConfig(k_recs=k)
        
        res = predict_batch_open(
            df,
            generator,
            item_db,
            prompt_cfg,
            system_prompt=system_prompt,
            catalogue_mapper=catalogue_mapper,
            title_to_mid=title_to_mid,
            min_sim=min_sim,
            progress=progress,
        )
        
        df["pred_mid"] = res.pred_mids
        df["ranked_mids"] = res.ranked_mids_list
        df["system_prompt"] = [system_prompt] * len(df)
        df["generator_response"] = res.model_responses
        df["valid_at_k"] = res.valid_at_k_list
        
    else:
        raise ValueError("predict_mode must be 'rank' or 'open'")
    
    return df


def run_zero_shot_ranking(
    df: pd.DataFrame,
    ranker: Ranker,
    k: int = 10,
    system_prompt: str | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Deprecated: use run_zero_shot(..., predict_mode='rank') instead.
    Kept for backward compatibility.
    """
    return run_zero_shot(
        df,
        ranker=ranker,
        predict_mode="rank",
        k=k,
        system_prompt=system_prompt,
        progress=progress,
    )


def run_zero_shot_generation(
    df: pd.DataFrame,
    generator: Generator,
    item_db: Dict[int, Dict[str, str]],
    k: int = 10,
    system_prompt: str | None = None,
    catalogue_mapper: Optional[CatalogMapper] = None,
    title_to_mid: Optional[Dict[str, int]] = None,
    progress: bool = False,
    min_sim: float = 0.65,
) -> pd.DataFrame:
    """
    Deprecated: use run_zero_shot(..., predict_mode='open') instead.
    Kept for backward compatibility.
    """
    return run_zero_shot(
        df,
        generator=generator,
        item_db=item_db,
        predict_mode="open",
        k=k,
        system_prompt=system_prompt,
        catalogue_mapper=catalogue_mapper,
        title_to_mid=title_to_mid,
        progress=progress,
        min_sim=min_sim,
    )

def evaluate_zero_shot(
    df: pd.DataFrame,
    k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate zero-shot predictions using recall and NDCG metrics.
    Assumes df contains 'ranked_mids' column (populated by run_zero_shot).
    """
    if "ranked_mids" not in df:
        raise ValueError("DataFrame must contain 'ranked_mids' column; call run_zero_shot first")

    ranked_mids = df["ranked_mids"].tolist()
    targets = df["target_mid"].astype(int).tolist()
    return mean_recall_ndcg(ranked_mids, targets, k=k)


def run_up5_placeholder(*args, **kwargs):
    """
    UP5 to be implemented.
    """
    raise NotImplementedError()
