from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from facter.models.ranker import Ranker
from facter.eval.metrics import mean_recall_ndcg


@dataclass(frozen=True)
class BaselineConfig:
    k: int = 10


def run_zero_shot_ranking(
    df: pd.DataFrame,
    ranker: Ranker,
    k: int = 10,
    system_prompt: str | None = None,
) -> List[List[int]]:
    """
    Returns ranked mids list per row (top-k by default).
    """
    ranked_mids: List[List[int]] = []
    for _, row in df.iterrows():
        ranked_idx = ranker.rank(row["prompt_rank"], row["candidate_titles"], system_prompt=system_prompt)
        topk_idx = ranked_idx[:k]
        mids = [int(row["candidate_mids"][i]) for i in topk_idx]
        ranked_mids.append(mids)
    return ranked_mids


def evaluate_zero_shot(
    df: pd.DataFrame,
    ranker: Ranker,
    k: int = 10,
) -> Dict[str, float]:
    ranked_mids = run_zero_shot_ranking(df, ranker, k=k, system_prompt=None)
    targets = df["target_mid"].astype(int).tolist()
    return mean_recall_ndcg(ranked_mids, targets, k=k)


def run_up5_placeholder(*args, **kwargs):
    """
    UP5 to be implemented.
    """
    raise NotImplementedError()
