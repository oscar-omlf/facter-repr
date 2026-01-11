from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

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
    progress: bool = False,
) -> List[List[int]]:
    """
    Returns ranked mids list per row (top-k by default).
    """
    preds: list[int] = []
    ranked_mids_list: list[list[int]] = []
    system_prompts_list: list[str] = []
    ranker_responses_list: list[str] = []


    it = df.iterrows()
    if progress:
        it = tqdm(it, total=len(df), desc="Baseline: zero-shot ranking")
    for _, row in it:
        ranked_idx, raw_response = ranker.rank(row["prompt_rank"], row["candidate_titles"], system_prompt=system_prompt)
        topk_idx = ranked_idx[:k]
        mids = [int(row["candidate_mids"][i]) for i in topk_idx]

        ranked_mids_list.append(mids)
        system_prompts_list.append(system_prompt)
        ranker_responses_list.append(raw_response)
        preds.append(mids[0])  # top-1

    df[f"pred_mid"] = preds
    df[f"ranked_mids"] = ranked_mids_list
    df[f"system_prompt"] = system_prompts_list
    df[f"ranker_response"] = ranker_responses_list

    return df


def evaluate_zero_shot(
    df: pd.DataFrame,
    ranker: Ranker,
    k: int = 10,
    progress: bool = False,
) -> Dict[str, float]:
    ranked_mids = run_zero_shot_ranking(df, ranker, k=k, system_prompt=None, progress=progress)["ranked_mids"].tolist()
    targets = df["target_mid"].astype(int).tolist()
    return mean_recall_ndcg(ranked_mids, targets, k=k)


def run_up5_placeholder(*args, **kwargs):
    """
    UP5 to be implemented.
    """
    raise NotImplementedError()
