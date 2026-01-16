from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

from facter.eval.metrics import mean_recall_ndcg
from facter.models.ranker import Ranker


@dataclass(frozen=True)
class BaselineConfig:
    k: int = 10


def run_zero_shot_ranking(
    df: pd.DataFrame,
    ranker: Ranker,
    k: int = 10,
    system_prompt: str | None = None,
    progress: bool = False,
) -> List[List[str]]:
    """
    Returns ranked mids list per row (top-k by default).
    """
    preds: list[str] = []
    ranked_mids_list: list[list[str]] = []
    system_prompts_list: list[str] = []
    ranker_responses_list: list[str] = []

    it = df.iterrows()
    if progress:
        it = tqdm(it, total=len(df), desc="Baseline: zero-shot ranking")
    for _, row in it:
        ranked_idx, raw_response = ranker.rank(
            row["prompt_rank"], row["candidate_titles"], system_prompt=system_prompt
        )
        topk_idx = ranked_idx[:k]
        mids = [str(row["candidate_mids"][i]) for i in topk_idx]

        ranked_mids_list.append(mids)
        system_prompts_list.append(system_prompt)
        ranker_responses_list.append(raw_response)
        preds.append(mids[0])  # top-1

    df["pred_mid"] = preds
    df["ranked_mids"] = ranked_mids_list
    df["system_prompt"] = system_prompts_list
    df["ranker_response"] = ranker_responses_list

    return df


def evaluate_zero_shot(
    df: pd.DataFrame,
    ranker: Ranker,
    k: int = 10,
    progress: bool = False,
) -> Dict[str, float]:
    if "ranked_mids" not in df:
        df = run_zero_shot_ranking(
            df, ranker, k=k, system_prompt=None, progress=progress
        )

    ranked_mids = df["ranked_mids"].tolist()
    targets = df["target_mid"].astype(str).tolist()
    return mean_recall_ndcg(ranked_mids, targets, k=k)


def run_up5_placeholder(*args, **kwargs):
    """
    UP5 to be implemented.
    """
    raise NotImplementedError()
