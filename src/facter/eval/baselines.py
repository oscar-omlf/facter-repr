from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import math
from collections import defaultdict, deque

import pandas as pd

from facter.models.ranker import Ranker
from facter.models.generator import Generator
from facter.eval.metrics import mean_recall_ndcg
from facter.eval.prediction import predict_batch_rank, predict_batch_open, build_title_to_mid_dict
from facter.fairness.context_encoder import ContextEncoder
from facter.eval.catalogue_map import CatalogueMapper
from facter.data.prompts import PromptConfig
from facter.fairness.scoring import NonconformityScorer


@dataclass(frozen=True)
class BaselineConfig:
    k: int = 10


def run_zero_shot(
    df: pd.DataFrame,
    ranker: Optional[Ranker] = None,
    generator: Optional[Generator] = None,
    scorer: Optional[NonconformityScorer] = None,
    context_encoder: Optional[ContextEncoder] = None,
    item_db: Optional[Dict[int, Dict[str, str]]] = None,
    neighbor_index: Optional[Any] = None,
    predict_mode: str = "rank",
    k: int = 10,
    system_prompt: str | None = None,
    progress: bool = False,
    catalogue_mapper: Optional[CatalogueMapper] = None,
    title_to_mid: Optional[Dict[str, int]] = None,
    min_sim: float = 0.65,
    threshold: float = 1,
    protected_cols: Optional[Sequence[str]] = None,
    buffer_size: int = 50,
) -> pd.DataFrame:
    """
    Unified zero-shot prediction: rank or open-generation based on predict_mode.

    IMPORTANT (Algorithm-1-aligned C):
      - Maintain a bounded FIFO violation buffer V (maxlen=buffer_size).
      - For each new point, filter V to the same protected group => C = |C|.
      - Use adj_thr = threshold + C/sqrt(n) (keeping your existing formula).
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

        if title_to_mid is None and catalogue_mapper is None:
            title_to_mid = build_title_to_mid_dict(item_db)

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
        df["pred_text"] = res.pred_texts
        df["ranked_mids"] = res.ranked_mids_list
        df["system_prompt"] = [system_prompt] * len(df)
        df["generator_response"] = res.model_responses
        df["valid_at_k"] = res.valid_at_k_list

    else:
        raise ValueError("predict_mode must be 'rank' or 'open'")

    if context_encoder is None:
        raise ValueError("context_encoder is required")
    if neighbor_index is None:
        raise ValueError("neighbor_index is required")

    context_emb = context_encoder.encode_df(df)
    neighbor_index.fit(df, context_emb)

    # Calculate S score for each prediction if scorer and neighbor_index are provided
    if scorer is not None and item_db is not None:
        S, d, delta, _pred_emb = scorer.compute(
            df,
            pred_mid_col="pred_mid",
            item_db=item_db,
            neighbor_index=neighbor_index
        )
        df["s_score"] = S
        df["d_score"] = d
        df["delta_score"] = delta

        # Algorithm-1 style: bounded FIFO buffer V of prior violations.
        protected_cols_tup = tuple(protected_cols or [])
        V = deque(maxlen=max(int(buffer_size), 0))  # stores group keys of past violations

        n = int(len(df))
        denom = math.sqrt(n) if n > 0 else 1.0

        adjusted_thresholds: List[float] = []
        is_violation_flags: List[bool] = []
        C_counts: List[int] = []
        V_lens: List[int] = []

        for _, row in df.iterrows():
            if protected_cols_tup:
                key = tuple(str(row[c]) for c in protected_cols_tup)
            else:
                key = ("__all__",)

            # C = |{ v in V : v.key == key }|
            C = sum(1 for kk in V if kk == key)

            adj_thr = float(threshold) + (float(C) / float(denom))
            is_violation = float(row["s_score"]) > adj_thr

            adjusted_thresholds.append(adj_thr)
            is_violation_flags.append(bool(is_violation))
            C_counts.append(int(C))
            V_lens.append(int(len(V)))

            if is_violation:
                V.append(key)

        df["C_count"] = C_counts               # debug: C per row
        df["V_len"] = V_lens                   # debug: buffer length per row
        df["adjusted_threshold"] = adjusted_thresholds
        df["is_violation"] = is_violation_flags

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
    catalogue_mapper: Optional[CatalogueMapper] = None,
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
