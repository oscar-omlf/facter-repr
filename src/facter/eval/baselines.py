"""Provide baseline evaluation utilities for the recommendation pipeline.

This module contains helper functions for running simple baselines (primarily
"zero-shot" ranking or open-generation) and evaluating their recommendation
metrics.

Some functions are kept for backward compatibility with older entry points.
"""

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
    """Store baseline configuration.

    Attributes:
        k (int): Number of recommendations/ranked items to consider.
    """
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
    """Run a unified zero-shot baseline in rank or open-generation mode.

    The function augments ``df`` in-place with prediction columns produced by
    either:
    - :func:`facter.eval.prediction.predict_batch_rank` (rank mode), or
    - :func:`facter.eval.prediction.predict_batch_open` (open mode).

    If a ``scorer`` is provided, the function also computes nonconformity and
    optionally flags violations using an adjusted threshold based on a bounded
    FIFO buffer over previous violations.

    The implemented adjustment is:

    $$\\text{adj\\_thr} = \\text{threshold} + \\frac{C}{\\sqrt{n}}$$

    where $n$ is ``len(df)`` and $C$ is the count of buffered prior violations
    whose protected-group key matches the current row.

    Args:
        df (pd.DataFrame): Input examples.
        ranker (Optional[Ranker]): Ranker used when ``predict_mode == 'rank'``.
        generator (Optional[Generator]): Generator used when
            ``predict_mode == 'open'``.
        scorer (Optional[NonconformityScorer]): If provided, computes
            nonconformity scores for predictions.
        context_encoder (Optional[ContextEncoder]): Context encoder used to
            embed rows prior to neighbor-index fitting.
        item_db (Optional[Dict[int, Dict[str, str]]]): Item metadata mapping.
        neighbor_index (Optional[Any]): Neighbor-index object expected to
            support ``fit(df, context_emb)``.
        predict_mode (str): Either ``'rank'`` or ``'open'``.
        k (int): Number of recommendations/ranked items to produce or consider.
        system_prompt (str | None): Optional system prompt passed through to
            the underlying model.
        progress (bool): Whether to show progress in underlying batched calls.
        catalogue_mapper (Optional[CatalogueMapper]): Optional mapper used in
            open-generation mode.
        title_to_mid (Optional[Dict[str, int]]): Optional fallback mapping used
            in open-generation mode.
        min_sim (float): Minimum similarity threshold passed through to
            catalogue mapping.
        threshold (float): Base threshold used for computing adjusted
            thresholds when ``scorer`` is provided.
        protected_cols (Optional[Sequence[str]]): Protected-attribute columns
            used to define the group key for the FIFO violation buffer.
        buffer_size (int): Maximum size of the FIFO violation buffer.

    Returns:
        pd.DataFrame: The input DataFrame with additional prediction and
        optional scoring/violation columns.

    Raises:
        ValueError: If required components for the selected ``predict_mode``
            are missing.
        ValueError: If ``context_encoder`` or ``neighbor_index`` is not
            provided.
        ValueError: If ``predict_mode`` is not ``'rank'`` or ``'open'``.
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
    """Run the deprecated rank-mode zero-shot baseline.

    This is a thin wrapper around :func:`run_zero_shot` with
    ``predict_mode='rank'``.

    Args:
        df (pd.DataFrame): Input examples.
        ranker (Ranker): Ranker used for prediction.
        k (int): Number of recommendations/ranked items to consider.
        system_prompt (str | None): Optional system prompt passed through to
            the ranker.
        progress (bool): Whether to show progress in the underlying call.

    Returns:
        pd.DataFrame: The input DataFrame with rank-mode prediction columns.
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
    """Run the deprecated open-generation zero-shot baseline.

    This is a thin wrapper around :func:`run_zero_shot` with
    ``predict_mode='open'``.

    Args:
        df (pd.DataFrame): Input examples.
        generator (Generator): Generator used for open-generation.
        item_db (Dict[int, Dict[str, str]]): Item metadata mapping.
        k (int): Number of generated titles to request.
        system_prompt (str | None): Optional system prompt passed through to
            the generator.
        catalogue_mapper (Optional[CatalogueMapper]): Optional embedding-based
            mapper used to map generated titles to catalogue items.
        title_to_mid (Optional[Dict[str, int]]): Optional fallback mapping used
            to map normalized titles to item ids.
        progress (bool): Whether to show progress in the underlying call.
        min_sim (float): Minimum similarity threshold passed through to
            catalogue mapping.

    Returns:
        pd.DataFrame: The input DataFrame with open-generation prediction
        columns.
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
    """Evaluate zero-shot predictions using recall and NDCG.

    The function expects the DataFrame to already contain a ``ranked_mids``
    column (as produced by :func:`run_zero_shot`). Targets are taken from the
    ``target_mid`` column.

    Args:
        df (pd.DataFrame): DataFrame containing predictions and targets.
        k (int): Cutoff used for Recall@k and NDCG@k.

    Returns:
        Dict[str, float]: Dictionary of metric values computed by
        :func:`facter.eval.metrics.mean_recall_ndcg`.

    Raises:
        ValueError: If ``ranked_mids`` is missing from ``df``.
    """
    if "ranked_mids" not in df:
        raise ValueError("DataFrame must contain 'ranked_mids' column; call run_zero_shot first")

    ranked_mids = df["ranked_mids"].tolist()
    targets = df["target_mid"].astype(int).tolist()
    return mean_recall_ndcg(ranked_mids, targets, k=k)


def run_up5_placeholder(*args, **kwargs):
    """Raise a placeholder error for an unimplemented UP5 baseline.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Raises:
        NotImplementedError: Always raised to indicate UP5 is not implemented.
    """
    raise NotImplementedError()
