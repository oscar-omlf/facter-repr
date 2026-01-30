"""Provide prediction helpers for ranking and open-generation evaluation.

This module contains small, reusable routines that run a model (ranker or text
generator) and post-process its output into a common ``PredictionResult``
structure.

These helpers are used by offline calibration, baselines, and online
monitoring.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import pandas as pd
from tqdm.auto import tqdm

from facter.models.ranker import Ranker
from facter.models.generator import Generator
from facter.eval.catalogue_map import CatalogueMapper
from facter.fairness.scoring import item_text
from facter.data.prompts import PromptConfig, build_open_prompt


@dataclass
class PredictionResult:
    """Hold model predictions and associated diagnostics.

    The same structure is used for both ranking mode (candidate re-ranking) and
    open-generation mode (generate titles, then map to catalogue item ids).

    Attributes:
        pred_mids (List[int]): Predicted item ids (one per example), where
            ``-1`` indicates that no valid item id could be produced.
        pred_texts (List[str]): Human-readable prediction text (one per
            example). In rank mode this is derived from ``item_db``; in open
            mode it may be a mapped title or a raw generated title.
        ranked_mids_list (List[List[int]]): Per-example list of item ids. In
            rank mode this is the full ranking over candidates; in open mode
            this is the list of mapped item ids in generation order.
        generated_titles_list (List[List[str]]): Per-example generated titles.
            Empty lists are used for rank mode.
        valid_at_k_list (List[float]): Per-example mapping success ratio used
            in open-generation mode. Defaults to ``0.0`` in rank mode.
        model_responses (List[str]): Raw model outputs serialized as strings.
    """
    pred_mids: List[int]
    pred_texts: List[str]
    ranked_mids_list: List[List[int]]  # For rank mode: all ranked mids; for open mode: mapped mids
    generated_titles_list: List[List[str]]  # Empty for rank mode; generated titles for open mode
    valid_at_k_list: List[float]  # For open mode: ratio of valid mappings
    model_responses: List[str]  # Raw responses (ranker or generator output)


def predict_single_rank(
    row: pd.Series,
    ranker: Ranker,
    item_db: Dict[int, Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> PredictionResult:
    """Predict one example in rank mode (re-rank provided candidates).

    The function calls ``ranker.rank`` on the row's rank prompt and candidate
    titles, then uses the top-ranked candidate as the prediction.

    Args:
        row (pd.Series): Input row expected to contain at least
            ``prompt_rank``, ``candidate_titles``, and ``candidate_mids``.
        ranker (Ranker): Ranker used to score and order candidates.
        item_db (Dict[int, Dict[str, str]]): Item metadata mapping used to
            format ``pred_text`` via ``item_text``.
        system_prompt (Optional[str]): Optional system prompt/context passed to
            the ranker.

    Returns:
        PredictionResult: Result object containing a single prediction.
    """
    candidates_titles: List[str] = row["candidate_titles"]
    candidate_mids: List[int] = row["candidate_mids"]

    ranked_idx, raw_response = ranker.rank(
        row["prompt_rank"], candidates_titles, system_prompt=system_prompt
    )

    best_idx = ranked_idx[0]
    pred_mid = int(candidate_mids[best_idx])
    pred_text = item_text(pred_mid, item_db)

    ranked_mids = [int(candidate_mids[idx]) for idx in ranked_idx]

    return PredictionResult(
        pred_mids=[pred_mid],
        pred_texts=[pred_text],
        ranked_mids_list=[ranked_mids],
        generated_titles_list=[[]],
        valid_at_k_list=[0.0],
        model_responses=[raw_response],
    )


def predict_single_open(
    row: pd.Series,
    generator: Generator,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    system_prompt: Optional[str] = None,
    catalogue_mapper: Optional[CatalogueMapper] = None,
    title_to_mid: Optional[Dict[str, int]] = None,
    min_sim: float = 0.65,
) -> PredictionResult:
    """Predict one example in open-generation mode (generate titles, then map).

    The function generates ``prompt_cfg.k_recs`` titles with
    ``generator.generate_topk`` and attempts to map generated titles to item ids
    using either a ``CatalogueMapper`` (preferred) or a fallback
    ``title_to_mid`` dictionary.

    Args:
        row (pd.Series): Input row. If it contains ``prompt_open`` (or
            ``prompt_gen``), that string is used directly; otherwise an open
            prompt is built from ``row.to_dict()`` and ``prompt_cfg``.
        generator (Generator): Text generator used to produce recommendation
            titles.
        item_db (Dict[int, Dict[str, str]]): Item metadata mapping used by
            ``item_text`` when a mapped ``mid`` is available.
        prompt_cfg (PromptConfig): Prompt configuration, including ``k_recs``.
        system_prompt (Optional[str]): Optional system prompt/context passed to
            the generator.
        catalogue_mapper (Optional[CatalogueMapper]): Mapper used to map
            generated titles to catalogue items via embedding similarity.
        title_to_mid (Optional[Dict[str, int]]): Fallback mapping from
            normalized title strings to item ids.
        min_sim (float): Similarity threshold passed through to
            ``CatalogueMapper.map_list`` when provided.

    Returns:
        PredictionResult: Result object containing a single prediction.

    Notes:
        If neither ``catalogue_mapper`` nor ``title_to_mid`` is provided, the
        returned ``pred_mid`` will be ``-1`` and ``pred_text`` will be taken
        directly from the generated titles (or ``"UNKNOWN_GENERATION"`` if no
        titles are produced).
    """
    open_prompt = row.get("prompt_open", row.get("prompt_gen", None))
    if open_prompt is None:
        # Fall back to building if needed
        open_prompt = build_open_prompt(row.to_dict(), prompt_cfg)

    titles = generator.generate_topk(
        [open_prompt], [system_prompt], k=prompt_cfg.k_recs
    )[0]

    mids: List[int] = []
    valid_at_k = 0.0
    pred_text = ""

    # Preferred: embedding-based catalog mapping (authors' approach)
    if catalogue_mapper is not None:
        map_res = catalogue_mapper.map_list(titles, k=prompt_cfg.k_recs, min_sim=min_sim)
        # keep valid mapped mids in rank order
        mids = [int(m) for m in getattr(map_res, "mapped_mids", []) if m is not None]
        valid_at_k = float(getattr(map_res, "valid_at_k", 0.0))

        # use canonical mapped title for pred_text if available
        mapped_titles = getattr(map_res, "mapped_titles", [])
        if mapped_titles and mapped_titles[0]:
            pred_text = str(mapped_titles[0])
        else:
            pred_text = titles[0] if titles else "UNKNOWN_GENERATION"

    # Fallback: exact normalized dict mapping (not paper-aligned, but keeps pipeline usable)
    elif title_to_mid is not None:
        for tt in titles:
            key = str(tt).strip().lower()
            mid = title_to_mid.get(key, -1)
            if mid != -1 and int(mid) not in mids:
                mids.append(int(mid))
        # crude "valid@k" proxy under dict mapping
        valid_at_k = float(min(len(mids), prompt_cfg.k_recs)) / float(prompt_cfg.k_recs) if prompt_cfg.k_recs else 0.0
        pred_text = item_text(int(mids[0]), item_db) if mids else (titles[0] if titles else "UNKNOWN_GENERATION")

    else:
        pred_text = titles[0] if titles else "UNKNOWN_GENERATION"

    pred_mid = mids[0] if mids else -1

    return PredictionResult(
        pred_mids=[int(pred_mid)],
        pred_texts=[pred_text],
        ranked_mids_list=[mids],
        generated_titles_list=[titles],
        valid_at_k_list=[valid_at_k],
        model_responses=[json.dumps(titles, ensure_ascii=False)],
    )


def predict_batch_rank(
    df: pd.DataFrame,
    ranker: Ranker,
    item_db: Dict[int, Dict[str, str]],
    system_prompt: Optional[str] = None,
    progress: bool = False,
) -> PredictionResult:
    """Predict a batch in rank mode.

    If the ``ranker`` exposes a ``rank_batch`` attribute, that path is used;
    otherwise the function falls back to iterating rows and calling
    ``ranker.rank``.

    Args:
        df (pd.DataFrame): Batch input expected to contain ``prompt_rank``,
            ``candidate_titles``, and ``candidate_mids``.
        ranker (Ranker): Ranker used to score and order candidates.
        item_db (Dict[int, Dict[str, str]]): Item metadata mapping used to
            format ``pred_text`` via ``item_text``.
        system_prompt (Optional[str]): Optional system prompt/context passed to
            the ranker.
        progress (bool): Whether to show a tqdm progress bar in the per-row
            fallback path.

    Returns:
        PredictionResult: Result object containing one prediction per row.
    """
    n = len(df)

    # Prepare inputs
    prompt_ranks: List[str] = []
    candidate_titles_list: List[List[str]] = []
    system_prompts: List[Optional[str]] = []
    it = df.iterrows()
    for _, row in it:
        prompt_ranks.append(row["prompt_rank"])
        candidate_titles_list.append(list(row["candidate_titles"]))
        system_prompts.append(system_prompt)

    ranked_mids_list: List[List[int]] = []
    model_responses_list: List[str] = []

    # Use batched path if available
    if hasattr(ranker, "rank_batch"):
        outputs = ranker.rank_batch(prompt_ranks, candidate_titles_list, system_prompts, progress)
        for i in range(n):
            top_idx = outputs[i][0]
            mids = [int(df.iloc[i]["candidate_mids"][j]) for j in top_idx]
            ranked_mids_list.append(mids)
        model_responses_list = [outputs[i][1] for i in range(n)]
    else:
        # Fallback to per-row
        it2 = df.iterrows()
        if progress:
            it2 = tqdm(it2, total=n, desc="Ranking (per-row)")
        for _, row in it2:
            ranked_idx, raw_response = ranker.rank(row["prompt_rank"], row["candidate_titles"], system_prompt=system_prompt)
            mids = [int(row["candidate_mids"][j]) for j in ranked_idx]
            ranked_mids_list.append(mids)
            model_responses_list.append(raw_response)

    pred_mids_list = [m[0] if m else -1 for m in ranked_mids_list]
    pred_texts_list = [item_text(mid, item_db) if mid != -1 else "" for mid in pred_mids_list]

    return PredictionResult(
        pred_mids=pred_mids_list,
        pred_texts=pred_texts_list,
        ranked_mids_list=ranked_mids_list,
        generated_titles_list=[[] for _ in range(n)],
        valid_at_k_list=[0.0 for _ in range(n)],
        model_responses=model_responses_list,
    )


def predict_batch_open(
    df: pd.DataFrame,
    generator: Generator,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    system_prompt: Optional[str] = None,
    catalogue_mapper: Optional[CatalogueMapper] = None,
    title_to_mid: Optional[Dict[str, int]] = None,
    min_sim: float = 0.65,
    progress: bool = False,
) -> PredictionResult:
    """Predict a batch in open-generation mode.

    The function builds one open prompt per row, performs a single batched
    ``generator.generate_topk`` call, and then maps each generated list of
    titles to item ids.

    If neither ``catalogue_mapper`` nor ``title_to_mid`` is provided, a fallback
    ``title_to_mid`` mapping is built from ``item_db``.

    Args:
        df (pd.DataFrame): Batch input.
        generator (Generator): Text generator used to produce recommendation
            titles.
        item_db (Dict[int, Dict[str, str]]): Item metadata mapping.
        prompt_cfg (PromptConfig): Prompt configuration, including ``k_recs``.
        system_prompt (Optional[str]): Optional system prompt/context passed to
            the generator.
        catalogue_mapper (Optional[CatalogueMapper]): Mapper used to map
            generated titles to catalogue items.
        title_to_mid (Optional[Dict[str, int]]): Fallback mapping from
            normalized title strings to item ids.
        min_sim (float): Similarity threshold passed through to
            ``CatalogueMapper.map_list`` when provided.
        progress (bool): Whether to display a progress bar during batched
            generation.

    Returns:
        PredictionResult: Result object containing one prediction per row.
    """
    n = len(df)

    # Build prompts for all rows first
    prompts: List[str] = []
    system_prompts: List[Optional[str]] = []
    it = df.iterrows()
    for _, row in it:
        open_prompt = row.get("prompt_open", row.get("prompt_gen", None))
        if open_prompt is None:
            open_prompt = build_open_prompt(row.to_dict(), prompt_cfg)
        prompts.append(open_prompt)
        system_prompts.append(system_prompt)

    # Single batched generation call
    gen_lists = generator.generate_topk(prompts, system_prompts, k=prompt_cfg.k_recs, progress=progress)

    pred_mids_list: List[int] = []
    pred_texts_list: List[str] = []
    ranked_mids_list: List[List[int]] = []
    generated_titles_list: List[List[str]] = []
    valid_at_k_list: List[float] = []
    model_responses_list: List[str] = []

    # Prepare fallback mapping if needed
    if title_to_mid is None and catalogue_mapper is None:
        title_to_mid = build_title_to_mid_dict(item_db)

    for i in range(n):
        titles = gen_lists[i]
        generated_titles_list.append(titles)
        model_responses_list.append(json.dumps(titles, ensure_ascii=False))

        mids: List[int] = []
        valid_at_k = 0.0
        pred_text = ""

        if catalogue_mapper is not None:
            map_res = catalogue_mapper.map_list(titles, k=prompt_cfg.k_recs, min_sim=min_sim)
            mids = [int(m) for m in getattr(map_res, "mapped_mids", []) if m is not None]
            valid_at_k = float(getattr(map_res, "valid_at_k", 0.0))
            mapped_titles = getattr(map_res, "mapped_titles", [])
            if mapped_titles and mapped_titles[0]:
                pred_text = str(mapped_titles[0])
            else:
                pred_text = titles[0] if titles else "UNKNOWN_GENERATION"
        elif title_to_mid is not None:
            for tt in titles:
                key = str(tt).strip().lower()
                mid = title_to_mid.get(key, -1)
                if mid != -1 and int(mid) not in mids:
                    mids.append(int(mid))
            valid_at_k = float(min(len(mids), prompt_cfg.k_recs)) / float(prompt_cfg.k_recs) if prompt_cfg.k_recs else 0.0
            pred_text = item_text(int(mids[0]), item_db) if mids else (titles[0] if titles else "UNKNOWN_GENERATION")
        else:
            pred_text = titles[0] if titles else "UNKNOWN_GENERATION"

        pred_mid = mids[0] if mids else -1

        pred_mids_list.append(int(pred_mid))
        pred_texts_list.append(pred_text)
        ranked_mids_list.append(mids)
        valid_at_k_list.append(valid_at_k)

    return PredictionResult(
        pred_mids=pred_mids_list,
        pred_texts=pred_texts_list,
        ranked_mids_list=ranked_mids_list,
        generated_titles_list=generated_titles_list,
        valid_at_k_list=valid_at_k_list,
        model_responses=model_responses_list,
    )


def build_title_to_mid_dict(item_db: Dict[int, Dict[str, str]]) -> Dict[str, int]:
    """Build a normalized title-to-id lookup from an item database.

    The mapping uses ``str(title).strip().lower()`` as the normalization.

    Args:
        item_db (Dict[int, Dict[str, str]]): Item metadata mapping that may
            contain a ``"title"`` field per item.

    Returns:
        Dict[str, int]: Mapping from normalized titles to item ids.
    """
    title_to_mid: Dict[str, int] = {}
    for mid, info in item_db.items():
        title_key = str(info.get("title", "")).strip().lower()
        title_to_mid[title_key] = int(mid)
    return title_to_mid
