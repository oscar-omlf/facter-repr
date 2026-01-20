"""
Reusable prediction utilities for rank and open-generation modes.
Used by: calibration, baselines, and online monitoring.
"""

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from facter.data.prompts import PromptConfig, build_open_prompt
from facter.eval.catalogue_map import CatalogueMapper
from facter.fairness.scoring import item_text
from facter.models.generator import Generator
from facter.models.ranker import Ranker


@dataclass
class PredictionResult:
    """Holds prediction results for a single or batch of examples."""

    pred_mids: List[int]
    pred_texts: List[str]
    ranked_mids_list: List[List[int]]
    generated_titles_list: List[List[str]]
    valid_at_k_list: List[float]
    model_responses: List[str]


def predict_single_rank(
    row: Union[pd.Series, Dict],
    ranker: Ranker,
    item_db: Dict[int, Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> PredictionResult:
    # Handle both Series and Dict access
    candidates_titles = row["candidate_titles"]
    candidate_mids = row["candidate_mids"]
    prompt_rank = row["prompt_rank"]

    ranked_idx, raw_response = ranker.rank(
        prompt_rank, candidates_titles, system_prompt=system_prompt
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
    row: Union[pd.Series, Dict],
    generator: Generator,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    system_prompt: Optional[str] = None,
    catalogue_mapper: Optional[CatalogueMapper] = None,
    title_to_mid: Optional[Dict[str, int]] = None,
    min_sim: float = 0.65,
) -> PredictionResult:
    # Robust prompt retrieval
    if "prompt_open" in row:
        open_prompt = row["prompt_open"]
    elif "prompt_gen" in row:
        open_prompt = row["prompt_gen"]
    else:
        open_prompt = build_open_prompt(row, prompt_cfg)

    titles = generator.generate_topk(
        [open_prompt], [system_prompt], k=prompt_cfg.k_recs
    )[0]

    mids: List[int] = []
    valid_at_k = 0.0
    pred_text = ""

    if catalogue_mapper is not None:
        map_res = catalogue_mapper.map_list(
            titles, k=prompt_cfg.k_recs, min_sim=min_sim
        )
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
        valid_at_k = (
            float(min(len(mids), prompt_cfg.k_recs)) / float(prompt_cfg.k_recs)
            if prompt_cfg.k_recs
            else 0.0
        )
        pred_text = (
            item_text(int(mids[0]), item_db)
            if mids
            else (titles[0] if titles else "UNKNOWN_GENERATION")
        )
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
    data: Union[pd.DataFrame, DataLoader, Iterable[Dict]],
    ranker: Ranker,
    item_db: Dict[int, Dict[str, str]],
    system_prompt: Optional[str] = None,
    progress: bool = False,
) -> PredictionResult:
    """
    Batch prediction in rank mode. Optimized for Tensor inputs from DataLoader.
    """
    ranked_mids_list: List[List[int]] = []
    model_responses_list: List[str] = []
    pred_mids_list: List[int] = []
    pred_texts_list: List[str] = []

    iterator = data
    if isinstance(data, pd.DataFrame):
        iterator = data.to_dict("records")

    if progress:
        total = len(data) if hasattr(data, "__len__") else None
        iterator = tqdm(iterator, total=total, desc="Ranking")

    for batch in iterator:
        # Determine if 'batch' is a single item or a collated batch
        # If it's a dict and 'prompt_rank' is a list/tensor with len > 1 (or 0), it's a batch
        is_batched = (
            isinstance(batch, dict)
            and (
                isinstance(batch.get("prompt_rank"), list)
                or isinstance(batch.get("prompt_rank"), torch.Tensor)
            )
            and not isinstance(batch.get("prompt_rank"), str)
        )

        if is_batched:
            prompt_ranks = batch["prompt_rank"]
            candidate_titles_list = batch["candidate_titles"]

            # Optimization: If candidate_mids is a Tensor, convert to list of lists once
            # to avoid frequent CPU<->GPU sync during the loop
            candidate_mids_raw = batch["candidate_mids"]
            if isinstance(candidate_mids_raw, torch.Tensor):
                candidate_mids_list = candidate_mids_raw.detach().cpu().tolist()
            else:
                candidate_mids_list = candidate_mids_raw

            sys_prompts = [system_prompt] * len(prompt_ranks)

            if hasattr(ranker, "rank_batch"):
                ranked_idx_list, raw_texts = ranker.rank_batch(
                    prompt_ranks, candidate_titles_list, sys_prompts
                )
            else:
                ranked_idx_list = []
                raw_texts = []
                for p, c, s in zip(prompt_ranks, candidate_titles_list, sys_prompts):
                    r_idx, r_txt = ranker.rank(p, c, system_prompt=s)
                    ranked_idx_list.append(r_idx)
                    raw_texts.append(r_txt)

            # Process results
            for i, top_idx in enumerate(ranked_idx_list):
                cand_mids = candidate_mids_list[i]
                # Map indices to MIDs
                # cand_mids is now a list of ints/floats
                mids = [int(cand_mids[j]) for j in top_idx]

                ranked_mids_list.append(mids)
                model_responses_list.append(raw_texts[i])

                best_mid = mids[0] if mids else -1
                pred_mids_list.append(best_mid)
                pred_texts_list.append(
                    item_text(best_mid, item_db) if best_mid != -1 else ""
                )

        else:
            # Single item
            res = predict_single_rank(batch, ranker, item_db, system_prompt)
            ranked_mids_list.extend(res.ranked_mids_list)
            model_responses_list.extend(res.model_responses)
            pred_mids_list.extend(res.pred_mids)
            pred_texts_list.extend(res.pred_texts)

    n = len(pred_mids_list)
    return PredictionResult(
        pred_mids=pred_mids_list,
        pred_texts=pred_texts_list,
        ranked_mids_list=ranked_mids_list,
        generated_titles_list=[[] for _ in range(n)],
        valid_at_k_list=[0.0 for _ in range(n)],
        model_responses=model_responses_list,
    )


def predict_batch_open(
    data: Union[pd.DataFrame, DataLoader, Iterable[Dict]],
    generator: Generator,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    system_prompt: Optional[str] = None,
    catalogue_mapper: Optional[CatalogueMapper] = None,
    title_to_mid: Optional[Dict[str, int]] = None,
    min_sim: float = 0.65,
    progress: bool = False,
) -> PredictionResult:
    if title_to_mid is None and catalogue_mapper is None:
        title_to_mid = build_title_to_mid_dict(item_db)

    pred_mids_list: List[int] = []
    pred_texts_list: List[str] = []
    ranked_mids_list: List[List[int]] = []
    generated_titles_list: List[List[str]] = []
    valid_at_k_list: List[float] = []
    model_responses_list: List[str] = []

    iterator = data
    if isinstance(data, pd.DataFrame):
        iterator = data.to_dict("records")

    if progress:
        total = len(data) if hasattr(data, "__len__") else None
        iterator = tqdm(iterator, total=total, desc="Generating")

    for batch in iterator:
        # Check if batched via tensor or list presence
        is_batched = isinstance(batch, dict) and (
            isinstance(batch.get("target_mid"), list)
            or isinstance(batch.get("target_mid"), torch.Tensor)
        )

        prompts = []
        if is_batched:
            # Calculate batch size safely
            # Note: with optimized collate, target_mid is a Tensor
            t_mid = batch.get("target_mid")
            batch_size = len(t_mid)

            for i in range(batch_size):
                if "prompt_open" in batch and batch["prompt_open"][i]:
                    prompts.append(batch["prompt_open"][i])
                elif "prompt_gen" in batch and batch["prompt_gen"][i]:
                    prompts.append(batch["prompt_gen"][i])
                else:
                    # Reconstruct row for prompt building
                    # Convert tensor values to python scalars for string formatting
                    row = {}
                    for k, v in batch.items():
                        val = v[i]
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        row[k] = val
                    prompts.append(build_open_prompt(row, prompt_cfg))
        else:
            if "prompt_open" in batch:
                prompts.append(batch["prompt_open"])
            elif "prompt_gen" in batch:
                prompts.append(batch["prompt_gen"])
            else:
                prompts.append(build_open_prompt(batch, prompt_cfg))

        sys_prompts = [system_prompt] * len(prompts)
        gen_lists = generator.generate_topk(prompts, sys_prompts, k=prompt_cfg.k_recs)

        for titles in gen_lists:
            generated_titles_list.append(titles)
            model_responses_list.append(json.dumps(titles, ensure_ascii=False))

            mids: List[int] = []
            valid_at_k = 0.0
            pred_text = ""

            if catalogue_mapper is not None:
                map_res = catalogue_mapper.map_list(
                    titles, k=prompt_cfg.k_recs, min_sim=min_sim
                )
                mids = [
                    int(m) for m in getattr(map_res, "mapped_mids", []) if m is not None
                ]
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
                valid_at_k = (
                    float(min(len(mids), prompt_cfg.k_recs)) / float(prompt_cfg.k_recs)
                    if prompt_cfg.k_recs
                    else 0.0
                )
                pred_text = (
                    item_text(int(mids[0]), item_db)
                    if mids
                    else (titles[0] if titles else "UNKNOWN_GENERATION")
                )
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
    title_to_mid: Dict[str, int] = {}
    for mid, info in item_db.items():
        title_key = str(info.get("title", "")).strip().lower()
        title_to_mid[title_key] = int(mid)
    return title_to_mid
