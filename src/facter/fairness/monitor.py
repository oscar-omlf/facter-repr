from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm

from facter.data.prompts import PromptConfig, build_open_prompt
from facter.models.generator import Generator
from facter.models.ranker import Ranker
from facter.fairness.online import OnlineScorer, CalibrationArtifacts
from facter.fairness.threshold_update import update_threshold_theorem2
from facter.prompting.repair import PromptRepairEngine
from facter.fairness.scoring import item_text


@dataclass(frozen=True)
class OnlineMonitorConfig:
    max_iterations: int = 5
    gamma: float = 0.95
    protected_key: str = "gender"  # which protected attribute to use for repair filtering


@dataclass
class OnlineIterationLog:
    iteration: int
    q_alpha: float
    violations: int
    mean_S: float
    mean_d: float
    mean_delta: float


class FACTEROnlineMonitor:
    """
    Implements the online phase loop (Sec. 3.3):
      - score each example
      - if violation: add to buffer + inject rules + update threshold (Theorem 2)
    """

    def __init__(
        self,
        ranker: Ranker,
        scorer: OnlineScorer,
        repair: PromptRepairEngine,
        cfg: OnlineMonitorConfig,
    ):
        self.ranker = ranker
        self.scorer = scorer
        self.repair = repair
        self.cfg = cfg

    def run(
        self,
        test_df: pd.DataFrame,
        item_db: Dict[int, Dict[str, str]],
        cal_artifacts: CalibrationArtifacts,
        q_alpha0: float,
        progress: bool = False,
        predict_mode: str = "rank",  # "rank" | "open"
        generator: Optional[Generator] = None,
        prompt_cfg: Optional[PromptConfig] = None,
        title_to_mid: Optional[Dict[str, int]] = None,
        catalog_mapper: Optional[Any] = None,
        min_sim: float = 0.65,
    ) -> Tuple[pd.DataFrame, list[OnlineIterationLog]]:
        """
        Online monitoring loop.

        Rank-mode:
        - rank candidates, choose top-1 mid
        - log ranked mids

        Open-mode:
        - generate top-k titles (JSON list)
        - map titles->mids using catalog_mapper (embedding NN + threshold) if provided
          (fallback: exact title_to_mid dict if provided)
        - score using pred_text (item_text(mapped_mid) if mapped else raw title)

        catalog_mapper is expected to expose:
          map_list(titles: List[str], k: int, min_sim: float) -> object with fields:
            - mapped_mids: List[Optional[int]]
            - mapped_titles: List[str]
            - valid_at_k: float
        """
        q = float(q_alpha0)
        logs: list[OnlineIterationLog] = []

        df = test_df.reset_index(drop=True).copy()

        if predict_mode == "open":
            if generator is None or prompt_cfg is None:
                raise ValueError("open mode requires generator and prompt_cfg")

        for t in range(1, self.cfg.max_iterations + 1):
            idx_iter = range(len(df))
            if progress:
                idx_iter = tqdm(idx_iter, total=len(df), desc=f"Online iter {t}/{self.cfg.max_iterations}")

            S_list: list[float] = []
            d_list: list[float] = []
            delta_list: list[float] = []
            viol = 0

            preds: list[int] = []
            ranked_mids_list: list[list[int]] = []
            generated_titles_list: list[list[str]] = []
            generated_mids_list: list[list[int]] = []

            # Open-mode diagnostics (optional but useful for matching their evaluation regime)
            valid_at_k_list: list[float] = []

            is_viol_list: list[bool] = []
            system_prompts_list: list[str] = []
            model_responses_list: list[str] = []

            for i in idx_iter:
                row = df.iloc[i]
                a_value = str(row[self.cfg.protected_key])

                system_prompt = self.repair.build_system_prompt(
                    a_value=a_value,
                    q_alpha=q,
                    iteration=t,
                    max_iterations=self.cfg.max_iterations,
                )
                system_prompts_list.append(system_prompt)

                pred_mid: int = -1
                pred_text: str = ""

                if predict_mode == "rank":
                    # NOTE: this assumes your Ranker.rank returns (ranked_idx, raw_response).
                    ranked_idx, raw_response = self.ranker.rank(
                        row["prompt_rank"], row["candidate_titles"], system_prompt=system_prompt
                    )
                    best_idx = ranked_idx[0]
                    pred_mid = int(row["candidate_mids"][best_idx])
                    preds.append(pred_mid)

                    ranked_mids = [int(row["candidate_mids"][idx]) for idx in ranked_idx]
                    ranked_mids_list.append(ranked_mids)

                    model_responses_list.append(raw_response)
                    generated_titles_list.append([])
                    generated_mids_list.append([])
                    valid_at_k_list.append(0.0)

                    pred_text = item_text(pred_mid, item_db)

                else:
                    # open generation (one-by-one; correct with streaming updates)
                    open_prompt = row.get("prompt_open", row.get("prompt_gen", None))
                    if open_prompt is None:
                        # fall back to building if needed
                        open_prompt = build_open_prompt(row.to_dict(), prompt_cfg)

                    # Call generator; prefer (prompts, system_prompt, k), fallback to (prompts, [system_prompt], k)
                    try:
                        titles = generator.generate_topk([open_prompt], system_prompt, k=prompt_cfg.k_recs)[0]
                    except TypeError:
                        titles = generator.generate_topk([open_prompt], [system_prompt], k=prompt_cfg.k_recs)[0]

                    generated_titles_list.append(titles)
                    model_responses_list.append(json.dumps(titles, ensure_ascii=False))

                    mids: list[int] = []
                    valid_at_k = 0.0

                    # Preferred: embedding-based catalog mapping (authors' approach)
                    if catalog_mapper is not None:
                        map_res = catalog_mapper.map_list(titles, k=prompt_cfg.k_recs, min_sim=min_sim)
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

                    generated_mids_list.append(mids)
                    valid_at_k_list.append(valid_at_k)

                    pred_mid = mids[0] if mids else -1
                    preds.append(int(pred_mid))

                    # For compatibility with existing metric code, we still populate ranked_mids_list
                    ranked_mids_list.append(mids)

                    if pred_mid != -1 and not pred_text:
                        pred_text = item_text(int(pred_mid), item_db)

                s, d, delta = self.scorer.score_one(
                    row=row,
                    pred_mid=(pred_mid if pred_mid != -1 else None),
                    pred_text=pred_text,
                    item_db=item_db,
                    cal=cal_artifacts,
                    target_mid=int(row["target_mid"]),
                )

                S_list.append(s)
                d_list.append(d)
                delta_list.append(delta)

                is_violation = s > q
                is_viol_list.append(is_violation)

                if is_violation:
                    viol += 1
                    if pred_mid != -1:
                        self.repair.add_violation(protected_value=a_value, pred_mid=pred_mid)
                    else:
                        self.repair.add_violation(protected_value=a_value, pred_title=pred_text)

                    q = update_threshold_theorem2(q_t=q, s_t=s, gamma=self.cfg.gamma)

            df[f"pred_mid_iter{t}"] = preds
            df[f"ranked_mids_iter{t}"] = ranked_mids_list
            df[f"generated_titles_iter{t}"] = generated_titles_list
            df[f"generated_mids_iter{t}"] = generated_mids_list
            df[f"valid_at_k_iter{t}"] = valid_at_k_list
            df[f"system_prompt_iter{t}"] = system_prompts_list
            df[f"model_response_iter{t}"] = model_responses_list

            df[f"S_iter{t}"] = S_list
            df[f"d_iter{t}"] = d_list
            df[f"delta_iter{t}"] = delta_list
            df[f"is_violation_iter{t}"] = is_viol_list

            logs.append(
                OnlineIterationLog(
                    iteration=t,
                    q_alpha=q,
                    violations=viol,
                    mean_S=float(np.mean(S_list)) if S_list else 0.0,
                    mean_d=float(np.mean(d_list)) if d_list else 0.0,
                    mean_delta=float(np.mean(delta_list)) if delta_list else 0.0,
                )
            )

        return df, logs
