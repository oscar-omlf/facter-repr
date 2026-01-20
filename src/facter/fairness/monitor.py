from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from facter.data.prompts import PromptConfig
from facter.models.generator import Generator
from facter.models.ranker import Ranker
from facter.eval.prediction import predict_single_rank, predict_single_open
from facter.fairness.online import OnlineScorer, CalibrationArtifacts
from facter.fairness.threshold_update import update_threshold_theorem2
from facter.prompting.repair import PromptRepairEngine


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
        catalogue_mapper: Optional[Any] = None,
        group_cols: Optional[Tuple[str, ...]] = None,
        min_sim: float = 0.65,
    ) -> Tuple[pd.DataFrame, list[OnlineIterationLog]]:
        """
        Online monitoring loop.

        Rank-mode:
        - rank candidates, choose top-1 mid
        - log ranked mids

        Open-mode:
        - generate top-k titles (JSON list)
        - map titles->mids using catalogue_mapper (embedding NN + threshold) if provided
          (fallback: exact title_to_mid dict if provided)
        - score using pred_text (item_text(mapped_mid) if mapped else raw title)

        catalogue_mapper is expected to expose:
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
                cols = group_cols if group_cols is not None else (self.cfg.protected_key,)
                attrs = {c: str(row[c]) for c in cols}

                system_prompt = self.repair.build_system_prompt(
                    attrs=attrs,
                    q_alpha=q,
                    iteration=t,
                    max_iterations=self.cfg.max_iterations,
                )
                
                system_prompts_list.append(system_prompt)

                # Use unified prediction functions from prediction.py
                if predict_mode == "rank":
                    pred_result = predict_single_rank(row, self.ranker, item_db, system_prompt)
                    pred_mid = pred_result.pred_mids[0]
                    pred_text = pred_result.pred_texts[0]
                    preds.append(pred_mid)
                    ranked_mids_list.append(pred_result.ranked_mids_list[0])
                    model_responses_list.append(pred_result.model_responses[0])
                    generated_titles_list.append([])
                    generated_mids_list.append([])
                    valid_at_k_list.append(0.0)

                else:  # predict_mode == "open"
                    pred_result = predict_single_open(
                        row, generator, item_db, prompt_cfg, system_prompt,
                        catalogue_mapper, title_to_mid, min_sim
                    )
                    pred_mid = pred_result.pred_mids[0]
                    pred_text = pred_result.pred_texts[0]
                    preds.append(pred_mid)
                    generated_titles_list.append(pred_result.generated_titles_list[0])
                    generated_mids_list.append(pred_result.ranked_mids_list[0])  # mapped mids
                    valid_at_k_list.append(pred_result.valid_at_k_list[0])
                    model_responses_list.append(pred_result.model_responses[0])
                    # For compatibility with existing metric code
                    ranked_mids_list.append(pred_result.ranked_mids_list[0])

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
                        self.repair.add_violation(attrs=attrs, pred_mid=pred_mid)
                    else:
                        self.repair.add_violation(attrs=attrs, pred_title=pred_text)

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
