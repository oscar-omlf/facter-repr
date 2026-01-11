from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm

from facter.models.ranker import Ranker
from facter.fairness.online import OnlineScorer, CalibrationArtifacts, OnlineScoringConfig
from facter.fairness.threshold_update import update_threshold_theorem2
from facter.prompting.repair import PromptRepairEngine, PromptRepairConfig


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
    ) -> Tuple[pd.DataFrame, list[OnlineIterationLog]]:
        q = float(q_alpha0)
        logs: list[OnlineIterationLog] = []

        df = test_df.reset_index(drop=True).copy()

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
            is_viol_list: list[bool] = []
            system_prompts_list: list[str] = []
            ranker_responses_list: list[str] = []

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

                # rank candidates
                ranked_idx, raw_response = self.ranker.rank(row["prompt_rank"], row["candidate_titles"], system_prompt=system_prompt)
                best_idx = ranked_idx[0]
                pred_mid = int(row["candidate_mids"][best_idx])
                preds.append(pred_mid)
                
                # save full ranked list of mids
                ranked_mids = [int(row["candidate_mids"][idx]) for idx in ranked_idx]
                ranked_mids_list.append(ranked_mids)
                ranker_responses_list.append(raw_response)

                s, d, delta = self.scorer.score_one(
                    row=row,
                    pred_mid=pred_mid,
                    item_db=item_db,
                    cal=cal_artifacts,
                    target_mid=int(row["target_mid"]),  # available in offline eval
                )

                S_list.append(s)
                d_list.append(d)
                delta_list.append(delta)

                is_violation = s > q
                is_viol_list.append(is_violation)

                if is_violation:
                    viol += 1
                    # store violation entry for repair (Eq.10 protocol)
                    self.repair.add_violation(protected_value=a_value, pred_mid=pred_mid)
                    # update threshold via Theorem 2 piecewise update
                    q = update_threshold_theorem2(q_t=q, s_t=s, gamma=self.cfg.gamma)

            df[f"pred_mid_iter{t}"] = preds
            df[f"ranked_mids_iter{t}"] = ranked_mids_list
            df[f"system_prompt_iter{t}"] = system_prompts_list
            df[f"ranker_response_iter{t}"] = ranker_responses_list
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