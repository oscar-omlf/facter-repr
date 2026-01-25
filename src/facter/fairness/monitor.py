from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

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
    protected_key: str = "gender"


@dataclass
class OnlineIterationLog:
    iteration: int
    q_alpha_start: float
    q_alpha_end: float
    q_trace: List[float]          # Q values: [start] + one entry per update
    q_update_steps: List[int]     # indices i where Q changed (row index in df)
    violations: int               # violations under dynamic Q (what you already counted)
    violations_at_Q0: int         # counterfactual: violations if threshold fixed at Q0 for this iteration
    mean_S: float
    mean_d: float
    mean_delta: float


class FACTEROnlineMonitor:
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
        predict_mode: str = "rank",
        generator: Optional[Generator] = None,
        prompt_cfg: Optional[PromptConfig] = None,
        title_to_mid: Optional[Dict[str, int]] = None,
        catalogue_mapper: Optional[Any] = None,
        group_cols: Optional[Tuple[str, ...]] = None,
        min_sim: float = 0.65,
    ) -> Tuple[pd.DataFrame, list[OnlineIterationLog]]:

        q = float(q_alpha0)       # dynamic Q (carries across iterations)
        q0 = float(q_alpha0)      # fixed baseline Q0 for counterfactual counting
        logs: list[OnlineIterationLog] = []

        df = test_df.reset_index(drop=True).copy()
        df["group_attrs"] = df[list(group_cols) if group_cols is not None else [self.cfg.protected_key]].astype(str).agg("_".join, axis=1)

        if predict_mode == "open":
            if generator is None or prompt_cfg is None:
                raise ValueError("open mode requires generator and prompt_cfg")

        for t in range(1, self.cfg.max_iterations + 1):
            q_start = float(q)
            q_trace: List[float] = [q_start]
            q_update_steps: List[int] = []

            idx_iter = range(len(df))
            if progress:
                idx_iter = tqdm(idx_iter, total=len(df), desc=f"Online iter {t}/{self.cfg.max_iterations}")

            S_list: list[float] = []
            d_list: list[float] = []
            delta_list: list[float] = []

            # logging Q per datapoint to “see the trajectory”
            q_before_list: list[float] = []
            q_after_list: list[float] = []

            viol_dynamic = 0
            viol_at_q0 = 0

            preds: list[int] = []
            ranked_mids_list: list[list[int]] = []
            generated_titles_list: list[list[str]] = []
            generated_mids_list: list[list[int]] = []
            valid_at_k_list: list[float] = []
            is_viol_list: list[bool] = []
            system_prompts_list: list[str] = []
            model_responses_list: list[str] = []

            for i in idx_iter:
                row = df.iloc[i]
                cols = group_cols if group_cols is not None else (self.cfg.protected_key,)
                attrs = {c: str(row[c]) for c in cols}

                # record q before scoring this point
                q_before = float(q)

                system_prompt = self.repair.build_system_prompt(
                    attrs=attrs,
                    q_alpha=q_before,
                    iteration=t,
                    max_iterations=self.cfg.max_iterations,
                )
                system_prompts_list.append(system_prompt)

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

                else:
                    pred_result = predict_single_open(
                        row, generator, item_db, prompt_cfg, system_prompt,
                        catalogue_mapper, title_to_mid, min_sim
                    )
                    pred_mid = pred_result.pred_mids[0]
                    pred_text = pred_result.pred_texts[0]
                    preds.append(pred_mid)
                    generated_titles_list.append(pred_result.generated_titles_list[0])
                    generated_mids_list.append(pred_result.ranked_mids_list[0])
                    valid_at_k_list.append(pred_result.valid_at_k_list[0])
                    model_responses_list.append(pred_result.model_responses[0])
                    ranked_mids_list.append(pred_result.ranked_mids_list[0])

                s, d, delta = self.scorer.score_one(
                    row=row,
                    pred_mid=(pred_mid if pred_mid != -1 else None),
                    pred_text=pred_text,
                    item_db=item_db,
                    cal=cal_artifacts,
                    target_mid=int(row["target_mid"]),
                )

                S_list.append(float(s))
                d_list.append(float(d))
                delta_list.append(float(delta))

                # counterfactual: violations if Q was fixed at Q0
                if float(s) > q0:
                    viol_at_q0 += 1

                is_violation = float(s) > q_before
                is_viol_list.append(bool(is_violation))

                if is_violation:
                    viol_dynamic += 1

                    if pred_mid != -1:
                        self.repair.add_violation(attrs=attrs, pred_mid=pred_mid)
                    else:
                        self.repair.add_violation(attrs=attrs, pred_title=pred_text)

                    # update Q and log it “whenever it changes”
                    q = update_threshold_theorem2(q_t=q_before, s_t=float(s), gamma=self.cfg.gamma)
                    q_trace.append(float(q))
                    q_update_steps.append(int(i))

                # record q after this point
                q_after = float(q)
                q_before_list.append(q_before)
                q_after_list.append(q_after)

            # persist outputs
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

            # Q trajectory columns
            df[f"q_before_iter{t}"] = q_before_list
            df[f"q_after_iter{t}"] = q_after_list

            q_end = float(q)

            logs.append(
                OnlineIterationLog(
                    iteration=t,
                    q_alpha_start=q_start,
                    q_alpha_end=q_end,
                    q_trace=q_trace,
                    q_update_steps=q_update_steps,
                    violations=int(viol_dynamic),
                    violations_at_Q0=int(viol_at_q0),
                    mean_S=float(np.mean(S_list)) if S_list else 0.0,
                    mean_d=float(np.mean(d_list)) if d_list else 0.0,
                    mean_delta=float(np.mean(delta_list)) if delta_list else 0.0,
                )
            )

        return df, logs
