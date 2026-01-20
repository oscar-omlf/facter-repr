from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from facter.data.prompts import PromptConfig
from facter.eval.prediction import predict_single_open, predict_single_rank
from facter.fairness.calibration import OfflineCalibrationResult
from facter.fairness.context_encoder import ContextEncoder
from facter.fairness.online import CalibrationArtifacts, OnlineScorer
from facter.fairness.threshold_update import update_threshold_theorem2
from facter.models.generator import Generator
from facter.models.ranker import Ranker
from facter.prompting.repair import PromptRepairEngine


@dataclass(frozen=True)
class OnlineMonitorConfig:
    max_iterations: int = 5
    gamma: float = 0.95

    # which protected attribute to use for repair filtering
    protected_key: str = "gender"


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
    Implements the online phase loop (Sec. 3.3).
    Optimized to pre-compute context embeddings (user history) which are static
    across repair iterations.
    """

    def __init__(
        self,
        ranker: Ranker,
        scorer: OnlineScorer,
        repair: PromptRepairEngine,
        cfg: OnlineMonitorConfig,
        context_encoder: ContextEncoder,
    ):
        self.ranker = ranker
        self.scorer = scorer
        self.repair = repair
        self.cfg = cfg
        self.context_encoder = context_encoder

    def run(
        self,
        test_df: pd.DataFrame,
        item_db: Dict[int, Dict[str, str]],
        cal_artifacts: Any,  # Can be OfflineCalibrationResult or CalibrationArtifacts
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
        # Prepare lightweight artifacts object if passed the full offline result
        if isinstance(cal_artifacts, OfflineCalibrationResult):
            cal_art = CalibrationArtifacts(
                cal_df=cal_artifacts.cal_df,
                cal_context_emb=cal_artifacts.cal_context_emb,
                cal_pred_emb=cal_artifacts.cal_pred_emb,
                cal_group_ids=cal_artifacts.cal_group_ids,
                group_code_map=cal_artifacts.group_code_map,
                q_alpha0=cal_artifacts.q_alpha0,
            )
        else:
            cal_art = cal_artifacts

        q = float(q_alpha0)
        logs: list[OnlineIterationLog] = []

        df = test_df.reset_index(drop=True).copy()

        # --- OPTIMIZATION: Pre-compute Context Embeddings & Group IDs ---
        # The user history does not change during prompt repair.
        # We encode all of them at once (Batch Mode) to save time.
        if progress:
            print(
                "[OnlineMonitor] Pre-computing user history embeddings...", flush=True
            )

        test_context_embs = self.context_encoder.encode_df(df)

        # Pre-compute Group IDs for fast lookup in scoring
        cols = group_cols if group_cols is not None else (self.cfg.protected_key,)

        # Optimize string concatenation using vectorized operations
        test_group_strs = df[cols[0]].astype(str)
        # Efficiently concat multiple columns
        for c in cols[1:]:
            test_group_strs = test_group_strs.str.cat(df[c].astype(str), sep="_")

        # map to int IDs, fill unknown with -1
        test_group_ids = (
            test_group_strs.map(cal_art.group_code_map).fillna(-1).astype(int).tolist()
        )

        # Convert DF to list of records for faster iteration than .iloc
        records = df.to_dict("records")
        # ----------------------------------------------------------------

        if predict_mode == "open":
            if generator is None or prompt_cfg is None:
                raise ValueError("open mode requires generator and prompt_cfg")

        # Outer loop: Iterations (Sequential)
        for t in range(1, self.cfg.max_iterations + 1):
            idx_iter = range(len(df))
            if progress:
                idx_iter = tqdm(
                    idx_iter,
                    total=len(df),
                    desc=f"Online iter {t}/{self.cfg.max_iterations}",
                )

            S_list: list[float] = []
            d_list: list[float] = []
            delta_list: list[float] = []
            viol = 0

            # Result collectors
            preds: list[int] = []
            ranked_mids_list: list[list[int]] = []
            generated_titles_list: list[list[str]] = []
            generated_mids_list: list[list[int]] = []
            valid_at_k_list: list[float] = []
            is_viol_list: list[bool] = []
            system_prompts_list: list[str] = []
            model_responses_list: list[str] = []

            # Inner loop: Examples (Sequential)
            for i in idx_iter:
                row = records[i]  # Fast access
                # row_s is a Series wrapper, needed if helper funcs strictly require Series
                # row_s = pd.Series(row)

                # 1. Build Prompt (Repair)
                attrs = {c: str(row[c]) for c in cols}

                system_prompt = self.repair.build_system_prompt(
                    attrs=attrs,
                    q_alpha=q,
                    iteration=t,
                    max_iterations=self.cfg.max_iterations,
                    predict_mode=predict_mode,
                )
                system_prompts_list.append(system_prompt)

                # 2. Predict (Single)
                if predict_mode == "rank":
                    pred_result = predict_single_rank(
                        row, self.ranker, item_db, system_prompt
                    )
                    pred_mid = pred_result.pred_mids[0]
                    pred_text = pred_result.pred_texts[0]
                    preds.append(pred_mid)
                    ranked_mids_list.append(pred_result.ranked_mids_list[0])
                    model_responses_list.append(pred_result.model_responses[0])
                    # placeholders
                    generated_titles_list.append([])
                    generated_mids_list.append([])
                    valid_at_k_list.append(0.0)

                else:  # open
                    pred_result = predict_single_open(
                        row,
                        generator,
                        item_db,
                        prompt_cfg,
                        system_prompt,
                        catalogue_mapper,
                        title_to_mid,
                        min_sim,
                    )
                    pred_mid = pred_result.pred_mids[0]
                    pred_text = pred_result.pred_texts[0]
                    preds.append(pred_mid)
                    generated_titles_list.append(pred_result.generated_titles_list[0])
                    generated_mids_list.append(pred_result.ranked_mids_list[0])
                    valid_at_k_list.append(pred_result.valid_at_k_list[0])
                    model_responses_list.append(pred_result.model_responses[0])
                    ranked_mids_list.append(pred_result.ranked_mids_list[0])

                # 3. Score (GPU Optimized with Pre-computed context)
                s, d, delta = self.scorer.score_one(
                    row=row,
                    pred_mid=(pred_mid if pred_mid != -1 else None),
                    pred_text=pred_text,
                    item_db=item_db,
                    cal=cal_art,
                    precomputed_context_emb=test_context_embs[i],  # <--- OPTIMIZATION
                    precomputed_group_id=test_group_ids[i],  # <--- OPTIMIZATION
                    target_mid=int(row["target_mid"]),
                )

                S_list.append(s)
                d_list.append(d)
                delta_list.append(delta)

                is_violation = s > q
                is_viol_list.append(is_violation)

                # 4. Update (Threshold & Buffer)
                if is_violation:
                    viol += 1
                    if pred_mid != -1:
                        self.repair.add_violation(attrs=attrs, pred_mid=pred_mid)
                    else:
                        self.repair.add_violation(attrs=attrs, pred_title=pred_text)

                    q = update_threshold_theorem2(q_t=q, s_t=s, gamma=self.cfg.gamma)

            # Log results for this iteration
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
