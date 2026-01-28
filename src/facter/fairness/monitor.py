"""Implement the FACTER online monitoring loop.

This module implements the repository's online phase: for each test example it
builds an iteration-specific system prompt, produces a recommendation, scores
it, and updates state when a score exceeds the current threshold.

The implementation records per-iteration diagnostics (scores, thresholds, and
violation counts) and returns an augmented ``DataFrame`` plus a list of
per-iteration logs.

Paper context:
    The overall control-flow (monitoring + violation-triggered prompt repair +
    threshold adaptation) matches the high-level online loop described in the
    paper. (Paper: Sec. 3.3 / Alg. 1)

TODO(doc): Clarify the exact correspondence between the paper's update rule and
the repository's threshold update helper; treat the implementation as
canonical.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List
from collections import deque
import math

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
    """Configure the online monitoring loop.

    ``gamma`` parameterizes the threshold adaptation used when violations are
    detected.

    Paper context:
        The paper describes an online phase where violations drive updates.
        (Paper: Sec. 3.3)

    Attributes:
        max_iterations (int): Maximum number of online repair/monitoring rounds.
        gamma (float): Threshold update smoothing/decay factor passed to
            ``update_threshold_theorem2``.
        protected_key (str): Fallback protected attribute column name to use
            when ``group_cols`` is not provided to ``FACTEROnlineMonitor.run``.
    """

    max_iterations: int = 5
    gamma: float = 0.95
    protected_key: str = "gender"


@dataclass
class OnlineIterationLog:
    """Store summary statistics and diagnostics for one online iteration.

    This is produced once per iteration of ``FACTEROnlineMonitor.run``.

    Attributes:
        iteration (int): 1-indexed iteration number.
        q_alpha_start (float): Threshold at the start of the iteration.
        q_alpha_end (float): Threshold at the end of the iteration.
        q_trace (List[float]): Threshold values after each in-iteration update.
        q_update_steps (List[int]): Row indices in the iteration where the
            threshold was updated.
        violations (int): Count of raw violations under the dynamic threshold
            (``S > Q_t`` at the time each sample is processed).
        violations_at_Q0 (int): Count of raw violations under the fixed initial
            threshold ``Q0``.
        violations_corr_dynamic (int): Count of "corrected" violations under the
            dynamic threshold plus a FIFO-based correction term.
        violations_corr_at_Q0 (int): Count of "corrected" violations under the
            fixed threshold plus a FIFO-based correction term.
        mean_S (float): Mean of per-row fairness scores ``S`` for the iteration.
        mean_d (float): Mean of per-row predictive error term ``d`` for the
            iteration.
        mean_delta (float): Mean of per-row fairness penalty term ``delta`` for
            the iteration.
    """

    iteration: int
    q_alpha_start: float
    q_alpha_end: float
    q_trace: List[float]
    q_update_steps: List[int]
    violations: int                    # raw: S > Q_t (dynamic)
    violations_at_Q0: int              # raw: S > Q0 (fixed)
    violations_corr_dynamic: int       # corrected: S > Q_t + C/sqrt(n)
    violations_corr_at_Q0: int         # corrected: S > Q0 + C/sqrt(n)
    mean_S: float
    mean_d: float
    mean_delta: float


class FACTEROnlineMonitor:
    """Run the online monitoring and prompt-repair loop.

    The monitor repeatedly:
    1) builds an iteration-specific system prompt via ``PromptRepairEngine``,
    2) obtains a recommendation via a ranker or generator,
    3) computes a fairness-aware score using ``OnlineScorer``,
    4) records violations and updates the threshold when violations occur.

    Paper context:
        This follows the paper's online loop at a high level: violation
        detection triggers prompt updates and threshold adaptation.
        (Paper: Sec. 3.3 / Alg. 1)

    Note:
        This class does not itself implement the scoring function; it delegates
        to ``OnlineScorer`` and uses ``update_threshold_theorem2`` to update the
        threshold.
    """

    def __init__(
        self,
        ranker: Ranker,
        scorer: OnlineScorer,
        repair: PromptRepairEngine,
        cfg: OnlineMonitorConfig,
    ):
        """Initialize the online monitor.

        Args:
            ranker (Ranker): Ranker used when ``predict_mode='rank'``.
            scorer (OnlineScorer): Scorer used to compute ``(S, d, delta)`` for a
                single row.
            repair (PromptRepairEngine): Prompt repair engine used to build
                system prompts and store violations.
            cfg (OnlineMonitorConfig): Online monitoring configuration.
        """
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
        """Execute the iterative online monitoring loop.

        This is the online phase that (a) generates outputs under a
        fairness-instruction prompt, (b) computes a fairness-aware score, and
        (c) updates the prompt/threshold on violations.

        Paper context:
            The online loop structure and "violation -> update" pattern are
            described in the paper. (Paper: Sec. 3.3 / Alg. 1)

        The returned DataFrame is a copy of ``test_df`` augmented with per-row
        predictions, prompts, scores, thresholds, and violation indicators for
        each iteration.

        Args:
            test_df (pd.DataFrame): Test split input.
            item_db (Dict[int, Dict[str, str]]): Item metadata mapping passed to
                prediction and scoring utilities.
            cal_artifacts (CalibrationArtifacts): Offline calibration artifacts
                used by ``OnlineScorer``.
            q_alpha0 (float): Initial threshold used to initialize the dynamic
                threshold ``q`` and the fixed baseline ``q0``.
            progress (bool): Whether to display a tqdm progress bar.
            predict_mode (str): Prediction mode. When ``'rank'``, uses
                ``predict_single_rank``; otherwise uses ``predict_single_open``.
            generator (Optional[Generator]): Generator required for
                ``predict_mode='open'``.
            prompt_cfg (Optional[PromptConfig]): Prompt configuration required
                for ``predict_mode='open'``.
            title_to_mid (Optional[Dict[str, int]]): Title-to-item-id map used in
                open generation.
            catalogue_mapper (Optional[Any]): Catalogue mapper passed through to
                ``predict_single_open``.
            group_cols (Optional[Tuple[str, ...]]): Protected attribute columns
                used to form group keys. If not provided, falls back to
                ``cfg.protected_key``.
            min_sim (float): Similarity threshold passed through to
                ``predict_single_open``.

        Returns:
            Tuple[pd.DataFrame, list[OnlineIterationLog]]: A tuple of
            ``(df, logs)`` where ``df`` is the augmented DataFrame and ``logs``
            contains per-iteration summary diagnostics.

        Raises:
            ValueError: If ``predict_mode`` is ``'open'`` and ``generator`` or
                ``prompt_cfg`` is not provided.
        """

        q = float(q_alpha0)   # dynamic Q carried across iterations
        q0 = float(q_alpha0)  # fixed baseline Q0 for counterfactual counting
        logs: list[OnlineIterationLog] = []

        df = test_df.reset_index(drop=True).copy()
        cols = tuple(group_cols) if group_cols is not None else (self.cfg.protected_key,)
        df["group_attrs"] = df[list(cols)].astype(str).agg("_".join, axis=1)

        if predict_mode == "open" and (generator is None or prompt_cfg is None):
            raise ValueError("open mode requires generator and prompt_cfg")

        # For corrected-violation metrics: FIFO buffers of *prior corrected violations* (baseline-style)
        buf_len = max(int(getattr(self.repair.cfg, "buffer_size", 50)), 0)
        Vcorr_dyn = deque(maxlen=buf_len)
        Vcorr_q0 = deque(maxlen=buf_len)

        n = int(len(df))
        denom = math.sqrt(n) if n > 0 else 1.0

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

            q_before_list: list[float] = []
            q_after_list: list[float] = []

            # corrected diagnostics per-row (optional but useful)
            C_corr_dyn_list: list[int] = []
            thr_corr_dyn_list: list[float] = []
            viol_corr_dyn_list: list[bool] = []

            C_corr_q0_list: list[int] = []
            thr_corr_q0_list: list[float] = []
            viol_corr_q0_list: list[bool] = []

            viol_dynamic = 0
            viol_at_q0 = 0
            viol_corr_dynamic = 0
            viol_corr_at_q0 = 0

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
                attrs = {c: str(row[c]) for c in cols}
                key = tuple(attrs[c] for c in cols)

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

                s = float(s)
                S_list.append(s)
                d_list.append(float(d))
                delta_list.append(float(delta))

                # Raw counters
                if s > q0:
                    viol_at_q0 += 1

                is_violation_raw = s > q_before
                is_viol_list.append(bool(is_violation_raw))
                if is_violation_raw:
                    viol_dynamic += 1

                    if pred_mid != -1:
                        self.repair.add_violation(attrs=attrs, pred_mid=pred_mid)
                    else:
                        self.repair.add_violation(attrs=attrs, pred_title=pred_text)

                    q = update_threshold_theorem2(q_t=q_before, s_t=s, gamma=self.cfg.gamma)
                    q_trace.append(float(q))
                    q_update_steps.append(int(i))

                # Corrected counters (baseline-style FIFO over prior corrected violations)
                C_dyn = sum(1 for kk in Vcorr_dyn if kk == key)
                thr_dyn = q_before + (float(C_dyn) / float(denom))
                is_viol_corr_dyn = s > thr_dyn
                if is_viol_corr_dyn:
                    viol_corr_dynamic += 1
                    Vcorr_dyn.append(key)

                C0 = sum(1 for kk in Vcorr_q0 if kk == key)
                thr0 = q0 + (float(C0) / float(denom))
                is_viol_corr_q0 = s > thr0
                if is_viol_corr_q0:
                    viol_corr_at_q0 += 1
                    Vcorr_q0.append(key)

                # store per-row diagnostics
                q_after = float(q)
                q_before_list.append(q_before)
                q_after_list.append(q_after)

                C_corr_dyn_list.append(int(C_dyn))
                thr_corr_dyn_list.append(float(thr_dyn))
                viol_corr_dyn_list.append(bool(is_viol_corr_dyn))

                C_corr_q0_list.append(int(C0))
                thr_corr_q0_list.append(float(thr0))
                viol_corr_q0_list.append(bool(is_viol_corr_q0))

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

            df[f"q_before_iter{t}"] = q_before_list
            df[f"q_after_iter{t}"] = q_after_list

            # corrected diagnostics columns
            df[f"C_corr_dyn_iter{t}"] = C_corr_dyn_list
            df[f"thr_corr_dyn_iter{t}"] = thr_corr_dyn_list
            df[f"is_violation_corr_dyn_iter{t}"] = viol_corr_dyn_list

            df[f"C_corr_q0_iter{t}"] = C_corr_q0_list
            df[f"thr_corr_q0_iter{t}"] = thr_corr_q0_list
            df[f"is_violation_corr_q0_iter{t}"] = viol_corr_q0_list

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
                    violations_corr_dynamic=int(viol_corr_dynamic),
                    violations_corr_at_Q0=int(viol_corr_at_q0),
                    mean_S=float(np.mean(S_list)) if S_list else 0.0,
                    mean_d=float(np.mean(d_list)) if d_list else 0.0,
                    mean_delta=float(np.mean(delta_list)) if delta_list else 0.0,
                )
            )

        return df, logs
