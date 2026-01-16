from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm

from facter.data.prompts import PromptConfig, build_open_prompt
from facter.models.ranker import Ranker
from facter.models.embedder import TextEmbedder
from facter.models.generator import Generator
from facter.fairness.context_encoder import ContextEncoder
from facter.fairness.neighbors import CrossGroupNeighborIndex, NeighborConfig
from facter.fairness.scoring import NonconformityScorer, ScoreConfig, item_text
from facter.fairness.conformal import conformal_quantile


@dataclass(frozen=True)
class OfflineCalibConfig:
    alpha: float = 0.10
    lambda_fairness: float = 0.7
    tau_rho: float = 0.90
    tau_x_l2: float | None = None
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    top_k_neighbors: int | None = None  # optional top-k storage in W


@dataclass(frozen=True)
class OfflineCalibrationResult:
    cal_df: pd.DataFrame
    cal_context_emb: np.ndarray
    cal_pred_mid: np.ndarray
    cal_pred_text: list[str]
    cal_pred_emb: np.ndarray
    scores_S: np.ndarray
    scores_d: np.ndarray
    scores_delta: np.ndarray
    q_alpha0: float


class OfflineCalibrator:
    """
    Implements Offline calibration phase (Sec. 3.2):
      - Enc(x) + W (Eq.4)
      - score S_i (Eq.5)
      - conformal quantile Q_alpha^(0) (Eq.6)
    """

    def __init__(
        self,
        ranker: Ranker,
        embedder: TextEmbedder,
        context_encoder: ContextEncoder,
        cfg: OfflineCalibConfig,
    ):
        self.ranker = ranker
        self.embedder = embedder
        self.context_encoder = context_encoder
        self.cfg = cfg

    def _predict_top1_mid(
        self, row: pd.Series, system_prompt: Optional[str]
    ) -> Tuple[int, List[int], str, List[int]]:
        """Returns (top1_mid, all_ranked_mids, raw_response, ranked_indices)"""
        candidates_titles: List[str] = row["candidate_titles"]
        candidate_mids: List[int] = row["candidate_mids"]

        # Get ranked indices from ranker
        ranked_idx, raw_response = self.ranker.rank(
            row["prompt_rank"], candidates_titles, system_prompt=system_prompt
        )

        best_idx = ranked_idx[0]
        top1_mid = int(candidate_mids[best_idx])

        return top1_mid, raw_response

    def run(
        self,
        cal_df: pd.DataFrame,
        item_db: Dict[int, Dict[str, str]],
        system_prompt: Optional[str] = None,
        progress: bool = False,
        predict_mode: str = "rank",  # "rank" | "open"
        generator: Optional[Generator] = None,
        prompt_cfg: Optional[PromptConfig] = None,
    ) -> OfflineCalibrationResult:
        """
        Offline calibration supporting rank-mode and open-generation mode.

        Rank-mode:
        - Use ranker to select top-1 mid from candidate set.
        - pred_text is item_text(pred_mid).

        Open-mode:
        - Use generator to produce top-k titles.
        - Map top-1 title to mid if possible (optional).
        - pred_text is item_text(mapped_mid) if mapped else raw title.

        Returns OfflineCalibrationResult with cal_pred_emb computed from pred_text,
        so online scoring uses the same output embedding space.
        """
        df = cal_df.reset_index(drop=True).copy()

        # 1) Enc(x)
        context_emb = self.context_encoder.encode_df(df)  # [N,D] normalized

        pred_mids_list: list[int] = []
        pred_texts_list: list[str] = []
        raw_responses_list: list[str] = []

        it = range(len(df))
        if progress:
            it = tqdm(it, total=len(df), desc=f"Offline: predict (mode={predict_mode})")

        if predict_mode == "rank":
            for i in it:
                top1_mid, raw_response = self._predict_top1_mid(
                    df.iloc[i], system_prompt
                )
                pred_mids_list.append(int(top1_mid))
                pred_texts_list.append(item_text(int(top1_mid), item_db))
                raw_responses_list.append(raw_response)

        elif predict_mode == "open":
            if generator is None or prompt_cfg is None:
                raise ValueError("open mode requires generator and prompt_cfg")

            # Build prompts
            prompts = [df["prompt_gen"].iloc[i] for i in it]
            system_prompts = [system_prompt] * len(df)

            # Generate
            gen_lists = generator.generate_topk(
                prompts, system_prompts, k=prompt_cfg.k_recs
            )

            # Minimal title->mid mapping (exact match on item_db title)
            # You can replace this with the more robust normalization index used in the script.
            title_to_mid: Dict[str, int] = {}
            for mid, info in item_db.items():
                title_to_mid[str(info.get("title", "")).strip().lower()] = int(mid)

            for titles in gen_lists:
                top1_title = (titles[0] if titles else "").strip()
                raw_responses_list.append(json.dumps(titles, ensure_ascii=False))

                mid = title_to_mid.get(top1_title.lower(), -1)
                pred_mids_list.append(int(mid))

                if mid != -1:
                    pred_texts_list.append(item_text(int(mid), item_db))
                else:
                    pred_texts_list.append(
                        top1_title if top1_title else "UNKNOWN_GENERATION"
                    )

        else:
            raise ValueError("predict_mode must be 'rank' or 'open'")

        pred_mids = np.array(pred_mids_list, dtype=np.int64)

        # Store for logging / reuse
        df["system_prompt"] = system_prompt
        df["ranker_response"] = raw_responses_list
        df["pred_mid"] = pred_mids
        df["pred_text"] = pred_texts_list

        # 3) Fit neighbor index
        ncfg = NeighborConfig(
            protected_cols=self.cfg.protected_cols,
            tau_rho=self.cfg.tau_rho,
            tau_x_l2=self.cfg.tau_x_l2,
            top_k=self.cfg.top_k_neighbors,
        )
        nidx = CrossGroupNeighborIndex(ncfg)
        nidx.fit(df, context_emb)

        # 4) Compute S_i using pred_text in open mode
        scfg = ScoreConfig(
            lambda_fairness=self.cfg.lambda_fairness, tau_rho=self.cfg.tau_rho
        )
        scorer = NonconformityScorer(self.embedder, scfg)

        if predict_mode == "rank":
            S, d, delta, pred_emb = scorer.compute(
                df, pred_mid_col="pred_mid", item_db=item_db, neighbor_index=nidx
            )
        else:
            S, d, delta, pred_emb = scorer.compute(
                df,
                pred_mid_col=None,
                item_db=item_db,
                neighbor_index=nidx,
                pred_text_col="pred_text",
            )

        # 5) Quantile
        q0 = conformal_quantile(S, alpha=self.cfg.alpha)

        return OfflineCalibrationResult(
            cal_df=df,
            cal_context_emb=context_emb,
            cal_pred_mid=pred_mids,
            cal_pred_text=pred_texts_list,
            cal_pred_emb=pred_emb,
            scores_S=S,
            scores_d=d,
            scores_delta=delta,
            q_alpha0=float(q0),
        )
