from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from facter.data.prompts import PromptConfig
from facter.models.ranker import Ranker
from facter.models.embedder import TextEmbedder
from facter.models.generator import Generator
from facter.eval.catalogue_map import CatalogueMapper
from facter.eval.prediction import predict_batch_rank, predict_batch_open, build_title_to_mid_dict
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

    def run(
        self,
        cal_df: pd.DataFrame,
        item_db: Dict[int, Dict[str, str]],
        system_prompt: Optional[str] = None,
        progress: bool = False,
        predict_mode: str = "rank",  # "rank" | "open"
        generator: Optional[Generator] = None,
        prompt_cfg: Optional[PromptConfig] = None,
        catalogue_mapper: Optional[CatalogueMapper] = None,
        min_sim: float = 0.65,
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

        # 2) Predict using unified prediction module
        title_to_mid: Optional[Dict[str, int]] = None
        if predict_mode == "rank":
            pred_result = predict_batch_rank(df, self.ranker, item_db, system_prompt, progress)
        elif predict_mode == "open":
            if generator is None or prompt_cfg is None:
                raise ValueError("open mode requires generator and prompt_cfg")
            pred_result = predict_batch_open(df, generator, item_db, prompt_cfg, system_prompt,
                                             catalogue_mapper, title_to_mid, min_sim, progress)
        else:
            raise ValueError("predict_mode must be 'rank' or 'open'")

        pred_mids = np.array(pred_result.pred_mids, dtype=np.int64)

        # Store for logging / reuse
        df["system_prompt"] = system_prompt
        df["ranker_response"] = pred_result.model_responses
        df["pred_mid"] = pred_mids
        df["pred_text"] = pred_result.pred_texts
        # In open mode, persist fraction of generated titles mapped to catalog
        if predict_mode == "open":
            df["valid_at_k"] = pred_result.valid_at_k_list

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
            cal_pred_text=pred_result.pred_texts,
            cal_pred_emb=pred_emb,
            scores_S=S,
            scores_d=d,
            scores_delta=delta,
            q_alpha0=float(q0),
        )
