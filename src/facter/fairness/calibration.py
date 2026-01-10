from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from facter.models.ranker import Ranker
from facter.models.embedder import TextEmbedder
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

    def _predict_top1_mid(self, row: pd.Series, system_prompt: Optional[str]) -> int:
        candidates_titles: List[str] = row["candidate_titles"]
        ranked_idx = self.ranker.rank(row["prompt_rank"], candidates_titles, system_prompt=system_prompt)
        best_idx = ranked_idx[0]
        return int(row["candidate_mids"][best_idx])

    def run(
        self,
        cal_df: pd.DataFrame,
        item_db: Dict[int, Dict[str, str]],
        system_prompt: Optional[str] = None,
        progress: bool = False,
    ) -> OfflineCalibrationResult:
        df = cal_df.reset_index(drop=True).copy()

        # 1) Enc(x)
        context_emb = self.context_encoder.encode_df(df)  # [N,D] normalized

        # 2) Predict hat{y}_i (ranking-based, choose top-1)
        # pred_mids = np.array([self._predict_top1_mid(df.iloc[i], system_prompt) for i in range(len(df))], dtype=np.int64)
        pred_mids_list = []
        it = range(len(df))
        if progress:
            it = tqdm(it, total=len(df), desc="Offline: rank top-1 (calibration)")
        for i in it:
            pred_mids_list.append(self._predict_top1_mid(df.iloc[i], system_prompt))
        pred_mids = np.array(pred_mids_list, dtype=np.int64)

        # 3) Fit W index
        ncfg = NeighborConfig(
            protected_cols=self.cfg.protected_cols,
            tau_rho=self.cfg.tau_rho,
            tau_x_l2=self.cfg.tau_x_l2,
            top_k=self.cfg.top_k_neighbors,
        )
        nidx = CrossGroupNeighborIndex(ncfg)
        nidx.fit(df, context_emb)

        # 4) Compute S_i
        # Add a column to reuse NonconformityScorer
        df["pred_mid"] = pred_mids
        scfg = ScoreConfig(lambda_fairness=self.cfg.lambda_fairness, tau_rho=self.cfg.tau_rho)
        scorer = NonconformityScorer(self.embedder, scfg)

        S, d, delta = scorer.compute(df, "pred_mid", item_db, nidx)

        # 5) Compute Q_alpha^(0)
        q0 = conformal_quantile(S, alpha=self.cfg.alpha)

        # Precompute pred embeddings for online use
        pred_texts = [item_text(int(m), item_db) for m in pred_mids.tolist()]
        pred_emb = self.embedder.encode_texts(pred_texts)

        return OfflineCalibrationResult(
            cal_df=df.drop(columns=["pred_mid"]),
            cal_context_emb=context_emb,
            cal_pred_mid=pred_mids,
            cal_pred_emb=pred_emb,
            scores_S=S,
            scores_d=d,
            scores_delta=delta,
            q_alpha0=float(q0),
        )
