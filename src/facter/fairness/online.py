from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from facter.models.embedder import TextEmbedder
from facter.fairness.context_encoder import ContextEncoder
from facter.fairness.scoring import item_text


@dataclass(frozen=True)
class OnlineScoringConfig:
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    tau_rho: float = 0.90
    lambda_fairness: float = 0.7


@dataclass(frozen=True)
class CalibrationArtifacts:
    """
    What online scoring needs from offline calibration.
    """
    cal_df: pd.DataFrame                  # must include protected cols
    cal_context_emb: np.ndarray           # [N, D], normalized
    cal_pred_emb: np.ndarray              # [N, M], normalized
    q_alpha0: float


class OnlineScorer:
    """
    Implements Eq.(9): S_new = d_new + lambda * Δ_new
    using N(z_new) defined by context similarity >= tau_rho and cross-group. :contentReference[oaicite:7]{index=7}
    """
    def __init__(
        self,
        embedder: TextEmbedder,
        context_encoder: ContextEncoder,
        cfg: OnlineScoringConfig,
    ):
        self.embedder = embedder
        self.context_encoder = context_encoder
        self.cfg = cfg

    def score_one(
        self,
        row: pd.Series,
        pred_mid: int,
        item_db: Dict[int, Dict[str, str]],
        cal: CalibrationArtifacts,
        target_mid: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Returns (S_new, d_new, delta_new).
        If target_mid is None, d_new := 0.0 (deployment-like mode).
        """
        # Context embedding for new point
        df_one = pd.DataFrame([row.to_dict()])
        x_new = self.context_encoder.encode_df(df_one)[0]  # [D] normalized
        sims = cal.cal_context_emb @ x_new  # cosine since normalized

        # Cross-group mask
        a_new = tuple(str(row[c]) for c in self.cfg.protected_cols)
        a_cal = cal.cal_df[list(self.cfg.protected_cols)].astype(str).agg("_".join, axis=1).to_numpy()
        a_new_key = "_".join(a_new)
        cross = a_cal != a_new_key

        # Similarity gate for neighborhood N(z_new)
        neigh_mask = (sims >= self.cfg.tau_rho) & cross
        neigh_idx = np.where(neigh_mask)[0]

        # Pred embedding
        pred_txt = item_text(pred_mid, item_db)
        pred_emb = self.embedder.encode_texts([pred_txt])[0]  # [M], normalized

        # Δ_new
        if neigh_idx.size == 0:
            delta_new = 0.0
        else:
            diffs = cal.cal_pred_emb[neigh_idx] - pred_emb
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            delta_new = float(np.max(dists))

        # d_new
        if target_mid is None:
            d_new = 0.0
        else:
            ref_txt = item_text(int(target_mid), item_db)
            ref_emb = self.embedder.encode_texts([ref_txt])[0]
            d_new = float(1.0 - float(np.sum(pred_emb * ref_emb)))

        s_new = float(d_new + self.cfg.lambda_fairness * delta_new)
        return s_new, float(d_new), float(delta_new)
