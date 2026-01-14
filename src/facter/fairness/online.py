from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


from facter.models.embedder import TextEmbedder
from facter.fairness.context_encoder import ContextEncoder
from facter.fairness.scoring import item_text


@dataclass(frozen=True)
class OnlineScoringConfig:
    protected_cols: Tuple[str, ...] = ("gender",)
    tau_rho: float = 0.90
    tau_x_l2: float | None = None  # locality constraint
    lambda_fairness: float = 0.7


@dataclass(frozen=True)
class CalibrationArtifacts:
    cal_df: pd.DataFrame            # must include protected cols
    cal_context_emb: np.ndarray     # [N, D], normalized
    cal_pred_emb: np.ndarray        # [N, D], normalized  (fix comment)
    q_alpha0: float


class OnlineScorer:
    def __init__(self, embedder: TextEmbedder, context_encoder: ContextEncoder, cfg: OnlineScoringConfig):
        self.embedder = embedder
        self.context_encoder = context_encoder
        self.cfg = cfg

    def score_one(
        self,
        row: pd.Series,
        pred_mid: Optional[int],
        item_db: Dict[int, Dict[str, str]],
        cal: CalibrationArtifacts,
        target_mid: Optional[int] = None,
        pred_text: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """
        Online scoring for a single example.
        Supports either:
        - rank mode: pred_mid provided, pred_text None
        - open mode: pred_text provided (optionally also a mapped pred_mid)
        """
        df_one = pd.DataFrame([row.to_dict()])
        x_new = self.context_encoder.encode_df(df_one)[0]  # [D] normalized
        sims = cal.cal_context_emb @ x_new

        # Cross-group mask
        a_new = tuple(str(row[c]) for c in self.cfg.protected_cols)
        a_cal = (
            cal.cal_df[list(self.cfg.protected_cols)]
            .astype(str)
            .agg("_".join, axis=1)
            .to_numpy()
        )
        cross = a_cal != "_".join(a_new)

        # Optional locality gate (embedding L2 radius τx)
        if self.cfg.tau_x_l2 is not None:
            cos_min = 1.0 - (self.cfg.tau_x_l2 ** 2) / 2.0
        else:
            cos_min = -np.inf

        neigh_mask = cross & (sims >= self.cfg.tau_rho) & (sims >= cos_min)
        neigh_idx = np.where(neigh_mask)[0]

        # Prediction embedding: from pred_text if provided, else from item_db mid
        if pred_text is not None:
            pred_txt = str(pred_text)
        else:
            if pred_mid is None:
                raise ValueError("Either pred_text or pred_mid must be provided.")
            pred_txt = item_text(int(pred_mid), item_db)

        pred_emb = self.embedder.encode_texts([pred_txt])[0]  # [D], normalized

        # \Delta_new
        if neigh_idx.size == 0:
            delta_new = 0.0
        else:
            diffs = cal.cal_pred_emb[neigh_idx] - pred_emb
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            delta_new = float(np.max(dists))

        # d_new (requires ground-truth target_mid for offline eval)
        if target_mid is None:
            d_new = 0.0
        else:
            ref_txt = item_text(int(target_mid), item_db)
            ref_emb = self.embedder.encode_texts([ref_txt])[0]
            d_new = float(1.0 - float(np.sum(pred_emb * ref_emb)))

        s_new = float(d_new + self.cfg.lambda_fairness * delta_new)
        return s_new, float(d_new), float(delta_new)
