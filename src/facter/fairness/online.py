from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


from facter.models.embedder import TextEmbedder
from facter.models.item_embedder import ItemEmbedder
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
    def __init__(self, embedder: TextEmbedder, item_embedder: ItemEmbedder, context_encoder: ContextEncoder, cfg: OnlineScoringConfig):
        self.embedder = embedder
        self.item_embedder = item_embedder
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
        cross = cal.cal_df["group_attrs"] != df_one["group_attrs"].iloc[0]

        # Optional locality gate (embedding L2 radius τx)
        if self.cfg.tau_x_l2 is not None:
            cos_min = 1.0 - (self.cfg.tau_x_l2 ** 2) / 2.0
        else:
            cos_min = -np.inf

        neigh_mask = cross & (sims >= self.cfg.tau_rho) & (sims >= cos_min)
        neigh_idx = np.where(neigh_mask)[0]

        # Prediction embedding: from pred_text if provided, else from item_db mid
        if pred_mid is not None and self.item_embedder is not None:
            pred_emb = self.item_embedder.get_embedding(pred_mid)
        else:
            if pred_text is None:
                raise ValueError("Either pred_mid or pred_text must be provided")
            pred_emb = self.embedder.encode_text(pred_text)

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
            if self.item_embedder is not None:
                ref_emb = self.item_embedder.get_embedding(target_mid)
            else:
                ref_text = item_text(target_mid, item_db)
                ref_emb = self.embedder.encode_text(ref_text)

            d_new = float(1.0 - float(np.sum(pred_emb * ref_emb)))

        s_new = float(d_new + self.cfg.lambda_fairness * delta_new)
        return s_new, float(d_new), float(delta_new)
