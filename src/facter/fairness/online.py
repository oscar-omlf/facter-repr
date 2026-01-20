from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import pandas as pd
from facter.fairness.scoring import item_text
from facter.models.embedder import TextEmbedder


@dataclass(frozen=True)
class OnlineScoringConfig:
    protected_cols: Tuple[str, ...] = ("gender",)
    tau_rho: float = 0.90
    tau_x_l2: float | None = None  # locality constraint
    lambda_fairness: float = 0.7


@dataclass(frozen=True)
class CalibrationArtifacts:
    cal_df: pd.DataFrame
    cal_context_emb: torch.Tensor  # [N, D]
    cal_pred_emb: torch.Tensor  # [N, D]
    cal_group_ids: torch.Tensor  # [N] Int tensor for fast group comparison
    group_code_map: Dict[str, int]  # "val1_val2" -> int ID
    q_alpha0: float


class OnlineScorer:
    def __init__(
        self,
        embedder: TextEmbedder,
        cfg: OnlineScoringConfig,
        # Note: context_encoder is removed from init as it is no longer used here
    ):
        self.embedder = embedder
        self.cfg = cfg

    def score_one(
        self,
        row: pd.Series,
        pred_mid: Optional[int],
        item_db: Dict[int, Dict[str, str]],
        cal: CalibrationArtifacts,
        precomputed_context_emb: torch.Tensor,
        target_mid: Optional[int] = None,
        pred_text: Optional[str] = None,
        precomputed_group_id: int = -1,
    ) -> Tuple[float, float, float]:
        """
        Online scoring for a single example using GPU acceleration.
        Uses precomputed_context_emb (history) to avoid re-running BERT.
        """
        # 1. Context Embedding (Pre-computed)
        # x_new is passed in directly. It is already on the correct device.
        x_new = precomputed_context_emb

        # 2. Similarity with Calibration Set (Matrix-Vector)
        # cal_context_emb [N, D] @ x_new [D] -> [N]
        sims = torch.mv(cal.cal_context_emb, x_new)

        # 3. Cross-Group Mask (Fast Integer Lookup)
        # Use precomputed group ID if available
        if precomputed_group_id != -1:
            gid_new = precomputed_group_id
        else:
            # Fallback (slower)
            a_vals = [str(row[c]) for c in self.cfg.protected_cols]
            a_key = "_".join(a_vals)
            gid_new = cal.group_code_map.get(a_key, -1)

        # Mask: True where cal_group != current_group
        cross_mask = cal.cal_group_ids != gid_new

        # 4. Locality Gate
        if self.cfg.tau_x_l2 is not None:
            cos_min = 1.0 - (self.cfg.tau_x_l2**2) / 2.0
            locality_mask = sims >= cos_min
            valid_mask = cross_mask & locality_mask

        else:
            valid_mask = cross_mask

        # 5. Rho Gate
        rho_mask = sims >= self.cfg.tau_rho
        neigh_mask = valid_mask & rho_mask

        # 6. Prediction Embedding (Dynamic - must be computed)
        if pred_text is not None:
            pred_txt = str(pred_text)
        else:
            if pred_mid is None:
                raise ValueError("Either pred_text or pred_mid must be provided.")
            pred_txt = item_text(int(pred_mid), item_db)

        pred_emb = self.embedder.encode_text(pred_txt)  # [D] Tensor

        # 7. Delta Calculation (Vectorized max)
        # ||a - b|| = sqrt(2 - 2<a,b>) for normalized vectors
        cos_sims_pred = torch.mv(cal.cal_pred_emb, pred_emb)
        dists = torch.sqrt(torch.clamp(2.0 * (1.0 - cos_sims_pred), min=0.0))

        # Apply mask: set invalid to -1 so they don't affect max
        dists_masked = dists.clone()
        dists_masked[~neigh_mask] = -1.0

        delta_new_t = torch.max(dists_masked)
        delta_new = float(delta_new_t.item())
        if delta_new < 0:
            delta_new = 0.0  # happens if no neighbors found

        # 8. d_new (Relevance)
        if target_mid is None:
            d_new = 0.0
        else:
            ref_txt = item_text(int(target_mid), item_db)
            ref_emb = self.embedder.encode_text(ref_txt)
            # 1 - cos
            d_new = 1.0 - float(torch.dot(pred_emb, ref_emb).item())

        s_new = d_new + self.cfg.lambda_fairness * delta_new
        return s_new, d_new, delta_new
