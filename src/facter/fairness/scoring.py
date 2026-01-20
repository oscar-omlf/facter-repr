from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from facter.fairness.neighbors import CrossGroupNeighborIndex
from facter.models.embedder import TextEmbedder


def item_text(mid: int, item_db: Dict[int, Dict[str, str]]) -> str:
    info = item_db.get(int(mid))
    if info is None:
        return f"UNKNOWN_ITEM_{mid}"
    title = info.get("title", f"UNKNOWN_ITEM_{mid}")
    genres = info.get("genres", "")
    # stable, content-richer string than title alone
    return f"{title} :: {genres}" if genres else title


@dataclass(frozen=True)
class ScoreConfig:
    lambda_fairness: float = 0.7
    tau_rho: float = 0.90  # should match NeighborConfig.tau_rho


class NonconformityScorer:
    def __init__(self, embedder: TextEmbedder, cfg: ScoreConfig):
        self.embedder = embedder
        self.cfg = cfg

    def compute(
        self,
        df: pd.DataFrame,
        pred_mid_col: Optional[str],
        item_db: Dict[int, Dict[str, str]],
        neighbor_index: CrossGroupNeighborIndex,
        pred_text_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Eq.(5) score for each row i, supporting either:
        - ranking mode: pred_mid_col is provided (predictions are item IDs)
        - open mode: pred_text_col is provided (predictions are strings)

        Returns: (S, d, delta, pred_emb)
        """
        # 1. Encode References
        ref_mids = df["target_mid"].astype(int).tolist()
        ref_texts = [item_text(m, item_db) for m in ref_mids]

        # Embedder now returns Tensor on GPU
        ref_emb = self.embedder.encode_texts(ref_texts)  # [N, D]

        # 2. Encode Predictions
        if pred_text_col is not None:
            pred_texts = df[pred_text_col].astype(str).tolist()
        else:
            if pred_mid_col is None:
                raise ValueError(
                    "Either pred_mid_col or pred_text_col must be provided."
                )
            pred_mids = df[pred_mid_col].astype(int).tolist()
            pred_texts = [item_text(m, item_db) for m in pred_mids]

        pred_emb = self.embedder.encode_texts(pred_texts)  # [N, D]

        # 3. Compute d_i = 1 - cos(pred, ref)
        # Element-wise multiplication + sum over dim 1
        cos_pr = torch.sum(pred_emb * ref_emb, dim=1)
        d = 1.0 - cos_pr

        # 4. Compute Delta (Vectorized)
        # \Delta_i = max_{j: W_ij > τρ} ||pred_i - pred_j||_2

        # Get boolean mask [N, N] where M_ij is True if j is a valid neighbor of i
        neighbor_mask = neighbor_index.get_mask_for_delta()

        # Compute Pairwise Euclidean Distance Matrix for predictions
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        # Since embs are normalized, ||a||=1, so dist^2 = 2 - 2<a,b>
        # pairwise_cos = pred_emb @ pred_emb.T
        # dists = sqrt(2 * (1 - pairwise_cos))
        # Note: Clamp to 0 to avoid sqrt(-eps)
        pairwise_cos = torch.matmul(pred_emb, pred_emb.T)
        dists = torch.sqrt(torch.clamp(2.0 * (1.0 - pairwise_cos), min=0.0))

        # Apply mask: we want max over Valid neighbors.
        # Set invalid locations to -1 (since distances are >= 0) so they don't affect max
        # If a row has NO valid neighbors, max will be -1 (we clamp to 0 later)
        masked_dists = dists.clone()
        masked_dists[~neighbor_mask] = -1.0

        delta_vals, _ = torch.max(masked_dists, dim=1)
        delta = torch.clamp(delta_vals, min=0.0)

        # 5. Final Score
        S = d + self.cfg.lambda_fairness * delta

        # Return to CPU/Numpy for logging/dataframe storage
        return (
            S.cpu().numpy(),
            d.cpu().numpy(),
            delta.cpu().numpy(),
            pred_emb.cpu().numpy(),
        )
