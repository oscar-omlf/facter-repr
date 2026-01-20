from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class NeighborConfig:
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    tau_rho: float = 0.90  # neighbor eligibility gate for \Delta_i and N(z_new)
    tau_x_l2: float | None = None  # optional locality constraint on context embeddings
    top_k: int | None = (
        None  # optionally store only top-k cross-group neighbors per point
    )


class CrossGroupNeighborIndex:
    """
    Represents W implicitly.
    Stores cross-group neighbors j for each calibration i, and their W_ij=cos(e^x_i,e^x_j),
    respecting optional locality (tau_x_l2) and optional top_k truncation.

    Paper Eq.(4): W_ij = cos(e^x_i,e^x_j) if a_i != a_j and ||x_i-x_j|| <= tau_x else 0.
    We proxy ||x_i-x_j|| with embedding L2 if tau_x_l2 is set.
    """

    def __init__(self, cfg: NeighborConfig):
        self.cfg = cfg
        # These will be tensors on the same device as input embeddings
        self._sim_matrix: torch.Tensor | None = None
        self._valid_mask: torch.Tensor | None = None  # Cross-group & Locality mask

    def fit(self, df: pd.DataFrame, context_emb: torch.Tensor) -> None:
        """
        Args:
            df: DataFrame containing protected attributes.
            context_emb: [N, D] tensor on GPU (normalized).
        """

        device = context_emb.device

        # 1. Compute Cosine Similarity Matrix [N, N]
        # Since context_emb is normalized, cos_sim = MM^T
        self._sim_matrix = torch.matmul(context_emb, context_emb.T)

        # Mask out self-loops (sim = -1.0 or just ignore later)
        self._sim_matrix.fill_diagonal_(-1.0)

        # 2. Construct Cross-Group Mask [N, N]
        # Create a unique integer ID for each group combination
        # We do this on CPU via Pandas, then move to GPU
        cols = list(self.cfg.protected_cols)
        a_series = df[cols[0]].astype(str)

        for c in cols[1:]:
            a_series = a_series.str.cat(df[c].astype(str), sep="_")

        # Factorize returns (codes, uniques)
        group_codes, _ = pd.factorize(a_series)
        group_ids = torch.tensor(group_codes, device=device, dtype=torch.long)

        # Broadcast comparison: [N, 1] != [1, N] -> [N, N] boolean matrix
        # True where groups are DIFFERENT
        cross_group_mask = group_ids.unsqueeze(1) != group_ids.unsqueeze(0)

        # 3. Locality Constraint (tau_x_l2)
        # L2 <= tau <=> Cos >= 1 - tau^2/2
        if self.cfg.tau_x_l2 is not None:
            cos_min = 1.0 - (self.cfg.tau_x_l2**2) / 2.0
            locality_mask = self._sim_matrix >= cos_min
            self._valid_mask = cross_group_mask & locality_mask

        else:
            self._valid_mask = cross_group_mask

        # 4. Apply Top-K (Optional)
        if self.cfg.top_k is not None:
            # We want to keep indices with the highest similarity within the valid mask
            # Mask invalid entries with -inf so they don't get selected
            masked_sims = self._sim_matrix.clone()
            masked_sims[~self._valid_mask] = -float("inf")

            # Get top-k values/indices
            _, topk_indices = torch.topk(masked_sims, k=self.cfg.top_k, dim=1)

            # Create a new mask from these indices
            new_mask = torch.zeros_like(self._valid_mask)

            # Scatter 1s into the topk positions
            new_mask.scatter_(1, topk_indices, True)

            # Update valid mask
            self._valid_mask = self._valid_mask & new_mask

    def get_mask_for_delta(self) -> torch.Tensor:
        """
        Returns the mask of neighbors eligible for delta calculation.
        Condition: Valid Neighbor AND Sim > tau_rho
        """
        rho_mask = self._sim_matrix > self.cfg.tau_rho
        return self._valid_mask & rho_mask

    def get_sims(self) -> torch.Tensor:
        return self._sim_matrix
