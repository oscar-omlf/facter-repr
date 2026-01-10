from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """
    emb assumed L2-normalized row-wise.
    returns [N, N] cosine similarities.
    """
    return emb @ emb.T


@dataclass(frozen=True)
class NeighborConfig:
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    tau_rho: float = 0.90  # similarity threshold used later for \Delta_i filtering
    tau_x_l2: float | None = None  # optional locality constraint on context embeddings
    top_k: int | None = None  # optional: store only top-k cross-group neighbors per point


class CrossGroupNeighborIndex:
    """
    Represents W implicitly. Stores cross-group neighbors and their W_ij = cos(e_i^x, e_j^x).
    Paper Eq.(4): W_ij = cos(e^x_i, e^x_j) if a_i != a_j and ||x_i-x_j|| <= tau_x else 0.
    We approximate x-distance with embedding L2 if tau_x_l2 is set.
    """

    def __init__(self, cfg: NeighborConfig):
        self.cfg = cfg
        self._neighbors: List[np.ndarray] = []
        self._sims: List[np.ndarray] = []
        self._a: List[Tuple[str, ...]] = []

    def fit(self, df: pd.DataFrame, context_emb: np.ndarray) -> None:
        """
        df: calibration dataframe with protected columns
        context_emb: [N, D] normalized context embeddings
        """
        n = len(df)
        if context_emb.shape[0] != n:
            raise ValueError("context_emb rows must match df length")

        # protected attribute tuple per row
        a = [
            tuple(str(df.iloc[i][c]) for c in self.cfg.protected_cols)
            for i in range(n)
        ]
        self._a = a

        sims = cosine_sim_matrix(context_emb)
        np.fill_diagonal(sims, -1.0)  # exclude self

        # optional locality constraint using embedding L2 distance
        if self.cfg.tau_x_l2 is not None:
            # Since embeddings are normalized, L2^2 = 2 - 2*cos
            # So L2 <= tau_x_l2  <=> cos >= 1 - tau_x_l2^2/2
            cos_min = 1.0 - (self.cfg.tau_x_l2 ** 2) / 2.0
        else:
            cos_min = -np.inf

        self._neighbors = []
        self._sims = []

        for i in range(n):
            # cross-group indices
            mask_cross = np.array([a[j] != a[i] for j in range(n)], dtype=bool)
            mask_sim = sims[i] >= cos_min
            mask = mask_cross & mask_sim

            idx = np.where(mask)[0]
            s = sims[i, idx]

            # optional: keep only top_k by cosine
            if self.cfg.top_k is not None and len(idx) > self.cfg.top_k:
                top = np.argsort(-s)[: self.cfg.top_k]
                idx = idx[top]
                s = s[top]

            self._neighbors.append(idx.astype(np.int64))
            self._sims.append(s.astype(np.float32))

    def neighbors_of(self, i: int) -> np.ndarray:
        return self._neighbors[i]

    def sims_of(self, i: int) -> np.ndarray:
        return self._sims[i]

    def eligible_neighbors_for_delta(self, i: int) -> np.ndarray:
        """
        Returns neighbor indices j with W_ij > tau_rho (paper's \tau_\rho gate used in \Delta).
        """
        idx = self._neighbors[i]
        s = self._sims[i]
        return idx[s > self.cfg.tau_rho]
