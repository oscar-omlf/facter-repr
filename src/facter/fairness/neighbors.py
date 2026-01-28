"""Build and query cross-group neighbor sets in embedding space.

This module constructs an index of cross-group neighbors for each calibration
example, based on cosine similarity between context embeddings. The resulting
neighbor sets are used downstream when computing disparity penalties such as
$\\Delta_i$ / $\\Delta_{\\mathrm{new}}$.

Where applicable, the implementation is inspired by the paper's similarity-matrix
construction for cross-group comparisons. (Paper: Sec. 3.2 / Eq. 4)
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """Compute a cosine similarity matrix from row-wise normalized embeddings.

    Args:
        emb (np.ndarray): Embedding matrix assumed to be L2-normalized per row.

    Returns:
        np.ndarray: Pairwise cosine similarities between all rows.
    """
    return emb @ emb.T


@dataclass(frozen=True)
class NeighborConfig:
    """Configure cross-group neighbor construction.

    Attributes:
        protected_cols (Tuple[str, ...]): Columns whose values define the
            protected-attribute tuple used for grouping.
        tau_rho (float): Cosine-similarity threshold used to decide whether a
            stored neighbor is eligible for the $\\Delta$ computation.
        tau_x_l2 (float | None): Optional locality constraint in L2 distance in
            the (normalized) context-embedding space.
        top_k (int | None): Optional cap on the number of stored cross-group
            neighbors per point.
    """

    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    tau_rho: float = 0.90  # neighbor eligibility gate for \\Delta_i and N(z_new)
    tau_x_l2: float | None = None  # optional locality constraint on context embeddings
    top_k: int | None = None  # optionally store only top-k cross-group neighbors per point


class CrossGroupNeighborIndex:
    """Store cross-group neighbors for each calibration example.

    This class represents a sparse cross-group similarity structure by storing,
    for each calibration index $i$, the indices $j$ that are considered
    cross-group neighbors and their associated cosine similarities.

    If a locality constraint is enabled via ``tau_x_l2``, it is applied in the
    *context embedding* space using the identity (for normalized vectors)
    $\\lVert x_i - x_j \\rVert_2^2 = 2 - 2\\cos(x_i, x_j)$.

    (Paper: Sec. 3.2 / Eq. 4)

    TODO(doc): The paper defines the locality condition using the original
    context vectors $x_i$; this implementation applies it to the encoded context
    embeddings passed to :meth:`fit`.
    """

    def __init__(self, cfg: NeighborConfig):
        """Initialize an empty neighbor index.

        Args:
            cfg (NeighborConfig): Configuration controlling grouping, locality,
                and truncation.
        """
        self.cfg = cfg
        self._neighbors: List[np.ndarray] = []
        self._sims: List[np.ndarray] = []
        self._group_id: np.ndarray | None = None  # integer group ids
        self._a_tuple: List[Tuple[str, ...]] = []

    @property
    def a_tuples(self) -> List[Tuple[str, ...]]:
        """Return the protected-attribute tuple for each fitted row.

        Returns:
            List[Tuple[str, ...]]: One tuple per row, with entries coerced to
            strings.
        """
        return self._a_tuple

    def fit(self, df: pd.DataFrame, context_emb: np.ndarray) -> None:
        """Build the cross-group neighbor lists from a calibration dataframe.

        Args:
            df (pd.DataFrame): Calibration dataframe containing the columns
                specified by ``NeighborConfig.protected_cols``.
            context_emb (np.ndarray): Context embedding matrix aligned with
                ``df``.

        Raises:
            ValueError: If ``context_emb`` does not have the same number of rows
                as ``df``.
        """
        n = len(df)
        if context_emb.shape[0] != n:
            raise ValueError("context_emb rows must match df length")

        # protected attribute tuple per row
        a_tuple = [
            tuple(str(df.iloc[i][c]) for c in self.cfg.protected_cols)
            for i in range(n)
        ]
        self._a_tuple = a_tuple

        # map tuples to integer group ids for fast cross-group masking
        uniq = {t: idx for idx, t in enumerate(sorted(set(a_tuple)))}
        group_id = np.array([uniq[t] for t in a_tuple], dtype=np.int64)
        self._group_id = group_id

        sims = cosine_sim_matrix(context_emb)
        np.fill_diagonal(sims, -1.0)  # exclude self

        # optional locality constraint
        if self.cfg.tau_x_l2 is not None:
            # normalized embeddings: L2^2 = 2 - 2*cos -> L2 <= tau => cos >= 1 - tau^2/2
            cos_min = 1.0 - (self.cfg.tau_x_l2 ** 2) / 2.0
        else:
            cos_min = -np.inf

        self._neighbors = []
        self._sims = []

        for i in range(n):
            # cross-group: group_id != group_id[i]
            mask_cross = group_id != group_id[i]
            mask_sim = sims[i] >= cos_min
            mask = mask_cross & mask_sim

            idx = np.where(mask)[0]
            s = sims[i, idx]

            if self.cfg.top_k is not None and len(idx) > self.cfg.top_k:
                top = np.argsort(-s)[: self.cfg.top_k]
                idx = idx[top]
                s = s[top]

            self._neighbors.append(idx.astype(np.int64))
            self._sims.append(s.astype(np.float32))

    def neighbors_of(self, i: int) -> np.ndarray:
        """Return stored neighbor indices for row ``i``.

        Args:
            i (int): Row index.

        Returns:
            np.ndarray: Array of neighbor indices.
        """
        return self._neighbors[i]

    def sims_of(self, i: int) -> np.ndarray:
        """Return stored cosine similarities for row ``i``.

        Args:
            i (int): Row index.

        Returns:
            np.ndarray: Cosine similarities aligned with :meth:`neighbors_of`.
        """
        return self._sims[i]

    def eligible_neighbors_for_delta(self, i: int) -> np.ndarray:
        """Return neighbors eligible for the $\\Delta$ computation.

        This applies the ``tau_rho`` threshold to the stored similarities.

        (Paper: Sec. 3.2 / Eq. 5)

        Args:
            i (int): Row index.

        Returns:
            np.ndarray: Neighbor indices whose similarity exceeds ``tau_rho``.
        """
        idx = self._neighbors[i]
        s = self._sims[i]
        return idx[s > self.cfg.tau_rho]
