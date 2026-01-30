"""Compute fairness-aware nonconformity scores for calibration and evaluation.

This module provides utilities for computing a per-example score that combines a
predictive-error term with a cross-group disparity penalty derived from a
cross-group neighbor index.

The main scoring routine in :class:`NonconformityScorer` implements the same
structure as the fairness-aware nonconformity score described in the paper
(Paper: Sec. 3.2 / Eq. 5).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from facter.models.embedder import TextEmbedder
from facter.fairness.neighbors import CrossGroupNeighborIndex

if TYPE_CHECKING:
    from facter.models.item_embedder import ItemEmbedder

def item_text(mid: int, item_db: Dict[int, Dict[str, str]]) -> str:
    """Build a stable text representation for an item id.

    The returned string is derived from the item metadata (e.g., title and
    genres) and is used as input to a text embedder when an ``ItemEmbedder`` is
    not available.

    Args:
        mid (int): Item id.
        item_db (Dict[int, Dict[str, str]]): Mapping from item id to metadata
            fields.

    Returns:
        str: Text representation for the item.
    """
    info = item_db.get(int(mid))
    if info is None:
        return f"UNKNOWN_ITEM_{mid}"
    title = info.get("title", f"UNKNOWN_ITEM_{mid}")
    genres = info.get("genres", "")
    # stable, content-richer string than title alone
    return f"{title} :: {genres}" if genres else title


@dataclass(frozen=True)
class ScoreConfig:
    """Configure nonconformity scoring.

    Attributes:
        lambda_fairness (float): Weight applied to the cross-group disparity
            term.
        tau_rho (float): Cosine-similarity threshold intended to match the
            neighbor index configuration.
    """

    lambda_fairness: float = 0.7
    tau_rho: float = 0.90  # should match NeighborConfig.tau_rho


class NonconformityScorer:
    """Compute fairness-aware nonconformity scores for a batch of examples.

    The nonconformity score combines an embedding-based predictive-error term
    with a maximum-distance penalty over cross-group neighbors.

    (Paper: Sec. 3.2 / Eq. 5)
    """

    def __init__(self, embedder: TextEmbedder, cfg: ScoreConfig, item_embedder: Optional["ItemEmbedder"] = None):
        """Initialize the scorer.

        Args:
            embedder (TextEmbedder): Text embedder used to convert strings into
                vector representations.
            cfg (ScoreConfig): Hyperparameters controlling the score.
            item_embedder (Optional[ItemEmbedder]): Optional embedder used when
                predictions/targets are item ids.
        """
        self.embedder = embedder
        self.cfg = cfg
        self.item_embedder = item_embedder

    def compute(
        self,
        df: pd.DataFrame,
        pred_mid_col: Optional[str],
        item_db: Dict[int, Dict[str, str]],
        neighbor_index: CrossGroupNeighborIndex,
        pred_text_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-row nonconformity score components.

        This method supports two prediction representations:

        - "rank" mode: predicted items are provided as ids in ``pred_mid_col``.
        - "open" mode: predicted outputs are provided as strings in
          ``pred_text_col``.

        It returns the full score $S$ alongside its components:

                - $d$: predictive error computed as $1-\\cos(\\mathrm{pred},\\mathrm{ref})$.
                - $\\Delta$: maximum L2 distance between the current prediction embedding
          and embeddings of cross-group neighbors.

        (Paper: Sec. 3.2 / Eq. 5)

        Args:
            df (pd.DataFrame): Input dataframe. Must include a ``target_mid``
                column.
            pred_mid_col (Optional[str]): Name of the column containing
                predicted item ids. Used when ``pred_text_col`` is not
                provided.
            item_db (Dict[int, Dict[str, str]]): Item metadata used to construct
                reference/prediction text when an ``ItemEmbedder`` is not
                available.
            neighbor_index (CrossGroupNeighborIndex): Index providing eligible
                neighbor ids for each row.
            pred_text_col (Optional[str]): Name of the column containing
                predicted strings for "open" mode.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple
            ``(S, d, delta, pred_emb)``.

        Raises:
            ValueError: If neither ``pred_mid_col`` nor ``pred_text_col`` is
                provided.
        """
        ref_mids = df["target_mid"].astype(int).tolist()
        
        # Use ItemEmbedder if available for ref items (always mids)
        if self.item_embedder is not None:
            ref_emb = self.item_embedder.get_embeddings(ref_mids)  # [N,D]
        else:
            ref_texts = [item_text(m, item_db) for m in ref_mids]
            ref_emb = self.embedder.encode_texts(ref_texts)  # [N,D]

        if pred_text_col is not None:
            # Open mode: predictions are text strings, use text embedder
            pred_texts = df[pred_text_col].astype(str).tolist()
            pred_emb = self.embedder.encode_texts(pred_texts)  # [N,D]
        else:
            # Rank mode: predictions are item IDs
            if pred_mid_col is None:
                raise ValueError("Either pred_mid_col or pred_text_col must be provided.")
            pred_mids = df[pred_mid_col].astype(int).tolist()
            
            # Use ItemEmbedder if available
            if self.item_embedder is not None:
                pred_emb = self.item_embedder.get_embeddings(pred_mids)  # [N,D]
            else:
                pred_texts = [item_text(m, item_db) for m in pred_mids]
                pred_emb = self.embedder.encode_texts(pred_texts)  # [N,D]

        # d_i = 1 - cos(pred, ref)
        cos_pr = np.sum(pred_emb * ref_emb, axis=1)
        d = (1.0 - cos_pr).astype(np.float32)

    # \\Delta_i = max_{j: W_ij > τρ} ||pred_i - pred_j||_2
        n = len(df)
        delta = np.zeros(n, dtype=np.float32)
        for i in range(n):
            js = neighbor_index.eligible_neighbors_for_delta(i)
            if js.size == 0:
                continue
            diffs = pred_emb[js] - pred_emb[i]
            dist = np.sqrt(np.sum(diffs * diffs, axis=1))
            delta[i] = float(np.max(dist))

        S = (d + self.cfg.lambda_fairness * delta).astype(np.float32)
        return S, d, delta, pred_emb
