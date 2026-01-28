"""Compute fairness-aware online scores for new predictions.

This module defines a small scoring utility that computes an online
fairness-aware score $S_{\mathrm{new}}$ by combining an optional predictive-error
term $d_{\mathrm{new}}$ with a cross-group disparity term $\\Delta_{\\mathrm{new}}$.

The implementation follows the same decomposition used in the method description
for online evaluation (Paper: Sec. 3.3 / Eq. 9), but operational details (e.g.
how groups are represented in the input frame) are driven by code.
"""

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
    """Configure online fairness scoring.

    TODO(doc): clarify whether ``protected_cols`` is used directly in this module
    or only kept for parity with other configuration objects.

    Attributes:
        protected_cols (Tuple[str, ...]): Names of protected-attribute columns.
        tau_rho (float): Cosine-similarity threshold used to select cross-group
            neighbors.
        tau_x_l2 (float | None): Optional locality constraint in L2 distance in
            the (normalized) context-embedding space.
        lambda_fairness (float): Weight applied to the disparity term
            $\\Delta_{\\mathrm{new}}$.
    """

    protected_cols: Tuple[str, ...] = ("gender",)
    tau_rho: float = 0.90
    tau_x_l2: float | None = None  # locality constraint
    lambda_fairness: float = 0.7


@dataclass(frozen=True)
class CalibrationArtifacts:
    """Bundle calibration data and embeddings needed for online scoring.

    Attributes:
        cal_df (pd.DataFrame): Calibration examples as a dataframe.
        cal_context_emb (np.ndarray): Calibration context embeddings.
        cal_pred_emb (np.ndarray): Embeddings of calibration predictions.
        q_alpha0 (float): Initial conformal threshold value.
    """

    cal_df: pd.DataFrame            # must include protected cols
    cal_context_emb: np.ndarray     # [N, D]; treated as cosine-ready (typically normalized)
    cal_pred_emb: np.ndarray        # [N, D]; treated as cosine-ready (typically normalized)
    q_alpha0: float


class OnlineScorer:
    """Score individual examples against calibration artifacts in the online phase.

    The score returned by :meth:`score_one` is:

    $$
    S_{\mathrm{new}} = d_{\mathrm{new}} + \lambda\,\Delta_{\mathrm{new}},
    $$

    where $\\Delta_{\\mathrm{new}}$ is computed from cross-group neighbors selected
    in context-embedding space and $d_{\mathrm{new}}$ is optional (only computed
    when a target item is provided).
    """

    def __init__(self, embedder: TextEmbedder, item_embedder: ItemEmbedder, context_encoder: ContextEncoder, cfg: OnlineScoringConfig):
        """Initialize the scorer.

        Args:
            embedder (TextEmbedder): Text embedder used when scoring from raw
                text.
            item_embedder (ItemEmbedder): Item embedder used when scoring from
                item ids.
            context_encoder (ContextEncoder): Encoder that produces context
                embeddings from input rows.
            cfg (OnlineScoringConfig): Configuration controlling neighbor
                selection and score weighting.
        """
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
        """Compute the fairness-aware score for a single example.

        This method selects cross-group neighbors from the calibration set using
        cosine similarity in the context-embedding space, then computes a
        disparity term $\Delta_{\mathrm{new}}$ based on distances between the
        current prediction embedding and neighbor prediction embeddings.

        If a ``target_mid`` is provided, it also computes a predictive-error term
    $d_{\mathrm{new}}$ as $1-\\cos(\\mathrm{pred}, \\mathrm{ref})$.

        (Paper: Sec. 3.3 / Eq. 9)

        Args:
            row (pd.Series): Input example to score.
            pred_mid (Optional[int]): Predicted item id for "rank" mode.
            item_db (Dict[int, Dict[str, str]]): Item metadata used to build
                reference text when an ``ItemEmbedder`` is not available.
            cal (CalibrationArtifacts): Calibration dataframe and embeddings.
            target_mid (Optional[int]): Optional ground-truth item id. When not
                provided, the predictive-error term is set to 0.
            pred_text (Optional[str]): Predicted text for "open" mode.

        Returns:
            Tuple[float, float, float]: A tuple ``(s_new, d_new, delta_new)``.

        Raises:
            ValueError: If neither ``pred_mid`` nor ``pred_text`` is provided.
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
