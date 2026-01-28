"""Run the offline calibration phase for FACTER.

This module implements the offline phase that:
1) encodes user contexts,
2) generates model predictions,
3) computes fairness-aware nonconformity scores, and
4) computes an initial conformal threshold ``q_alpha0``.

The calibration flow corresponds to the paper's offline calibration phase.
(Paper: Sec. 3.2 / Eq. 5 / Eq. 6)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from facter.data.prompts import PromptConfig
from facter.models.ranker import Ranker
from facter.models.embedder import TextEmbedder
from facter.models.generator import Generator
from facter.eval.catalogue_map import CatalogueMapper
from facter.eval.prediction import predict_batch_rank, predict_batch_open, build_title_to_mid_dict
from facter.fairness.context_encoder import ContextEncoder
from facter.fairness.neighbors import CrossGroupNeighborIndex, NeighborConfig
from facter.fairness.scoring import NonconformityScorer, ScoreConfig, item_text
from facter.fairness.conformal import conformal_quantile


@dataclass(frozen=True)
class OfflineCalibConfig:
    """Configure the offline calibration phase.

    Attributes:
        alpha (float): Miscoverage level used for the conformal quantile.
        lambda_fairness (float): Fairness penalty weight passed to the scoring
            configuration.
        tau_rho (float): Context-similarity threshold used by the neighbor index
            and scoring.
        tau_x_l2 (float | None): Optional L2 radius used when constructing the
            neighborhood (see ``NeighborConfig``).
        protected_cols (Tuple[str, ...]): Protected attribute columns used to
            define cross-group neighborhoods.
        top_k_neighbors (int | None): Optional top-k storage for neighbor
            relationships.
    """

    alpha: float = 0.10
    lambda_fairness: float = 0.7
    tau_rho: float = 0.90
    tau_x_l2: float | None = None
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    top_k_neighbors: int | None = None  # optional top-k storage in W


@dataclass(frozen=True)
class OfflineCalibrationResult:
    """Collect artifacts and outputs from offline calibration.

    Attributes:
        cal_df (pd.DataFrame): Calibration DataFrame augmented with predictions
            and metadata.
        cal_context_emb (np.ndarray): Context embeddings computed from
            ``cal_df``.
        cal_pred_mid (np.ndarray): Predicted item IDs.
        cal_pred_text (list[str]): Predicted output text used for embedding.
        cal_pred_emb (np.ndarray): Embeddings of predicted outputs.
        scores_S (np.ndarray): Nonconformity scores.
        scores_d (np.ndarray): Predictive error component produced by the
            scorer.
        scores_delta (np.ndarray): Fairness penalty component produced by the
            scorer.
        q_alpha0 (float): Initial conformal threshold computed from ``scores_S``.
    """

    cal_df: pd.DataFrame
    cal_context_emb: np.ndarray
    cal_pred_mid: np.ndarray
    cal_pred_text: list[str]
    cal_pred_emb: np.ndarray
    scores_S: np.ndarray
    scores_d: np.ndarray
    scores_delta: np.ndarray
    q_alpha0: float


class OfflineCalibrator:
    """Run offline calibration to compute the initial conformal threshold.

    This class orchestrates context encoding, prediction, neighborhood
    construction, scoring, and conformal quantile computation.

    Paper alignment:
        - Fairness-aware nonconformity scoring (Paper: Sec. 3.2 / Eq. 5)
        - Conformal quantile threshold $Q_\\alpha(0)$ (Paper: Sec. 3.2 / Eq. 6)

    TODO(doc): The module uses a neighbor index built from encoded contexts;
    link to the precise paper definition if the codebase's neighborhood
    construction exactly matches Eq. 4.
    """

    def __init__(
        self,
        ranker: Ranker,
        embedder: TextEmbedder,
        context_encoder: ContextEncoder,
        cfg: OfflineCalibConfig,
    ):
        """Initialize the offline calibrator.

        Args:
            ranker (Ranker): Ranker used when ``predict_mode='rank'``.
            embedder (TextEmbedder): Embedder used by the
                ``NonconformityScorer``.
            context_encoder (ContextEncoder): Encoder used to embed user
                contexts.
            cfg (OfflineCalibConfig): Offline calibration configuration.
        """
        self.ranker = ranker
        self.embedder = embedder
        self.context_encoder = context_encoder
        self.cfg = cfg

    def run(
        self,
        cal_df: pd.DataFrame,
        item_db: Dict[int, Dict[str, str]],
        system_prompt: Optional[str] = None,
        progress: bool = False,
        predict_mode: str = "rank",  # "rank" | "open"
        generator: Optional[Generator] = None,
        prompt_cfg: Optional[PromptConfig] = None,
        catalogue_mapper: Optional[CatalogueMapper] = None,
        min_sim: float = 0.65,
    ) -> OfflineCalibrationResult:
        """Run offline calibration supporting rank and open-generation modes.

        This method:
        1) encodes contexts,
        2) generates predictions via the prediction utilities,
        3) fits a cross-group neighbor index, and
        4) computes nonconformity scores and the conformal quantile.

        Args:
            cal_df (pd.DataFrame): Calibration split input.
            item_db (Dict[int, Dict[str, str]]): Item metadata mapping.
            system_prompt (Optional[str]): Optional system prompt passed through
                to prediction.
            progress (bool): Whether to display a tqdm progress bar.
            predict_mode (str): Prediction mode ("rank" or "open").
            generator (Optional[Generator]): Generator required when
                ``predict_mode='open'``.
            prompt_cfg (Optional[PromptConfig]): Prompt configuration required
                when ``predict_mode='open'``.
            catalogue_mapper (Optional[CatalogueMapper]): Catalogue mapper
                passed through to open prediction.
            min_sim (float): Similarity threshold passed through to open
                prediction.

        Returns:
            OfflineCalibrationResult: Calibration outputs and artifacts.

        Raises:
            ValueError: If ``predict_mode`` is ``'open'`` and ``generator`` or
                ``prompt_cfg`` is not provided.
            ValueError: If ``predict_mode`` is not one of ``'rank'`` or
                ``'open'``.
        """
        df = cal_df.reset_index(drop=True).copy()
        df["group_attrs"] = df[list(self.cfg.protected_cols)].astype(str).agg("_".join, axis=1)

        # 1) Enc(x)
        context_emb = self.context_encoder.encode_df(df)  # [N,D] normalized

        # 2) Predict using unified prediction module
        title_to_mid: Optional[Dict[str, int]] = None
        if predict_mode == "rank":
            pred_result = predict_batch_rank(df, self.ranker, item_db, system_prompt, progress)
        elif predict_mode == "open":
            if generator is None or prompt_cfg is None:
                raise ValueError("open mode requires generator and prompt_cfg")
            pred_result = predict_batch_open(df, generator, item_db, prompt_cfg, system_prompt,
                                             catalogue_mapper, title_to_mid, min_sim, progress)
        else:
            raise ValueError("predict_mode must be 'rank' or 'open'")

        pred_mids = np.array(pred_result.pred_mids, dtype=np.int64)

        # Store for logging / reuse
        df["system_prompt"] = system_prompt
        df["ranker_response"] = pred_result.model_responses
        df["pred_mid"] = pred_mids
        df["pred_text"] = pred_result.pred_texts
        # In open mode, persist fraction of generated titles mapped to catalog
        if predict_mode == "open":
            df["valid_at_k"] = pred_result.valid_at_k_list

        # 3) Fit neighbor index
        ncfg = NeighborConfig(
            protected_cols=self.cfg.protected_cols,
            tau_rho=self.cfg.tau_rho,
            tau_x_l2=self.cfg.tau_x_l2,
            top_k=self.cfg.top_k_neighbors,
        )
        nidx = CrossGroupNeighborIndex(ncfg)
        nidx.fit(df, context_emb)

        # 4) Compute S_i using pred_text in open mode
        scfg = ScoreConfig(
            lambda_fairness=self.cfg.lambda_fairness, tau_rho=self.cfg.tau_rho
        )
        scorer = NonconformityScorer(self.embedder, scfg)

        if predict_mode == "rank":
            S, d, delta, pred_emb = scorer.compute(
                df, pred_mid_col="pred_mid", item_db=item_db, neighbor_index=nidx
            )
        else:
            S, d, delta, pred_emb = scorer.compute(
                df,
                pred_mid_col=None,
                item_db=item_db,
                neighbor_index=nidx,
                pred_text_col="pred_text",
            )

        # 5) Quantile
        q0 = conformal_quantile(S, alpha=self.cfg.alpha)

        return OfflineCalibrationResult(
            cal_df=df,
            cal_context_emb=context_emb,
            cal_pred_mid=pred_mids,
            cal_pred_text=pred_result.pred_texts,
            cal_pred_emb=pred_emb,
            scores_S=S,
            scores_d=d,
            scores_delta=delta,
            q_alpha0=float(q0),
        )
