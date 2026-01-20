from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from facter.data.prompts import PromptConfig
from facter.eval.catalogue_map import CatalogueMapper
from facter.eval.prediction import (
    predict_batch_open,
    predict_batch_rank,
)
from facter.fairness.conformal import conformal_quantile
from facter.fairness.context_encoder import ContextEncoder
from facter.fairness.neighbors import CrossGroupNeighborIndex, NeighborConfig
from facter.fairness.scoring import NonconformityScorer, ScoreConfig
from facter.models.embedder import TextEmbedder
from facter.models.generator import Generator
from facter.models.ranker import Ranker
from tqdm import tqdm


@dataclass(frozen=True)
class OfflineCalibConfig:
    alpha: float = 0.10
    lambda_fairness: float = 0.7
    tau_rho: float = 0.90
    tau_x_l2: float | None = None
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    top_k_neighbors: int | None = None


@dataclass(frozen=True)
class OfflineCalibrationResult:
    cal_df: pd.DataFrame
    cal_context_emb: torch.Tensor
    cal_pred_mid: np.ndarray
    cal_pred_text: list[str]
    cal_pred_emb: torch.Tensor
    scores_S: np.ndarray
    scores_d: np.ndarray
    scores_delta: np.ndarray
    q_alpha0: float
    cal_group_ids: torch.Tensor
    group_code_map: Dict[str, int]


class OfflineCalibrator:
    def __init__(
        self,
        ranker: Ranker,
        embedder: TextEmbedder,
        context_encoder: ContextEncoder,
        cfg: OfflineCalibConfig,
    ):
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
        predict_mode: str = "rank",
        generator: Optional[Generator] = None,
        prompt_cfg: Optional[PromptConfig] = None,
        catalogue_mapper: Optional[CatalogueMapper] = None,
        min_sim: float = 0.65,
        batch_loader: Optional[DataLoader] = None,
    ) -> OfflineCalibrationResult:
        # We need the dataframe for metadata (protected groups)
        df = cal_df.reset_index(drop=True).copy()

        # --- OPTIMIZED EXECUTION ---
        # Instead of encoding context for the whole DF at once, and predicting via loader separately,
        # we iterate the loader ONCE to do both. This saves GPU memory.

        # Storage for batched results
        all_pred_mids: List[int] = []
        all_pred_texts: List[str] = []
        all_model_responses: List[str] = []
        all_valid_at_k: List[float] = []
        context_embs_list: List[torch.Tensor] = []

        # Use the loader if provided, otherwise create a lightweight iterator from DF
        if batch_loader is not None:
            iterator = batch_loader
        else:
            # Fallback: simple list of dicts if no loader
            iterator = df.to_dict("records")

        # --- SINGLE PASS LOOP ---
        if progress:
            # try to guess length
            total = len(batch_loader) if batch_loader is not None else len(df)
            iterator = tqdm(iterator, total=total, desc="Calibrating (Pred+Ctx)")

        for batch in iterator:
            # 1. Run Prediction on Batch
            if predict_mode == "rank":
                # predict_batch_rank can handle the batch dict directly
                res = predict_batch_rank(
                    [batch]
                    if not isinstance(batch, dict) or "uid" in batch
                    else batch,  # handle single vs batch
                    self.ranker,
                    item_db,
                    system_prompt,
                    progress=False,  # disable inner progress
                )
            elif predict_mode == "open":
                res = predict_batch_open(
                    [batch] if not isinstance(batch, dict) or "uid" in batch else batch,
                    generator,
                    item_db,
                    prompt_cfg,
                    system_prompt,
                    catalogue_mapper,
                    None,
                    min_sim,
                    progress=False,
                )
            else:
                raise ValueError("predict_mode must be 'rank' or 'open'")

            all_pred_mids.extend(res.pred_mids)
            all_pred_texts.extend(res.pred_texts)
            all_model_responses.extend(res.model_responses)

            if predict_mode == "open":
                all_valid_at_k.extend(res.valid_at_k_list)

            # 2. Run Context Encoding on Batch
            # Handle both collated batch (dict with list) and single item (dict)
            if isinstance(batch, dict) and "history_titles" in batch:
                h_titles = batch["history_titles"]
                # if it's a single item (list of strings), wrap in list for batch
                if (
                    isinstance(h_titles, list)
                    and len(h_titles) > 0
                    and isinstance(h_titles[0], str)
                ):
                    # Check if it is a single example history or a batch of histories
                    # In dict_collate_fn, history_titles is a List[List[str]] (batch)
                    # But if we iterate df.to_dict('records'), it is List[str] (single)
                    if batch_loader is None:
                        # Single item mode
                        emb_batch = self.context_encoder.encode_batch([h_titles])
                    else:
                        # Batch mode
                        emb_batch = self.context_encoder.encode_batch(h_titles)
                else:
                    # Likely batch mode List[List[str]]
                    emb_batch = self.context_encoder.encode_batch(h_titles)

                context_embs_list.append(emb_batch)

        pred_mids = np.array(all_pred_mids, dtype=np.int64)
        df["system_prompt"] = system_prompt
        df["ranker_response"] = all_model_responses
        df["pred_mid"] = pred_mids
        df["pred_text"] = all_pred_texts

        if predict_mode == "open":
            df["valid_at_k"] = all_valid_at_k

        context_emb = torch.cat(context_embs_list, dim=0)

        # 3. Fit Neighbor Index (GPU Optimized)
        ncfg = NeighborConfig(
            protected_cols=self.cfg.protected_cols,
            tau_rho=self.cfg.tau_rho,
            tau_x_l2=self.cfg.tau_x_l2,
            top_k=self.cfg.top_k_neighbors,
        )
        nidx = CrossGroupNeighborIndex(ncfg)
        nidx.fit(df, context_emb)

        # 4. Compute Scores
        scfg = ScoreConfig(
            lambda_fairness=self.cfg.lambda_fairness, tau_rho=self.cfg.tau_rho
        )
        scorer = NonconformityScorer(self.embedder, scfg)

        if predict_mode == "rank":
            S, d, delta, pred_emb_np = scorer.compute(
                df, pred_mid_col="pred_mid", item_db=item_db, neighbor_index=nidx
            )
        else:
            S, d, delta, pred_emb_np = scorer.compute(
                df,
                pred_mid_col=None,
                item_db=item_db,
                neighbor_index=nidx,
                pred_text_col="pred_text",
            )

        pred_emb_t = torch.tensor(
            pred_emb_np, dtype=torch.float32, device=self.embedder.cfg.device
        )

        q0 = conformal_quantile(S, alpha=self.cfg.alpha)

        # 5. Group ID Mapping
        a_series = df[list(self.cfg.protected_cols)].astype(str).agg("_".join, axis=1)
        unique_groups = sorted(a_series.unique())
        group_code_map = {g: i for i, g in enumerate(unique_groups)}
        group_ids_np = a_series.map(group_code_map).to_numpy(dtype=np.int64)
        cal_group_ids = torch.tensor(
            group_ids_np, dtype=torch.long, device=self.embedder.cfg.device
        )

        return OfflineCalibrationResult(
            cal_df=df,
            cal_context_emb=context_emb,
            cal_pred_mid=pred_mids,
            cal_pred_text=all_pred_texts,
            cal_pred_emb=pred_emb_t,
            scores_S=S,
            scores_d=d,
            scores_delta=delta,
            q_alpha0=float(q0),
            cal_group_ids=cal_group_ids,
            group_code_map=group_code_map,
        )
