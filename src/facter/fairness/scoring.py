from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from facter.models.embedder import TextEmbedder
from facter.fairness.neighbors import CrossGroupNeighborIndex

if TYPE_CHECKING:
    from facter.models.item_embedder import ItemEmbedder

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
    def __init__(self, embedder: TextEmbedder, cfg: ScoreConfig, item_embedder: Optional["ItemEmbedder"] = None):
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
        """
        Computes Eq.(5) score for each row i, supporting either:
        - ranking mode: pred_mid_col is provided (predictions are item IDs)
        - open mode: pred_text_col is provided (predictions are strings)

        Returns: (S, d, delta, pred_emb)
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

        # \Delta_i = max_{j: W_ij > τρ} ||pred_i - pred_j||_2
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
