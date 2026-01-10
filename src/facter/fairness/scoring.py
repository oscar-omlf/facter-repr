from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from facter.models.embedder import TextEmbedder
from facter.fairness.neighbors import CrossGroupNeighborIndex


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
        pred_mid_col: str,
        item_db: Dict[int, Dict[str, str]],
        neighbor_index: CrossGroupNeighborIndex,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Eq.(5) score for each row i.

        df must include:
          - target_mid (reference y_i)
          - pred_mid_col (predicted hat{y}_i)
        """
        pred_mids = df[pred_mid_col].astype(int).tolist()
        ref_mids = df["target_mid"].astype(int).tolist()

        pred_texts = [item_text(m, item_db) for m in pred_mids]
        ref_texts = [item_text(m, item_db) for m in ref_mids]

        # Embeddings are normalized (EmbedderConfig.normalize=True), so cosine = dot
        pred_emb = self.embedder.encode_texts(pred_texts)  # [N, D]
        ref_emb = self.embedder.encode_texts(ref_texts)    # [N, D]

        # d_i = 1 - cos(pred, ref)
        cos_pr = np.sum(pred_emb * ref_emb, axis=1)
        d = (1.0 - cos_pr).astype(np.float32)

        # \Delta_i = max_{j: W_ij > tau_rho} || pred_i - pred_j ||_2
        n = len(df)
        delta = np.zeros(n, dtype=np.float32)

        for i in range(n):
            js = neighbor_index.eligible_neighbors_for_delta(i)
            if js.size == 0:
                delta[i] = 0.0
                continue
            diffs = pred_emb[js] - pred_emb[i]
            # L2 distances
            dist = np.sqrt(np.sum(diffs * diffs, axis=1))
            delta[i] = float(np.max(dist))

        S = (d + self.cfg.lambda_fairness * delta).astype(np.float32)
        return S, d, delta
