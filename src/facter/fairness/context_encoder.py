from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd
import torch

from facter.models.embedder import TextEmbedder


@dataclass(frozen=True)
class ContextEncodingConfig:
    max_history_items: int = 5


class ContextEncoder:
    def __init__(self, embedder: TextEmbedder, cfg: ContextEncodingConfig):
        self.embedder = embedder
        self.cfg = cfg

    def _context_text(self, history_titles: Sequence[str]) -> str:
        if self.cfg.max_history_items <= 0:
            hist = []
        else:
            hist = list(history_titles)[-self.cfg.max_history_items :]

        lines = ["History (most recent last):"]
        lines.extend([f"- {t}" for t in hist])
        return "\n".join(lines)

    def encode_batch(self, history_titles_batch: List[List[str]]) -> torch.Tensor:
        """
        Encodes a batch of history lists directly.
        """
        texts: List[str] = [self._context_text(h) for h in history_titles_batch]
        return self.embedder.encode_texts(texts)

    def encode_df(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Encodes the entire dataframe.
        Warning: For large datasets, use encode_batch within a loader loop instead.
        """
        return self.encode_batch(df["history_titles"].tolist())
