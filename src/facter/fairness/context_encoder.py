from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from facter.models.embedder import TextEmbedder


@dataclass(frozen=True)
class ContextEncodingConfig:
    """
    Enc(x): encode non-sensitive user context.
    For now: encode only history titles (no demographics).
    """
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


    def encode_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        df must include history_titles column (list[str]).
        Returns float32 array [N, D]. Assumes embedder returns normalized embeddings.
        """
        texts: List[str] = [self._context_text(h) for h in df["history_titles"].tolist()]
        return self.embedder.encode_texts(texts)
