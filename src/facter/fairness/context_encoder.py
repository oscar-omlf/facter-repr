"""Encode user context into embeddings for fairness-aware evaluation.

This module defines a lightweight context encoder that turns a user's interaction
history into a text prompt and then embeds it using a provided :class:`TextEmbedder`.

The implementation currently encodes only the contents of the ``history_titles``
field and does not incorporate demographic columns.

In the paper's notation, this corresponds to constructing a context embedding
$e_x = \mathrm{Enc}(x)$ as part of offline preprocessing. (Paper: Sec. 3.2 / Stage A)
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from facter.models.embedder import TextEmbedder


@dataclass(frozen=True)
class ContextEncodingConfig:
    """Configure how a user context is converted to text for embedding.

    The configuration affects how many history items are included when producing
    the text passed to the embedder.

    The paper describes encoding a user context $x$ into an embedding via
    $\mathrm{Enc}(x)$. This config controls one specific implementation detail of
    that step: how much interaction history is included. (Paper: Sec. 3.2 / Stage A)

    TODO(doc): clarify whether "most recent" ordering is always guaranteed by
    the upstream data pipeline.

    Attributes:
        max_history_items (int): Maximum number of history titles to include.
            If non-positive, no history titles are included.
    """

    max_history_items: int = 5


class ContextEncoder:
    def __init__(self, embedder: TextEmbedder, cfg: ContextEncodingConfig):
        """Initialize the context encoder.

        Args:
            embedder (TextEmbedder): Embedder used to convert constructed text
                into vectors.
            cfg (ContextEncodingConfig): Configuration controlling text
                construction.
        """
        self.embedder = embedder
        self.cfg = cfg

    def _context_text(self, history_titles: Sequence[str]) -> str:
        """Format history titles into a single text block.

        Args:
            history_titles (Sequence[str]): Sequence of previously interacted
                item titles.

        Returns:
            str: Multi-line text representation of the history.
        """
        if self.cfg.max_history_items <= 0:
            hist = []
        else:
            hist = list(history_titles)[-self.cfg.max_history_items :]

        lines = ["History (most recent last):"]
        lines.extend([f"- {t}" for t in hist])
        return "\n".join(lines)


    def encode_df(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a dataframe of examples into context embeddings.

        Args:
            df (pd.DataFrame): Input dataframe. Must include a
                ``history_titles`` column. The encoder does not consume protected
                attributes from ``df``.

        Returns:
            np.ndarray: Embeddings returned by the underlying embedder.

        Raises:
            KeyError: If ``history_titles`` is missing from ``df``.
        """
        texts: List[str] = [self._context_text(h) for h in df["history_titles"].tolist()]
        return self.embedder.encode_texts(texts)
