"""Provide an item-level embedding cache built on top of :class:`TextEmbedder`.

This module defines a small wrapper that pre-computes embeddings for all items
in an item metadata database and exposes fast lookup by item id (``mid``).
"""

from typing import Dict, Sequence

import numpy as np
from facter.models.embedder import TextEmbedder


def item_text(mid: int, item_db: Dict[int, Dict[str, str]]) -> str:
    """Build a canonical text representation for an item.

    The representation is derived from the item's metadata in ``item_db`` and
    is used as input to the underlying text embedder.

    Args:
        mid (int): Item id.
        item_db (Dict[int, Dict[str, str]]): Mapping from item id to a metadata
            dictionary. The metadata dictionary is expected to contain keys
            like ``"title"`` and optionally ``"genres"``.

    Returns:
        str: Title-only text if no genres are present; otherwise
        ``"{title} :: {genres}"``. For unknown ids, returns a placeholder
        string beginning with ``"UNKNOWN_ITEM_"``.
    """
    info = item_db.get(int(mid))
    if info is None:
        return f"UNKNOWN_ITEM_{mid}"
    title = info.get("title", f"UNKNOWN_ITEM_{mid}")
    genres = info.get("genres", "")
    return f"{title} :: {genres}" if genres else title


class ItemEmbedder:
    """Pre-compute and cache embeddings for items in an item database.

    The class pre-encodes all items found in ``item_db`` at construction time
    using the provided :class:`~facter.models.embedder.TextEmbedder`. It keeps
    an in-memory mapping from ``mid`` to the corresponding embedding.

    Note:
        The code creates an in-memory cache of embeddings. Depending on the
        number of items and embedding dimensionality, this may consume
        substantial memory.
    """
    
    def __init__(self, embedder: TextEmbedder, item_db: Dict[int, Dict[str, str]]):
        """Initialize the item embedder and pre-compute embeddings.

        Args:
            embedder (TextEmbedder): Underlying text embedder used to create
                item embeddings.
            item_db (Dict[int, Dict[str, str]]): Item metadata database used by
                :func:`item_text` to construct embedding inputs.
        """
        self.embedder = embedder
        self.item_db = item_db
        self._item_embeddings: Dict[int, np.ndarray] = {}
        self._precompute_all()
    
    def _precompute_all(self) -> None:
        """Pre-compute embeddings for all items in the database."""
        mids = sorted(self.item_db.keys())
        texts = [item_text(int(mid), self.item_db) for mid in mids]
        
        # Batch encode all items at once
        embeddings = self.embedder.encode_texts(texts)
        
        # Store in dict for fast lookup
        for mid, emb in zip(mids, embeddings):
            self._item_embeddings[int(mid)] = emb
    
    def get_embedding(self, mid: int) -> np.ndarray:
        """Get the embedding for a single item id.

        If the item id was not present at construction time, this method falls
        back to on-demand encoding using :func:`item_text` and the underlying
        text embedder.

        Args:
            mid (int): Item id.

        Returns:
            np.ndarray: Embedding vector for the item.
        """
        mid = int(mid)
        if mid not in self._item_embeddings:
            # Fall back to on-demand encoding for unknown items
            text = item_text(mid, self.item_db)
            emb = self.embedder.encode_text(text)
            self._item_embeddings[mid] = emb
        return self._item_embeddings[mid]
    
    def get_embeddings(self, mids: Sequence[int]) -> np.ndarray:
        """Get embeddings for multiple items.

        Args:
            mids (Sequence[int]): Item ids.

        Returns:
            np.ndarray: Stacked embeddings for the requested ids.
        """
        embeddings = [self.get_embedding(int(mid)) for mid in mids]
        return np.stack(embeddings, axis=0)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Embed an arbitrary text string using the underlying text embedder.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Embedding vector.
        """
        return self.embedder.encode_text(text)
    
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple text strings using the underlying text embedder.

        Args:
            texts (Sequence[str]): Input texts.

        Returns:
            np.ndarray: Stacked embeddings.
        """
        return self.embedder.encode_texts(texts)
