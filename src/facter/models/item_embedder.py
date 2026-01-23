"""
Specialized embedder for item database with pre-computed embeddings.
Avoids redundant embedding computation for items.
"""
from typing import Dict, List, Sequence
import numpy as np
from facter.models.embedder import TextEmbedder


def item_text(mid: int, item_db: Dict[int, Dict[str, str]]) -> str:
    """Generate canonical text representation for an item."""
    info = item_db.get(int(mid))
    if info is None:
        return f"UNKNOWN_ITEM_{mid}"
    title = info.get("title", f"UNKNOWN_ITEM_{mid}")
    genres = info.get("genres", "")
    return f"{title} :: {genres}" if genres else title


class ItemEmbedder:
    """
    Pre-computes and caches embeddings for all items in the database.
    Provides O(1) lookup for item embeddings by mid.
    """
    
    def __init__(self, embedder: TextEmbedder, item_db: Dict[int, Dict[str, str]]):
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
        """Get embedding for a single item by mid."""
        mid = int(mid)
        if mid not in self._item_embeddings:
            # Fall back to on-demand encoding for unknown items
            text = item_text(mid, self.item_db)
            emb = self.embedder.encode_text(text)
            self._item_embeddings[mid] = emb
        return self._item_embeddings[mid]
    
    def get_embeddings(self, mids: Sequence[int]) -> np.ndarray:
        """Get embeddings for multiple items. Returns shape [N, D]."""
        embeddings = [self.get_embedding(int(mid)) for mid in mids]
        return np.stack(embeddings, axis=0)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Fall back to text embedder for arbitrary text (e.g., generated titles)."""
        return self.embedder.encode_text(text)
    
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Fall back to text embedder for arbitrary texts."""
        return self.embedder.encode_texts(texts)
