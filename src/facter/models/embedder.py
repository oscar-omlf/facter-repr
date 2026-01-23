import hashlib
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]  # Shorter hash for efficiency


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = "paraphrase-mpnet-base-v2"
    device: str = "cuda"  # "cuda" | "cpu"
    batch_size: int = 512
    normalize: bool = True
    cache_dir: Path = Path("data/cache/embeddings")
    progress: bool = False
    embeddings_per_file: int = 1000  # Batch embeddings into larger files
    max_memory_cache: int = 10000  # Max embeddings to keep in memory


class TextEmbedder:
    """
    SentenceTransformer wrapper with deterministic batching + disk cache.
    Optimized for large datasets by batching embeddings into larger files.

    API:
      encode_texts(texts: list[str]) -> np.ndarray shape [N, D]
      encode_text(text: str) -> np.ndarray shape [D]
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)

        # Cache manifest: maps sha256(text) -> (batch_file_id, index_in_batch)
        self._manifest_path = self.cfg.cache_dir / "manifest.pkl"
        self._manifest: Dict[str, Tuple[int, int]] = {}
        self._batch_counter = 0
        
        if self._manifest_path.exists():
            with self._manifest_path.open("rb") as f:
                data = pickle.load(f)
                self._manifest = data.get("manifest", {})
                self._batch_counter = data.get("batch_counter", 0)
        
        # In-memory cache for recently accessed embeddings (LRU)
        self._memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        
        # Buffer for new embeddings to batch before writing
        self._write_buffer: List[Tuple[str, np.ndarray]] = []
        self._current_batch_id = self._batch_counter

    def _save_manifest(self) -> None:
        """Save manifest to disk using pickle for efficiency."""
        with self._manifest_path.open("wb") as f:
            pickle.dump({
                "manifest": self._manifest,
                "batch_counter": self._batch_counter
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _flush_write_buffer(self) -> None:
        """Write buffered embeddings to disk in batches."""
        if not self._write_buffer:
            return
        
        batch_file = self.cfg.cache_dir / f"batch_{self._current_batch_id}.npz"
        embeddings = np.stack([emb for _, emb in self._write_buffer], axis=0)
        
        # Save batch
        np.savez_compressed(batch_file, embeddings=embeddings)
        
        # Update manifest
        for idx, (key, emb) in enumerate(self._write_buffer):
            self._manifest[key] = (self._current_batch_id, idx)
            # Add to memory cache
            self._memory_cache[key] = emb
            if len(self._memory_cache) > self.cfg.max_memory_cache:
                self._memory_cache.popitem(last=False)  # Remove oldest
        
        self._write_buffer.clear()
        self._batch_counter += 1
        self._current_batch_id = self._batch_counter
        self._save_manifest()
    
    def _load_batch_file(self, batch_id: int) -> np.ndarray:
        """Load a batch file containing multiple embeddings."""
        batch_file = self.cfg.cache_dir / f"batch_{batch_id}.npz"
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file {batch_file} not found")
        data = np.load(batch_file)
        return data["embeddings"]

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Returns embeddings as float32 numpy array of shape [N, D].
        Uses batched disk cache and memory cache for efficiency.
        """
        keys = [_sha256_text(t) for t in texts]

        # Load cached embeddings efficiently
        cached_vecs: Dict[str, np.ndarray] = {}
        missing_texts: List[str] = []
        missing_keys: List[str] = []
        
        # Group cache lookups by batch file to minimize I/O
        batch_loads: Dict[int, List[Tuple[str, int]]] = {}  # batch_id -> [(key, index), ...]

        for t, k in zip(texts, keys):
            # Check memory cache first
            if k in self._memory_cache:
                cached_vecs[k] = self._memory_cache[k]
                # Move to end (LRU)
                self._memory_cache.move_to_end(k)
                continue
            
            # Check manifest
            location = self._manifest.get(k)
            if location is None:
                missing_texts.append(t)
                missing_keys.append(k)
                continue
            
            batch_id, idx = location
            if batch_id not in batch_loads:
                batch_loads[batch_id] = []
            batch_loads[batch_id].append((k, idx))
        
        # Load batches efficiently
        for batch_id, indices in batch_loads.items():
            try:
                batch_embs = self._load_batch_file(batch_id)
                for k, idx in indices:
                    emb = batch_embs[idx]
                    cached_vecs[k] = emb
                    # Add to memory cache
                    self._memory_cache[k] = emb
                    if len(self._memory_cache) > self.cfg.max_memory_cache:
                        self._memory_cache.popitem(last=False)
            except (FileNotFoundError, KeyError, IndexError) as e:
                # Batch file corrupted or missing, recompute
                for k, idx in indices:
                    if k in keys:
                        text_idx = keys.index(k)
                        missing_texts.append(texts[text_idx])
                        missing_keys.append(k)

        # Encode missing in batches
        if missing_texts:
            new_embs = self.model.encode(
                list(missing_texts),
                batch_size=self.cfg.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.cfg.normalize,
            ).astype(np.float32)

            # Add to write buffer and cache
            for k, vec in zip(missing_keys, new_embs):
                self._write_buffer.append((k, vec))
                cached_vecs[k] = vec
                
                # Flush buffer when it reaches batch size
                if len(self._write_buffer) >= self.cfg.embeddings_per_file:
                    self._flush_write_buffer()

        # Reconstruct in original order
        out = np.stack([cached_vecs[k] for k in keys], axis=0).astype(np.float32)
        return out
    
    def flush(self) -> None:
        """Flush any remaining embeddings in the write buffer to disk."""
        self._flush_write_buffer()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush remaining embeddings."""
        self.flush()
        return False
