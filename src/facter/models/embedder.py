import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = "paraphrase-mpnet-base-v2"
    device: str = "cuda"  # "cuda" | "cpu"
    batch_size: int = 512
    normalize: bool = True
    cache_dir: Path = Path("data/cache/embeddings")
    progress: bool = False


class TextEmbedder:
    """
    SentenceTransformer wrapper with deterministic batching + disk cache.

    API:
      encode_texts(texts: list[str]) -> np.ndarray shape [N, D]
      encode_text(text: str) -> np.ndarray shape [D]
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)

        # Cache manifest: maps sha256(text) -> filename (npz)
        self._manifest_path = self.cfg.cache_dir / "manifest.json"
        self._manifest: Dict[str, str] = {}
        if self._manifest_path.exists():
            # Keep it simple; if corrupt, user can delete cache dir
            import json
            with self._manifest_path.open("r", encoding="utf-8") as f:
                self._manifest = json.load(f)

    def _save_manifest(self) -> None:
        import json
        with self._manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, sort_keys=True)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Returns embeddings as float32 numpy array of shape [N, D].
        Uses disk cache per unique text.
        """
        # Deduplicate inputs while preserving order
        keys = [_sha256_text(t) for t in texts]

        # Load cached
        cached_vecs: Dict[str, np.ndarray] = {}
        missing_texts: List[str] = []
        missing_keys: List[str] = []

        for t, k in tqdm(zip(texts, keys), total=len(texts), desc="Loading cached embeddings", leave=False, disable=not self.cfg.progress):
            fname = self._manifest.get(k)
            if fname is None:
                missing_texts.append(t)
                missing_keys.append(k)
                continue
            fpath = self.cfg.cache_dir / fname
            if not fpath.exists():
                # Manifest entry stale
                missing_texts.append(t)
                missing_keys.append(k)
                self._manifest.pop(k, None)
                continue
            arr = np.load(fpath)["emb"]
            cached_vecs[k] = arr

        # Encode missing in batches
        if missing_texts:
            new_embs = self.model.encode(
                list(missing_texts),
                batch_size=self.cfg.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.cfg.normalize,
            ).astype(np.float32)

            for k, vec in tqdm(zip(missing_keys, new_embs), total=len(missing_keys), desc="Caching new embeddings", leave=False, disable=not self.cfg.progress):
                fname = f"{k}.npz"
                fpath = self.cfg.cache_dir / fname
                np.savez_compressed(fpath, emb=vec)
                self._manifest[k] = fname
                cached_vecs[k] = vec

            self._save_manifest()

        # Reconstruct in original order
        out = np.stack([cached_vecs[k] for k in keys], axis=0).astype(np.float32)
        return out
