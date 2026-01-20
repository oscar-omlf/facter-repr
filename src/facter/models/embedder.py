import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = "paraphrase-mpnet-base-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    normalize: bool = True
    cache_dir: Path = Path("data/cache/embeddings")
    progress: bool = False


class TextEmbedder:
    """
    SentenceTransformer wrapper that optimizes for both speed (batching)
    and reproducibility (disk caching).
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
        self.model.eval()

        # Cache manifest: maps sha256(text) -> filename (npz)
        self._manifest_path = self.cfg.cache_dir / "manifest.json"
        self._manifest: Dict[str, str] = {}

        if self._manifest_path.exists():
            with self._manifest_path.open("r", encoding="utf-8") as f:
                self._manifest = json.load(f)

    def _save_manifest(self) -> None:
        with self._manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, sort_keys=True)

    def encode_text(self, text: str) -> torch.Tensor:
        """Returns [D] tensor."""
        # Wrap in list to reuse the robust batch logic
        return self.encode_texts([text])[0]

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """
        Returns embeddings as FloatTensor of shape [N, D] on self.cfg.device.

        Optimization:
        1. Check memory/disk cache for all texts.
        2. Identify texts that are missing.
        3. Encode ONLY the missing texts in a GPU batch (fast).
        4. Save missing texts to disk (reproducibility).
        5. Reassemble and return.
        """
        if not texts:
            return torch.empty(0, device=self.cfg.device)

        keys = [_sha256_text(t) for t in texts]

        # 1. Separate Hits (Cache) and Misses (Compute)
        cached_tensors = {}
        missing_indices = []
        missing_texts = []

        # Optimization: Collect all cached numpy arrays to move to GPU in one batch
        hit_indices = []
        hit_arrays = []

        # Check manifest first
        for i, k in enumerate(
            tqdm(
                keys,
                total=len(keys),
                desc="Checking cached embeddings",
                leave=False,
                disable=not self.cfg.progress,
            )
        ):
            fname = self._manifest.get(k)
            found = False
            if fname:
                fpath = self.cfg.cache_dir / fname
                if fpath.exists():
                    try:
                        # Load from disk
                        arr = np.load(fpath)["emb"]
                        # Convert to tensor immediately
                        hit_indices.append(i)
                        hit_arrays.append(arr)
                        # cached_tensors[i] = torch.from_numpy(arr).to(self.cfg.device)
                        found = True
                    except Exception:
                        pass  # File corrupt, treat as missing

            if not found:
                missing_indices.append(i)
                missing_texts.append(texts[i])

        # Move all hits to GPU in one go
        if hit_arrays:
            # Stack numpy first (fast CPU op)
            hits_block = np.stack(hit_arrays)

            # Single transfer to device
            hits_tensor = torch.from_numpy(hits_block).to(self.cfg.device)

            # Distribute back to the dict
            for local_idx, global_idx in enumerate(hit_indices):
                cached_tensors[global_idx] = hits_tensor[local_idx]

        # 2. Compute Misses in Batch (GPU Optimization)
        if missing_texts:
            with torch.no_grad():
                # SentenceTransformer handles internal batching if list is large
                # convert_to_tensor=True keeps it on GPU, avoiding CPU transfer
                new_embs = self.model.encode(
                    missing_texts,
                    batch_size=self.cfg.batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=self.cfg.normalize,
                    show_progress_bar=self.cfg.progress,
                )

            # 3. Save Misses to Disk (Reproducibility)
            # We must move to CPU/Numpy to save to disk, but keep GPU tensor for return
            new_embs_np = new_embs.cpu().numpy()

            for local_idx, global_idx in enumerate(missing_indices):
                k = keys[global_idx]
                vec_np = new_embs_np[local_idx]
                vec_torch = new_embs[local_idx]

                # Store in our result map
                cached_tensors[global_idx] = vec_torch

                # Write to disk
                fname = f"{k}.npz"
                np.savez_compressed(self.cfg.cache_dir / fname, emb=vec_np)
                self._manifest[k] = fname

            self._save_manifest()

        # 4. Reassemble in original order
        # We construct the list of tensors in the order of input `texts`
        final_list = [cached_tensors[i] for i in range(len(texts))]

        # Stack into a single tensor [N, D]
        return torch.stack(final_list)
