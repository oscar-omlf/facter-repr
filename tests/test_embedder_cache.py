from pathlib import Path

import numpy as np
import torch

from facter.models.embedder import EmbedderConfig, TextEmbedder


def test_embedder_shapes_and_cache(tmp_path: Path):
    # Isolate cache directory for test
    cfg = EmbedderConfig(
        model_name="paraphrase-mpnet-base-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
        normalize=True,
        cache_dir=tmp_path / "emb_cache",
    )
    emb = TextEmbedder(cfg)

    texts = ["hello world", "goodbye world", "hello world"]
    v1 = emb.encode_texts(texts)
    assert v1.ndim == 2
    assert v1.shape[0] == 3
    assert v1.shape[1] > 100  # mpnet embeddings are high-d

    # Second call should return exactly same vectors (from cache)
    v2 = emb.encode_texts(texts)
    assert np.allclose(v1, v2)

    # Cache files should exist
    assert (cfg.cache_dir / "manifest.json").exists()
    npz_files = list(cfg.cache_dir.glob("*.npz"))
    assert len(npz_files) >= 2  # hello and goodbye


def test_embedder_single_text(tmp_path: Path):
    cfg = EmbedderConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=tmp_path / "emb_cache",
    )
    emb = TextEmbedder(cfg)
    v = emb.encode_text("one text")
    assert v.ndim == 1
    assert v.shape[0] > 100
