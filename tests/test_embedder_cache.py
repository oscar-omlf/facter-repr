from pathlib import Path

import numpy as np
import torch

from facter.models.embedder import EmbedderConfig, TextEmbedder


def test_embedder_shapes_and_cache(tmp_path: Path):
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
    assert v1.shape[1] > 100

    v2 = emb.encode_texts(texts)
    assert np.allclose(v1, v2)

    if cfg.cache_dir.exists():
        list(cfg.cache_dir.rglob("*"))
