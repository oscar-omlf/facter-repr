import numpy as np
import pandas as pd

from facter.fairness.neighbors import CrossGroupNeighborIndex, NeighborConfig
from facter.fairness.scoring import NonconformityScorer, ScoreConfig


class DummyEmbedder:
    """
    Minimal stand-in for TextEmbedder: encode_texts(texts)->np.ndarray
    """
    def __init__(self, mapping):
        self.mapping = mapping

    def encode_texts(self, texts):
        return np.stack([self.mapping[t] for t in texts], axis=0).astype(np.float32)


def test_nonconformity_score_components():
    # Two items A and B in 2D unit space
    A = np.array([1.0, 0.0], dtype=np.float32)
    B = np.array([0.0, 1.0], dtype=np.float32)
    mapping = {
        "A": A,
        "B": B,
        "A :: G": A,
        "B :: G": B,
    }
    emb = DummyEmbedder(mapping)

    # Two rows; predicted equals reference => d=0
    df = pd.DataFrame({
        "target_mid": [1, 2],
        "pred_mid": [1, 2],
        "gender": ["M", "F"],
        "age": [25, 25],
        "occupation": [1, 1],
    })
    item_db = {
        1: {"title": "A", "genres": "G"},
        2: {"title": "B", "genres": "G"},
    }

    # Context embeddings: make them identical so they are neighbors, cross-group
    context = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    ncfg = NeighborConfig(protected_cols=("gender",), tau_rho=0.0)
    nidx = CrossGroupNeighborIndex(ncfg)
    nidx.fit(df, context)

    scfg = ScoreConfig(lambda_fairness=1.0, tau_rho=0.0)
    scorer = NonconformityScorer(emb, scfg)  # type: ignore[arg-type]

    S, d, delta = scorer.compute(df, "pred_mid", item_db, nidx)

    # d should be 0 because pred==ref and cosine=1
    assert np.allclose(d, np.array([0.0, 0.0], dtype=np.float32))

    # delta: distance between A and B is sqrt(2)
    assert np.isclose(delta[0], np.sqrt(2), atol=1e-6)
    assert np.isclose(delta[1], np.sqrt(2), atol=1e-6)

    # S = d + lambda*delta = sqrt(2)
    assert np.allclose(S, delta)
