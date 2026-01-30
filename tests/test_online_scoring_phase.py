import numpy as np
import pandas as pd

from facter.fairness.context_encoder import ContextEncoder, ContextEncodingConfig
from facter.fairness.online import OnlineScorer, OnlineScoringConfig, CalibrationArtifacts


class DummyEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode_texts(self, texts):
        return np.stack([self.mapping[t] for t in texts], axis=0).astype(np.float32)

    def encode_text(self, text):
        return self.encode_texts([text])[0]


class DummyItemEmbedder:
    def __init__(self, mid_to_vec):
        self.mid_to_vec = mid_to_vec

    def get_embedding(self, mid: int):
        return self.mid_to_vec[int(mid)]


def test_online_scorer_returns_expected_components():
    mapping = {
        "History (most recent last):": np.array([1.0, 0.0], dtype=np.float32),
        "A :: G": np.array([1.0, 0.0], dtype=np.float32),
        "B :: G": np.array([0.0, 1.0], dtype=np.float32),
    }
    emb = DummyEmbedder(mapping)
    ctx = ContextEncoder(emb, ContextEncodingConfig(max_history_items=0))  # type: ignore[arg-type]

    # Add group_attrs because OnlineScorer now uses it for cross-group selection
    cal_df = pd.DataFrame(
        {
            "gender": ["M", "F"],
            "age": [25, 25],
            "occupation": [1, 1],
            "history_titles": [["x"], ["y"]],
            "group_attrs": ["M", "F"],
        }
    )
    cal_context = np.stack(
        [mapping["History (most recent last):"], mapping["History (most recent last):"]]
    )
    cal_pred = np.stack([mapping["A :: G"], mapping["B :: G"]])

    cal = CalibrationArtifacts(
        cal_df=cal_df,
        cal_context_emb=cal_context,
        cal_pred_emb=cal_pred,
        q_alpha0=0.0,
    )

    item_emb = DummyItemEmbedder({1: mapping["A :: G"], 2: mapping["B :: G"]})

    scorer = OnlineScorer(
        embedder=emb, 
        item_embedder=item_emb,
        context_encoder=ctx,
        cfg=OnlineScoringConfig(tau_rho=0.0, lambda_fairness=1.0),
    )

    row = pd.Series(
        {
            "gender": "M",
            "age": 25,
            "occupation": 1,
            "history_titles": ["z"],
            "group_attrs": "M",
        }
    )
    item_db = {1: {"title": "A", "genres": "G"}, 2: {"title": "B", "genres": "G"}}

    # predict A, reference A => d=0
    s, d, delta = scorer.score_one(row, pred_mid=1, item_db=item_db, cal=cal, target_mid=1)
    assert abs(d - 0.0) < 1e-6
    # cross-group neighbor exists (the F row), so delta = ||A - B|| = sqrt(2)
    assert abs(delta - np.sqrt(2)) < 1e-6
    assert abs(s - np.sqrt(2)) < 1e-6
