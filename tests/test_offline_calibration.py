import numpy as np
import pandas as pd

from facter.fairness.context_encoder import ContextEncoder, ContextEncodingConfig
from facter.fairness.calibration import OfflineCalibrator, OfflineCalibConfig
from facter.models.ranker import Ranker


class DummyRanker:
    def rank(self, prompt_rank, candidate_titles, system_prompt=None):
        # Always rank in given order
        return list(range(len(candidate_titles))), 42


class DummyEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode_texts(self, texts):
        return np.stack([self.mapping[t] for t in texts], axis=0).astype(np.float32)


def test_offline_calibration_produces_q0():
    # Make embeddings deterministic and normalized
    v = {
        "History (most recent last):": np.array([1.0, 0.0], dtype=np.float32),
        "X :: G": np.array([1.0, 0.0], dtype=np.float32),
        "Y :: G": np.array([0.0, 1.0], dtype=np.float32),
    }
    emb = DummyEmbedder(v)

    # ContextEncoder with max_history_items=0 => constant context text
    ctx = ContextEncoder(emb, ContextEncodingConfig(max_history_items=0))  # type: ignore[arg-type]

    # calibration df
    df = pd.DataFrame(
        {
            "gender": ["M", "F"],
            "age": [25, 25],
            "occupation": [1, 1],
            "history_titles": [["a"], ["b"]],
            "candidate_mids": [[1, 2], [1, 2]],
            "candidate_titles": [["X", "Y"], ["X", "Y"]],
            "prompt_rank": ["p1", "p2"],
            "target_mid": [1, 2],
            "target_title": ["X", "Y"],
        }
    )
    item_db = {1: {"title": "X", "genres": "G"}, 2: {"title": "Y", "genres": "G"}}

    cal = OfflineCalibrator(
        ranker=DummyRanker(),
        embedder=emb,  # type: ignore[arg-type]
        context_encoder=ctx,
        cfg=OfflineCalibConfig(alpha=0.5, lambda_fairness=0.0, tau_rho=0.0),
    )
    res = cal.run(df, item_db=item_db, system_prompt=None)

    assert isinstance(res.q_alpha0, float)
    assert res.cal_pred_mid.shape[0] == 2
    assert res.cal_pred_emb.shape[0] == 2
    assert res.scores_S.shape[0] == 2
