import numpy as np
import pandas as pd

from facter.eval.baselines import run_zero_shot, evaluate_zero_shot
from facter.fairness.context_encoder import ContextEncoder, ContextEncodingConfig


class DummyRanker:
    def rank(self, prompt_rank, candidate_titles, system_prompt=None):
        return list(range(len(candidate_titles))), "raw"


class DummyEmbedder:
    def encode_texts(self, texts):
        # return deterministic 2D embeddings
        out = []
        for t in texts:
            out.append(np.array([float(len(t) % 3), 1.0], dtype=np.float32))
        return np.stack(out, axis=0)


class DummyNeighborIndex:
    def fit(self, df, context_emb):
        return None


def test_zero_shot_runner_and_eval():
    df = pd.DataFrame(
        {
            "prompt_rank": ["p1", "p2"],
            "candidate_titles": [["A", "B", "C"], ["A", "B", "C"]],
            "candidate_mids": [[1, 2, 3], [1, 2, 3]],
            "target_mid": [1, 2],
            "history_titles": [["x"], ["y"]],  # required by ContextEncoder
        }
    )

    r = DummyRanker()
    ctx = ContextEncoder(DummyEmbedder(), ContextEncodingConfig(max_history_items=0))  # type: ignore[arg-type]
    nidx = DummyNeighborIndex()

    out = run_zero_shot(
        df.copy(),
        ranker=r,
        scorer=None,
        context_encoder=ctx,
        neighbor_index=nidx,
        item_db={},
        predict_mode="rank",
        k=2,
        system_prompt=None,
    )

    # ranked_mids contains full permutation; check top-k matches expectation
    assert [xs[:2] for xs in out["ranked_mids"].tolist()] == [[1, 2], [1, 2]]

    metrics = evaluate_zero_shot(out, k=2)
    assert "Recall@2" in metrics and "NDCG@2" in metrics
