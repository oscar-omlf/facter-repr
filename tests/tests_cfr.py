import numpy as np
import pandas as pd

from facter.eval.counterfactual import compute_cfr, CFRConfig
from facter.data.prompts import PromptConfig, build_ranking_prompt


class DummyRanker:
    """
    If prompt contains "gender: F", reverse the candidate order; otherwise keep order.
    """
    def rank(self, prompt_rank, candidate_titles, system_prompt=None):
        n = len(candidate_titles)
        if "gender: F" in prompt_rank:
            return list(reversed(range(n)))
        return list(range(n))


class DummyEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode_texts(self, texts):
        return np.stack([self.mapping[t] for t in texts], axis=0).astype(np.float32)


def test_cfr_increases_when_flip_changes_ranking():
    # Two items with orthogonal embeddings
    item_db = {
        1: {"title": "A", "genres": "G"},
        2: {"title": "B", "genres": "G"},
    }
    mapping = {
        "A :: G": np.array([1.0, 0.0], dtype=np.float32),
        "B :: G": np.array([0.0, 1.0], dtype=np.float32),
    }
    emb = DummyEmbedder(mapping)
    ranker = DummyRanker()

    pcfg = PromptConfig(k_recs=10, include_demographics=True, domain="movie")

    row = {
        "gender": "M",
        "age": 25,
        "occupation": 1,
        "history_titles": ["X", "Y", "Z", "W", "V"],
    }
    candidate_titles = ["A", "B"]
    prompt_m = build_ranking_prompt(row, candidate_titles, pcfg)

    df = pd.DataFrame({
        "gender": ["M"],
        "age": [25],
        "occupation": [1],
        "history_titles": [["X", "Y", "Z", "W", "V"]],
        "candidate_titles": [candidate_titles],
        "candidate_mids": [[1, 2]],
        "prompt_rank": [prompt_m],
        "target_mid": [1],
    })

    cfr = compute_cfr(
        df=df,
        ranker=ranker,
        embedder=emb,  # type: ignore[arg-type]
        item_db=item_db,
        prompt_cfg=pcfg,
        cfg=CFRConfig(flip_attr="gender", k=2),
        system_prompt=None,
    )

    # Flip M->F reverses ranking => output embedding changes => distance > 0
    assert cfr > 0.0
