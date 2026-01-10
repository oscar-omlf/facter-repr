import pandas as pd

from facter.eval.baselines import run_zero_shot_ranking, evaluate_zero_shot


class DummyRanker:
    def rank(self, prompt_rank, candidate_titles, system_prompt=None):
        return list(range(len(candidate_titles)))


def test_zero_shot_runner_and_eval():
    df = pd.DataFrame({
        "prompt_rank": ["p1", "p2"],
        "candidate_titles": [["A", "B", "C"], ["A", "B", "C"]],
        "candidate_mids": [[1, 2, 3], [1, 2, 3]],
        "target_mid": [1, 2],
    })
    r = DummyRanker()
    ranked = run_zero_shot_ranking(df, r, k=2, system_prompt=None)
    assert ranked == [[1, 2], [1, 2]]

    metrics = evaluate_zero_shot(df, r, k=2)
    assert "Recall@2" in metrics and "NDCG@2" in metrics
