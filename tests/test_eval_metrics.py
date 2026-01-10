from facter.eval.metrics import ndcg_at_k, recall_at_k, mean_recall_ndcg, count_violations


def test_recall_at_k_single_relevant():
    ranked = [5, 3, 2, 9]
    assert recall_at_k(ranked, {2}, k=3) == 1.0
    assert recall_at_k(ranked, {2}, k=2) == 0.0


def test_ndcg_at_k_single_relevant():
    ranked = [5, 3, 2, 9]
    # relevant at rank 3 => DCG = 1/log2(4) = 0.5, IDCG=1
    nd = ndcg_at_k(ranked, {2}, k=4)
    assert abs(nd - 0.5) < 1e-9


def test_mean_metrics():
    ranked_lists = [[1, 2, 3], [3, 2, 1]]
    targets = [2, 2]
    out = mean_recall_ndcg(ranked_lists, targets, k=2)
    assert out["Recall@2"] == 1.0
    # ndcg: first list has target at rank2 -> 1/log2(3)=0.6309; second at rank2 -> same
    assert out["NDCG@2"] > 0.6


def test_count_violations():
    scores = [0.1, 0.2, 0.5]
    assert count_violations(scores, q_alpha=0.2) == 1  # strictly >
