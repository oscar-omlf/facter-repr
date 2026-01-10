import numpy as np

from facter.fairness.conformal import conformal_quantile


def test_conformal_quantile_order_statistic():
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    # n=4
    # alpha=0.25 => k=ceil(0.75*(5))=ceil(3.75)=4 => threshold = max score = 0.4
    q = conformal_quantile(scores, alpha=0.25)
    assert np.isclose(q, 0.4)

    # alpha=0.5 => k=ceil(0.5*(5))=ceil(2.5)=3 => 0.3
    q2 = conformal_quantile(scores, alpha=0.5)
    assert np.isclose(q2, 0.3)
