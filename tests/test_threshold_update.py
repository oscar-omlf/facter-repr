from facter.fairness.threshold_update import update_threshold_theorem2


def test_threshold_updates_only_when_s_exceeds_q():
    q0 = 1.0
    gamma = 0.9

    # no update if s <= q
    q1 = update_threshold_theorem2(q_t=q0, s_t=0.5, gamma=gamma)
    assert q1 == q0

    # update if s > q
    q2 = update_threshold_theorem2(q_t=q0, s_t=2.0, gamma=gamma)
    assert abs(q2 - (0.9 * 1.0 + 0.1 * 2.0)) < 1e-9
