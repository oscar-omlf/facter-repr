"""Update online conformal threshold after detecting a fairness violation.

This module contains a small helper used by the online monitoring loop to adjust
the current conformal threshold $Q_\alpha(t)$ when a new fairness score exceeds
it.

The update rule implemented in :func:`update_threshold_theorem2` matches the
main-paper threshold adaptation step (Paper: Eq. 11) and the convergence analysis
in the appendix (Paper: Thm A.2).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdUpdateConfig:
    """Store hyperparameters for threshold updates.

    Attributes:
        gamma (float): Exponential decay factor $\gamma \in (0, 1)$ controlling how
            aggressively the threshold is updated after a violation.
    """

    gamma: float = 0.95


def update_threshold_theorem2(q_t: float, s_t: float, gamma: float) -> float:
    """Update the threshold when a new score exceeds the current value.

    This function applies a piecewise update that only changes the threshold when
    the new fairness score exceeds it.

    (Paper: Eq. 11 / Thm A.2)

    Args:
        q_t (float): Current threshold value $q_t$.
        s_t (float): Current fairness/nonconformity score $s_t$.
        gamma (float): Exponential decay factor $\gamma$.

    Returns:
        float: The updated threshold value.
    """
    if s_t > q_t:
        return gamma * q_t + (1.0 - gamma) * s_t
    return q_t
