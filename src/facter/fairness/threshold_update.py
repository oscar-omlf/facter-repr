from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdUpdateConfig:
    gamma: float = 0.95


def update_threshold_theorem2(q_t: float, s_t: float, gamma: float) -> float:
    """
    Appendix 1.2 Theorem 2 piecewise rule (as you noted):
      if s_t > q_t: q_{t+1} = gamma q_t + (1-gamma) s_t
      else:         q_{t+1} = q_t

    This updates Q when the score is ABOVE the threshold.
    """
    if s_t > q_t:
        return gamma * q_t + (1.0 - gamma) * s_t
    return q_t
