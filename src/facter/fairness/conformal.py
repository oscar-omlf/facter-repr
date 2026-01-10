import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = 0.10  # miscoverage level


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Implements Eq.(6) in an order-statistic form:
      Q = S_(ceil((1-alpha)(n+1)))
    with clamping to [1, n].
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array")
    n = len(scores)
    if n == 0:
        raise ValueError("scores is empty")

    sorted_scores = np.sort(scores)
    k = int(math.ceil((1.0 - alpha) * (n + 1)))
    k = max(1, min(k, n))
    return float(sorted_scores[k - 1])


class ConformalQuantileCalibrator:
    def __init__(self, cfg: ConformalConfig):
        self.cfg = cfg

    def fit(self, scores: np.ndarray) -> float:
        return conformal_quantile(scores, self.cfg.alpha)
