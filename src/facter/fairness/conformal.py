"""Compute conformal quantile thresholds from nonconformity scores.

This module implements a simple conformal quantile rule used to convert a set of
calibration nonconformity scores into a scalar threshold $Q_\alpha(0)$.

The quantile computation in :func:`conformal_quantile` follows the order
statistic form described in the paper for the offline calibration threshold
(Paper: Sec. 3.2 / Eq. 6).
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ConformalConfig:
    """Configure conformal quantile computation.

    Attributes:
        alpha (float): Miscoverage level $\\alpha$.
    """

    alpha: float = 0.10  # miscoverage level


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile threshold for a set of scores.

    This function sorts the calibration scores and selects the order statistic
    at index $\\lceil (1-\\alpha)(n+1) \\rceil$, clamped to the range ``[1, n]``.

    (Paper: Sec. 3.2 / Eq. 6)

    Args:
        scores (np.ndarray): One-dimensional array of calibration scores.
        alpha (float): Miscoverage level $\\alpha$.

    Returns:
        float: Selected quantile value.

    Raises:
        ValueError: If ``scores`` is not 1D or is empty.
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
        """Initialize the calibrator.

        Args:
            cfg (ConformalConfig): Configuration containing the miscoverage
                level.
        """
        self.cfg = cfg

    def fit(self, scores: np.ndarray) -> float:
        """Compute the conformal quantile threshold for the provided scores.

        Args:
            scores (np.ndarray): One-dimensional array of calibration scores.

        Returns:
            float: Conformal quantile threshold.

        Raises:
            ValueError: If ``scores`` is not 1D or is empty.
        """
        return conformal_quantile(scores, self.cfg.alpha)
