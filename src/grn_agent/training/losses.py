"""Training losses for binary edge presence."""

from __future__ import annotations

import numpy as np


def binary_bce(prob_present: float, y_true: int, eps: float = 1e-10) -> float:
    p = float(np.clip(prob_present, eps, 1.0 - eps))
    y = 1.0 if int(y_true) == 1 else 0.0
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
