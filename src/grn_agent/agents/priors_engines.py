"""
Wrappers for GRNBoost2 / GENIE3 / PIDC / SCENIC.

v0 implements train-mask-scoped **Python proxies** so splits are leakage-safe without R deps.
Replace `run_*` bodies with subprocess calls to R/java tools when available.
"""

from __future__ import annotations

import numpy as np


def _masked_pearson(expr: np.ndarray, i: int, j: int) -> float:
    """expr: cells x genes, subset already masked to train rows."""
    a = expr[:, i].astype(np.float64)
    b = expr[:, j].astype(np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def proxy_grnboost(expr_train: np.ndarray, ti: int, gi: int) -> float:
    """Map correlation to [0,1] as stand-in for GRNBoost2 edge weight."""
    r = abs(_masked_pearson(expr_train, ti, gi))
    return float(min(1.0, max(0.0, r)))


def proxy_genie3(expr_train: np.ndarray, ti: int, gi: int) -> float:
    r = abs(_masked_pearson(expr_train, ti, gi))
    return float(min(1.0, (r + 0.1) / 1.1 * 0.95))


def proxy_pidc(expr_train: np.ndarray, ti: int, gi: int) -> float:
    r = abs(_masked_pearson(expr_train, ti, gi))
    return float(min(1.0, r * 0.85))


def proxy_scenic(expr_train: np.ndarray, ti: int, gi: int) -> float:
    return float(0.5 * abs(_masked_pearson(expr_train, ti, gi)))


def bootstrap_stability(
    expr_full: np.ndarray,
    train_mask: np.ndarray,
    ti: int,
    gi: int,
    n_boot: int,
    seed: int,
) -> float:
    """Variance of proxy ensemble across bootstrap train row resamples (lower var => higher stability)."""
    rng = np.random.default_rng(seed)
    train_idx = np.where(train_mask)[0]
    if len(train_idx) < 5:
        return 0.0
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(train_idx, size=len(train_idx), replace=True)
        sub = expr_full[samp, :]
        v = (
            proxy_grnboost(sub, ti, gi)
            + proxy_genie3(sub, ti, gi)
            + proxy_pidc(sub, ti, gi)
        ) / 3.0
        vals.append(v)
    vals_arr = np.array(vals)
    return float(max(0.0, 1.0 - float(vals_arr.std())))
