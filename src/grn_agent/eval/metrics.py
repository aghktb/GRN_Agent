from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    probs: (N, 3) predicted probabilities.
    labels: (N,) integer class 0,1,2
    Multiclass ECE: average L1 deviation of max-prob bin from true freq.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        ece += np.mean(m) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def auc_pr_proxy(scores: np.ndarray, binary_y: np.ndarray) -> float:
    """Unweighted AUCPR via trapezoid on precision-recall steps (simple)."""
    order = np.argsort(-scores)
    scores = scores[order]
    y = binary_y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    recall = tp / max(1, np.sum(y))
    precision = tp / np.maximum(tp + fp, 1)
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(precision, recall))


def precision_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(y[order].mean()) if k > 0 else 0.0


def recall_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    pos = float(np.sum(y))
    if pos <= 0:
        return 0.0
    order = np.argsort(-scores)[:k]
    tp = float(np.sum(y[order]))
    return tp / pos


def multiclass_aupr_macro(probs: np.ndarray, labels: np.ndarray) -> float:
    n_classes = probs.shape[1]
    vals: list[float] = []
    for c in range(n_classes):
        y = (labels == c).astype(np.int64)
        vals.append(auc_pr_proxy(probs[:, c], y))
    return float(np.mean(vals))


def multiclass_aupr_micro(probs: np.ndarray, labels: np.ndarray) -> float:
    n_classes = probs.shape[1]
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    for c in range(n_classes):
        ys.append((labels == c).astype(np.int64))
        ss.append(probs[:, c])
    y_flat = np.concatenate(ys, axis=0)
    s_flat = np.concatenate(ss, axis=0)
    return auc_pr_proxy(s_flat, y_flat)
