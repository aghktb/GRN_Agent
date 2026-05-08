from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from grn_agent.schemas import ScoredEdge


def calibrate_edges_temperature(edges: list[ScoredEdge], temperature: float = 1.2) -> list[ScoredEdge]:
    """Temperature scaling on the logit of p(present)."""
    ts = max(1e-4, float(temperature))
    out: list[ScoredEdge] = []
    for e in edges:
        p = float(np.clip(e.p_present, 1e-6, 1.0 - 1e-6))
        logit = float(np.log(p / (1.0 - p)))
        logit /= ts
        p2 = float(1.0 / (1.0 + np.exp(-logit)))
        out.append(
            e.model_copy(
                update={
                    "p_present": p2,
                    "logit": logit,
                    "confidence_score": p2,
                }
            )
        )
    return out


def fit_isotonic_on_confidence(val_edges: list[ScoredEdge], val_correct: np.ndarray) -> IsotonicRegression:
    """val_correct: binary 1 if prediction matches label."""
    conf = np.array([e.confidence_score for e in val_edges])
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(conf, val_correct)
    return ir


def apply_isotonic(ir: IsotonicRegression, edges: list[ScoredEdge]) -> list[ScoredEdge]:
    out = []
    for e in edges:
        c = float(ir.transform([e.confidence_score])[0])
        out.append(e.model_copy(update={"confidence_score": c, "p_present": c}))
    return out
