import numpy as np

from grn_agent.eval.metrics import multiclass_aupr_macro, multiclass_aupr_micro, recall_at_k


def test_recall_at_k_bounds():
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
    y = np.array([1, 0, 1, 0], dtype=np.int64)
    r = recall_at_k(scores, y, 2)
    assert 0.0 <= r <= 1.0


def test_multiclass_aupr_outputs_valid_range():
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 2, 0], dtype=np.int64)
    macro = multiclass_aupr_macro(probs, labels)
    micro = multiclass_aupr_micro(probs, labels)
    assert 0.0 <= macro <= 1.0
    assert 0.0 <= micro <= 1.0

