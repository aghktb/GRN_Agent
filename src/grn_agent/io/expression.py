"""Load expression matrices for ingest / features (numpy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_expression_npy(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def load_expression_csv(path: str | Path) -> np.ndarray:
    import pandas as pd

    return pd.read_csv(path, index_col=0).to_numpy(dtype=np.float64)


def save_expression_npy(path: str | Path, matrix: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
