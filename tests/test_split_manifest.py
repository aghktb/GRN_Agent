import pandas as pd
import pytest

from grn_agent.eval.splits import validate_fold_no_leakage
from grn_agent.io.split_manifest import load_split_manifest
from grn_agent.schemas import SplitStrategy


def test_load_split_manifest_csv(tmp_path):
    p = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {
                "split_name": "leave_one_tf_out",
                "fold_id": "f1",
                "subset": "train",
                "source_tf": "A",
                "target_gene": "B",
                "cell_type": "x",
                "species": "human",
                "tf_frequency_bucket": "high",
            }
        ]
    ).to_csv(p, index=False)
    m = load_split_manifest(p)
    assert len(m.rows) == 1
    assert m.rows[0].split_name.value == "leave_one_tf_out"


def test_validate_fold_detects_loto_tf_overlap(tmp_path):
    p = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {"split_name": "leave_one_tf_out", "fold_id": "f1", "subset": "train", "source_tf": "A", "target_gene": "B"},
            {"split_name": "leave_one_tf_out", "fold_id": "f1", "subset": "test", "source_tf": "A", "target_gene": "C"},
        ]
    ).to_csv(p, index=False)
    m = load_split_manifest(p)
    with pytest.raises(ValueError):
        validate_fold_no_leakage(m, SplitStrategy.leave_one_tf_out, "f1")

