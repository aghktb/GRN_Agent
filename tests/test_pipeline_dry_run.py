import json

import yaml

from grn_agent.models.eager.checkpoint import save_minimal_eager_for_tests
from grn_agent.pipeline.run import run_pipeline


def test_pipeline_dry_run(tmp_path):
    ckpt = tmp_path / "eager.pt"
    save_minimal_eager_for_tests(ckpt)
    cfg = {
        "run_id": "t1",
        "artifact_root": str(tmp_path),
        "seed": 3,
        "eval_track": "track1_no_literature",
        "dataset": {
            "mode": "synthetic",
            "dataset_id": "DS",
            "species": "human",
            "n_cells": 40,
            "n_genes": 20,
            "gene_prefix": "G",
            "tf_list": ["G0", "G1"],
        },
        "candidates": {"min_pearson": 0.0, "max_edges_per_tf": 5},
        "calibration": {"temperature": 1.0},
        "decode": {"max_regulators_per_target": 3, "min_confidence": 0.0},
        "scoring": {"checkpoint": str(ckpt)},
    }
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.dump(cfg), encoding="utf-8")
    out = run_pipeline(cpath)
    assert (out / "exports/network.json").is_file()
    assert (out / "exports/network.csv").is_file()
    assert (out / "evidence_graphs.jsonl").is_file()


def test_pipeline_inference_filter_restricts_to_heldout_tf(tmp_path):
    ckpt = tmp_path / "eager.pt"
    save_minimal_eager_for_tests(ckpt)
    splitp = tmp_path / "split.csv"
    splitp.write_text(
        "split_name,fold_id,subset,source_tf,target_gene\n"
        "leave_one_tf_out,f1,test,G0,G2\n"
        "leave_one_tf_out,f1,train,G1,G3\n",
        encoding="utf-8",
    )
    cfg = {
        "run_id": "heldout_only",
        "artifact_root": str(tmp_path),
        "seed": 3,
        "eval_track": "track1_no_literature",
        "dataset": {
            "mode": "synthetic",
            "dataset_id": "DS",
            "species": "human",
            "n_cells": 30,
            "n_genes": 8,
            "gene_prefix": "G",
            "tf_list": ["G0", "G1"],
        },
        "inference_filter": {
            "split_manifest": str(splitp),
            "strategy": "leave_one_tf_out",
            "fold_id": "f1",
            "subset": "test",
            "target_universe": "all_genes",
        },
        "candidates": {"min_pearson": 0.0, "max_edges_per_tf": 4},
        "calibration": {"temperature": 1.0},
        "decode": {"max_regulators_per_target": 3, "min_confidence": 0.0},
        "scoring": {"checkpoint": str(ckpt)},
    }
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.dump(cfg), encoding="utf-8")
    out = run_pipeline(cpath)
    assert (out / "exports/scored_edges.csv").is_file()
    rows = [
        json.loads(line)
        for line in (out / "evidence_graphs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert {r["edge"]["source_tf"] for r in rows} == {"G0"}
