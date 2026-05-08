import yaml

from grn_agent.pipeline.run import run_pipeline


def test_stop_after_evidence_graphs_no_exports(tmp_path):
    cfg = {
        "run_id": "stop_test",
        "artifact_root": str(tmp_path),
        "seed": 1,
        "stop_after": "evidence_graphs",
        "eval_track": "track1_no_literature",
        "dataset": {
            "mode": "synthetic",
            "dataset_id": "DS",
            "species": "human",
            "n_cells": 20,
            "n_genes": 15,
            "gene_prefix": "G",
            "tf_list": ["G0", "G1"],
        },
        "candidates": {"min_pearson": 0.0, "max_edges_per_tf": 3},
    }
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.dump(cfg), encoding="utf-8")
    out = run_pipeline(cpath)
    assert (out / "evidence_graphs.jsonl").is_file()
    assert not (out / "exports/network.csv").is_file()
