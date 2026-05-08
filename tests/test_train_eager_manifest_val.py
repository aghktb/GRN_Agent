from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch


def _graph(tf: str, tg: str, corr: float) -> dict:
    return {
        "context": {
            "context_id": "ctx1",
            "cell_type": "synthetic",
            "module_genes": [tf, tg],
            "candidate_tfs": [tf],
            "cell_indices": [0],
            "metadata": {},
        },
        "edge": {"source_tf": tf, "target_gene": tg, "context_id": "ctx1"},
        "nodes": [
            {"node_id": "n_tf", "node_type": "expression", "label": tf, "payload": {"role": "TF"}},
            {"node_id": "n_target", "node_type": "expression", "label": tg, "payload": {"role": "target"}},
            {
                "node_id": "n_ctx",
                "node_type": "expression",
                "label": "context",
                "payload": {"n_cells": 10, "n_module_genes": 2},
            },
            {
                "node_id": "ev_expr",
                "node_type": "expression",
                "label": "expression_evidence",
                "payload": {
                    "z_t": corr,
                    "z_g": corr,
                    "activity_t": corr,
                    "mean_expr_t": 1.0,
                    "mean_expr_g": 1.0,
                    "dropout_t": 0.1,
                    "dropout_g": 0.1,
                },
            },
            {
                "node_id": "ev_network",
                "node_type": "correlation",
                "label": "network_evidence",
                "payload": {
                    "pearson_r": corr,
                    "partial_corr": corr / 2.0,
                    "in_same_module": 1.0,
                    "k_hop_distance": 1.0,
                    "shared_neighbors": 2.0,
                },
            },
        ],
        "relations": [
            {"src_id": "n_tf", "dst_id": "ev_expr", "relation": "in_context"},
            {"src_id": "n_target", "dst_id": "ev_expr", "relation": "in_context"},
            {"src_id": "n_tf", "dst_id": "ev_network", "relation": "supports_activation"},
            {"src_id": "ev_network", "dst_id": "n_target", "relation": "supports_activation"},
        ],
        "evidence": {"correlation": corr},
    }


def _write_graphs(path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_train_eager_uses_explicit_manifest_val_set(tmp_path):
    train_graphs = tmp_path / "train_graphs.jsonl"
    train_y = tmp_path / "train_y.npz"
    val_graphs = tmp_path / "val_graphs.jsonl"
    val_y = tmp_path / "val_y.npz"
    ckpt = tmp_path / "eager.pt"

    _write_graphs(
        train_graphs,
        [
            _graph("TF_A", "G1", 0.8),
            _graph("TF_A", "G2", -0.3),
            _graph("TF_B", "G3", 0.6),
            _graph("TF_B", "G4", -0.4),
        ],
    )
    np.savez(train_y, y=np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32))

    _write_graphs(
        val_graphs,
        [
            _graph("TF_C", "G5", 0.7),
            _graph("TF_C", "G6", -0.2),
        ],
    )
    np.savez(val_y, y=np.asarray([1.0, 0.0], dtype=np.float32))

    subprocess.run(
        [
            sys.executable,
            "scripts/train_eager.py",
            "--graphs-jsonl",
            str(train_graphs),
            "--y-npz",
            str(train_y),
            "--val-graphs-jsonl",
            str(val_graphs),
            "--val-y-npz",
            str(val_y),
            "--out",
            str(ckpt),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        check=True,
    )

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    extra = raw.get("extra", {})
    assert extra["validation_mode"] == "manifest"
    assert int(extra["n_train"]) == 4
    assert int(extra["n_val"]) == 2
