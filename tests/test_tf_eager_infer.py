from __future__ import annotations

import json

import pandas as pd
import torch

from scripts.infer_tf_eager import (
    aggregate_duplicate_scores,
    load_tf_eager_checkpoint,
    score_windows,
    select_network_rows,
    write_flat_evidence_jsonl,
    write_scores_csv,
)
from grn_agent.models.tf_eager import TfEagerConfig, TfEagerWindowModel


def _window_record() -> dict:
    return {
        "schema": "tf_eager_window_v1",
        "source_tf": "tf_a",
        "context": {
            "context_id": "ctx_1",
            "cell_type": "stem",
            "species": "mouse",
            "dataset_id": "ds",
        },
        "window_index": 0,
        "genes": [
            {
                "target_gene": "gene_b",
                "label": 1,
                "sample_weight": 1.0,
                "candidate_bucket": "split_positive",
                "evidence": {
                    "correlation": 0.5,
                    "motif_present": True,
                    "accessibility": 1.0,
                    "ensemble_prior": 0.7,
                },
                "expression": {"z_t": 0.2, "z_g": 0.1, "activity_t": 0.5},
                "network": {"pearson_r": 0.5},
                "motif": {"motif_present": True, "motif_score": 8.0},
                "accessibility": {"peak_accessibility": 1.0},
                "linkage": {},
                "prior": {"ensemble_prior": 0.7},
                "orthology": {},
            },
            {
                "target_gene": "gene_c",
                "label": 0,
                "sample_weight": 1.0,
                "candidate_bucket": "background",
                "evidence": {"correlation": 0.0},
                "expression": {"z_t": 0.2, "z_g": -0.1, "activity_t": 0.0},
                "network": {"pearson_r": 0.0},
                "motif": {},
                "accessibility": {},
                "linkage": {},
                "prior": {},
                "orthology": {},
            },
        ],
    }


def _save_checkpoint(path):
    cfg = TfEagerConfig(d_model=16, n_heads=4, n_encoder_layers=1, dropout=0.0)
    model = TfEagerWindowModel(cfg)
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "window_size": 100}, path)


def test_tf_eager_checkpoint_score_and_csv_exports(tmp_path):
    ckpt = tmp_path / "tf_eager.pt"
    _save_checkpoint(ckpt)

    model = load_tf_eager_checkpoint(ckpt, "cpu")
    rows = score_windows([_window_record()], model, device="cpu")

    assert len(rows) == 2
    assert rows[0]["source_tf"] == "TF_A"
    assert rows[0]["target_gene"] == "GENE_B"
    assert 0.0 <= rows[0]["p_present"] <= 1.0
    assert "mechanism_reasoning" in rows[0]

    scored = tmp_path / "scored.csv"
    write_scores_csv(rows, scored)
    df = pd.read_csv(scored)
    assert list(df["target_gene"]) == ["GENE_B", "GENE_C"]
    assert "p_present" in df.columns


def test_tf_eager_network_selection_and_flat_evidence(tmp_path):
    rows = [
        {"source_tf": "TF_A", "target_gene": "G1", "context_id": "c", "p_present": 0.9},
        {"source_tf": "TF_A", "target_gene": "G2", "context_id": "c", "p_present": 0.2},
        {"source_tf": "TF_B", "target_gene": "G3", "context_id": "c", "p_present": 0.1},
    ]
    selected = select_network_rows(rows, threshold=0.8, topk_per_tf=1)
    pairs = {(r["source_tf"], r["target_gene"]) for r in selected}
    assert pairs == {("TF_A", "G1"), ("TF_B", "G3")}

    evidence = tmp_path / "evidence.jsonl"
    write_flat_evidence_jsonl([_window_record()], evidence)
    records = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
    assert records[0]["schema"] == "tf_eager_flat_edge_v1"
    assert records[0]["edge"] == {"source_tf": "TF_A", "target_gene": "GENE_B"}
    assert records[0]["context"]["metadata"]["species"] == "mouse"


def test_aggregate_duplicate_scores_uses_normalized_vote_count():
    rows = [
        {"source_tf": "TF_A", "target_gene": "G1", "context_id": "c", "p_present": 0.9, "logit": 2.0},
        {"source_tf": "TF_A", "target_gene": "G1", "context_id": "c", "p_present": 0.4, "logit": 0.5},
        {"source_tf": "TF_A", "target_gene": "G1", "context_id": "c", "p_present": 0.2, "logit": -1.0},
        {"source_tf": "TF_A", "target_gene": "G2", "context_id": "c", "p_present": 0.1, "logit": -2.0},
    ]

    aggregated = aggregate_duplicate_scores(rows, vote_threshold=0.3)
    by_gene = {r["target_gene"]: r for r in aggregated}

    assert by_gene["G1"]["window_vote_count"] == 2
    assert by_gene["G1"]["window_vote_total"] == 3
    assert by_gene["G1"]["window_vote_fraction"] == 2 / 3
    assert by_gene["G1"]["p_present"] == 0.5
    assert by_gene["G1"]["p_present_mean"] == 0.5
    assert by_gene["G2"]["window_vote_fraction"] == 0.0
    assert by_gene["G2"]["p_present"] == 0.1
