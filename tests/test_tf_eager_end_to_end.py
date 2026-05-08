from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import yaml


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def test_tf_eager_cli_end_to_end_train_infer_eval(tmp_path):
    expr = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "cell0": [1.0, 1.1, 0.0, 3.0, 2.8, 0.1, 5.0, 5.1, 0.2],
            "cell1": [2.0, 2.1, 0.2, 2.0, 1.9, 0.0, 4.0, 4.2, 0.1],
            "cell2": [3.0, 3.2, 0.1, 1.0, 1.1, 0.3, 3.0, 3.1, 0.0],
            "cell3": [4.0, 4.1, 0.0, 0.5, 0.4, 0.2, 2.0, 2.1, 0.2],
            "cell4": [5.0, 5.2, 0.2, 0.2, 0.3, 0.1, 1.0, 1.1, 0.0],
            "cell5": [6.0, 6.1, 0.1, 0.1, 0.0, 0.2, 0.5, 0.4, 0.1],
        },
        index=["TF_A", "GENE_B", "GENE_C", "TF_D", "GENE_E", "GENE_F", "TF_G", "GENE_H", "GENE_I"],
    ).to_csv(expr)

    tf_file = tmp_path / "tfs.csv"
    tf_file.write_text("TF_A\nTF_D\nTF_G\n", encoding="utf-8")

    motif_hits = tmp_path / "motif_hits.tsv"
    pd.DataFrame(
        [
            {
                "source_tf": "TF_A",
                "target_gene": "GENE_B",
                "motif_present": 1,
                "max_score_pct": 92.0,
                "peak_count": 2,
            },
            {
                "source_tf": "TF_D",
                "target_gene": "GENE_E",
                "motif_present": 1,
                "max_score_pct": 88.0,
                "peak_count": 1,
            },
            {
                "source_tf": "TF_G",
                "target_gene": "GENE_H",
                "motif_present": 1,
                "max_score_pct": 95.0,
                "peak_count": 3,
            },
        ]
    ).to_csv(motif_hits, sep="\t", index=False)

    promoter_accessibility = tmp_path / "promoter_accessibility.bed"
    promoter_accessibility.write_text(
        "GENE_B\t1.2\n"
        "GENE_E\t0.8\n"
        "GENE_H\t1.7\n"
        "GENE_I\t0.4\n",
        encoding="utf-8",
    )

    multimodal_manifest = tmp_path / "multimodal_manifest.json"
    multimodal_manifest.write_text(
        json.dumps(
            {
                "dataset_id": "tiny",
                "output_paths": {
                    "motif_hits": str(motif_hits),
                    "promoter_accessibility": str(promoter_accessibility),
                },
                "qc_report": {"source": "synthetic_acquisition_fixture"},
            }
        ),
        encoding="utf-8",
    )

    gold = tmp_path / "gold.csv"
    gold.write_text(
        "source_tf,target_gene\n"
        "TF_A,GENE_B\n"
        "TF_D,GENE_E\n"
        "TF_G,GENE_H\n",
        encoding="utf-8",
    )

    split = tmp_path / "split.csv"
    split.write_text(
        "split_name,fold_id,subset,source_tf,target_gene\n"
        "leave_one_tf_out,f1,train,TF_A,GENE_B\n"
        "leave_one_tf_out,f1,train,TF_D,GENE_E\n"
        "leave_one_tf_out,f1,test,TF_G,GENE_H\n",
        encoding="utf-8",
    )

    train_windows = tmp_path / "train_windows.jsonl"
    test_windows = tmp_path / "test_windows.jsonl"
    ckpt = tmp_path / "tf_eager.pt"
    scored = tmp_path / "scored.csv"
    network = tmp_path / "network.csv"
    flat_evidence = tmp_path / "flat_evidence.jsonl"
    report = tmp_path / "eval_report.json"

    base_cfg = {
        "seed": 7,
        "disable_priors": True,
        "use_ortholog_lookup": False,
        "multimodal_manifest": str(multimodal_manifest),
        "dataset": {
            "mode": "beeline_csv",
            "dataset_id": "tiny",
            "species": "mouse",
            "expression_path": str(expr),
            "tf_file": str(tf_file),
            "modalities": ["scrna"],
        },
        "cell_context": {"cell_type": "synthetic"},
        "candidates": {
            "topk_corr": 2,
            "bottomk_corr": 1,
            "max_edges_per_tf": 3,
            "train_subgraph_bootstraps": 2,
            "rescue_motif": False,
            "rescue_accessibility": False,
        },
        "scoring": {"device": "cpu"},
        "tf_eager": {
            "gold_edges": str(gold),
            "split_manifest": str(split),
            "strategy": "leave_one_tf_out",
            "fold_id": "f1",
            "subset": "train",
            "windows_jsonl": str(train_windows),
            "checkpoint": str(ckpt),
            "train": {"epochs": 1, "val_frac": 0.5, "device": "cpu", "lr": 0.001},
        },
    }
    cfg = tmp_path / "tf_eager.yml"
    cfg.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    _run([sys.executable, "scripts/build_tf_eager_windows.py", "--config", str(cfg)])
    assert train_windows.is_file()
    assert train_windows.read_text(encoding="utf-8").strip()
    train_records = [json.loads(line) for line in train_windows.read_text(encoding="utf-8").splitlines()]
    assert any(r["evidence_graph"]["modality_mask"]["motif"] for r in train_records)
    assert any(r["evidence_graph"]["modality_mask"]["acc"] for r in train_records)
    assert any(r["evidence_graph"]["modality_mask"]["link"] for r in train_records)
    tf_a_gene_b = next(
        g
        for r in train_records
        if r["source_tf"] == "TF_A"
        for g in r["genes"]
        if g["target_gene"] == "GENE_B"
    )
    assert tf_a_gene_b["motif"]["motif_present"] is True
    assert tf_a_gene_b["motif"]["motif_score"] == 92.0
    assert tf_a_gene_b["accessibility"]["peak_accessibility"] == 1.2
    assert tf_a_gene_b["linkage"]["peak_to_gene_linked"] is True
    tf_a_gene_h = next(
        g
        for r in train_records
        if r["source_tf"] == "TF_A"
        for g in r["genes"]
        if g["target_gene"] == "GENE_H"
    )
    assert tf_a_gene_h["negative_class"] == "ambiguous"
    assert tf_a_gene_h["sample_weight"] == 0.0

    _run([sys.executable, "scripts/train_tf_eager.py", "--config", str(cfg)])
    assert ckpt.is_file()

    test_cfg = dict(base_cfg)
    test_tf_eager = dict(base_cfg["tf_eager"])
    test_tf_eager.update(
        {
            "subset": "test",
            "windows_jsonl": str(test_windows),
            "infer": {
                "windows_jsonl": str(test_windows),
                "checkpoint": str(ckpt),
                "scored_csv": str(scored),
                "network_csv": str(network),
                "evidence_jsonl": str(flat_evidence),
                "threshold": 0.0,
                "topk_per_tf": 1,
                "device": "cpu",
            },
        }
    )
    test_cfg["tf_eager"] = test_tf_eager
    cfg.write_text(yaml.safe_dump(test_cfg, sort_keys=False), encoding="utf-8")

    _run([sys.executable, "scripts/build_tf_eager_windows.py", "--config", str(cfg)])
    assert test_windows.is_file()
    test_records = [json.loads(line) for line in test_windows.read_text(encoding="utf-8").splitlines()]
    tf_g_gene_h = next(
        g
        for r in test_records
        if r["source_tf"] == "TF_G"
        for g in r["genes"]
        if g["target_gene"] == "GENE_H"
    )
    assert tf_g_gene_h["motif"]["motif_present"] is True
    assert tf_g_gene_h["accessibility"]["peak_accessibility"] == 1.7
    assert tf_g_gene_h["linkage"]["linkage_score"] == 1.7

    _run([sys.executable, "scripts/infer_tf_eager.py", "--config", str(cfg)])
    scored_df = pd.read_csv(scored)
    assert {"source_tf", "target_gene", "p_present", "logit"}.issubset(scored_df.columns)
    assert set(scored_df["source_tf"]) == {"TF_G"}
    positive_row = scored_df.loc[scored_df["target_gene"] == "GENE_H"].iloc[0]
    assert positive_row["motif_present"] is True or positive_row["motif_present"] == 1
    assert positive_row["accessibility"] == 1.7
    assert flat_evidence.is_file()
    flat_records = [json.loads(line) for line in flat_evidence.read_text(encoding="utf-8").splitlines()]
    flat_positive = next(r for r in flat_records if r["edge"]["target_gene"] == "GENE_H")
    assert flat_positive["evidence"]["motif_present"] is True
    assert flat_positive["evidence"]["accessibility"] == 1.7

    _run(
        [
            sys.executable,
            "scripts/eval_grn_agent.py",
            "--scored-csv",
            str(scored),
            "--evidence-jsonl",
            str(flat_evidence),
            "--gold-edges",
            str(gold),
            "--split-manifest",
            str(split),
            "--strategy",
            "leave_one_tf_out",
            "--fold-id",
            "f1",
            "--subset",
            "test",
            "--negative-ratio",
            "1",
            "--out-report",
            str(report),
        ]
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["subset"] == "test"
    metrics = payload["results_by_ratio"]["1.0"]
    assert metrics["n_matched"] >= 2
    assert metrics["n_positive"] >= 1
    assert metrics["n_negative"] >= 1
