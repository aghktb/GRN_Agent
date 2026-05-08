from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import yaml


def test_integrated_tf_eager_workflow_script(tmp_path):
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
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "source_tf,target_gene\nTF_A,GENE_B\nTF_D,GENE_E\nTF_G,GENE_H\n",
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
    motif_hits = tmp_path / "motif_hits.tsv"
    pd.DataFrame(
        [
            {"source_tf": "TF_A", "target_gene": "GENE_B", "motif_present": 1, "max_score_pct": 92.0, "peak_count": 2},
            {"source_tf": "TF_D", "target_gene": "GENE_E", "motif_present": 1, "max_score_pct": 88.0, "peak_count": 1},
            {"source_tf": "TF_G", "target_gene": "GENE_H", "motif_present": 1, "max_score_pct": 95.0, "peak_count": 3},
        ]
    ).to_csv(motif_hits, sep="\t", index=False)
    promoter_accessibility = tmp_path / "promoter_accessibility.bed"
    promoter_accessibility.write_text("GENE_B\t1.2\nGENE_E\t0.8\nGENE_H\t1.7\n", encoding="utf-8")
    manifest = tmp_path / "multimodal_manifest.json"
    manifest.write_text(
        json.dumps({"output_paths": {"motif_hits": str(motif_hits), "promoter_accessibility": str(promoter_accessibility)}}),
        encoding="utf-8",
    )

    cfg = {
        "workflow": {"id": "tf_eager_wf", "artifact_root": str(tmp_path), "seed": 7},
        "multimodal_manifest": str(manifest),
        "disable_priors": True,
        "use_ortholog_lookup": False,
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
            "rescue_motif": True,
            "rescue_accessibility": True,
            "rescue_max_per_tf": 2,
        },
        "scoring": {"device": "cpu"},
        "split": {"enabled": False, "gold_edges": str(gold), "out": str(split), "fold_id": "f1"},
        "tf_eager": {"strategy": "leave_one_tf_out", "fold_id": "f1"},
        "train_tf_eager": {"enabled": True, "epochs": 1, "val_frac": 0.5, "device": "cpu", "lr": 0.001},
        "infer_tf_eager": {"threshold": 0.0, "topk_per_tf": 1, "device": "cpu"},
        "evaluation": {"gold_edges": str(gold), "negative_ratio": 1},
    }
    cfg_path = tmp_path / "workflow.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/run_integrated_tf_eager_workflow.py", "--config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    wf_dir = tmp_path / "tf_eager_wf"
    train_windows = wf_dir / "tf_eager" / "train_windows.jsonl"
    test_windows = wf_dir / "tf_eager" / "test_windows.jsonl"
    ckpt = wf_dir / "tf_eager" / "tf_eager.pt"
    scored = wf_dir / "tf_eager" / "test_scored_edges.csv"
    network = wf_dir / "tf_eager" / "test_network.csv"
    evidence = wf_dir / "tf_eager" / "test_flat_evidence.jsonl"
    report = wf_dir / "evaluation" / "eval_test_by_ratio.json"

    for path in (train_windows, test_windows, ckpt, scored, network, evidence, report):
        assert path.is_file(), path
        assert wf_dir in path.parents

    test_records = [json.loads(line) for line in test_windows.read_text(encoding="utf-8").splitlines()]
    assert any(r["evidence_graph"]["modality_mask"]["motif"] for r in test_records)
    assert any(r["evidence_graph"]["modality_mask"]["acc"] for r in test_records)
    scored_df = pd.read_csv(scored)
    assert set(scored_df["source_tf"]) == {"TF_G"}
    assert float(scored_df.loc[scored_df["target_gene"] == "GENE_H", "accessibility"].iloc[0]) == 1.7
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["prediction_kind"] == "all_scored_edges"
    assert payload["results_by_ratio"]["1.0"]["n_positive"] >= 1


def test_integrated_tf_eager_workflow_runs_acquisition_stage(tmp_path):
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
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "source_tf,target_gene\nTF_A,GENE_B\nTF_D,GENE_E\nTF_G,GENE_H\n",
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
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        "\n".join(
            [
                'chr1\tfixture\tgene\t1000\t1200\t.\t+\t.\tgene_id "TF_A"; gene_name "TF_A";',
                'chr1\tfixture\tgene\t1300\t1500\t.\t+\t.\tgene_id "GENE_B"; gene_name "GENE_B";',
                'chr1\tfixture\tgene\t1800\t2000\t.\t+\t.\tgene_id "GENE_C"; gene_name "GENE_C";',
                'chr1\tfixture\tgene\t3000\t3200\t.\t+\t.\tgene_id "TF_D"; gene_name "TF_D";',
                'chr1\tfixture\tgene\t3400\t3600\t.\t+\t.\tgene_id "GENE_E"; gene_name "GENE_E";',
                'chr1\tfixture\tgene\t3900\t4100\t.\t+\t.\tgene_id "GENE_F"; gene_name "GENE_F";',
                'chr1\tfixture\tgene\t5000\t5200\t.\t+\t.\tgene_id "TF_G"; gene_name "TF_G";',
                'chr1\tfixture\tgene\t5400\t5600\t.\t+\t.\tgene_id "GENE_H"; gene_name "GENE_H";',
                'chr1\tfixture\tgene\t5900\t6100\t.\t+\t.\tgene_id "GENE_I"; gene_name "GENE_I";',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t1250\t1350\tp_gene_b\t2\nchr1\t3350\t3450\tp_gene_e\t3\nchr1\t5350\t5450\tp_gene_h\t4\n", encoding="utf-8")

    cfg = {
        "workflow": {"id": "tf_eager_acq_wf", "artifact_root": str(tmp_path), "seed": 7},
        "acquisition": {
            "enabled": True,
            "expr": str(expr),
            "species": "mouse",
            "cell_type": "synthetic",
            "dataset_id": "tiny_acq",
            "gold_network": str(gold),
            "atac_file": str(peaks),
            "genome": "mouse",
            "gtf": str(gtf),
            "skip_motif": True,
            "no_auto_genome": True,
            "reuse_if_exists": False,
        },
        "disable_priors": True,
        "use_ortholog_lookup": False,
        "dataset": {
            "mode": "beeline_csv",
            "dataset_id": "tiny",
            "species": "mouse",
            "expression_path": str(expr),
            "tf_file": str(tf_file),
            "modalities": ["scrna", "atac"],
        },
        "cell_context": {"cell_type": "synthetic"},
        "candidates": {
            "topk_corr": 2,
            "bottomk_corr": 1,
            "max_edges_per_tf": 3,
            "rescue_motif": False,
            "rescue_accessibility": True,
            "rescue_max_per_tf": 2,
        },
        "scoring": {"device": "cpu"},
        "split": {"enabled": False, "gold_edges": str(gold), "out": str(split), "fold_id": "f1"},
        "tf_eager": {"strategy": "leave_one_tf_out", "fold_id": "f1"},
        "train_tf_eager": {"enabled": True, "epochs": 1, "val_frac": 0.5, "device": "cpu", "lr": 0.001},
        "infer_tf_eager": {"threshold": 0.0, "topk_per_tf": 1, "device": "cpu"},
        "evaluation": {"gold_edges": str(gold), "negative_ratio": 1},
    }
    cfg_path = tmp_path / "workflow_acq.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/run_integrated_tf_eager_workflow.py", "--config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    wf_dir = tmp_path / "tf_eager_acq_wf"
    manifest = wf_dir / "acquisition" / "multimodal_manifest.json"
    test_windows = wf_dir / "tf_eager" / "test_windows.jsonl"
    scored = wf_dir / "tf_eager" / "test_scored_edges.csv"
    report = wf_dir / "evaluation" / "eval_test_by_ratio.json"
    assert manifest.is_file()
    assert wf_dir in manifest.parents
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    promoter_path = manifest_payload["paths"]["promoter_accessibility"]
    assert promoter_path.endswith("promoter_accessibility.bed")
    assert test_windows.is_file()
    records = [json.loads(line) for line in test_windows.read_text(encoding="utf-8").splitlines()]
    assert any(r["evidence_graph"]["modality_mask"]["acc"] for r in records)
    gene_h = next(g for r in records for g in r["genes"] if g["target_gene"] == "GENE_H")
    assert gene_h["accessibility"]["peak_accessibility"] > 0.0
    assert gene_h["linkage"]["peak_to_gene_linked"] is True
    scored_df = pd.read_csv(scored)
    assert float(scored_df.loc[scored_df["target_gene"] == "GENE_H", "accessibility"].iloc[0]) > 0.0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["results_by_ratio"]["1.0"]["n_positive"] >= 1


def test_integrated_tf_eager_workflow_supports_blind_single_dataset_inference(tmp_path):
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
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "source_tf,target_gene\nTF_A,GENE_B\nTF_D,GENE_E\nTF_G,GENE_H\n",
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

    train_cfg = {
        "workflow": {"id": "tf_eager_train_wf", "artifact_root": str(tmp_path), "seed": 7},
        "disable_priors": True,
        "use_ortholog_lookup": False,
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
            "train_window_neighbors": 2,
        },
        "scoring": {"device": "cpu"},
        "split": {"enabled": False, "gold_edges": str(gold), "out": str(split), "fold_id": "f1"},
        "tf_eager": {"strategy": "leave_one_tf_out", "fold_id": "f1"},
        "train_tf_eager": {"enabled": True, "epochs": 1, "val_frac": 0.5, "device": "cpu", "lr": 0.001},
        "infer_tf_eager": {"threshold": 0.0, "topk_per_tf": 1, "device": "cpu"},
        "evaluation": {"gold_edges": str(gold), "negative_ratio": 1},
    }
    train_cfg_path = tmp_path / "train_workflow.yml"
    train_cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
    subprocess.run(
        [sys.executable, "scripts/run_integrated_tf_eager_workflow.py", "--config", str(train_cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    checkpoint = tmp_path / "tf_eager_train_wf" / "tf_eager" / "tf_eager.pt"
    assert checkpoint.is_file()

    blind_cfg = {
        "workflow": {
            "id": "tf_eager_blind_wf",
            "artifact_root": str(tmp_path),
            "seed": 11,
            "inference_evaluation_only": True,
        },
        "disable_priors": True,
        "use_ortholog_lookup": False,
        "dataset": {
            "mode": "beeline_csv",
            "dataset_id": "tiny_blind",
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
            "train_window_neighbors": 2,
            "blind_ensure_coverage": True,
        },
        "scoring": {"device": "cpu"},
        "build_test_windows": {
            "enabled": True,
            "reuse_if_exists": False,
            "tf_workers": 2,
            "build_device": "cpu",
        },
        "tf_eager": {
            "strategy": "leave_one_tf_out",
            "fold_id": "f1",
            "checkpoint": str(checkpoint),
        },
        "infer_tf_eager": {"threshold": 0.0, "topk_per_tf": 2, "device": "cpu"},
        "evaluation": {"enabled": True},
    }
    blind_cfg_path = tmp_path / "blind_workflow.yml"
    blind_cfg_path.write_text(yaml.safe_dump(blind_cfg, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/run_integrated_tf_eager_workflow.py", "--config", str(blind_cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    wf_dir = tmp_path / "tf_eager_blind_wf"
    test_windows = wf_dir / "tf_eager" / "test_windows.jsonl"
    scored = wf_dir / "tf_eager" / "test_scored_edges.csv"
    evidence = wf_dir / "tf_eager" / "test_flat_evidence.jsonl"
    report = wf_dir / "evaluation" / "eval_test_by_ratio.json"

    assert test_windows.is_file()
    assert scored.is_file()
    assert evidence.is_file()
    assert not report.exists()

    records = [json.loads(line) for line in test_windows.read_text(encoding="utf-8").splitlines()]
    assert records
    assert all(rec["context"]["cell_type"] == "synthetic" for rec in records)
    scored_df = pd.read_csv(scored)
    assert set(scored_df["source_tf"]) == {"TF_A", "TF_D", "TF_G"}
