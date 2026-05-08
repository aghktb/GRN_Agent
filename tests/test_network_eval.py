import json

import pandas as pd

from grn_agent.eval.network_eval import evaluate_network_vs_labels, evaluate_network_vs_weak_labels, evaluate_network_with_manifest


def test_evaluate_network_matches(tmp_path):
    eg = {
        "context": {
            "context_id": "ctx1",
            "cell_type": "t",
            "module_genes": ["A", "B"],
            "candidate_tfs": ["A"],
            "cell_indices": [0],
            "metadata": {},
        },
        "edge": {"source_tf": "A", "target_gene": "B", "context_id": "ctx1"},
        "nodes": [],
        "relations": [],
        "evidence": {"correlation": 0.5},
    }
    jpath = tmp_path / "e.jsonl"
    jpath.write_text(json.dumps(eg) + "\n", encoding="utf-8")
    df = pd.DataFrame([
        {
            "source_tf": "A",
            "target_gene": "B",
            "confidence_score": 0.8,
            "p_present": 0.8,
            "mechanism_reasoning": "x",
        }
    ])
    csvp = tmp_path / "n.csv"
    df.to_csv(csvp, index=False)
    m = evaluate_network_vs_weak_labels(csvp, jpath)
    assert m["n_matched"] == 1
    assert "accuracy" in m


def test_evaluate_network_vs_gold_file(tmp_path):
    eg = {
        "context": {
            "context_id": "ctx1",
            "cell_type": "t",
            "module_genes": ["A", "B"],
            "candidate_tfs": ["A"],
            "cell_indices": [0],
            "metadata": {},
        },
        "edge": {"source_tf": "A", "target_gene": "B", "context_id": "ctx1"},
        "nodes": [],
        "relations": [],
        "evidence": {"correlation": 0.5},
    }
    jpath = tmp_path / "e.jsonl"
    jpath.write_text(json.dumps(eg) + "\n", encoding="utf-8")
    goldp = tmp_path / "gold.csv"
    pd.DataFrame([{"source_tf": "A", "target_gene": "B", "regulation_type": "Activation"}]).to_csv(goldp, index=False)
    df = pd.DataFrame([
        {
            "source_tf": "A",
            "target_gene": "B",
            "confidence_score": 0.8,
            "p_present": 0.8,
            "mechanism_reasoning": "x",
        }
    ])
    csvp = tmp_path / "n.csv"
    df.to_csv(csvp, index=False)
    m = evaluate_network_vs_labels(csvp, jpath, gold_edges=goldp)
    assert m["n_matched"] == 1
    assert m["label_source"] == "gold_file"


def test_evaluate_network_vs_gold_file_negative_ratio(tmp_path):
    records = []
    preds = []
    gold_rows = []
    for gene, score, is_pos in [("B", 0.9, True), ("C", 0.8, False), ("D", 0.2, False), ("E", 0.1, False)]:
        records.append(
            {
                "context": {
                    "context_id": "ctx1",
                    "cell_type": "t",
                    "module_genes": ["A", gene],
                    "candidate_tfs": ["A"],
                    "cell_indices": [0],
                    "metadata": {},
                },
                "edge": {"source_tf": "A", "target_gene": gene, "context_id": "ctx1"},
                "nodes": [],
                "relations": [],
                "evidence": {"correlation": 0.5},
            }
        )
        preds.append(
            {
                "source_tf": "A",
                "target_gene": gene,
                "confidence_score": score,
                "p_present": score,
                "mechanism_reasoning": "x",
            }
        )
        if is_pos:
            gold_rows.append({"source_tf": "A", "target_gene": gene, "regulation_type": "Activation"})
    jpath = tmp_path / "e.jsonl"
    jpath.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    csvp = tmp_path / "n.csv"
    pd.DataFrame(preds).to_csv(csvp, index=False)
    goldp = tmp_path / "gold.csv"
    pd.DataFrame(gold_rows).to_csv(goldp, index=False)

    m = evaluate_network_vs_labels(csvp, jpath, gold_edges=goldp, negative_ratio=2, negative_repeats=3, seed=1)
    assert m["n_positive"] == 1
    assert m["n_negative"] == 2
    assert m["negative_ratio"] == 2.0
    assert m["negative_sampling_repeats"] == 3
    assert "negative_sampling_metric_std" in m


def test_evaluate_network_with_manifest_filters_subset(tmp_path):
    eg = {
        "context": {
            "context_id": "ctx1",
            "cell_type": "ctA",
            "module_genes": ["A", "B", "C"],
            "candidate_tfs": ["A"],
            "cell_indices": [0],
            "metadata": {"species": "human"},
        },
        "edge": {"source_tf": "A", "target_gene": "B", "context_id": "ctx1"},
        "nodes": [],
        "relations": [],
        "evidence": {"correlation": 0.4},
    }
    jpath = tmp_path / "e.jsonl"
    jpath.write_text(json.dumps(eg) + "\n", encoding="utf-8")
    csvp = tmp_path / "n.csv"
    pd.DataFrame([
        {"source_tf": "A", "target_gene": "B", "confidence_score": 0.9, "p_present": 0.8, "mechanism_reasoning": "x"},
        {"source_tf": "A", "target_gene": "C", "confidence_score": 0.6, "p_present": 0.6, "mechanism_reasoning": "x"},
    ]).to_csv(csvp, index=False)
    splitp = tmp_path / "split.csv"
    pd.DataFrame([
        {
            "split_name": "leave_one_tf_out",
            "fold_id": "f1",
            "subset": "test",
            "source_tf": "A",
            "target_gene": "B",
            "cell_type": "ctA",
            "species": "human",
            "tf_frequency_bucket": "high",
        }
    ]).to_csv(splitp, index=False)
    out = evaluate_network_with_manifest(
        csvp,
        jpath,
        split_manifest=splitp,
        strategy="leave_one_tf_out",
        fold_id="f1",
        subset="test",
    )
    assert out["n_matched"] == 1
    assert out["n_excluded_outside_subset"] >= 1
