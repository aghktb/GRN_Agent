import pandas as pd
import subprocess
import sys

from scripts.make_tf_holdout_split_manifest import _load_expression_gene_universe, _load_unique_edges


def test_load_unique_edges_gene1_gene2_aliases(tmp_path):
    p = tmp_path / "gold.csv"
    pd.DataFrame(
        [
            {"Gene1": "tf_a", "Gene2": "gene_b"},
            {"Gene1": "tf_a", "Gene2": "gene_b"},
            {"Gene1": "tf_c", "Gene2": "gene_d"},
        ]
    ).to_csv(p, index=False)

    pairs, meta = _load_unique_edges(p)

    assert pairs == [("TF_A", "GENE_B"), ("TF_C", "GENE_D")]
    assert meta["dataset_id"] is None


def test_load_expression_gene_universe_prefers_axis_with_better_gold_overlap(tmp_path):
    expr = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "cell_1": [1.0, 2.0, 3.0],
            "cell_2": [0.5, 1.5, 2.5],
        },
        index=["tf_a", "gene_b", "gene_c"],
    ).to_csv(expr)

    genes = _load_expression_gene_universe(expr, {"TF_A", "GENE_B", "GENE_X"})
    assert genes == {"TF_A", "GENE_B", "GENE_C"}


def test_manifest_filters_gold_edges_to_expression_universe(tmp_path):
    gold = tmp_path / "gold.csv"
    pd.DataFrame(
        [
            {"source_tf": "TF_A", "target_gene": "GENE_B"},
            {"source_tf": "TF_A", "target_gene": "GENE_X"},
            {"source_tf": "TF_Z", "target_gene": "GENE_B"},
        ]
    ).to_csv(gold, index=False)
    expr = tmp_path / "expr.csv"
    pd.DataFrame({"cell1": [1.0, 2.0]}, index=["TF_A", "GENE_B"]).to_csv(expr)
    out = tmp_path / "split.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/make_tf_holdout_split_manifest.py",
            "--gold-edges",
            str(gold),
            "--out",
            str(out),
            "--fold-id",
            "f1",
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.1",
            "--test-ratio",
            "0.2",
            "--expression-path",
            str(expr),
        ],
        check=True,
    )

    df = pd.read_csv(out)
    pairs = {(str(r.source_tf), str(r.target_gene)) for r in df.itertuples(index=False)}
    assert pairs == {("TF_A", "GENE_B")}


def test_expression_node_split_uses_expressed_tf_universe(tmp_path):
    gold = tmp_path / "gold.csv"
    pd.DataFrame(
        [
            {"source_tf": "TF_A", "target_gene": "GENE_1"},
            {"source_tf": "TF_A", "target_gene": "GENE_2"},
            {"source_tf": "TF_A", "target_gene": "GENE_3"},
            {"source_tf": "TF_B", "target_gene": "GENE_1"},
            {"source_tf": "TF_C", "target_gene": "GENE_1"},
            {"source_tf": "TF_D", "target_gene": "GENE_1"},
        ]
    ).to_csv(gold, index=False)
    expr = tmp_path / "expr.csv"
    pd.DataFrame(
        {"cell1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
        index=["TF_A", "TF_B", "TF_C", "TF_D", "GENE_1", "GENE_2", "GENE_3"],
    ).to_csv(expr)
    tf_file = tmp_path / "tfs.csv"
    pd.DataFrame({"TF": ["TF_A", "TF_B", "TF_C", "TF_D", "TF_Z"]}).to_csv(tf_file, index=False)
    out = tmp_path / "split.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/make_tf_holdout_split_manifest.py",
            "--gold-edges",
            str(gold),
            "--out",
            str(out),
            "--fold-id",
            "f1",
            "--train-ratio",
            "0.5",
            "--val-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
            "--expression-path",
            str(expr),
            "--tf-file",
            str(tf_file),
            "--node-split-mode",
            "expression",
            "--seed",
            "7",
        ],
        check=True,
    )

    df = pd.read_csv(out)
    tf_subsets = df.groupby("source_tf")["subset"].nunique().to_dict()
    assert set(df["source_tf"]) == {"TF_A", "TF_B", "TF_C", "TF_D"}
    assert all(n == 1 for n in tf_subsets.values())
    assert {"source_tf_subset", "target_gene_subset"} <= set(df.columns)
    assert df["source_tf_subset"].equals(df["subset"])


def test_expression_node_split_drops_gold_source_tf_missing_from_tf_file(tmp_path):
    gold = tmp_path / "gold.csv"
    pd.DataFrame(
        [
            {"source_tf": "RUVBL1", "target_gene": "PRMT5"},
            {"source_tf": "RUVBL1", "target_gene": "ANG"},
        ]
    ).to_csv(gold, index=False)
    expr = tmp_path / "expr.csv"
    pd.DataFrame(
        {"cell1": [1.0, 2.0, 3.0]},
        index=["RUVBL1", "PRMT5", "ANG"],
    ).to_csv(expr)
    tf_file = tmp_path / "tfs.csv"
    pd.DataFrame({"TF": ["AHR", "ADNP"]}).to_csv(tf_file, index=False)
    out = tmp_path / "split.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/make_tf_holdout_split_manifest.py",
            "--gold-edges",
            str(gold),
            "--out",
            str(out),
            "--fold-id",
            "f1",
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.1",
            "--test-ratio",
            "0.2",
            "--expression-path",
            str(expr),
            "--tf-file",
            str(tf_file),
            "--node-split-mode",
            "expression",
            "--seed",
            "7",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert not out.exists()
