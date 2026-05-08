from __future__ import annotations
from collections import OrderedDict

import numpy as np

from scripts.build_tf_eager_windows import (
    ContextStats,
    _cap_tf_subgraph,
    _full_neighborhood_targets_for_tf,
    _label_and_sample_tf_genes,
    _should_add_coverage_window,
    _subgraph_bootstraps_for_subset,
    _subgraph_normalized_arrays,
    _use_without_replacement_chunks,
    _without_replacement_tf_subgraph_chunks,
)


def test_tf_subgraph_cap_random_samples_without_gold_priority():
    targets = OrderedDict((f"G{i}", "coexpression_neighbor") for i in range(150))
    targets["P1"] = "split_positive"
    targets["P2"] = "split_positive"

    sampled = _cap_tf_subgraph(
        targets,
        rng=np.random.default_rng(7),
        max_count=100,
    )

    assert len(sampled) == 100
    assert list(sampled.keys()) != list(targets.keys())[:100]


def test_without_replacement_tf_subgraph_chunks_do_not_overlap():
    targets = OrderedDict((f"G{i}", "coexpression_neighbor") for i in range(10))

    chunks = _without_replacement_tf_subgraph_chunks(
        targets,
        rng=np.random.default_rng(7),
        max_count=4,
        n_chunks=3,
    )

    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    flattened = [gene for chunk in chunks for gene in chunk]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(targets)


def test_full_neighborhood_adds_high_corr_rescue_and_low_corr_background():
    genes = ["TF1", "P1", "P2", "N1", "N2"]
    stats = ContextStats(
        context={"context_id": "ctx"},
        gene_symbols=genes,
        gene_to_idx={g: i for i, g in enumerate(genes)},
        module_genes=genes,
        module_set=set(genes),
        module_idxs=np.arange(len(genes), dtype=np.int64),
        sub=np.zeros((3, len(genes))),
        z_sub=np.zeros((3, len(genes))),
        denom=2.0,
        global_mean=np.zeros(len(genes)),
        global_std=np.ones(len(genes)),
        ctx_mean=np.zeros(len(genes)),
        ctx_dropout=np.zeros(len(genes)),
    )
    corr = np.asarray([1.0, 0.9, 0.8, 0.7, 0.0], dtype=np.float64)

    targets, skipped = _full_neighborhood_targets_for_tf(
        stats,
        "TF1",
        corr,
        allowed_positive={("TF1", "P1"), ("TF1", "P2")},
        positive_global={("TF1", "P1"), ("TF1", "P2")},
        loader=None,
        corr_threshold=0.6,
        max_count=4,
        rng=np.random.default_rng(7),
        rescue_accessibility=False,
        rescue_motif=False,
    )

    assert skipped == 0
    assert targets["N1"] == "coexpression_neighbor"
    assert "N2" in targets
    assert targets["N2"] == "low_coexpression_background"
    assert len(targets) == 4


def test_subgraph_normalization_zscores_each_cell_across_selected_genes():
    genes = ["TF1", "G1", "G2"]
    sub = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [5.0, 5.0, 5.0]], dtype=np.float64)
    stats = ContextStats(
        context={"context_id": "ctx"},
        gene_symbols=genes,
        gene_to_idx={g: i for i, g in enumerate(genes)},
        module_genes=genes,
        module_set=set(genes),
        module_idxs=np.arange(len(genes), dtype=np.int64),
        sub=sub,
        z_sub=np.zeros_like(sub),
        denom=3.0,
        global_mean=sub.mean(axis=0),
        global_std=sub.std(axis=0),
        ctx_mean=sub.mean(axis=0),
        ctx_dropout=(sub == 0).mean(axis=0),
    )

    corr, ctx_mean, global_mean, global_std, _dropout = _subgraph_normalized_arrays(
        stats,
        "TF1",
        OrderedDict([("G1", "coexpression_neighbor"), ("G2", "low_coexpression_background")]),
        np.zeros(len(genes)),
    )

    row_z = (sub - sub.mean(axis=1, keepdims=True)) / (sub.std(axis=1, keepdims=True) + 1e-8)
    assert np.allclose(ctx_mean, row_z.mean(axis=0))
    assert np.allclose(global_mean, np.zeros(len(genes)))
    assert np.all(global_std > 0)
    assert corr.shape == (len(genes),)


def test_negative_sampling_uses_requested_weighted_ratio():
    genes = OrderedDict()
    genes["P1"] = {"target_gene": "P1", "evidence": {"correlation": 0.9}}
    genes["P2"] = {"target_gene": "P2", "evidence": {"correlation": 0.8}}
    for i in range(20):
        genes[f"N{i}"] = {"target_gene": f"N{i}", "evidence": {"correlation": 0.0}}

    sampled = _label_and_sample_tf_genes(
        genes,
        tf="TF1",
        allowed_positive={("TF1", "P1"), ("TF1", "P2")},
        positive_global={("TF1", "P1"), ("TF1", "P2")},
        rng=np.random.default_rng(7),
        negative_ratio=5,
        corr_threshold=0.25,
        motif_score_threshold=0.0,
        acc_threshold=0.0,
    )

    weighted = [g for g in sampled if float(g.get("sample_weight", 0.0)) > 0.0]
    assert sum(int(g["label"]) for g in weighted) == 2
    assert sum(1 for g in weighted if int(g["label"]) == 0) == 10
    assert [int(g["label"]) for g in weighted] != sorted([int(g["label"]) for g in weighted], reverse=True)


def test_coverage_windows_are_not_added_for_validation():
    assert _should_add_coverage_window(
        subset="train",
        blind=False,
        blind_ensure_coverage=True,
        eval_ensure_coverage=True,
    ) is False
    assert _should_add_coverage_window(
        subset="val",
        blind=False,
        blind_ensure_coverage=True,
        eval_ensure_coverage=True,
    ) is False
    assert _should_add_coverage_window(
        subset="test",
        blind=False,
        blind_ensure_coverage=True,
        eval_ensure_coverage=True,
    ) is True
    assert _should_add_coverage_window(
        subset="blind",
        blind=True,
        blind_ensure_coverage=True,
        eval_ensure_coverage=False,
    ) is True


def test_validation_uses_configured_subgraph_bootstraps():
    assert _subgraph_bootstraps_for_subset(
        subset="train",
        blind=False,
        train_subgraph_bootstraps=5,
        val_subgraph_bootstraps=3,
        test_subgraph_bootstraps=4,
    ) == 5
    assert _subgraph_bootstraps_for_subset(
        subset="val",
        blind=False,
        train_subgraph_bootstraps=5,
        val_subgraph_bootstraps=3,
        test_subgraph_bootstraps=4,
    ) == 3
    assert _subgraph_bootstraps_for_subset(
        subset="test",
        blind=False,
        train_subgraph_bootstraps=5,
        val_subgraph_bootstraps=3,
        test_subgraph_bootstraps=4,
    ) == 4
    assert _subgraph_bootstraps_for_subset(
        subset="blind",
        blind=True,
        train_subgraph_bootstraps=5,
        val_subgraph_bootstraps=3,
        test_subgraph_bootstraps=4,
    ) == 5


def test_eval_subsets_use_without_replacement_chunks():
    assert _use_without_replacement_chunks(subset="train", blind=False) is True
    assert _use_without_replacement_chunks(subset="val", blind=False) is True
    assert _use_without_replacement_chunks(subset="test", blind=False) is True
    assert _use_without_replacement_chunks(subset="blind", blind=True) is True
