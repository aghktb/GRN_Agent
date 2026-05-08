import pandas as pd

from grn_agent.io.gold_edges import load_gold_edge_labels
from grn_agent.training.examples import build_graphs_with_gold


def test_load_gold_csv_binary_map(tmp_path):
    p = tmp_path / "g.csv"
    pd.DataFrame(
        [
            {"source_tf": "A", "target_gene": "B", "regulation_type": "Activation"},
            {"source_tf": "A", "target_gene": "C", "regulation_type": "None"},
        ]
    ).to_csv(p, index=False)
    g = load_gold_edge_labels(p)
    assert g[("A", "B")] == 1
    assert g[("A", "C")] == 0


def test_load_gold_csv_gene1_gene2_aliases(tmp_path):
    p = tmp_path / "g.csv"
    pd.DataFrame([{"Gene1": "tf_a", "Gene2": "target_b"}]).to_csv(p, index=False)
    g = load_gold_edge_labels(p)
    assert g == {("TF_A", "TARGET_B"): 1}


def test_build_graphs_with_gold_filters(tmp_path):
    from grn_agent.schemas import CandidateEdge, CellContext, EvidenceGraph

    ctx = CellContext(context_id="c", module_genes=["A", "B"], candidate_tfs=["A"], cell_indices=[0])
    eg1 = EvidenceGraph(
        context=ctx,
        edge=CandidateEdge(source_tf="A", target_gene="B", context_id="c"),
        evidence={"correlation": -0.9},
    )
    eg2 = EvidenceGraph(
        context=ctx,
        edge=CandidateEdge(source_tf="A", target_gene="Z", context_id="c"),
        evidence={"correlation": 0.9},
    )
    p = tmp_path / "g.csv"
    pd.DataFrame([{"source_tf": "A", "target_gene": "B", "regulation_type": "Repression"}]).to_csv(p, index=False)
    gold = load_gold_edge_labels(p)
    gs, y = build_graphs_with_gold([eg1, eg2], gold)
    assert len(gs) == 1
    assert int(y[0]) == 1


def test_stratified_negative_sampling_uses_requested_buckets():
    import importlib.util
    import random
    from pathlib import Path

    from grn_agent.schemas import CandidateEdge, CellContext, EvidenceGraph

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_training_pairs.py"
    spec = importlib.util.spec_from_file_location("build_training_pairs", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ctx = CellContext(
        context_id="c",
        module_genes=["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11"],
        candidate_tfs=["TF1", "TF2", "TF3", "TF4", "TF5"],
        cell_indices=[0],
    )
    positives = [
        EvidenceGraph(
            context=ctx,
            edge=CandidateEdge(source_tf=tf, target_gene=target, context_id="c"),
            evidence={"correlation": 0.7, "ensemble_prior": 0.8, "motif_present": True, "accessibility": 0.7},
        )
        for tf, target in [("TF1", "G1"), ("TF2", "G2"), ("TF3", "G3")]
    ]

    negatives = []
    for tf, target in [
        ("TF1", "G4"),
        ("TF1", "G5"),
        ("TF2", "G4"),
        ("TF2", "G5"),
        ("TF3", "G4"),
        ("TF3", "G5"),
    ]:
        negatives.append(
            EvidenceGraph(
                context=ctx,
                edge=CandidateEdge(source_tf=tf, target_gene=target, context_id="c"),
                evidence={"correlation": 0.0, "ensemble_prior": 0.05, "motif_present": False, "accessibility": 0.0},
            )
        )
    for tf, target in [("TF1", "G2"), ("TF1", "G3"), ("TF2", "G1"), ("TF2", "G3"), ("TF3", "G1")]:
        negatives.append(
            EvidenceGraph(
                context=ctx,
                edge=CandidateEdge(source_tf=tf, target_gene=target, context_id="c"),
                evidence={"correlation": 0.0, "ensemble_prior": 0.05, "motif_present": False, "accessibility": 0.0},
            )
        )
    for tf, target in [
        ("TF1", "G6"),
        ("TF2", "G6"),
        ("TF3", "G6"),
        ("TF1", "G8"),
        ("TF2", "G9"),
        ("TF3", "G10"),
        ("TF1", "G11"),
    ]:
        negatives.append(
            EvidenceGraph(
                context=ctx,
                edge=CandidateEdge(source_tf=tf, target_gene=target, context_id="c"),
                evidence={"correlation": 0.0, "ensemble_prior": 0.05, "motif_present": False, "accessibility": 0.0},
            )
        )
    negatives.append(
        EvidenceGraph(
            context=ctx,
            edge=CandidateEdge(source_tf="TF2", target_gene="G7", context_id="c"),
            evidence={"correlation": 0.25, "ensemble_prior": 0.1, "motif_present": True, "accessibility": 0.0},
        )
    )
    by_pair = {mod._pair_key(g.edge.source_tf, g.edge.target_gene): g for g in [*positives, *negatives]}
    negs = mod._stratified_binary_negatives(
        positives,
        ratio=5,
        rng=random.Random(7),
        positive_edges={("TF1", "G1"), ("TF2", "G2"), ("TF3", "G3")},
        by_pair=by_pair,
    )

    assert len(negs) == 15
    buckets = [g.evidence.get("negative_sampling_bucket") for g in negs]
    assert buckets.count("same_tf") == 6
    assert buckets.count("same_gene") >= 4
    assert buckets.count("background") >= 4
    assert buckets.count("decoy_conflict") == 1
    assert all(mod._pair_key(g.edge.source_tf, g.edge.target_gene) not in {("TF1", "G1"), ("TF2", "G2"), ("TF3", "G3")} for g in negs)
    assert all(mod._pair_key(g.edge.source_tf, g.edge.target_gene) in by_pair for g in negs)


def test_stratified_negative_sampling_does_not_synthesize_missing_negatives():
    import importlib.util
    import random
    from pathlib import Path

    from grn_agent.schemas import CandidateEdge, CellContext, EvidenceGraph

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_training_pairs.py"
    spec = importlib.util.spec_from_file_location("build_training_pairs", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ctx = CellContext(
        context_id="c",
        module_genes=["G1", "G2", "G3"],
        candidate_tfs=["TF1", "TF2"],
        cell_indices=[0],
    )
    pos = EvidenceGraph(
        context=ctx,
        edge=CandidateEdge(source_tf="TF1", target_gene="G1", context_id="c"),
        evidence={"correlation": 0.7, "ensemble_prior": 0.8, "motif_present": True, "accessibility": 0.7},
    )
    negs = mod._stratified_binary_negatives(
        [pos],
        ratio=5,
        rng=random.Random(7),
        positive_edges={("TF1", "G1")},
        by_pair={("TF1", "G1"): pos},
    )
    assert negs == []


def test_ambiguous_high_evidence_edges_are_not_sampled_as_negatives():
    import importlib.util
    import random
    from pathlib import Path

    from grn_agent.schemas import CandidateEdge, CellContext, EvidenceGraph

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_training_pairs.py"
    spec = importlib.util.spec_from_file_location("build_training_pairs", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ctx = CellContext(context_id="c", module_genes=["G1", "G2"], candidate_tfs=["TF1"], cell_indices=[0])
    pos = EvidenceGraph(
        context=ctx,
        edge=CandidateEdge(source_tf="TF1", target_gene="G1", context_id="c"),
        evidence={"correlation": 0.7, "motif_present": True, "accessibility": 1.0},
    )
    ambiguous = EvidenceGraph(
        context=ctx,
        edge=CandidateEdge(source_tf="TF1", target_gene="G2", context_id="c"),
        evidence={"correlation": 0.6, "motif_present": True, "accessibility": 1.0},
    )
    negs = mod._stratified_binary_negatives(
        [pos],
        ratio=1,
        rng=random.Random(7),
        positive_edges={("TF1", "G1")},
        by_pair={("TF1", "G1"): pos, ("TF1", "G2"): ambiguous},
    )
    assert negs == []


def test_none_motif_and_accessibility_count_as_absent_negative_signal():
    import importlib.util
    import random
    from pathlib import Path

    from grn_agent.schemas import CandidateEdge, CellContext, EvidenceGraph

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_training_pairs.py"
    spec = importlib.util.spec_from_file_location("build_training_pairs", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ctx = CellContext(context_id="c", module_genes=["G1", "G2"], candidate_tfs=["TF1"], cell_indices=[0])
    pos = EvidenceGraph(
        context=ctx,
        edge=CandidateEdge(source_tf="TF1", target_gene="G1", context_id="c"),
        evidence={"correlation": 0.7, "motif_present": True, "accessibility": 1.0},
    )
    absent = EvidenceGraph(
        context=ctx,
        edge=CandidateEdge(source_tf="TF1", target_gene="G2", context_id="c"),
        evidence={"correlation": 0.05, "motif_present": None, "accessibility": None},
    )
    negs = mod._stratified_binary_negatives(
        [pos],
        ratio=1,
        rng=random.Random(7),
        positive_edges={("TF1", "G1")},
        by_pair={("TF1", "G1"): pos, ("TF1", "G2"): absent},
    )
    assert len(negs) == 1
    assert negs[0].edge.target_gene == "G2"
