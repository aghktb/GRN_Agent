#!/usr/bin/env python3
"""Build EAGER training files: matching evidence_graphs.jsonl subset and binary y (.npz)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from grn_agent.eval.splits import validate_fold_no_leakage
from grn_agent.io import load_split_manifest
from grn_agent.io.gold_edges import load_gold_edge_labels
from grn_agent.pipeline.config import load_yaml_config
from grn_agent.schemas import EvidenceGraph, SplitStrategy, SplitSubset
from grn_agent.training.examples import build_graphs_with_gold, label_binary_from_evidence_graph


def _pair_key(tf: str, tg: str) -> tuple[str, str]:
    return (str(tf).strip().upper(), str(tg).strip().upper())


def _as_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _has_other_omics(eg: EvidenceGraph) -> bool:
    e = eg.evidence
    return "motif_present" in e or "accessibility" in e


TAU_CORR_LOW = 0.20
TAU_ACC_LOW = 0.0
TAU_CORR_HIGH = 0.20
TAU_ACC_HIGH = 0.0
EXPRESSION_NEGATIVE_WEIGHT = 0.5


def _dataset_modalities(graphs: list[EvidenceGraph]) -> str:
    has_motif = any(g.evidence.get("motif_present") is not None for g in graphs)
    has_accessibility = any(g.evidence.get("accessibility") is not None for g in graphs)
    if has_motif and has_accessibility:
        return "expression_motif_accessibility"
    if has_motif:
        return "expression_motif"
    return "expression_only"


def _linkage_weak_or_absent(eg: EvidenceGraph) -> bool:
    for node in eg.nodes:
        if node.node_id == "ev_binding":
            linked = node.payload.get("peak_to_gene_linked")
            return linked is not True
    return True


def _motif_present(v: object) -> bool:
    return v is True


def _motif_absent(v: object) -> bool:
    return v is False or v is None


def _detectable_expression(eg: EvidenceGraph) -> bool:
    for node in eg.nodes:
        if node.node_id == "ev_expr":
            payload = node.payload
            return payload.get("mean_expr_t") is not None and payload.get("mean_expr_g") is not None
    return eg.evidence.get("z_t") is not None and eg.evidence.get("z_g") is not None


def _is_reliable_negative(eg: EvidenceGraph, modality_case: str, corr_low_threshold: float = TAU_CORR_LOW) -> bool:
    e = eg.evidence
    corr = abs(_as_float(e.get("correlation"), 0.0))
    motif = e.get("motif_present")
    acc = _as_float(e.get("accessibility"), 0.0)
    if modality_case == "expression_only":
        return corr <= corr_low_threshold and _detectable_expression(eg)
    if modality_case == "expression_motif":
        return corr < TAU_CORR_LOW and _motif_absent(motif)
    return corr < TAU_CORR_LOW and _motif_absent(motif) and acc <= TAU_ACC_LOW and _linkage_weak_or_absent(eg)


def _is_strong_evidence_all_modalities(eg: EvidenceGraph) -> bool:
    e = eg.evidence
    corr = abs(_as_float(e.get("correlation"), 0.0))
    motif = e.get("motif_present")
    acc = _as_float(e.get("accessibility"), 0.0)
    return corr > TAU_CORR_HIGH and _motif_present(motif) and acc > TAU_ACC_HIGH and not _linkage_weak_or_absent(eg)


def _is_ambiguous_negative(eg: EvidenceGraph) -> bool:
    e = eg.evidence
    corr = abs(_as_float(e.get("correlation"), 0.0))
    motif = e.get("motif_present")
    acc = _as_float(e.get("accessibility"), 0.0)
    return corr > TAU_CORR_HIGH and _motif_present(motif) and acc > TAU_ACC_HIGH


def _is_decoy_conflict_negative(eg: EvidenceGraph) -> bool:
    if not _has_other_omics(eg):
        return False
    if _is_strong_evidence_all_modalities(eg) or _is_ambiguous_negative(eg):
        return False
    e = eg.evidence
    corr = abs(_as_float(e.get("correlation"), 0.0))
    motif = e.get("motif_present")
    acc = _as_float(e.get("accessibility"), 0.0)
    return (
        (_motif_present(motif) and acc <= TAU_ACC_LOW)
        or (corr > TAU_CORR_HIGH and _motif_absent(motif))
        or (acc > TAU_ACC_HIGH and _motif_absent(motif))
    )


def _tag_negative(eg: EvidenceGraph, bucket: str) -> EvidenceGraph:
    ev = dict(eg.evidence)
    ev["negative_sampling_bucket"] = bucket
    return eg.model_copy(update={"evidence": ev})


def _bucket_counts(total: int, include_decoy: bool) -> dict[str, int]:
    weights = {
        "same_tf": 0.40,
        "same_gene": 0.25,
        "background": 0.25,
    }
    if include_decoy:
        weights["decoy_conflict"] = 0.10
    else:
        # Redistribute the decoy share while preserving the other bucket proportions.
        weights = {k: v / 0.90 for k, v in weights.items()}
    raw = {k: total * v for k, v in weights.items()}
    counts = {k: int(raw[k]) for k in raw}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def _stratified_binary_negatives(
    positives: list[EvidenceGraph],
    ratio: int,
    rng: random.Random,
    positive_edges: set[tuple[str, str]],
    by_pair: dict[tuple[str, str], EvidenceGraph],
) -> list[EvidenceGraph]:
    if ratio <= 0 or not positives:
        return []
    positive_tfs = {eg.edge.source_tf.strip().upper() for eg in positives}
    positive_targets = {eg.edge.target_gene.strip().upper() for eg in positives}
    modality_case = _dataset_modalities(list(by_pair.values()))
    candidate_corrs = [
        abs(_as_float(eg.evidence.get("correlation"), 0.0))
        for key, eg in by_pair.items()
        if key not in positive_edges and key[0] in positive_tfs
    ]
    expr_corr_threshold = (
        float(np.quantile(candidate_corrs, 0.30))
        if modality_case == "expression_only" and candidate_corrs
        else TAU_CORR_LOW
    )
    include_decoy = any(_has_other_omics(g) for g in by_pair.values())
    counts = _bucket_counts(len(positives) * ratio, include_decoy=include_decoy)
    pools: dict[str, list[EvidenceGraph]] = {k: [] for k in counts}
    decoys: list[EvidenceGraph] = []
    fill_candidates: list[EvidenceGraph] = []
    tf_background_shared: list[EvidenceGraph] = []

    for key, eg in by_pair.items():
        if key in positive_edges:
            continue
        tf, target = key
        if tf not in positive_tfs:
            continue
        if _is_ambiguous_negative(eg):
            continue
        if include_decoy and _is_decoy_conflict_negative(eg):
            decoys.append(eg)
            fill_candidates.append(eg)
            continue
        if _is_reliable_negative(eg, modality_case, expr_corr_threshold):
            fill_candidates.append(eg)
            if target in positive_targets:
                pools["same_gene"].append(eg)
            else:
                tf_background_shared.append(eg)

    if tf_background_shared:
        rng.shuffle(tf_background_shared)
        same_tf_target = counts.get("same_tf", 0)
        background_target = counts.get("background", 0)
        shared_target = same_tf_target + background_target
        if len(tf_background_shared) <= shared_target and shared_target > 0:
            n_same_tf = int(round(len(tf_background_shared) * same_tf_target / shared_target))
        else:
            n_same_tf = min(same_tf_target, len(tf_background_shared))
        pools["same_tf"].extend(tf_background_shared[:n_same_tf])
        pools["background"].extend(tf_background_shared[n_same_tf:])

    out: list[EvidenceGraph] = []
    used: set[tuple[str, str]] = set()

    def add_from_pool(bucket: str, n: int) -> None:
        rng.shuffle(pools[bucket])
        for eg in pools[bucket]:
            if len([g for g in out if g.evidence.get("negative_sampling_bucket") == bucket]) >= n:
                break
            key = _pair_key(eg.edge.source_tf, eg.edge.target_gene)
            if key in used or key in positive_edges:
                continue
            out.append(_tag_negative(eg, bucket))
            used.add(key)

    for bucket in ("same_gene", "background", "same_tf"):
        add_from_pool(bucket, counts.get(bucket, 0))

    if include_decoy and counts.get("decoy_conflict", 0) > 0:
        rng.shuffle(decoys)
        for eg in decoys:
            if len([g for g in out if g.evidence.get("negative_sampling_bucket") == "decoy_conflict"]) >= counts["decoy_conflict"]:
                break
            key = _pair_key(eg.edge.source_tf, eg.edge.target_gene)
            if key in used or key in positive_edges:
                continue
            ev = dict(eg.evidence)
            ev["negative_sampling_bucket"] = "decoy_conflict"
            out.append(eg.model_copy(update={"evidence": ev}))
            used.add(key)

    target_total = len(positives) * ratio
    if len(out) < target_total:
        fill_pool: list[EvidenceGraph] = list(fill_candidates)
        for bucket in ("same_tf", "same_gene", "background"):
            fill_pool.extend(pools.get(bucket, []))
        if include_decoy:
            fill_pool.extend(decoys)
        rng.shuffle(fill_pool)
        for eg in fill_pool:
            if len(out) >= target_total:
                break
            key = _pair_key(eg.edge.source_tf, eg.edge.target_gene)
            if key in used or key in positive_edges:
                continue
            out.append(_tag_negative(eg, "available_fill"))
            used.add(key)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="Optional YAML config; CLI args override config values")
    ap.add_argument("--evidence-jsonl", default="", help="evidence_graphs.jsonl from run")
    ap.add_argument("--out-prefix", default="", help="Prefix for {prefix}_graphs.jsonl and {prefix}_y.npz")
    ap.add_argument(
        "--gold-edges",
        default="",
        help="CSV/TSV with source_tf, target_gene, optional label / regulation_type (binary 0/1 or signed→present)",
    )
    ap.add_argument("--split-manifest", default="", help="CSV/TSV split manifest for fold/subset filtering")
    ap.add_argument("--strategy", default="", help="Split strategy name in manifest")
    ap.add_argument("--fold-id", default="", help="Fold id in manifest")
    ap.add_argument("--subset", default="", choices=["", "train", "val", "test"])
    ap.add_argument("--negative-ratio", type=int, default=None, help="Synthetic negatives per positive edge")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config) if args.config.strip() else {}

    def _cfg(key: str, default):
        if key in cfg:
            return cfg[key]
        alt = key.replace("_", "-")
        if alt in cfg:
            return cfg[alt]
        return default

    evidence_jsonl = str(args.evidence_jsonl or _cfg("evidence_jsonl", ""))
    out_prefix = str(args.out_prefix or _cfg("out_prefix", ""))
    gold_edges = str(args.gold_edges or _cfg("gold_edges", ""))
    split_manifest = str(args.split_manifest or _cfg("split_manifest", ""))
    strategy = str(args.strategy or _cfg("strategy", ""))
    fold_id = str(args.fold_id or _cfg("fold_id", ""))
    subset = str(args.subset or _cfg("subset", "train"))
    negative_ratio = int(args.negative_ratio if args.negative_ratio is not None else _cfg("negative_ratio", 0))
    seed = int(args.seed if args.seed is not None else _cfg("seed", 42))

    if not evidence_jsonl.strip() or not out_prefix.strip():
        raise SystemExit("Missing required args: --evidence-jsonl and --out-prefix (or set in --config)")

    rng = random.Random(seed)

    graphs: list[EvidenceGraph] = []
    with open(evidence_jsonl, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            graphs.append(EvidenceGraph.model_validate(json.loads(line)))
    all_graphs = list(graphs)

    if split_manifest.strip():
        if not strategy.strip() or not fold_id.strip():
            raise SystemExit("--strategy and --fold-id are required with --split-manifest")
        manifest = load_split_manifest(split_manifest)
        strategy_e = SplitStrategy(strategy.strip())
        subset_e = SplitSubset(subset.strip())
        validate_fold_no_leakage(manifest, strategy_e, fold_id.strip())
        allowed = {
            _pair_key(r.source_tf, r.target_gene)
            for r in manifest.rows
            if r.split_name == strategy_e and r.fold_id == fold_id.strip() and r.subset == subset_e
        }
        graphs = [g for g in graphs if _pair_key(g.edge.source_tf, g.edge.target_gene) in allowed]
        if not graphs:
            raise SystemExit("No evidence graphs match the requested manifest strategy/fold/subset.")

    prefix = Path(out_prefix)
    prefix = prefix.parent / prefix.name

    if gold_edges.strip():
        gold = load_gold_edge_labels(gold_edges)
        sel, y = build_graphs_with_gold(graphs, gold)
        weights = np.ones(len(sel), dtype=np.float32)
        if negative_ratio > 0:
            positive_edges = {k for k, v in gold.items() if int(v) == 1}
            by_pair = {_pair_key(g.edge.source_tf, g.edge.target_gene): g for g in all_graphs}
            modality_case = _dataset_modalities(list(by_pair.values()))
            base_graphs = list(sel)
            base_y = y.tolist()
            positives = [eg for eg, lab in zip(base_graphs, base_y) if int(lab) == 1]
            negatives = _stratified_binary_negatives(
                positives,
                negative_ratio,
                rng,
                positive_edges,
                by_pair,
            )
            requested_negatives = len(positives) * negative_ratio
            if len(negatives) < requested_negatives:
                print(
                    "Warning: requested "
                    f"{requested_negatives} negatives for ratio {negative_ratio}:1, "
                    f"but only {len(negatives)} qualifying real evidence-graph negatives were available.",
                    file=sys.stderr,
                    flush=True,
            )
            sel.extend(negatives)
            base_y.extend([0.0] * len(negatives))
            neg_weights = [
                EXPRESSION_NEGATIVE_WEIGHT
                if g.evidence.get("negative_sampling_bucket") in {"same_tf", "same_gene", "background", "available_fill"}
                and modality_case == "expression_only"
                else 1.0
                for g in negatives
            ]
            weights = np.concatenate([weights, np.asarray(neg_weights, dtype=np.float32)])
            y = np.asarray(base_y, dtype=np.float32)
    else:
        sel = graphs
        y = np.array([label_binary_from_evidence_graph(g) for g in graphs], dtype=np.float32)
        weights = np.ones(len(sel), dtype=np.float32)

    out_jsonl = Path(str(out_prefix) + "_graphs.jsonl")
    out_y = Path(str(out_prefix) + "_y.npz")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for g in sel:
            fp.write(json.dumps(g.model_dump(mode="json")) + "\n")
    np.savez(out_y, y=y, sample_weight=weights)
    print(f"Wrote {out_jsonl} ({len(sel)} graphs) and {out_y} y={y.shape}")


if __name__ == "__main__":
    main()
