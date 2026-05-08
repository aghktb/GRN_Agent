#!/usr/bin/env python3
"""Create a strict TF-heldout split manifest with train/val/test edge exclusivity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random

import pandas as pd

from grn_agent.eval.splits import validate_fold_no_leakage
from grn_agent.io.split_manifest import load_split_manifest
from grn_agent.pipeline.config import load_yaml_config
from grn_agent.schemas import SplitStrategy


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _pick_col(df: pd.DataFrame, *aliases: str, required: bool = True) -> str | None:
    cmap = {_normalize_col(c): c for c in df.columns}
    for a in aliases:
        k = _normalize_col(a)
        if k in cmap:
            return cmap[k]
    if required:
        raise ValueError(f"Missing required column one of {aliases}; got {list(df.columns)}")
    return None


def _load_unique_edges(path: Path) -> tuple[list[tuple[str, str]], dict[str, object]]:
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    c_tf = _pick_col(df, "source_tf", "tf", "source", "regulator", "gene1")
    c_tg = _pick_col(df, "target_gene", "target", "gene", "gene2")
    c_lab = _pick_col(df, "lab_id", required=False)
    c_ds = _pick_col(df, "dataset_id", required=False)
    c_ct = _pick_col(df, "cell_type", required=False)
    c_sp = _pick_col(df, "species", required=False)
    c_tfb = _pick_col(df, "tf_frequency_bucket", required=False)
    c_cut = _pick_col(df, "time_cutoff_year", required=False)

    pairs = {
        (str(r[c_tf]).strip().upper(), str(r[c_tg]).strip().upper())
        for _, r in df.iterrows()
        if str(r[c_tf]).strip() and str(r[c_tg]).strip()
    }
    meta = {
        "lab_id": (str(df[c_lab].dropna().iloc[0]).strip() if c_lab and len(df[c_lab].dropna()) else None),
        "dataset_id": (str(df[c_ds].dropna().iloc[0]).strip() if c_ds and len(df[c_ds].dropna()) else None),
        "cell_type": (str(df[c_ct].dropna().iloc[0]).strip() if c_ct and len(df[c_ct].dropna()) else None),
        "species": (str(df[c_sp].dropna().iloc[0]).strip() if c_sp and len(df[c_sp].dropna()) else None),
        "tf_frequency_bucket": (str(df[c_tfb].dropna().iloc[0]).strip() if c_tfb and len(df[c_tfb].dropna()) else None),
        "time_cutoff_year": (
            int(df[c_cut].dropna().iloc[0]) if c_cut and len(df[c_cut].dropna()) else None
        ),
    }
    return sorted(pairs), meta


def _clean_symbol(value: object) -> str:
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    return text


def _load_expression_gene_universe(path: Path, symbols_hint: set[str]) -> set[str]:
    """
    Load gene symbols from expression data.

    For BEELINE-style CSV (genes x cells), symbols are in the first column/index.
    If orientation differs (cells x genes), infer the most likely axis by overlap
    against gold symbols.
    """
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".tab"}:
        sep = "\t" if suffix in {".tsv", ".tab"} else ","
        df = pd.read_csv(path, sep=sep, index_col=0)
        idx_syms = {_clean_symbol(v) for v in df.index}
        col_syms = {_clean_symbol(v) for v in df.columns}
        idx_syms.discard("")
        col_syms.discard("")
        idx_overlap = len(idx_syms & symbols_hint)
        col_overlap = len(col_syms & symbols_hint)
        return idx_syms if idx_overlap >= col_overlap else col_syms
    if suffix in {".txt", ".list"}:
        syms = {_clean_symbol(line) for line in path.read_text(encoding="utf-8").splitlines()}
        syms.discard("")
        return syms
    raise ValueError(
        "Unsupported expression_path format for split filtering. "
        "Use .csv/.tsv/.tab (matrix) or .txt/.list (one gene per line)."
    )


def _load_tf_universe(path: Path) -> set[str]:
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        return set()
    c_tf = _pick_col(df, "tf", "source_tf", "regulator", "gene", "gene_symbol", required=False)
    values = df[c_tf] if c_tf is not None else df.iloc[:, 0]
    out = {_clean_symbol(v) for v in values}
    out.discard("")
    return out


def _assign_items_by_ratio(
    items: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    shuffled = sorted(set(items))
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)
    else:
        n_train = min(max(n_train, 1), n)
        n_val = min(max(n_val, 0), max(0, n - n_train))
    train_end = n_train
    val_end = train_end + n_val

    out: dict[str, str] = {}
    for item in shuffled[:train_end]:
        out[item] = "train"
    for item in shuffled[train_end:val_end]:
        out[item] = "val"
    for item in shuffled[val_end:]:
        out[item] = "test"
    return out


def _assign_tfs_by_edge_ratio(
    tf_to_count: dict[str, int],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    total = sum(tf_to_count.values())
    train_target = total * train_ratio
    val_target = total * val_ratio
    test_target = total - train_target - val_target
    tfs = sorted(tf_to_count.items(), key=lambda kv: (-kv[1], kv[0]))
    out: dict[str, str] = {}
    sums = {"train": 0, "val": 0, "test": 0}
    targets = {"train": train_target, "val": val_target, "test": test_target}

    def score(curr: dict[str, int]) -> float:
        return (
            abs(curr["train"] - targets["train"])
            + abs(curr["val"] - targets["val"])
            + abs(curr["test"] - targets["test"])
        )

    for tf, c in tfs:
        best_subset = "test"
        best_score = None
        for subset in ("train", "val", "test"):
            trial = dict(sums)
            trial[subset] += c
            s = score(trial)
            if best_score is None or s < best_score:
                best_score = s
                best_subset = subset
        out[tf] = best_subset
        sums[best_subset] += c

    # Ensure each subset has at least one TF, then improve by 1-TF local moves.
    if "test" not in out.values() and out:
        tf_move = min((tf for tf, s in out.items() if s == "train"), key=lambda x: tf_to_count[x], default=None)
        if tf_move is None:
            tf_move = min(out.keys(), key=lambda x: tf_to_count[x])
        sums[out[tf_move]] -= tf_to_count[tf_move]
        out[tf_move] = "test"
        sums["test"] += tf_to_count[tf_move]
    if "val" not in out.values() and len(out) >= 2:
        candidates = [tf for tf, s in out.items() if s == "train"]
        if candidates:
            tf_move = min(candidates, key=lambda x: tf_to_count[x])
            sums[out[tf_move]] -= tf_to_count[tf_move]
            out[tf_move] = "val"
            sums["val"] += tf_to_count[tf_move]

    improved = True
    while improved:
        improved = False
        base_score = score(sums)
        for tf, c in sorted(tf_to_count.items(), key=lambda kv: kv[1]):
            src = out[tf]
            for dst in ("train", "val", "test"):
                if dst == src:
                    continue
                # Keep at least one TF in every subset.
                src_tf_count = sum(1 for t, s in out.items() if s == src)
                if src_tf_count <= 1:
                    continue
                trial = dict(sums)
                trial[src] -= c
                trial[dst] += c
                s = score(trial)
                if s + 1e-9 < base_score:
                    out[tf] = dst
                    sums = trial
                    improved = True
                    base_score = s
                    break
            if improved:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="Optional YAML config; CLI args override config values")
    ap.add_argument("--gold-edges", default="", help="CSV/TSV with source_tf,target_gene (and optional metadata cols)")
    ap.add_argument("--out", default="", help="Output split manifest CSV path")
    ap.add_argument("--fold-id", default="")
    ap.add_argument(
        "--expression-path",
        default="",
        help="Optional expression matrix/genes file; when set, keep only gold edges with TF and target in this gene universe",
    )
    ap.add_argument("--tf-file", default="", help="Optional TF list; used with --node-split-mode expression")
    ap.add_argument(
        "--node-split-mode",
        default="",
        choices=["gold_edge_balanced", "expression"],
        help=(
            "gold_edge_balanced splits observed gold TFs by edge counts. "
            "expression splits expressed TFs first, then assigns gold edges by source TF partition."
        ),
    )
    ap.add_argument(
        "--target-gene-policy",
        default="",
        choices=["audit", "same_subset"],
        help=(
            "With node-split-mode=expression, audit records target_gene_subset but assigns by source TF. "
            "same_subset keeps only gold edges whose source TF and target gene landed in the same subset."
        ),
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--val-ratio", type=float, default=None)
    ap.add_argument("--test-ratio", type=float, default=None)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config) if args.config.strip() else {}

    def _cfg(key: str, default):
        if key in cfg:
            return cfg[key]
        alt = key.replace("_", "-")
        if alt in cfg:
            return cfg[alt]
        return default

    gold_edges = str(args.gold_edges or _cfg("gold_edges", ""))
    out = str(args.out or _cfg("out", ""))
    fold_id = str(args.fold_id or _cfg("fold_id", "tf_holdout_701020"))
    expression_path = str(args.expression_path or _cfg("expression_path", _cfg("expr", "")))
    tf_file = str(args.tf_file or _cfg("tf_file", ""))
    node_split_mode = str(args.node_split_mode or _cfg("node_split_mode", "gold_edge_balanced"))
    target_gene_policy = str(args.target_gene_policy or _cfg("target_gene_policy", "audit"))
    seed = int(args.seed if args.seed is not None else _cfg("seed", 0))
    train_ratio = float(args.train_ratio if args.train_ratio is not None else _cfg("train_ratio", 0.7))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else _cfg("val_ratio", 0.1))
    test_ratio = float(args.test_ratio if args.test_ratio is not None else _cfg("test_ratio", 0.2))

    if not gold_edges.strip() or not out.strip():
        raise SystemExit("Missing required args: --gold-edges and --out (or set in --config)")

    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r <= 0.0 for r in ratios):
        raise SystemExit("All ratios must be > 0")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("Ratios must sum to 1.0")

    pairs, meta = _load_unique_edges(Path(gold_edges))
    if not pairs:
        raise SystemExit("No edges found in --gold-edges")
    n_before_filter = len(pairs)
    gene_universe: set[str] = set()
    if expression_path.strip():
        symbols_hint = {s for p in pairs for s in p}
        gene_universe = _load_expression_gene_universe(Path(expression_path), symbols_hint)
        pairs = [(tf, tg) for tf, tg in pairs if tf in gene_universe and tg in gene_universe]
        if not pairs:
            raise SystemExit(
                "No gold edges remained after filtering to expression gene universe; "
                "check --expression-path and symbol harmonization."
            )
    target_gene_subset: dict[str, str] = {}
    if node_split_mode == "expression":
        if not expression_path.strip():
            raise SystemExit("node_split_mode=expression requires --expression-path")
        if not tf_file.strip():
            raise SystemExit("node_split_mode=expression requires --tf-file")
        expressed_tfs = sorted(gene_universe & _load_tf_universe(Path(tf_file)))
        if not expressed_tfs:
            raise SystemExit("No TFs from --tf-file were found in the expression gene universe")
        tf_subset = _assign_items_by_ratio(expressed_tfs, train_ratio, val_ratio, seed)
        target_gene_subset = _assign_items_by_ratio(sorted(gene_universe), train_ratio, val_ratio, seed + 1)
        pairs = [(tf, tg) for tf, tg in pairs if tf in tf_subset]
        if target_gene_policy == "same_subset":
            pairs = [(tf, tg) for tf, tg in pairs if tf_subset[tf] == target_gene_subset.get(tg)]
        if not pairs:
            raise SystemExit("No gold edges remained after applying expression node splits")
    else:
        tf_to_count: dict[str, int] = defaultdict(int)
        for tf, _ in pairs:
            tf_to_count[tf] += 1
        tf_subset = _assign_tfs_by_edge_ratio(tf_to_count, train_ratio, val_ratio)

    rows: list[dict[str, object]] = []
    for tf, tg in pairs:
        subset = tf_subset[tf]
        row = {
            "split_name": "leave_one_tf_out",
            "fold_id": fold_id,
            "subset": subset,
            "source_tf": tf,
            "target_gene": tg,
            "lab_id": meta["lab_id"],
            "dataset_id": meta["dataset_id"],
            "cell_type": meta["cell_type"],
            "species": meta["species"],
            "tf_frequency_bucket": meta["tf_frequency_bucket"],
            "time_cutoff_year": meta["time_cutoff_year"],
        }
        if node_split_mode == "expression":
            row["source_tf_subset"] = subset
            row["target_gene_subset"] = target_gene_subset.get(tg, "")
        rows.append(row)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    manifest = load_split_manifest(out_path)
    validate_fold_no_leakage(manifest, strategy=SplitStrategy.leave_one_tf_out, fold_id=fold_id)

    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        counts[str(r["subset"])] += 1
    total = len(rows)
    print(
        f"Wrote {out_path} with {total} unique edges: "
        f"train={counts['train']} ({counts['train']/total:.3f}), "
        f"val={counts['val']} ({counts['val']/total:.3f}), "
        f"test={counts['test']} ({counts['test']/total:.3f})"
    )
    if node_split_mode == "expression":
        subset_tfs: dict[str, set[str]] = defaultdict(set)
        subset_targets: dict[str, int] = defaultdict(int)
        for tf, subset in tf_subset.items():
            subset_tfs[subset].add(tf)
        for _, subset in target_gene_subset.items():
            subset_targets[subset] += 1
        print(
            "Expression node split: "
            f"expressed_tfs={len(tf_subset)} "
            f"tf_train={len(subset_tfs['train'])} tf_val={len(subset_tfs['val'])} tf_test={len(subset_tfs['test'])} "
            f"genes_train={subset_targets['train']} genes_val={subset_targets['val']} genes_test={subset_targets['test']} "
            f"target_gene_policy={target_gene_policy}"
        )
    if expression_path.strip():
        removed = n_before_filter - total
        print(
            f"Expression-universe filter applied from {expression_path}: "
            f"kept={total}, removed={removed}",
            flush=True,
        )


if __name__ == "__main__":
    main()
