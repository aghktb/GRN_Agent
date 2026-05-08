"""
Compare exported network CSV to gold edge labels (binary present / absent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import random

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score
import torch
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from grn_agent.eval.metrics import (
    expected_calibration_error,
    precision_at_k,
    recall_at_k,
)
from grn_agent.io.gold_edges import inspect_gold_label_mode, load_gold_edge_labels
from grn_agent.schemas import EvidenceGraph, SplitStrategy, SplitSubset
from grn_agent.training.examples import label_binary_from_evidence_graph


def load_evidence_graphs(path: str | Path) -> list[EvidenceGraph]:
    graphs: list[EvidenceGraph] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            graphs.append(EvidenceGraph.model_validate(json.loads(line)))
    return graphs


def load_evidence_pair_records(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            edge = obj.get("edge") or {}
            k = (str(edge.get("source_tf", "")), str(edge.get("target_gene", "")))
            if not k[0] or not k[1]:
                continue
            ctx = obj.get("context") or {}
            meta = ctx.get("metadata") or {}
            records[k] = {
                "cell_type": ctx.get("cell_type"),
                "species": meta.get("species"),
                "evidence": obj.get("evidence") or {},
            }
    return records


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _pick_col(df: pd.DataFrame, *aliases: str, required: bool = True) -> str | None:
    cmap = {_normalize_col(c): c for c in df.columns}
    for name in aliases:
        key = _normalize_col(name)
        if key in cmap:
            return cmap[key]
    if required:
        raise ValueError(f"Missing required column one of {aliases}; got {list(df.columns)}")
    return None


def _clean_optional(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_split_subset_records(
    path: str | Path,
    strategy: str,
    fold_id: str,
    subset: str,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], dict[str, str | None]]]:
    p = Path(path)
    sep = "\t" if p.suffix.lower() in (".tsv", ".tab") else ","
    header = pd.read_csv(p, sep=sep, nrows=0)
    c_split = _pick_col(header, "split_name", "strategy")
    c_fold = _pick_col(header, "fold_id", "fold")
    c_subset = _pick_col(header, "subset")
    c_tf = _pick_col(header, "source_tf", "tf", "source")
    c_tg = _pick_col(header, "target_gene", "target", "gene")
    c_ct = _pick_col(header, "cell_type", required=False)
    c_sp = _pick_col(header, "species", required=False)
    c_tfb = _pick_col(header, "tf_frequency_bucket", required=False)
    usecols = [c for c in (c_split, c_fold, c_subset, c_tf, c_tg, c_ct, c_sp, c_tfb) if c is not None]
    df = pd.read_csv(p, sep=sep, dtype=str, usecols=usecols)
    mask = (
        df[c_split].astype(str).str.strip().eq(strategy)
        & df[c_fold].astype(str).str.strip().eq(fold_id)
        & df[c_subset].astype(str).str.strip().eq(subset)
    )
    sub_df = df.loc[mask]
    allowed: set[tuple[str, str]] = set()
    lookup: dict[tuple[str, str], dict[str, str | None]] = {}
    for _, row in sub_df.iterrows():
        pair = (str(row[c_tf]).strip().upper(), str(row[c_tg]).strip().upper())
        allowed.add(pair)
        lookup[pair] = {
            "cell_type": _clean_optional(row[c_ct]) if c_ct is not None else None,
            "species": _clean_optional(row[c_sp]) if c_sp is not None else None,
            "tf_frequency_bucket": _clean_optional(row[c_tfb]) if c_tfb is not None else None,
        }
    return allowed, lookup


def _safe_auroc_binary(y_t: np.ndarray, scores: np.ndarray) -> float | None:
    if len(y_t) == 0 or len(np.unique(y_t)) < 2:
        return None
    try:
        return float(roc_auc_score(y_t, scores))
    except ValueError:
        return None


def _binary_metrics(y_t: np.ndarray, p_present: np.ndarray, k_values: list[int] | None = None) -> dict[str, Any]:
    y_t = y_t.astype(np.int64)
    p_present = p_present.astype(np.float64)
    probs2 = np.stack([1.0 - p_present, p_present], axis=1)
    n_pos = int((y_t == 1).sum())
    n_neg = int((y_t == 0).sum())
    has_both_classes = n_pos > 0 and n_neg > 0
    y_true_t = torch.tensor(y_t, dtype=torch.int64)
    y_prob_t = torch.tensor(p_present, dtype=torch.float32)
    acc_m = BinaryAccuracy()
    pre_m = BinaryPrecision()
    rec_m = BinaryRecall()
    f1_m = BinaryF1Score()
    acc = float(acc_m(y_prob_t, y_true_t).item())
    prec = float(pre_m(y_prob_t, y_true_t).item())
    rec = float(rec_m(y_prob_t, y_true_t).item())
    f1 = float(f1_m(y_prob_t, y_true_t).item())
    ap: float | None
    roc: float | None
    if has_both_classes:
        auprc_m = BinaryAveragePrecision()
        auroc_m = BinaryAUROC()
        ap = float(auprc_m(y_prob_t, y_true_t).item())
        roc = _safe_auroc_binary(y_t, p_present)
        if roc is None:
            roc = float(auroc_m(y_prob_t, y_true_t).item())
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_t, p_present)
    else:
        # Ranking metrics are not meaningful for single-class subsets.
        ap = None
        roc = None
        precision_curve = np.asarray([], dtype=np.float64)
        recall_curve = np.asarray([], dtype=np.float64)
        thresholds = np.asarray([], dtype=np.float64)
    out: dict[str, Any] = {
        "n_matched": int(len(y_t)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "has_both_classes": bool(has_both_classes),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "precision_macro": prec,
        "recall_macro": rec,
        "precision_micro": prec,
        "recall_micro": rec,
        "f1_macro": f1,
        "f1_weighted": f1,
        "ece": expected_calibration_error(probs2, y_t),
        "brier": float(np.mean((p_present - y_t) ** 2)),
        "aucpr_macro": ap,
        "aucpr_micro": ap,
        "auprc_macro": ap,
        "auprc_micro": ap,
        "auroc_macro": roc,
        "auroc_micro": roc,
        "auroc": roc,
        "task_mode": "binary_presence",
    }
    if has_both_classes:
        out["pr_curve"] = {
            "precision": [float(x) for x in precision_curve.tolist()],
            "recall": [float(x) for x in recall_curve.tolist()],
            "thresholds": [float(x) for x in thresholds.tolist()],
            "positive_prevalence": float(n_pos / max(1, len(y_t))),
        }
    if not has_both_classes:
        out["metric_warning"] = "Single-class subset: AUROC/AUPRC are undefined; collect both positives and negatives."
    ks = k_values or [10, 50, 100]
    pk: dict[str, float] = {}
    rk: dict[str, float] = {}
    for k in ks:
        k_eff = min(int(k), len(y_t))
        pk[f"present@{k}"] = precision_at_k(p_present, y_t, k_eff)
        rk[f"present@{k}"] = recall_at_k(p_present, y_t, k_eff)
    out["precision_at_k"] = pk
    out["recall_at_k"] = rk
    return out


def _sample_rows_by_negative_ratio(
    rows: list[dict[str, Any]],
    *,
    negative_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    positives = [r for r in rows if int(r["y_true"]) == 1]
    negatives = [r for r in rows if int(r["y_true"]) == 0]
    rng = random.Random(seed)
    n_neg = min(len(negatives), int(round(float(negative_ratio) * len(positives))))
    sampled_negatives = rng.sample(negatives, n_neg) if n_neg > 0 else []
    return positives + sampled_negatives


def _metrics_from_rows(
    rows: list[dict[str, Any]],
    *,
    k_values: list[int] | None,
    label_source: str,
    negative_ratio: float | None,
) -> dict[str, Any]:
    if not rows:
        out = {"n_matched": 0, "error": "no_rows_after_negative_ratio_sampling"}
        if negative_ratio is not None and negative_ratio > 0:
            out["negative_ratio"] = float(negative_ratio)
        return out
    df = pd.DataFrame(rows)
    y_t = df["y_true"].to_numpy(dtype=np.int64)
    p_present = df["p_present"].to_numpy(dtype=np.float64)
    out = _binary_metrics(y_t, p_present, k_values=k_values)
    out["label_source"] = label_source
    if negative_ratio is not None and negative_ratio > 0:
        out["negative_ratio"] = float(negative_ratio)
    return out


def _is_metric_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f))


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _aggregate_repeated_reports(reports: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    if not reports:
        return {"n_matched": 0, "error": "no_repeated_reports"}
    if len(reports) == 1:
        out = dict(reports[0])
        out["negative_sampling_repeats"] = 1
        out["negative_sampling_seed"] = int(seed)
        return out

    out = dict(reports[0])
    stds: dict[str, Any] = {}
    metadata_keys = {
        "n_matched",
        "n_positive",
        "n_negative",
        "has_both_classes",
        "task_mode",
        "label_source",
        "negative_ratio",
        "strategy",
        "fold_id",
        "subset",
        "n_allowed_pairs",
        "n_excluded_outside_subset",
        "n_candidate_pairs",
        "n_prediction_pairs",
        "n_prediction_pairs_in_subset",
        "n_prediction_pairs_in_allowed_manifest",
        "n_eval_pairs_with_prediction",
        "n_eval_pairs_missing_prediction",
    }
    for key in sorted({k for r in reports for k in r}):
        if key in metadata_keys or key in {"precision_at_k", "recall_at_k", "robustness_slices"}:
            continue
        values = [float(r[key]) for r in reports if _is_metric_number(r.get(key))]
        if len(values) == len(reports):
            mean, std = _mean_std(values)
            out[key] = mean
            stds[key] = std

    for nested_key in ("precision_at_k", "recall_at_k"):
        nested_means: dict[str, float] = {}
        nested_stds: dict[str, float] = {}
        nested_labels = sorted(
            {
                label
                for r in reports
                for label in ((r.get(nested_key) or {}).keys() if isinstance(r.get(nested_key), dict) else [])
            }
        )
        for label in nested_labels:
            values = [
                float((r.get(nested_key) or {}).get(label))
                for r in reports
                if _is_metric_number((r.get(nested_key) or {}).get(label))
            ]
            if len(values) == len(reports):
                mean, std = _mean_std(values)
                nested_means[label] = mean
                nested_stds[label] = std
        if nested_means:
            out[nested_key] = nested_means
            stds[nested_key] = nested_stds

    pr_curves = [r.get("pr_curve") for r in reports if isinstance(r.get("pr_curve"), dict)]
    if len(pr_curves) == len(reports) and pr_curves:
        recall_grid = np.linspace(0.0, 1.0, 201, dtype=np.float64)
        precision_rows: list[np.ndarray] = []
        prevalences: list[float] = []
        for curve in pr_curves:
            recall_vals = np.asarray(curve.get("recall") or [], dtype=np.float64)
            precision_vals = np.asarray(curve.get("precision") or [], dtype=np.float64)
            if recall_vals.size == 0 or precision_vals.size == 0 or recall_vals.size != precision_vals.size:
                continue
            order = np.argsort(recall_vals)
            recall_sorted = recall_vals[order]
            precision_sorted = precision_vals[order]
            precision_interp = np.interp(recall_grid, recall_sorted, precision_sorted)
            precision_rows.append(precision_interp)
            prev = curve.get("positive_prevalence")
            if _is_metric_number(prev):
                prevalences.append(float(prev))
        if precision_rows:
            precision_arr = np.vstack(precision_rows)
            out["pr_curve"] = {
                "recall": [float(x) for x in recall_grid.tolist()],
                "precision_mean": [float(x) for x in precision_arr.mean(axis=0).tolist()],
                "precision_std": [float(x) for x in precision_arr.std(axis=0, ddof=0).tolist()],
                "positive_prevalence_mean": float(np.mean(prevalences)) if prevalences else None,
            }

    out["negative_sampling_repeats"] = int(len(reports))
    out["negative_sampling_seed"] = int(seed)
    out["negative_sampling_metric_std"] = stds
    return out


def _slice_values(row: pd.Series, key: str, by_pair: dict[tuple[str, str], EvidenceGraph], manifest_lookup: dict[tuple[str, str], dict[str, str | None]]) -> str:
    pair = (str(row["source_tf"]), str(row["target_gene"]))
    if pair in manifest_lookup and manifest_lookup[pair].get(key):
        return str(manifest_lookup[pair][key])
    eg = by_pair.get(pair)
    if eg is None:
        return "unknown"
    if key == "cell_type":
        return str(eg.context.cell_type or "unknown")
    if key == "species":
        return str(eg.context.metadata.get("species") or "unknown")
    if key == "tf_frequency_bucket":
        return "unknown"
    return "unknown"


def _slice_values_from_record(
    row: pd.Series,
    key: str,
    by_pair: dict[tuple[str, str], dict[str, Any]],
    manifest_lookup: dict[tuple[str, str], dict[str, str | None]],
) -> str:
    pair = (str(row["source_tf"]), str(row["target_gene"]))
    if pair in manifest_lookup and manifest_lookup[pair].get(key):
        return str(manifest_lookup[pair][key])
    rec = by_pair.get(pair)
    if rec is None:
        return "unknown"
    if key in {"cell_type", "species"} and rec.get(key):
        return str(rec[key])
    if key == "tf_frequency_bucket":
        return "unknown"
    return "unknown"


def _y_from_gold(k: tuple[str, str], gold_map: dict[tuple[str, str], int] | None) -> int:
    if gold_map is None:
        raise ValueError("gold_map required")
    return int(gold_map.get(k, 0))


def _label_binary_from_evidence_record(rec: dict[str, Any]) -> int:
    c = (rec.get("evidence") or {}).get("correlation")
    if c is None:
        return 0
    return 1 if abs(float(c)) > 0.12 else 0


def evaluate_network_vs_labels(
    network_csv: str | Path,
    evidence_jsonl: str | Path,
    gold_edges: str | Path | None = None,
    k_values: list[int] | None = None,
    negative_ratio: float | None = None,
    negative_repeats: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Join predictions to rows on (source_tf, target_gene).
    Gold file supplies binary labels; missing (tf,g) in gold = 0.
    """
    pred_df = pd.read_csv(network_csv)
    graphs = load_evidence_graphs(evidence_jsonl)
    by_pair: dict[tuple[str, str], EvidenceGraph] = {
        (g.edge.source_tf, g.edge.target_gene): g for g in graphs
    }
    gold_map: dict[tuple[str, str], int] | None = None
    if gold_edges is not None:
        _ = inspect_gold_label_mode(gold_edges)
        gold_map = load_gold_edge_labels(gold_edges)

    rows: list[dict[str, Any]] = []
    for _, row in pred_df.iterrows():
        k = (str(row["source_tf"]), str(row["target_gene"]))
        if k not in by_pair:
            continue
        if gold_map is not None:
            y_true = _y_from_gold(k, gold_map)
        else:
            y_true = int(label_binary_from_evidence_graph(by_pair[k]))
        p_present = float(row["p_present"])
        rows.append(
            {
                "source_tf": k[0],
                "target_gene": k[1],
                "y_true": y_true,
                "y_pred": int(p_present >= 0.5),
                "p_present": p_present,
            }
        )

    if not rows:
        return {"n_matched": 0, "error": "no_rows_matched_evidence_or_gold"}

    if negative_ratio is not None and negative_ratio > 0:
        reports = [
            _metrics_from_rows(
                _sample_rows_by_negative_ratio(rows, negative_ratio=float(negative_ratio), seed=seed + i),
                k_values=k_values,
                label_source="gold_file" if gold_map is not None else "weak_correlation",
                negative_ratio=float(negative_ratio),
            )
            for i in range(max(1, int(negative_repeats)))
        ]
        return _aggregate_repeated_reports(reports, seed=seed)

    return _metrics_from_rows(
        rows,
        k_values=k_values,
        label_source="gold_file" if gold_map is not None else "weak_correlation",
        negative_ratio=negative_ratio,
    )


def evaluate_network_with_manifest(
    network_csv: str | Path,
    evidence_jsonl: str | Path,
    split_manifest: str | Path,
    strategy: str,
    fold_id: str,
    subset: str = "test",
    gold_edges: str | Path | None = None,
    k_values: list[int] | None = None,
    negative_ratio: float | None = None,
    negative_repeats: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    pred_df = pd.read_csv(network_csv)
    by_pair = load_evidence_pair_records(evidence_jsonl)
    pred_map: dict[tuple[str, str], float] = {
        (str(r["source_tf"]), str(r["target_gene"])): float(r["p_present"]) for _, r in pred_df.iterrows()
    }
    strat = SplitStrategy(strategy)
    sub = SplitSubset(subset)
    allowed, manifest_lookup = load_split_subset_records(split_manifest, strat.value, fold_id, sub.value)
    gold_map: dict[tuple[str, str], int] | None = None
    if gold_edges is not None:
        _ = inspect_gold_label_mode(gold_edges)
        gold_map = load_gold_edge_labels(gold_edges)

    subset_tfs_all = {tf for tf, _ in allowed}
    n_prediction_pairs_in_allowed_manifest = int(sum(1 for k in pred_map if k in allowed))
    n_prediction_pairs_for_subset_tfs = int(sum(1 for k in pred_map if k[0] in subset_tfs_all))
    if negative_ratio is not None and negative_ratio > 0:
        excluded = int(max(0, len(pred_map) - n_prediction_pairs_for_subset_tfs))
    else:
        excluded = int(max(0, len(pred_map) - n_prediction_pairs_in_allowed_manifest))
    if negative_ratio is not None and negative_ratio > 0:
        if gold_map is None:
            raise ValueError("negative_ratio sampling requires --gold-edges to define positives/negatives")
        allowed_pairs = [k for k in sorted(allowed) if k in by_pair]
        positives = [k for k in allowed_pairs if _y_from_gold(k, gold_map) == 1]
        neg_pool = [k for k in allowed_pairs if _y_from_gold(k, gold_map) == 0]
        if not neg_pool:
            # Common case: manifest built from positive gold edges only.
            # Backfill negatives from evidence-graph candidate universe for the same TF subset.
            subset_tfs = {tf for tf, _ in allowed_pairs}
            pos_set = set(positives)
            neg_pool = [
                k for k in by_pair
                if k[0] in subset_tfs and k not in pos_set and _y_from_gold(k, gold_map) == 0
            ]
        n_pos = len(positives)
    else:
        positives = []
        neg_pool = []
        n_pos = 0

    def _sample_eval_pairs(seed_i: int) -> list[tuple[str, str]]:
        if negative_ratio is None or negative_ratio <= 0:
            return [k for k in pred_map if k in allowed and k in by_pair]
        rng = random.Random(seed_i)
        n_neg = min(len(neg_pool), int(round(float(negative_ratio) * n_pos)))
        sampled_neg = rng.sample(neg_pool, n_neg) if n_neg > 0 else []
        return positives + sampled_neg

    def _report_for_pairs(eval_pairs: list[tuple[str, str]]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        n_eval_with_prediction = int(sum(1 for k in eval_pairs if k in pred_map))
        n_eval_missing_prediction = int(max(0, len(eval_pairs) - n_eval_with_prediction))
        for k in eval_pairs:
            if gold_map is not None:
                y_true = _y_from_gold(k, gold_map)
            else:
                y_true = _label_binary_from_evidence_record(by_pair[k])
            p_present = float(pred_map.get(k, 0.0))
            row = pd.Series({"source_tf": k[0], "target_gene": k[1]})
            rows.append(
                {
                    "source_tf": k[0],
                    "target_gene": k[1],
                    "y_true": y_true,
                    "y_pred": int(p_present >= 0.5),
                    "p_present": p_present,
                    "cell_type": _slice_values_from_record(row, "cell_type", by_pair, manifest_lookup),
                    "species": _slice_values_from_record(row, "species", by_pair, manifest_lookup),
                    "tf_frequency_bucket": _slice_values_from_record(row, "tf_frequency_bucket", by_pair, manifest_lookup),
                }
            )
        if not rows:
            return {
                "strategy": strategy,
                "fold_id": fold_id,
                "subset": subset,
                "n_allowed_pairs": int(len(allowed)),
                "n_excluded_outside_subset": int(excluded),
                "n_matched": 0,
                "error": "no_rows_matched_evidence_or_gold",
            }

        df = pd.DataFrame(rows)
        y_t = df["y_true"].to_numpy(dtype=np.int64)
        p_present = df["p_present"].to_numpy(dtype=np.float64)
        out = _binary_metrics(y_t, p_present, k_values=k_values)
        out.update(
            {
                "strategy": strategy,
                "fold_id": fold_id,
                "subset": subset,
                "label_source": "gold_file" if gold_map is not None else "weak_correlation",
                "n_allowed_pairs": int(len(allowed)),
                "n_excluded_outside_subset": int(excluded),
                "n_candidate_pairs": int(len(by_pair)),
                "n_prediction_pairs": int(len(pred_map)),
                "n_prediction_pairs_in_subset": n_prediction_pairs_for_subset_tfs,
                "n_prediction_pairs_in_allowed_manifest": n_prediction_pairs_in_allowed_manifest,
                "n_eval_pairs_with_prediction": n_eval_with_prediction,
                "n_eval_pairs_missing_prediction": n_eval_missing_prediction,
                "negative_ratio": (float(negative_ratio) if negative_ratio is not None else None),
            }
        )

        slices: dict[str, Any] = {}
        for key in ("tf_frequency_bucket", "species", "cell_type"):
            per_val: dict[str, Any] = {}
            for v, d in df.groupby(key):
                y_ts = d["y_true"].to_numpy(dtype=np.int64)
                p_s = d["p_present"].to_numpy(dtype=np.float64)
                m = _binary_metrics(y_ts, p_s, k_values=k_values)
                m["n"] = int(len(d))
                per_val[str(v)] = m
            slices[key] = per_val
        out["robustness_slices"] = slices
        return out

    reports = [_report_for_pairs(_sample_eval_pairs(seed + i)) for i in range(max(1, int(negative_repeats)))]
    if negative_ratio is not None and negative_ratio > 0:
        return _aggregate_repeated_reports(reports, seed=seed)
    return reports[0]


def evaluate_network_vs_weak_labels(
    network_csv: str | Path,
    evidence_jsonl: str | Path,
) -> dict[str, Any]:
    return evaluate_network_vs_labels(network_csv, evidence_jsonl, gold_edges=None)


def write_eval_report(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
