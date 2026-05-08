#!/usr/bin/env python3
"""Benchmark blind TF-EAGER outputs against fair-evaluation baselines."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAIR_DIR = ROOT / "data" / "fair_evaluation_output"
DEFAULT_EPR_PATH = ROOT / "data" / "EPR" / "epr_all_methods_all_datasets.csv"
DEFAULT_BLIND_ROOT = ROOT / "artifacts" / "blind_tf_eager_neg2" / "mESC_mHSC-L"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "fair_benchmark_vs_baselines"


def _normalize_pair(tf: str, target: str) -> tuple[str, str]:
    return str(tf).strip().upper(), str(target).strip().upper()


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _dataset_ids(fair_dir: Path) -> list[str]:
    ids: list[str] = []
    for path in sorted(fair_dir.glob("master_test_*.csv")):
        dataset_id = path.stem.replace("master_test_", "", 1)
        if dataset_id:
            ids.append(dataset_id)
    return ids


def _load_baseline_epr_map(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path, usecols=["Dataset", "Method", "epr@100"])
    out: dict[tuple[str, str], float] = {}
    for dataset, method, value in zip(df["Dataset"], df["Method"], df["epr@100"], strict=False):
        epr = _safe_float(value)
        if epr is None:
            continue
        out[(str(dataset).strip(), str(method).strip())] = epr
    return out


def _load_pairs(path: Path, *, tf_col: str = "TF", target_col: str = "Target") -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    return [_normalize_pair(tf, target) for tf, target in zip(df[tf_col], df[target_col], strict=False)]


def _load_prediction_map(path: Path) -> dict[tuple[str, str], float]:
    df = pd.read_csv(path, usecols=["source_tf", "target_gene", "p_present"])
    pred_map: dict[tuple[str, str], float] = {}
    for tf, target, score in zip(df["source_tf"], df["target_gene"], df["p_present"], strict=False):
        key = _normalize_pair(tf, target)
        value = _safe_float(score)
        if value is None:
            continue
        pred_map[key] = value
    return pred_map


def _compute_metrics(y_true: list[int], y_score: list[float]) -> tuple[float | None, float | None]:
    if not y_true or len(set(y_true)) < 2:
        return None, None
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def _precision_at_k(y_true: list[int], y_score: list[float], k: int) -> float | None:
    if not y_true or not y_score:
        return None
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    k_eff = min(max(int(k), 1), len(y_true))
    ranked = sorted(zip(y_true, y_score, strict=False), key=lambda item: item[1], reverse=True)

    # Tie-safe precision@k:
    # - fully include groups with strictly higher score than cutoff
    # - for the cutoff score group, include expected positives proportionally
    score_groups: dict[float, list[int]] = {}
    for label, score in ranked:
        score_groups.setdefault(float(score), []).append(int(label))
    sorted_scores = sorted(score_groups.keys(), reverse=True)

    taken = 0
    positives = 0.0
    for score in sorted_scores:
        labels = score_groups[score]
        group_n = len(labels)
        group_pos = sum(labels)
        if taken + group_n < k_eff:
            positives += group_pos
            taken += group_n
            continue
        remaining = k_eff - taken
        if remaining > 0:
            positives += (group_pos / float(group_n)) * remaining
            taken += remaining
        break
    return positives / float(k_eff)


def _baseline_rows_for_dataset(
    fair_dir: Path,
    dataset_id: str,
    baseline_epr_map: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(fair_dir.glob(f"*_results_{dataset_id}.csv")):
        if path.name.startswith("all_methods_results_"):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        method = str(row.get("method", path.stem.split("_results_", 1)[0])).strip()
        rows.append(
            {
                "dataset": dataset_id,
                "method": method,
                "auroc": _safe_float(row.get("auroc")),
                "aupr": _safe_float(row.get("aupr")),
                "epr_at_100": baseline_epr_map.get((dataset_id, method)),
                "n_positives": int(row.get("n_positives", 0) or 0),
                "n_negatives": int(row.get("n_negatives", 0) or 0),
                "n_neg_samples": int(row.get("n_neg_samples", 0) or 0),
                "source_path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def _our_row_for_dataset(fair_dir: Path, blind_root: Path, dataset_id: str) -> dict[str, object]:
    positives_path = fair_dir / f"master_test_{dataset_id}.csv"
    negatives_path = fair_dir / f"clean_evaluation_pool_all_pairs_{dataset_id}.csv"
    scored_path = blind_root / dataset_id / "blind_scored_edges.csv"
    positives = _load_pairs(positives_path)
    negatives = _load_pairs(negatives_path)
    pred_map = _load_prediction_map(scored_path) if scored_path.is_file() else {}

    pos_set = set(positives)
    neg_set = [pair for pair in negatives if pair not in pos_set]
    # Build evaluation list from the fair benchmark set only:
    # 1) intersection with scored edges keeps model-provided scores
    # 2) any fair-eval pair missing from scored edges is appended with score=0
    # This preserves full fair-eval coverage while preventing off-pool pairs
    # from influencing ranking.
    pos_scored = [pair for pair in positives if pair in pred_map]
    neg_scored = [pair for pair in neg_set if pair in pred_map]
    pos_missing = [pair for pair in positives if pair not in pred_map]
    neg_missing = [pair for pair in neg_set if pair not in pred_map]

    eval_pairs = pos_scored + neg_scored + pos_missing + neg_missing
    y_true = (
        [1] * len(pos_scored)
        + [0] * len(neg_scored)
        + [1] * len(pos_missing)
        + [0] * len(neg_missing)
    )
    y_score = (
        [pred_map[pair] for pair in pos_scored]
        + [pred_map[pair] for pair in neg_scored]
        + [0.0] * len(pos_missing)
        + [0.0] * len(neg_missing)
    )

    auroc, aupr = _compute_metrics(y_true, y_score)
    epr_at_100 = _precision_at_k(y_true, y_score, 100)
    pos_with_prediction = sum(1 for pair in positives if pair in pred_map)
    eval_with_prediction = sum(1 for pair in eval_pairs if pair in pred_map)

    return {
        "dataset": dataset_id,
        "method": "TF-EAGER-neg2",
        "auroc": auroc,
        "aupr": aupr,
        "epr_at_100": epr_at_100,
        "n_positives": len(positives),
        "n_negatives": len(neg_set),
        "n_neg_samples": len(neg_set),
        "prediction_pairs": len(pred_map),
        "positive_predictions_found": pos_with_prediction,
        "positive_predictions_missing": len(positives) - pos_with_prediction,
        "positive_coverage": (pos_with_prediction / len(positives)) if positives else None,
        "eval_predictions_found": eval_with_prediction,
        "eval_predictions_missing": len(eval_pairs) - eval_with_prediction,
        "eval_coverage": (eval_with_prediction / len(eval_pairs)) if eval_pairs else None,
        "source_path": str(scored_path.relative_to(ROOT)) if scored_path.is_file() else "",
    }


def _mean(values: Iterable[float | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fair-dir", type=Path, default=DEFAULT_FAIR_DIR)
    ap.add_argument("--baseline-epr-csv", type=Path, default=DEFAULT_EPR_PATH)
    ap.add_argument("--blind-root", type=Path, default=DEFAULT_BLIND_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    fair_dir = args.fair_dir
    baseline_epr_csv = args.baseline_epr_csv
    blind_root = args.blind_root
    out_dir = args.out_dir

    datasets = _dataset_ids(fair_dir)
    baseline_epr_map = _load_baseline_epr_map(baseline_epr_csv)
    comparison_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for dataset_id in datasets:
        our_row = _our_row_for_dataset(fair_dir, blind_root, dataset_id)
        comparison_rows.append(our_row)
        coverage_rows.append(
            {
                "dataset": dataset_id,
                "n_positives": our_row["n_positives"],
                "n_negatives": our_row["n_negatives"],
                "prediction_pairs": our_row["prediction_pairs"],
                "positive_predictions_found": our_row["positive_predictions_found"],
                "positive_predictions_missing": our_row["positive_predictions_missing"],
                "positive_coverage": our_row["positive_coverage"],
                "eval_predictions_found": our_row["eval_predictions_found"],
                "eval_predictions_missing": our_row["eval_predictions_missing"],
                "eval_coverage": our_row["eval_coverage"],
                "source_path": our_row["source_path"],
            }
        )
        comparison_rows.extend(_baseline_rows_for_dataset(fair_dir, dataset_id, baseline_epr_map))

    summary_rows: list[dict[str, object]] = []
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in comparison_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    for method, rows in sorted(by_method.items()):
        summary_rows.append(
            {
                "method": method,
                "datasets": len(rows),
                "mean_auroc": _mean(row.get("auroc") for row in rows),
                "mean_aupr": _mean(row.get("aupr") for row in rows),
                "mean_epr_at_100": _mean(row.get("epr_at_100") for row in rows),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            -1.0 if row["mean_epr_at_100"] is None else -float(row["mean_epr_at_100"]),
            -1.0 if row["mean_aupr"] is None else -float(row["mean_aupr"]),
            -1.0 if row["mean_auroc"] is None else -float(row["mean_auroc"]),
            str(row["method"]),
        )
    )

    per_dataset_path = out_dir / "fair_benchmark_vs_baselines_long.csv"
    summary_path = out_dir / "fair_benchmark_vs_baselines_summary.csv"
    coverage_path = out_dir / "fair_benchmark_coverage.csv"

    _write_csv(
        per_dataset_path,
        comparison_rows,
        [
            "dataset",
            "method",
            "auroc",
            "aupr",
            "epr_at_100",
            "n_positives",
            "n_negatives",
            "n_neg_samples",
            "prediction_pairs",
            "positive_predictions_found",
            "positive_predictions_missing",
            "positive_coverage",
            "eval_predictions_found",
            "eval_predictions_missing",
            "eval_coverage",
            "source_path",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        ["method", "datasets", "mean_auroc", "mean_aupr", "mean_epr_at_100"],
    )
    _write_csv(
        coverage_path,
        coverage_rows,
        [
            "dataset",
            "n_positives",
            "n_negatives",
            "prediction_pairs",
            "positive_predictions_found",
            "positive_predictions_missing",
            "positive_coverage",
            "eval_predictions_found",
            "eval_predictions_missing",
            "eval_coverage",
            "source_path",
        ],
    )

    print(f"Wrote {per_dataset_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {coverage_path}")


if __name__ == "__main__":
    main()
