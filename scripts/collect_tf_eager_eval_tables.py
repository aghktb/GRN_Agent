#!/usr/bin/env python3
"""Collect TF-EAGER evaluation JSON reports into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_COLUMNS = ("auroc", "auprc", "precision_at_10", "precision_at_50", "precision_at_100")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _fmt(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _ratio_label(value: Any) -> str:
    f = _as_float(value)
    if f is None:
        text = str(value).strip()
        return text
    if f.is_integer():
        return str(int(f))
    return str(f).rstrip("0").rstrip(".")


def _ratio_sort_key(label: str) -> tuple[int, float | str]:
    f = _as_float(label)
    if f is None:
        return (1, label)
    return (0, f)


def _split_dataset_name(dataset: str) -> tuple[str, str]:
    if "_" not in dataset:
        return dataset, ""
    cell, variant = dataset.split("_", 1)
    return cell, variant


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _pick_auprc(metrics: dict[str, Any]) -> float | None:
    for key in ("auprc", "auprc_macro", "aucpr_macro", "auprc_micro", "aucpr_micro"):
        value = _as_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _precision_at_k(metrics: dict[str, Any], k: int) -> float | None:
    pk = metrics.get("precision_at_k")
    if not isinstance(pk, dict):
        return None
    for key in (f"present@{k}", str(k), f"@{k}"):
        value = _as_float(pk.get(key))
        if value is not None:
            return value
    return None


def _report_rows(path: Path, source_type: str, workflow: str, dataset: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_ratio = payload.get("results_by_ratio")
    if not isinstance(by_ratio, dict):
        return []
    cell, variant = _split_dataset_name(dataset)
    rows: list[dict[str, Any]] = []
    for raw_ratio, metrics in sorted(by_ratio.items(), key=lambda item: _ratio_sort_key(_ratio_label(item[0]))):
        if not isinstance(metrics, dict):
            continue
        ratio = _ratio_label(metrics.get("negative_ratio", raw_ratio))
        n_matched = _as_float(metrics.get("n_matched"))
        n_pos = _as_float(metrics.get("n_positive"))
        n_neg = _as_float(metrics.get("n_negative"))
        positive_rate = (n_pos / n_matched) if n_pos is not None and n_matched not in (None, 0.0) else None
        rows.append(
            {
                "source_type": source_type,
                "workflow": workflow,
                "dataset": dataset,
                "cell": cell,
                "variant": variant,
                "negative_ratio": ratio,
                "ratio_label": f"1:{ratio}",
                "n_matched": n_matched,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "positive_rate": positive_rate,
                "auroc": _as_float(metrics.get("auroc", metrics.get("auroc_macro"))),
                "auprc": _pick_auprc(metrics),
                "precision_at_10": _precision_at_k(metrics, 10),
                "precision_at_50": _precision_at_k(metrics, 50),
                "precision_at_100": _precision_at_k(metrics, 100),
                "report_path": _relative_path(path),
            }
        )
    return rows


def _discover_blind_reports(root: Path) -> list[tuple[Path, str, str]]:
    reports: list[tuple[Path, str, str]] = []
    for path in sorted(root.rglob("blind_eval.json")):
        dataset = path.parent.name
        workflow = path.parent.parent.name if path.parent.parent != root.parent else root.name
        reports.append((path, workflow, dataset))
    return reports


def _discover_multicontext_reports(root: Path) -> list[tuple[Path, str, str]]:
    reports: list[tuple[Path, str, str]] = []
    for path in sorted(root.rglob("eval_test_by_ratio.json")):
        parts = path.parts
        if "contexts" not in parts:
            continue
        idx = parts.index("contexts")
        if idx == 0 or idx + 1 >= len(parts):
            continue
        workflow = parts[idx - 1]
        dataset = parts[idx + 1]
        reports.append((path, workflow, dataset))
    return reports


def _csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _write_long(rows: list[dict[str, Any]], path: Path, digits: int) -> None:
    fields = [
        "source_type",
        "dataset",
        "cell",
        "variant",
        "negative_ratio",
        "ratio_label",
        "n_matched",
        "n_positive",
        "n_negative",
        "positive_rate",
        *METRIC_COLUMNS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("n_matched", "n_positive", "n_negative"):
                value = out.get(key)
                out[key] = "" if value is None else str(int(value))
            for key in ("positive_rate", *METRIC_COLUMNS):
                out[key] = _fmt(out.get(key), digits)
            writer.writerow({key: out.get(key, "") for key in fields})


def _write_wide(rows: list[dict[str, Any]], path: Path, digits: int) -> None:
    ratios = sorted({str(row["negative_ratio"]) for row in rows}, key=_ratio_sort_key)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["source_type"]), str(row["dataset"]))
        grouped.setdefault(
            key,
            {
                "source_type": row["source_type"],
                "dataset": row["dataset"],
                "cell": row["cell"],
                "variant": row["variant"],
            },
        )
        ratio = str(row["negative_ratio"])
        for metric in METRIC_COLUMNS:
            grouped[key][f"{metric}_r{ratio}"] = row.get(metric)

    fields = ["source_type", "dataset", "cell", "variant"]
    for ratio in ratios:
        fields.extend(f"{metric}_r{ratio}" for metric in METRIC_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in sorted(grouped.values(), key=lambda r: (str(r["source_type"]), str(r["dataset"]))):
            out = dict(row)
            for ratio in ratios:
                for metric in METRIC_COLUMNS:
                    key = f"{metric}_r{ratio}"
                    out[key] = _fmt(out.get(key), digits)
            writer.writerow({key: out.get(key, "") for key in fields})


def _write_by_ratio(rows: list[dict[str, Any]], out_dir: Path, digits: int) -> list[Path]:
    ratio_dir = out_dir / "by_ratio"
    ratio_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_type",
        "dataset",
        "cell",
        "variant",
        "n_matched",
        "n_positive",
        "n_negative",
        "positive_rate",
        *METRIC_COLUMNS,
    ]
    out_paths: list[Path] = []
    for ratio in sorted({str(row["negative_ratio"]) for row in rows}, key=_ratio_sort_key):
        path = ratio_dir / f"negative_ratio_{ratio}.csv"
        out_paths.append(path)
        ratio_rows = [row for row in rows if str(row["negative_ratio"]) == ratio]
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            for row in sorted(ratio_rows, key=lambda r: (str(r["source_type"]), str(r["dataset"]))):
                out = dict(row)
                for key in ("n_matched", "n_positive", "n_negative"):
                    value = out.get(key)
                    out[key] = "" if value is None else str(int(value))
                for key in ("positive_rate", *METRIC_COLUMNS):
                    out[key] = _fmt(out.get(key), digits)
                writer.writerow({key: out.get(key, "") for key in fields})
    return out_paths


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return None
    return statistics.fmean(clean)


def _std(values: list[float]) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return None
    return statistics.stdev(clean)


def _write_summary(rows: list[dict[str, Any]], path: Path, digits: int) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_type"]), str(row["negative_ratio"]))].append(row)

    fields = ["source_type", "negative_ratio", "n_reports"]
    for metric in METRIC_COLUMNS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for (source_type, ratio), items in sorted(grouped.items(), key=lambda item: (item[0][0], _ratio_sort_key(item[0][1]))):
            out: dict[str, Any] = {
                "source_type": source_type,
                "negative_ratio": ratio,
                "n_reports": len(items),
            }
            for metric in METRIC_COLUMNS:
                values = [v for row in items if (v := _as_float(row.get(metric))) is not None]
                out[f"{metric}_mean"] = _fmt(_mean(values), digits)
                out[f"{metric}_std"] = _fmt(_std(values), digits)
            writer.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-root", default="artifacts/blind_tf_eager")
    ap.add_argument("--multicontext-root", default="artifacts/multicontext_tf_eager")
    ap.add_argument("--out-dir", default="artifacts/evaluation_tables")
    ap.add_argument("--exclude-multicontext-workflows", default="hHep_only")
    ap.add_argument("--multicontext-label", default="leave_tf_out_validation")
    ap.add_argument("--digits", type=int, default=4)
    args = ap.parse_args()

    blind_root = Path(args.blind_root)
    multicontext_root = Path(args.multicontext_root)
    out_dir = Path(args.out_dir)

    rows: list[dict[str, Any]] = []
    if blind_root.exists():
        for path, workflow, dataset in _discover_blind_reports(blind_root):
            rows.extend(_report_rows(path, "blind", workflow, dataset))
    if multicontext_root.exists():
        excluded_workflows = _csv_set(args.exclude_multicontext_workflows)
        for path, workflow, dataset in _discover_multicontext_reports(multicontext_root):
            if workflow in excluded_workflows:
                continue
            rows.extend(_report_rows(path, args.multicontext_label, args.multicontext_label, dataset))

    if not rows:
        raise SystemExit(
            f"No evaluation reports found under blind_root={blind_root} "
            f"or multicontext_root={multicontext_root}"
        )

    rows.sort(key=lambda r: (str(r["source_type"]), str(r["workflow"]), str(r["dataset"]), _ratio_sort_key(str(r["negative_ratio"]))))
    long_path = out_dir / "tf_eager_eval_metrics_long.csv"
    wide_path = out_dir / "tf_eager_eval_metrics_pivot.csv"
    summary_path = out_dir / "tf_eager_eval_metrics_summary.csv"
    _write_long(rows, long_path, args.digits)
    _write_wide(rows, wide_path, args.digits)
    ratio_paths = _write_by_ratio(rows, out_dir, args.digits)
    _write_summary(rows, summary_path, args.digits)
    print(
        "[collect-tf-eager-eval] "
        f"reports={len({row['report_path'] for row in rows})} rows={len(rows)} "
        f"long={long_path} pivot={wide_path} summary={summary_path} "
        f"by_ratio={','.join(str(p) for p in ratio_paths)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
