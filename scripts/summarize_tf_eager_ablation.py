#!/usr/bin/env python3
"""Aggregate TF-EAGER ablation evaluation reports by regulatory type."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

MODELS = {
    "full": "eval_test_by_ratio.json",
    "functional_only": "eval_test_by_ratio_functional_only.json",
    "single_stage": "eval_test_by_ratio_single_stage.json",
}
MODEL_LABELS = {
    "full": "Full",
    "functional_only": "Functional-only",
    "single_stage": "Single-stage",
}
METRICS = ("auprc", "auroc", "p_at_10", "brier", "recall")


def regulatory_type_from_context(context_name: str) -> str:
    rest = context_name.split("_", 1)[1] if "_" in context_name else context_name
    if rest.startswith("string"):
        return "string"
    if rest.startswith("specific_chipseq"):
        return "specific_chipseq"
    if rest.startswith("nonspecific_chipseq"):
        return "nonspecific_chipseq"
    return "other"


def _mean(values: list[float | None]) -> float | None:
    usable = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _ratio_payload(report: dict[str, Any], ratio: str) -> dict[str, Any] | None:
    by_ratio = report.get("results_by_ratio", {})
    if not isinstance(by_ratio, dict):
        return None
    return by_ratio.get(ratio) or by_ratio.get(str(float(ratio)))


def collect_rows(root: Path, ratio: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for ctx_dir in sorted((root / "contexts").iterdir()):
        if not ctx_dir.is_dir():
            continue
        ctx_name = ctx_dir.name
        reg_type = regulatory_type_from_context(ctx_name)
        eval_dir = ctx_dir / "evaluation"
        for model_key, filename in MODELS.items():
            report_path = eval_dir / filename
            if not report_path.is_file():
                missing.append(str(report_path))
                continue
            report = json.loads(report_path.read_text())
            payload = _ratio_payload(report, ratio)
            if payload is None:
                missing.append(f"{report_path}:missing_ratio={ratio}")
                continue
            precision_at_k = payload.get("precision_at_k", {}) or {}
            rows.append(
                {
                    "context": ctx_name,
                    "regulatory_type": reg_type,
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "auprc": payload.get("auprc_macro"),
                    "auroc": payload.get("auroc"),
                    "p_at_10": precision_at_k.get("present@10"),
                    "brier": payload.get("brier"),
                    "recall": payload.get("recall"),
                    "n_matched": payload.get("n_matched"),
                }
            )
    return rows, missing


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["regulatory_type"], row["model"])].append(row)

    out: list[dict[str, Any]] = []
    regulatory_types = sorted({row["regulatory_type"] for row in rows})
    model_order = list(MODELS.keys())

    for reg_type in regulatory_types:
        for model in model_order:
            bucket = grouped.get((reg_type, model), [])
            if not bucket:
                continue
            out.append(
                {
                    "regulatory_type": reg_type,
                    "n_contexts": len(bucket),
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    **{metric: _mean([item[metric] for item in bucket]) for metric in METRICS},
                }
            )

    for model in model_order:
        bucket = [row for row in rows if row["model"] == model]
        if not bucket:
            continue
        out.append(
            {
                "regulatory_type": "Overall",
                "n_contexts": len(bucket),
                "model": model,
                "model_label": MODEL_LABELS[model],
                **{metric: _mean([item[metric] for item in bucket]) for metric in METRICS},
            }
        )
    return out


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regulatory_type",
        "n_contexts",
        "model",
        "model_label",
        "auprc",
        "auroc",
        "p_at_10",
        "brier",
        "recall",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Regulatory Type & Model & n & AUPRC & AUROC & P@10 & Brier & Recall \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['regulatory_type']} & {row['model_label']} & {row['n_contexts']} & "
            f"{_fmt(row['auprc'])} & {_fmt(row['auroc'])} & {_fmt(row['p_at_10'])} & "
            f"{_fmt(row['brier'])} & {_fmt(row['recall'])} \\\\",
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Artifact root for one multicontext workflow run")
    ap.add_argument("--ratio", default="1.0", help="Negative ratio key to aggregate, e.g. 1.0")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-tex", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows, missing = collect_rows(root, args.ratio)
    if not rows:
        raise SystemExit(f"No evaluation rows found under {root}")
    agg = aggregate_rows(rows)
    write_csv(agg, Path(args.out_csv))
    write_latex(agg, Path(args.out_tex))
    print(f"[summarize-tf-eager-ablation] wrote_csv={args.out_csv} rows={len(agg)}")
    print(f"[summarize-tf-eager-ablation] wrote_tex={args.out_tex}")
    if missing:
        print(f"[summarize-tf-eager-ablation] missing_reports={len(missing)}")
        for item in missing[:10]:
            print(f"  missing: {item}")


if __name__ == "__main__":
    main()
