#!/usr/bin/env python3
"""Plot precision-recall curves from an evaluation report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _parse_formats(value: str) -> list[str]:
    formats = [item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()]
    return formats or ["png"]


def _ratio_sort_key(item: tuple[str, dict]) -> tuple[int, float | str]:
    label = str(item[0])
    try:
        return (0, float(label))
    except ValueError:
        return (1, label)


def _curve_payload(metrics: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float | None] | None:
    curve = metrics.get("pr_curve")
    if not isinstance(curve, dict):
        return None
    recall = np.asarray(curve.get("recall") or [], dtype=float)
    if "precision_mean" in curve:
        precision = np.asarray(curve.get("precision_mean") or [], dtype=float)
        precision_std = np.asarray(curve.get("precision_std") or [], dtype=float)
        prevalence = curve.get("positive_prevalence_mean")
    else:
        precision = np.asarray(curve.get("precision") or [], dtype=float)
        precision_std = None
        prevalence = curve.get("positive_prevalence")
    if recall.size == 0 or precision.size == 0 or recall.size != precision.size:
        return None
    prevalence_f = float(prevalence) if prevalence is not None else None
    return recall, precision, precision_std, prevalence_f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Evaluation report JSON")
    ap.add_argument("--out", required=True, help="Output image path stem or full path")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument(
        "--allow-missing-curves",
        action="store_true",
        help="Exit successfully with a skip message when the report has no PR-curve payloads",
    )
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    by_ratio = payload.get("results_by_ratio")
    if not isinstance(by_ratio, dict) or not by_ratio:
        raise SystemExit("Evaluation report does not contain results_by_ratio")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)
    palette = sns.color_palette("viridis", n_colors=len(by_ratio))
    plotted = 0
    for color, item in zip(palette, sorted(by_ratio.items(), key=_ratio_sort_key)):
        ratio, metrics = item
        curve_payload = _curve_payload(metrics if isinstance(metrics, dict) else {})
        if curve_payload is None:
            continue
        recall, precision, precision_std, prevalence = curve_payload
        label = f"1:{ratio}"
        ax.plot(recall, precision, color=color, linewidth=2.3, label=label)
        if precision_std is not None and precision_std.size == precision.size:
            lower = np.clip(precision - precision_std, 0.0, 1.0)
            upper = np.clip(precision + precision_std, 0.0, 1.0)
            ax.fill_between(recall, lower, upper, color=color, alpha=0.15, linewidth=0)
        if prevalence is not None:
            ax.hlines(
                prevalence,
                xmin=0.0,
                xmax=1.0,
                colors=color,
                linestyles="--",
                linewidth=1.0,
                alpha=0.55,
            )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        if args.allow_missing_curves:
            print("[plot-eval-pr-curve] skip=no_pr_curves_in_report", flush=True)
            return
        raise SystemExit("No PR curves were found in the evaluation report")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves by Negative Ratio")
    ax.legend(title="Ratio", frameon=False, loc="lower left")

    out_path = Path(args.out)
    stem_path = out_path.with_suffix("") if out_path.suffix else out_path
    for fmt in _parse_formats(args.formats):
        fig.savefig(stem_path.with_suffix(f".{fmt}"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
