#!/usr/bin/env python3
"""Plot TF-EAGER evaluation metrics across negative ratios and evaluation types."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "precision_at_10": "Precision@10",
    "precision_at_50": "Precision@50",
    "precision_at_100": "Precision@100",
}
EVAL_LABELS = {
    "blind": "Blind inference",
    "leave_tf_out_validation": "Leave-TF-out validation",
}
PALETTE = {
    "Blind inference": "#2F6F9F",
    "Leave-TF-out validation": "#B55A30",
}
FIGURE_RC = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
}
RATIO_ROBUSTNESS_LABELS = {
    "Blind inference": "Unseen cell-types",
    "Leave-TF-out validation": "Leave-one-TF-out",
}


def _parse_formats(value: str) -> list[str]:
    formats = [item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()]
    return formats or ["png"]


def _prepare_dataframe(path: Path, model: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "source_type" not in df.columns and "evaluation_type" in df.columns:
        df = df.rename(columns={"evaluation_type": "source_type"})
    required = {"source_type", "dataset", "negative_ratio", *METRIC_LABELS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {path}: {missing}")
    df = df.copy()
    if model is not None and "model" in df.columns:
        df = df[df["model"].astype(str) == model].copy()
    df["negative_ratio"] = pd.to_numeric(df["negative_ratio"], errors="coerce")
    df = df.dropna(subset=["negative_ratio"])
    df["negative_ratio"] = df["negative_ratio"].astype(int)
    df["ratio_label"] = "1:" + df["negative_ratio"].astype(str)
    df["evaluation_type"] = df["source_type"].map(EVAL_LABELS).fillna(df["source_type"])
    for metric in METRIC_LABELS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _ratio_order(df: pd.DataFrame) -> list[str]:
    ratios = sorted(int(v) for v in df["negative_ratio"].dropna().unique())
    return [f"1:{ratio}" for ratio in ratios]


def _plot_metric_means(df: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    ratio_order = _ratio_order(df)
    metric_order = list(METRIC_LABELS)
    long = df.melt(
        id_vars=["evaluation_type", "ratio_label"],
        value_vars=metric_order,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    long["metric_label"] = long["metric"].map(METRIC_LABELS)
    long["ratio_label"] = pd.Categorical(long["ratio_label"], categories=ratio_order, ordered=True)

    sns.set_theme(style="whitegrid", context="notebook", rc=FIGURE_RC)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, metric_order):
        sub = long[long["metric"] == metric]
        sns.pointplot(
            data=sub,
            x="ratio_label",
            y="value",
            hue="evaluation_type",
            order=ratio_order,
            palette=PALETTE,
            errorbar="sd",
            dodge=0.25,
            markers="o",
            linestyles="-",
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Positive:negative ratio")
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric == "auroc":
            ax.axhline(0.5, color="#666666", linewidth=1, linestyle="--")
            ax.set_ylim(0.0, 1.02)
        else:
            ax.set_ylim(0.0, 1.02)
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].axis("off")
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False, title="Evaluation type")
    fig.suptitle("TF-EAGER metrics across negative sampling ratios", y=1.02, fontsize=16)
    return _save(fig, out_dir, "metric_means_by_ratio", formats, dpi)


def _plot_distribution_boxes(df: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    ratio_order = _ratio_order(df)
    metric_order = ["auroc", "auprc"]
    long = df.melt(
        id_vars=["evaluation_type", "ratio_label"],
        value_vars=metric_order,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    long["metric_label"] = long["metric"].map(METRIC_LABELS)
    long["ratio_label"] = pd.Categorical(long["ratio_label"], categories=ratio_order, ordered=True)

    sns.set_theme(style="whitegrid", context="notebook", rc=FIGURE_RC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric in zip(axes, metric_order):
        sub = long[long["metric"] == metric]
        sns.boxplot(
            data=sub,
            x="ratio_label",
            y="value",
            hue="evaluation_type",
            order=ratio_order,
            palette=PALETTE,
            fliersize=2,
            linewidth=1.2,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="ratio_label",
            y="value",
            hue="evaluation_type",
            order=ratio_order,
            dodge=True,
            palette=PALETTE,
            alpha=0.35,
            size=2.4,
            legend=False,
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Positive:negative ratio")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(0.0, 1.02)
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower center", ncol=2, frameon=False, title="Evaluation type")
    fig.suptitle("Dataset-level AUROC/AUPRC distributions", y=1.04, fontsize=16)
    return _save(fig, out_dir, "auroc_auprc_distributions", formats, dpi)


def _plot_precision_profiles(df: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    ratio_order = _ratio_order(df)
    precision_metrics = ["precision_at_10", "precision_at_50", "precision_at_100"]
    long = df.melt(
        id_vars=["evaluation_type", "ratio_label"],
        value_vars=precision_metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    long["metric_label"] = long["metric"].map(METRIC_LABELS)
    long["ratio_label"] = pd.Categorical(long["ratio_label"], categories=ratio_order, ordered=True)

    sns.set_theme(style="whitegrid", context="notebook", rc=FIGURE_RC)
    eval_types = list(dict.fromkeys(long["evaluation_type"].tolist()))
    fig, axes = plt.subplots(1, len(eval_types), figsize=(6.5 * len(eval_types), 5), sharey=True, constrained_layout=True)
    if len(eval_types) == 1:
        axes = [axes]
    precision_palette = {"Precision@10": "#6A4C93", "Precision@50": "#198754", "Precision@100": "#D97706"}
    for ax, eval_type in zip(axes, eval_types):
        sub = long[long["evaluation_type"] == eval_type]
        sns.pointplot(
            data=sub,
            x="ratio_label",
            y="value",
            hue="metric_label",
            order=ratio_order,
            palette=precision_palette,
            errorbar="sd",
            dodge=0.2,
            markers="o",
            ax=ax,
        )
        ax.set_title(eval_type)
        ax.set_xlabel("Positive:negative ratio")
        ax.set_ylabel("Raw precision")
        ax.set_ylim(0.0, 1.02)
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, title="Metric")
    fig.suptitle("Raw precision@k profiles by evaluation type", y=1.05, fontsize=16)
    return _save(fig, out_dir, "precision_profiles_by_ratio", formats, dpi)


def _plot_precision_heatmaps(df: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    ratios = sorted(int(v) for v in df["negative_ratio"].dropna().unique())
    precision_metrics = ["precision_at_10", "precision_at_50", "precision_at_100"]
    summary = (
        df.groupby(["evaluation_type", "negative_ratio"], as_index=False)[precision_metrics]
        .mean(numeric_only=True)
        .sort_values(["evaluation_type", "negative_ratio"])
    )

    sns.set_theme(style="white", context="notebook", rc=FIGURE_RC)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for ax, metric in zip(axes, precision_metrics):
        pivot = summary.pivot(index="evaluation_type", columns="negative_ratio", values=metric)
        pivot = pivot.reindex(columns=ratios)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            cbar=metric == precision_metrics[-1],
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 10},
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Negative ratio")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Mean raw precision@k by ratio and evaluation type", y=1.05, fontsize=16)
    return _save(fig, out_dir, "precision_mean_heatmaps", formats, dpi)


def _plot_ratio_robustness(df: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    ratio_order = _ratio_order(df)
    metric_specs = [
        ("auprc", "AUPRC"),
        ("precision_at_10", "EP@10"),
        ("precision_at_100", "EP@100"),
    ]
    long = df.melt(
        id_vars=["evaluation_type", "ratio_label"],
        value_vars=[metric for metric, _ in metric_specs],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    long["ratio_label"] = pd.Categorical(long["ratio_label"], categories=ratio_order, ordered=True)
    long["presentation_label"] = long["evaluation_type"].map(RATIO_ROBUSTNESS_LABELS).fillna(long["evaluation_type"])

    line_palette = {
        "Leave-one-TF-out": "#B55A30",
        "Unseen cell-types": "#2F6F9F",
    }

    sns.set_theme(style="whitegrid", context="talk", rc=FIGURE_RC)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), sharex=True)
    panel_labels = ["A", "B", "C"]
    for idx, (ax, (metric, ylabel)) in enumerate(zip(axes, metric_specs)):
        sub = long[long["metric"] == metric]
        sns.pointplot(
            data=sub,
            x="ratio_label",
            y="value",
            hue="presentation_label",
            hue_order=list(line_palette),
            palette=line_palette,
            errorbar="sd",
            dodge=0.2,
            markers="o",
            linestyles="-",
            ax=ax,
        )
        ax.set_xlabel("Positive:negative ratio", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(ylabel, fontsize=17, pad=1)
        ax.tick_params(axis="both", labelsize=12)
        ax.text(-0.12, 1.02, panel_labels[idx], transform=ax.transAxes, fontsize=16, fontweight="bold")
        if ax.legend_:
            ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False, fontsize=15)
    fig.suptitle("Ratio Robustness Curves", y=0.845, fontsize=20)
    fig.tight_layout(rect=(0, 0.07, 1, 0.925))
    return _save(fig, out_dir, "ratio_robustness_curves", formats, dpi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/evaluation_tables/tf_eager_eval_metrics_long.csv")
    ap.add_argument("--out-dir", default="artifacts/evaluation_tables/figures")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--model", default="", help="Optional model filter for comparison tables, e.g. neg2")
    args = ap.parse_args()

    df = _prepare_dataframe(Path(args.input), model=args.model.strip() or None)
    out_dir = Path(args.out_dir)
    formats = _parse_formats(args.formats)

    paths: list[Path] = []
    paths.extend(_plot_metric_means(df, out_dir, formats, args.dpi))
    paths.extend(_plot_distribution_boxes(df, out_dir, formats, args.dpi))
    paths.extend(_plot_precision_profiles(df, out_dir, formats, args.dpi))
    paths.extend(_plot_precision_heatmaps(df, out_dir, formats, args.dpi))
    paths.extend(_plot_ratio_robustness(df, out_dir, formats, args.dpi))
    print("[plot-tf-eager-eval] wrote " + ", ".join(str(path) for path in paths), flush=True)


if __name__ == "__main__":
    main()
