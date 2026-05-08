#!/usr/bin/env python3
"""Aggregate and compare TF-EAGER neg1/neg2 validation and blind metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns


METRICS = ("auroc", "auprc", "precision_at_10", "precision_at_50", "precision_at_100")
BLIND_STD_METRICS = (
    "auroc_neg_sampling_std",
    "auprc_neg_sampling_std",
    "precision_at_10_neg_sampling_std",
    "precision_at_50_neg_sampling_std",
    "precision_at_100_neg_sampling_std",
)
METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "precision_at_10": "Precision@10",
    "precision_at_50": "Precision@50",
    "precision_at_100": "Precision@100",
    "auroc_neg_sampling_std": "AUROC std",
    "auprc_neg_sampling_std": "AUPRC std",
    "precision_at_10_neg_sampling_std": "Precision@10 std",
    "precision_at_50_neg_sampling_std": "Precision@50 std",
    "precision_at_100_neg_sampling_std": "Precision@100 std",
}
EVAL_LABELS = {
    "leave_tf_out_validation": "Leave-TF-out validation",
    "blind": "Blind inference",
}
MODEL_PALETTE = {
    "neg1": "#C05A2B",
    "neg2": "#2B6CB0",
}
REGULATORY_TYPE_LABELS = {
    "nonspecific_chipseq": "Non-Cell-Type Specific",
    "specific_chipseq": "Cell-type-specific ChIP-seq",
    "string": "STRING",
}
GENE_SET_ORDER = ("tf500", "tf1000", "500", "1000")
REGULATORY_TYPE_ORDER = ("nonspecific_chipseq", "specific_chipseq", "string")
ALL_RATIO_AUROC_AUPRC_CMAP = LinearSegmentedColormap.from_list(
    "all_ratio_auroc_auprc",
    ["#3B0F70", "#8C2981", "#DE4968", "#FE9F6D", "#FDE725"],
)
ALL_RATIO_PRECISION_CMAP = LinearSegmentedColormap.from_list(
    "all_ratio_precision",
    ["#1D3557", "#457B9D", "#A8DADC", "#F1FAEE", "#E9C46A", "#F4A261"],
)
EVAL_BACKGROUND = {
    "blind": "#E8F1FB",
    "leave_tf_out_validation": "#FBEBD9",
}
GENE_SET_COLORS = {
    "tf500": "#4C78A8",
    "tf1000": "#F58518",
    "500": "#54A24B",
    "1000": "#E45756",
}
BARGRAPH_RC = {
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
}


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


def _ratio_label(value: Any) -> str:
    out = _as_float(value)
    if out is None:
        return str(value).strip()
    if float(out).is_integer():
        return str(int(out))
    return str(out).rstrip("0").rstrip(".")


def _ratio_sort_key(value: Any) -> tuple[int, float | str]:
    out = _as_float(value)
    if out is None:
        return (1, str(value))
    return (0, out)


def _ordered_metric_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for metric in (*METRICS, *BLIND_STD_METRICS):
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["ratio_num"] = pd.to_numeric(df["negative_ratio"], errors="coerce")
    return df


def _column_order(df: pd.DataFrame) -> list[str]:
    pairs = (
        df[["regulatory_type", "gene_set"]]
        .drop_duplicates()
        .sort_values(
            by=["regulatory_type", "gene_set"],
            key=lambda s: s.map(
                {
                    **{v: i for i, v in enumerate(REGULATORY_TYPE_ORDER)},
                    **{v: i for i, v in enumerate(GENE_SET_ORDER)},
                }
            ).fillna(999)
            if s.name in {"regulatory_type", "gene_set"}
            else s
        )
    )
    return [f"{row.regulatory_type}|{row.gene_set}" for row in pairs.itertuples(index=False)]


def _row_order(df: pd.DataFrame) -> list[str]:
    view = df[["cell"]].drop_duplicates().sort_values(["cell"], kind="stable")
    return [str(row.cell) for row in view.itertuples(index=False)]


def _column_labels(column_order: list[str]) -> list[str]:
    labels: list[str] = []
    for column_key in column_order:
        regulatory_type, gene_set = column_key.split("|", 1)
        labels.append(gene_set.upper())
    return labels


def _add_group_headers(ax: plt.Axes, column_order: list[str]) -> None:
    groups: list[tuple[str, int, int]] = []
    start = 0
    while start < len(column_order):
        regulatory_type = column_order[start].split("|", 1)[0]
        end = start
        while end + 1 < len(column_order) and column_order[end + 1].split("|", 1)[0] == regulatory_type:
            end += 1
        groups.append((regulatory_type, start, end))
        start = end + 1
    for regulatory_type, start, end in groups:
        if start:
            ax.axvline(start, color="#D9D9D9", linewidth=1.0)
        x_center = (start + end + 1) / 2.0
        ax.text(
            x_center,
            1.06,
            REGULATORY_TYPE_LABELS.get(regulatory_type, regulatory_type),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def _cell_order_by_eval(df: pd.DataFrame) -> list[tuple[str, str]]:
    order: list[tuple[str, str]] = []
    for eval_type in ("blind", "leave_tf_out_validation"):
        sub = df[df["evaluation_type"] == eval_type]
        cells = sorted(sub["cell"].dropna().astype(str).unique().tolist())
        order.extend((eval_type, cell) for cell in cells)
    return order


def _plot_all_ratio_metric_panels(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    metrics: tuple[str, ...],
    stem: str,
    title: str,
    cmap: LinearSegmentedColormap,
) -> list[Path]:
    df = _ordered_metric_dataframe(rows)
    if df.empty:
        return []
    df = df.dropna(subset=["ratio_num"]).copy()
    df = df[df["model"].astype(str) == "neg2"].copy()
    if df.empty:
        return []
    eval_order = [eval_type for eval_type in ("blind", "leave_tf_out_validation") if eval_type in set(df["evaluation_type"].astype(str))]
    if not eval_order:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ratios = sorted(df["negative_ratio"].dropna().unique().tolist(), key=_ratio_sort_key)
    for ratio in ratios:
        ratio_df = df[df["negative_ratio"].astype(str) == str(ratio)].copy()
        if ratio_df.empty:
            continue
        x_order = _cell_order_by_eval(ratio_df)
        if not x_order:
            continue
        x_positions = list(range(len(x_order)))
        x_labels = [cell for _, cell in x_order]
        blind_count = sum(1 for eval_type, _cell in x_order if eval_type == "blind")
        gene_sets = [gene_set for gene_set in GENE_SET_ORDER if gene_set in set(ratio_df["gene_set"].astype(str))]
        reg_types = [reg for reg in REGULATORY_TYPE_ORDER if reg in set(ratio_df["regulatory_type"].astype(str))]
        if not gene_sets or not reg_types:
            continue
        sns.set_theme(style="whitegrid", context="talk", rc=BARGRAPH_RC)
        fig, axes = plt.subplots(
            len(metrics),
            len(reg_types),
            figsize=(8.0 * len(reg_types), 4.9 * len(metrics)),
            squeeze=False,
            constrained_layout=True,
        )
        width = min(0.8 / max(len(gene_sets), 1), 0.18)
        for metric_idx, metric in enumerate(metrics):
            for col_idx, regulatory_type in enumerate(reg_types):
                ax = axes[metric_idx][col_idx]
                sub = ratio_df[ratio_df["regulatory_type"].astype(str) == regulatory_type].copy()
                if sub.empty:
                    ax.axis("off")
                    continue
                ax.axvspan(-0.5, blind_count - 0.5, color=EVAL_BACKGROUND["blind"], alpha=0.85, zorder=0)
                ax.axvspan(blind_count - 0.5, len(x_positions) - 0.5, color=EVAL_BACKGROUND["leave_tf_out_validation"], alpha=0.85, zorder=0)
                for gene_idx, gene_set in enumerate(gene_sets):
                    gene_sub = sub[sub["gene_set"].astype(str) == gene_set][["evaluation_type", "cell", metric]].copy()
                    plot_df = pd.DataFrame(x_order, columns=["evaluation_type", "cell"]).merge(
                        gene_sub,
                        on=["evaluation_type", "cell"],
                        how="left",
                    )
                    offset = (gene_idx - (len(gene_sets) - 1) / 2.0) * width
                    bar_values = [0.0 if pd.isna(v) else float(v) for v in plot_df[metric].tolist()]
                    ax.bar(
                        [x + offset for x in x_positions],
                        bar_values,
                        width=width,
                        color=GENE_SET_COLORS.get(gene_set, "#4C4C4C"),
                        edgecolor="white",
                        linewidth=0.6,
                        label=gene_set.upper() if metric_idx == 0 and col_idx == 0 else None,
                        zorder=2,
                    )
                if blind_count > 0 and blind_count < len(x_positions):
                    ax.axvline(blind_count - 0.5, color="#7A7A7A", linewidth=1.0, linestyle="--", zorder=3)
                ax.set_title(REGULATORY_TYPE_LABELS.get(regulatory_type, regulatory_type), fontsize=18, pad=12)
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=16)
                ax.set_xlabel("")
                ax.set_ylim(0.0, 1.02)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=35, ha="right")
                ax.grid(axis="y", color="#CFCFCF", linewidth=0.8, alpha=0.8, zorder=1)
                ax.grid(axis="x", visible=False)
        gene_handles, gene_labels = axes[0][0].get_legend_handles_labels()
        eval_handles = [
            Patch(facecolor=EVAL_BACKGROUND["blind"], edgecolor="none", label="Blind"),
            Patch(facecolor=EVAL_BACKGROUND["leave_tf_out_validation"], edgecolor="none", label="Leave-TF-out"),
        ]
        if gene_handles:
            fig.legend(
                gene_handles,
                gene_labels,
                loc="upper center",
                bbox_to_anchor=(0.18, 1.115),
                ncol=len(gene_handles),
                frameon=False,
                title="Gene set",
            )
        fig.legend(
            eval_handles,
            [h.get_label() for h in eval_handles],
            loc="upper center",
            bbox_to_anchor=(0.84, 1.115),
            ncol=2,
            frameon=False,
            title="Evaluation",
        )
        fig.suptitle(f"{title} - 1:{_ratio_label(ratio)}", y=1.12, fontsize=24, fontweight="bold")
        path = out_dir / f"{stem}_ratio_{_ratio_label(ratio)}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _precision_at_k(metrics: dict[str, Any], k: int) -> float | None:
    pk = metrics.get("precision_at_k")
    if not isinstance(pk, dict):
        return None
    for key in (f"present@{k}", str(k), f"@{k}"):
        out = _as_float(pk.get(key))
        if out is not None:
            return out
    return None


def _pick_auprc(metrics: dict[str, Any]) -> float | None:
    for key in ("auprc", "auprc_macro", "aucpr_macro", "auprc_micro", "aucpr_micro"):
        out = _as_float(metrics.get(key))
        if out is not None:
            return out
    return None


def _split_dataset_name(dataset: str) -> tuple[str, str]:
    if "_" not in dataset:
        return dataset, ""
    cell, variant = dataset.split("_", 1)
    return cell, variant


def _parse_variant(variant: str) -> tuple[str, str]:
    text = str(variant or "").strip()
    if not text:
        return "", ""
    if text.startswith("string_"):
        return "string", text[len("string_") :]
    if "_" not in text:
        return text, ""
    head, tail = text.rsplit("_", 1)
    return head, tail


def _report_rows(path: Path, *, model: str, eval_type: str, dataset: str) -> list[dict[str, Any]]:
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
        ns_std = metrics.get("negative_sampling_metric_std", {})
        ns_std = ns_std if isinstance(ns_std, dict) else {}
        ns_std_pk = ns_std.get("precision_at_k", {})
        ns_std_pk = ns_std_pk if isinstance(ns_std_pk, dict) else {}
        rows.append(
            {
                "model": model,
                "evaluation_type": eval_type,
                "dataset": dataset,
                "cell": cell,
                "variant": variant,
                "regulatory_type": _parse_variant(variant)[0],
                "gene_set": _parse_variant(variant)[1],
                "negative_ratio": ratio,
                "auroc": _as_float(metrics.get("auroc", metrics.get("auroc_macro"))),
                "auprc": _pick_auprc(metrics),
                "precision_at_10": _precision_at_k(metrics, 10),
                "precision_at_50": _precision_at_k(metrics, 50),
                "precision_at_100": _precision_at_k(metrics, 100),
                "auroc_neg_sampling_std": _as_float(ns_std.get("auroc", ns_std.get("auroc_macro"))),
                "auprc_neg_sampling_std": _pick_auprc(ns_std),
                "precision_at_10_neg_sampling_std": _as_float(ns_std_pk.get("present@10", ns_std_pk.get("10"))),
                "precision_at_50_neg_sampling_std": _as_float(ns_std_pk.get("present@50", ns_std_pk.get("50"))),
                "precision_at_100_neg_sampling_std": _as_float(ns_std_pk.get("present@100", ns_std_pk.get("100"))),
                "report_path": str(path),
            }
        )
    return rows


def _discover_validation_rows(root: Path, model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("eval_test_by_ratio.json")):
        parts = path.parts
        if "contexts" not in parts:
            continue
        idx = parts.index("contexts")
        if idx + 1 >= len(parts):
            continue
        dataset = parts[idx + 1]
        rows.extend(_report_rows(path, model=model, eval_type="leave_tf_out_validation", dataset=dataset))
    return rows


def _discover_blind_rows(root: Path, model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("blind_eval.json")):
        dataset = path.parent.name
        rows.extend(_report_rows(path, model=model, eval_type="blind", dataset=dataset))
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "evaluation_type",
        "dataset",
        "cell",
        "variant",
        "regulatory_type",
        "gene_set",
        "negative_ratio",
        *METRICS,
        *BLIND_STD_METRICS,
        "report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for metric in (*METRICS, *BLIND_STD_METRICS):
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    grouped = (
        df.groupby(["model", "evaluation_type", "negative_ratio"], as_index=False)[list((*METRICS, *BLIND_STD_METRICS))]
        .mean(numeric_only=True)
        .sort_values(["evaluation_type", "model", "negative_ratio"], key=lambda s: s.map(_ratio_sort_key) if s.name == "negative_ratio" else s)
    )
    overall = (
        df.groupby(["model", "evaluation_type"], as_index=False)[list((*METRICS, *BLIND_STD_METRICS))]
        .mean(numeric_only=True)
        .assign(negative_ratio="overall_avg")
    )
    out = pd.concat([grouped, overall], ignore_index=True)
    out["evaluation_label"] = out["evaluation_type"].map(EVAL_LABELS).fillna(out["evaluation_type"])
    return out


def _write_summary_table(summary: pd.DataFrame, path: Path) -> None:
    cols = ["model", "evaluation_type", "negative_ratio", *METRICS, *BLIND_STD_METRICS]
    out = summary[cols].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _write_ratio_average_table(summary: pd.DataFrame, path: Path) -> None:
    cols = ["model", "evaluation_type", "negative_ratio", *METRICS, *BLIND_STD_METRICS]
    out = summary[summary["negative_ratio"].astype(str) != "overall_avg"][cols].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _write_pivot_table(summary: pd.DataFrame, path: Path) -> None:
    view = summary.copy()
    view["ratio_label"] = view["negative_ratio"].map(lambda x: f"1:{x}" if str(x) != "avg" else "avg")
    pivot = view.pivot_table(
        index=["evaluation_type", "model"],
        columns="ratio_label",
        values=list(METRICS),
        aggfunc="first",
    )
    pivot = pivot.sort_index(axis=1, level=[0, 1])
    pivot.to_csv(path)


def _write_detail_pivot(rows: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    for metric in (*METRICS, *BLIND_STD_METRICS):
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    pivot = df.pivot_table(
        index=["cell", "regulatory_type", "gene_set"],
        columns=["evaluation_type", "model", "negative_ratio"],
        values=list((*METRICS, *BLIND_STD_METRICS)),
        aggfunc="mean",
    )
    pivot = pivot.sort_index(axis=1, level=[0, 1, 2, 3])
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(path)


def _plot_ratio_lines(summary: pd.DataFrame, out_dir: Path) -> Path:
    df = summary[summary["negative_ratio"] != "overall_avg"].copy()
    df["negative_ratio_num"] = pd.to_numeric(df["negative_ratio"], errors="coerce")
    long = df.melt(
        id_vars=["model", "evaluation_type", "evaluation_label", "negative_ratio_num"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value", "negative_ratio_num"])
    long["metric_label"] = long["metric"].map(METRIC_LABELS)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    metric_order = list(METRICS)
    for ax, metric in zip(axes_flat, metric_order):
        sub = long[long["metric"] == metric]
        sns.lineplot(
            data=sub,
            x="negative_ratio_num",
            y="value",
            hue="model",
            style="evaluation_label",
            markers=True,
            dashes=True,
            palette=MODEL_PALETTE,
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Negative ratio")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(0.0, 1.02)
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].axis("off")
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Neg1 vs Neg2 by ratio and evaluation type", y=1.02, fontsize=16)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "neg_model_metric_lines.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_average_bars(summary: pd.DataFrame, out_dir: Path) -> Path:
    df = summary[summary["negative_ratio"] == "overall_avg"].copy()
    long = df.melt(
        id_vars=["model", "evaluation_type", "evaluation_label"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    long["metric_label"] = long["metric"].map(METRIC_LABELS)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, METRICS):
        sub = long[long["metric"] == metric]
        sns.barplot(
            data=sub,
            x="evaluation_label",
            y="value",
            hue="model",
            palette=MODEL_PALETTE,
            ax=ax,
        )
        ax.set_title(f"Average {METRIC_LABELS[metric]}")
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(axis="x", rotation=12)
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].axis("off")
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False, title="Model")
    fig.suptitle("Average Neg1 vs Neg2 performance", y=1.02, fontsize=16)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "neg_model_average_bars.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_blind_neg_sampling_std(summary: pd.DataFrame, out_dir: Path) -> Path:
    df = summary[(summary["evaluation_type"] == "blind") & (summary["negative_ratio"] != "overall_avg")].copy()
    if df.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "blind_negative_sampling_std.png"
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No blind negative-sampling std available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return path
    df["negative_ratio_num"] = pd.to_numeric(df["negative_ratio"], errors="coerce")
    long = df.melt(
        id_vars=["model", "negative_ratio_num"],
        value_vars=list(BLIND_STD_METRICS),
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value", "negative_ratio_num"])
    long["metric_label"] = long["metric"].map(METRIC_LABELS)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, BLIND_STD_METRICS):
        sub = long[long["metric"] == metric]
        sns.lineplot(
            data=sub,
            x="negative_ratio_num",
            y="value",
            hue="model",
            palette=MODEL_PALETTE,
            marker="o",
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Negative ratio")
        ax.set_ylabel("Std across repeats")
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].axis("off")
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False, title="Model")
    fig.suptitle("Blind-eval negative-sampling variability", y=1.02, fontsize=16)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "blind_negative_sampling_std.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_model_heatmaps(rows: list[dict[str, Any]], out_dir: Path, *, model: str) -> list[Path]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    for metric in (*METRICS, *BLIND_STD_METRICS):
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[df["model"] == model].copy()
    if df.empty:
        return []
    df["column_key"] = df["regulatory_type"].astype(str) + "|" + df["gene_set"].astype(str)
    paths: list[Path] = []
    sns.set_theme(style="white", context="notebook")
    ordered_cols = sorted(df["column_key"].dropna().unique().tolist())
    ordered_cells = sorted(df["cell"].dropna().unique().tolist())
    metrics_for_heatmap = list(METRICS)
    for eval_type in sorted(df["evaluation_type"].dropna().unique().tolist()):
        for ratio in sorted(df["negative_ratio"].dropna().unique().tolist(), key=_ratio_sort_key):
            sub = df[(df["evaluation_type"] == eval_type) & (df["negative_ratio"].astype(str) == str(ratio))]
            if sub.empty:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
            axes_flat = axes.ravel()
            for ax, metric in zip(axes_flat, metrics_for_heatmap):
                mat = sub.pivot_table(
                    index="cell",
                    columns="column_key",
                    values=metric,
                    aggfunc="mean",
                )
                if mat.empty:
                    ax.axis("off")
                    continue
                mat = mat.reindex(index=ordered_cells, columns=ordered_cols)
                sns.heatmap(
                    mat,
                    annot=True,
                    fmt=".2f",
                    cmap="YlGnBu",
                    vmin=0.0,
                    vmax=1.0,
                    linewidths=0.4,
                    linecolor="white",
                    cbar=metric == metrics_for_heatmap[-1],
                    annot_kws={"size": 8},
                    ax=ax,
                )
                ax.set_title(f"{METRIC_LABELS[metric]} ({model})")
                ax.set_xlabel("Regulatory type | gene set")
                ax.set_ylabel("Cell")
                ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="y", rotation=0)
            axes_flat[-1].axis("off")
            fig.suptitle(
                f"{model} {EVAL_LABELS.get(eval_type, eval_type)} ratio 1:{ratio}",
                y=1.02,
                fontsize=16,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{model}_heatmap_{eval_type}_ratio_{ratio}.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg1-validation-root", default="artifacts/multicontext_tf_eager/all_datacontext_contexts_neg1")
    ap.add_argument("--neg2-validation-root", default="artifacts/multicontext_tf_eager/all_datacontext_contexts_neg2")
    ap.add_argument("--neg1-blind-root", default="artifacts/blind_tf_eager_neg1")
    ap.add_argument("--neg2-blind-root", default="artifacts/blind_tf_eager_neg2")
    ap.add_argument("--out-dir", default="artifacts/evaluation_tables/neg1_neg2_comparison")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    rows.extend(_discover_validation_rows(Path(args.neg1_validation_root), "neg1"))
    rows.extend(_discover_validation_rows(Path(args.neg2_validation_root), "neg2"))
    rows.extend(_discover_blind_rows(Path(args.neg1_blind_root), "neg1"))
    rows.extend(_discover_blind_rows(Path(args.neg2_blind_root), "neg2"))
    if not rows:
        raise SystemExit("No neg1/neg2 validation or blind evaluation reports were found")

    rows.sort(
        key=lambda row: (
            str(row["evaluation_type"]),
            str(row["model"]),
            str(row["dataset"]),
            _ratio_sort_key(row["negative_ratio"]),
        )
    )
    out_dir = Path(args.out_dir)
    raw_path = out_dir / "neg1_neg2_metrics_long.csv"
    summary_path = out_dir / "neg1_neg2_summary.csv"
    ratio_avg_path = out_dir / "neg1_neg2_ratio_averages.csv"
    pivot_path = out_dir / "neg1_neg2_summary_pivot.csv"
    detail_pivot_path = out_dir / "neg1_neg2_detail_pivot.csv"
    fig_dir = out_dir / "figures"

    _write_csv(rows, raw_path)
    summary = _summary_dataframe(rows)
    _write_summary_table(summary, summary_path)
    _write_ratio_average_table(summary, ratio_avg_path)
    _write_pivot_table(summary, pivot_path)
    _write_detail_pivot(rows, detail_pivot_path)
    line_path = _plot_ratio_lines(summary, fig_dir)
    bar_path = _plot_average_bars(summary, fig_dir)
    std_path = _plot_blind_neg_sampling_std(summary, fig_dir)
    model_heatmap_paths = _plot_model_heatmaps(rows, fig_dir, model="neg2")
    combined_auc_paths = _plot_all_ratio_metric_panels(
        rows,
        fig_dir,
        metrics=("auroc", "auprc"),
        stem="combined_auroc_auprc_bargraph",
        title="Sampled_AUROC and Sampled AUPRC",
        cmap=ALL_RATIO_AUROC_AUPRC_CMAP,
    )
    combined_precision_paths = _plot_all_ratio_metric_panels(
        rows,
        fig_dir,
        metrics=("precision_at_10", "precision_at_50", "precision_at_100"),
        stem="combined_precision_at_k_bargraph",
        title="Precision@K",
        cmap=ALL_RATIO_PRECISION_CMAP,
    )
    print(
        "[compare-tf-eager-neg-models] "
        f"rows={len(rows)} raw={raw_path} summary={summary_path} ratio_avg={ratio_avg_path} pivot={pivot_path} detail_pivot={detail_pivot_path} "
        f"figures={line_path},{bar_path},{std_path}"
        f"{',' if model_heatmap_paths else ''}{','.join(str(p) for p in model_heatmap_paths)}"
        f"{',' if combined_auc_paths else ''}{','.join(str(p) for p in combined_auc_paths)}"
        f"{',' if combined_precision_paths else ''}{','.join(str(p) for p in combined_precision_paths)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
