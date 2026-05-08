#!/usr/bin/env python3
"""Wilcoxon analysis and benchmark heatmaps from fair benchmark table."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "fair_benchmark_vs_baselines" / "fair_benchmark_vs_baselines_long.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "fair_benchmark_vs_baselines"


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _method_label(method: str) -> str:
    m = str(method).strip()
    if m in {"TF-EAGER", "TF-EAGER-neg2"}:
        return "GRNAgent"
    return m


def _reg_type(dataset: str) -> str:
    d = str(dataset)
    if "nonspecific_chipseq" in d:
        return "Non-Cell-Type Specific"
    if "specific_chipseq" in d:
        return "Cell-type-specific ChIP-seq"
    if "string" in d:
        return "STRING"
    return "Other"


def _cell_type(dataset: str) -> str:
    d = str(dataset)
    for ct in ["mESC", "mHSC-L", "mHSC-E", "mHSC-GM", "mDC", "hESC", "hHep"]:
        if ct in d:
            return ct
    return "Other"


def _gene_set(dataset: str) -> str:
    m = re.search(r"(tf)?(500|1000)\b", str(dataset), re.IGNORECASE)
    if not m:
        return "Other"
    token = m.group(0).upper()
    if token == "500":
        return "500"
    if token == "1000":
        return "1000"
    return token.replace("TF", "TF")


def _load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = df["method"].map(_method_label)
    df["aupr"] = pd.to_numeric(df["aupr"], errors="coerce")
    df["epr_at_100"] = pd.to_numeric(df["epr_at_100"], errors="coerce")
    if "prediction_pairs" in df.columns:
        pred_pairs = pd.to_numeric(df["prediction_pairs"], errors="coerce").fillna(0)
        # If the model produced no scored edges, EPR@100 should be 0, not a tie-order artifact.
        no_pred_mask = (df["method"] == "GRNAgent") & (pred_pairs <= 0)
        df.loc[no_pred_mask, "epr_at_100"] = 0.0
    df["RegType"] = df["dataset"].map(_reg_type)
    df["CellType"] = df["dataset"].map(_cell_type)
    df["GeneSet"] = df["dataset"].map(_gene_set)
    return df


def _paired_wilcoxon(
    df: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairings = [(methods[0], methods[1]), (methods[0], methods[2]), (methods[1], methods[2])]

    for metric in metrics:
        metric_df = df[df["method"].isin(methods)][["dataset", "method", metric]].dropna()
        wide = metric_df.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")

        for m1, m2 in pairings:
            if m1 not in wide.columns or m2 not in wide.columns:
                rows.append(
                    {
                        "metric": metric,
                        "method_1": m1,
                        "method_2": m2,
                        "n_pairs": 0,
                        "wilcoxon_statistic": None,
                        "p_value": None,
                        "median_delta_method1_minus_method2": None,
                        "mean_delta_method1_minus_method2": None,
                    }
                )
                continue

            paired = wide[[m1, m2]].dropna()
            n_pairs = len(paired)
            if n_pairs == 0:
                stat = None
                p_value = None
            else:
                diff = paired[m1] - paired[m2]
                if np.allclose(diff.values, 0.0):
                    stat = 0.0
                    p_value = 1.0
                else:
                    test = wilcoxon(paired[m1], paired[m2], zero_method="wilcox", alternative="two-sided")
                    stat = _safe_float(test.statistic)
                    p_value = _safe_float(test.pvalue)

            rows.append(
                {
                    "metric": metric,
                    "method_1": m1,
                    "method_2": m2,
                    "n_pairs": n_pairs,
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                    "median_delta_method1_minus_method2": _safe_float((paired[m1] - paired[m2]).median())
                    if n_pairs
                    else None,
                    "mean_delta_method1_minus_method2": _safe_float((paired[m1] - paired[m2]).mean())
                    if n_pairs
                    else None,
                }
            )
    return pd.DataFrame(rows)


def _plot_combined_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_context("talk", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.dpi"] = 330

    metrics = [
        ("aupr", "AUPRC", 100.0, "AUPRC (%)"),
        ("epr_at_100", "EPR@100", 100.0, "EPR@100 (%)"),
    ]
    # Print-friendly monotonic palette with a lighter low end so abundant zero
    # cells remain visually pale instead of muddy.
    base = plt.get_cmap("cividis_r")
    cmap = LinearSegmentedColormap.from_list(
        "cividis_trimmed_r",
        base(np.linspace(0.18, 1.0, 256)),
    )
    gene_set_order = ["TF500", "TF1000", "500", "1000"]
    reg_type_order = ["Non-Cell-Type Specific", "Cell-type-specific ChIP-seq", "STRING"]
    cell_type_order = ["mESC", "mHSC-L", "mHSC-E", "mHSC-GM", "mDC", "hESC", "hHep"]

    methods_seen = [m for m in ["GRNAgent", "GRNFormer", "GNNLink"] if m in set(df["method"])]
    methods_seen += sorted([m for m in df["method"].dropna().unique() if m not in methods_seen])
    methods = methods_seen

    n_rows_per_metric = len(gene_set_order)
    n_cols = len(reg_type_order)
    total_rows = len(metrics) * n_rows_per_metric
    # A more compact canvas improves readability by reducing the large vertical
    # whitespace between successive heatmap rows.
    fig = plt.figure(figsize=(8.8 * n_cols, 1.85 * total_rows))

    # Keep typography readable, but avoid oversized text that forces excessive whitespace.
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "axes.labelweight": "bold",       # Bold axis labels
        "axes.titleweight": "bold",       # Bold titles
        "xtick.labelsize": 13,
        "ytick.labelsize": 14,
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.fontsize": 15,
        "legend.title_fontsize": 16,
        "text.color": "black",
        "font.weight": "bold",            # Bold text
    })

    gs_top = fig.add_gridspec(
        n_rows_per_metric,
        n_cols,
        left=0.11,
        right=0.855,
        bottom=0.53,
        top=0.945,
        hspace=0.06,
        wspace=0.035,
    )
    gs_bottom = fig.add_gridspec(
        n_rows_per_metric,
        n_cols,
        left=0.11,
        right=0.855,
        bottom=0.07,
        top=0.445,
        hspace=0.06,
        wspace=0.035,
    )

    axes = np.empty((total_rows, n_cols), dtype=object)
    for i in range(n_rows_per_metric):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs_top[i, j])
            axes[i + n_rows_per_metric, j] = fig.add_subplot(gs_bottom[i, j])

    cbar_ax_top = fig.add_axes([0.87, 0.56, 0.015, 0.34])
    cbar_ax_bottom = fig.add_axes([0.87, 0.085, 0.015, 0.34])

    def _relative_luminance(rgba: tuple[float, float, float, float]) -> float:
        r, g, b = rgba[:3]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for metric_idx, (metric_col, _raw_label, scale_factor, metric_label) in enumerate(metrics):
        norm = Normalize(vmin=0.0, vmax=1.0)
        cbar_ax = cbar_ax_top if metric_idx == 0 else cbar_ax_bottom
        cbar_added = False

        for row_idx, gs in enumerate(gene_set_order):
            for col_idx, rt in enumerate(reg_type_order):
                ax = axes[metric_idx * n_rows_per_metric + row_idx, col_idx]
                block = df[(df["GeneSet"] == gs) & (df["RegType"] == rt)]
                if block.empty:
                    ax.set_axis_off()
                    continue

                agg = (
                    block.groupby(["CellType", "method"], as_index=False)[metric_col]
                    .mean(numeric_only=True)
                    .dropna()
                )
                ct_present = [ct for ct in cell_type_order if ct in set(agg["CellType"])]
                heat = np.full((len(ct_present), len(methods)), np.nan)
                for i, ct in enumerate(ct_present):
                    row_vals = agg[agg["CellType"] == ct].set_index("method")[metric_col]
                    for j, m in enumerate(methods):
                        if m in row_vals.index:
                            heat[i, j] = float(row_vals[m]) * scale_factor

                valid_mask = ~np.isnan(heat)
                heat_color = np.full_like(heat, np.nan, dtype=float)
                if np.any(valid_mask):
                    vals = heat[valid_mask]
                    zero_mask = valid_mask & np.isclose(heat, 0.0)
                    nonzero_mask = valid_mask & ~np.isclose(heat, 0.0)
                    # Exact zeros should always map to the lowest legend color.
                    heat_color[zero_mask] = 0.0
                    if np.any(nonzero_mask):
                        nz_vals = heat[nonzero_mask]
                        if len(nz_vals) == 1 or np.allclose(nz_vals, nz_vals[0]):
                            heat_color[nonzero_mask] = 1.0
                        else:
                            heat_color[nonzero_mask] = pd.Series(nz_vals).rank(
                                method="average", pct=True
                            ).to_numpy()

                annot = np.empty(heat.shape, dtype=object)
                for i in range(heat.shape[0]):
                    for j in range(heat.shape[1]):
                        val = heat[i, j]
                        if np.isnan(val) or np.isclose(val, 0.0):
                            annot[i, j] = ""
                        else:
                            annot[i, j] = f"{val:.2f}"

                sns.heatmap(
                    heat_color,
                    ax=ax,
                    cmap='magma_r',
                    vmin=0.0,
                    vmax=1.0,
                    square=False,
                    linewidths=0.45,
                    linecolor="#8a8a8a",
                    annot=annot,
                    fmt="",
                    annot_kws={"fontsize": 14, "fontweight": "bold"},
                    cbar=(not cbar_added and col_idx == n_cols - 1 and row_idx == 0),
                    cbar_ax=cbar_ax if (not cbar_added and col_idx == n_cols - 1 and row_idx == 0) else None,
                    cbar_kws={"label": f"{metric_label} relative rank (within panel)"},
                    mask=np.isnan(heat_color),
                )
                if not cbar_added and col_idx == n_cols - 1 and row_idx == 0:
                    cbar = ax.collections[0].colorbar
                    if cbar is not None:
                        cbar.set_ticks([0.0, 0.5, 1.0])
                        cbar.set_ticklabels(["lowest", "middle", "highest"])
                    cbar_added = True

                text_idx = 0
                for i in range(heat_color.shape[0]):
                    for j in range(heat_color.shape[1]):
                        if np.isnan(heat_color[i, j]):
                            continue
                        if text_idx >= len(ax.texts):
                            break
                        rgba = cmap(norm(float(heat_color[i, j])))
                        luminance = _relative_luminance(rgba)
                        ax.texts[text_idx].set_color("black" if luminance >= 0.52 else "white")
                        text_idx += 1

                if row_idx == 0:
                    ax.set_title(rt, fontsize=18, fontweight="bold", pad=10)
                else:
                    ax.set_title("")

                if col_idx == 0:
                    ax.set_ylabel(f"{gs}\nCell Type", fontsize=15, fontweight="bold", labelpad=18)
                    ax.set_yticks(np.arange(len(ct_present)) + 0.5)
                    ax.set_yticklabels(ct_present, rotation=0, fontsize=14)
                else:
                    ax.set_ylabel("")
                    ax.set_yticks([])
                    ax.set_yticklabels([])

                if row_idx == n_rows_per_metric - 1:
                    ax.set_xticks(np.arange(len(methods)) + 0.5)
                    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=13)
                    if metric_idx == len(metrics) - 1:
                        ax.set_xlabel("Method", fontsize=15, fontweight="bold", labelpad=4)
                    else:
                        ax.set_xlabel("")
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_xlabel("")

    fig.text(0.038, 0.75, "AUPRC (%)", fontsize=18, fontweight="bold", rotation=90, va="center")
    fig.text(0.038, 0.275, "EPR@100 (%)", fontsize=18, fontweight="bold", rotation=90, va="center")
    fig.text(0.075, 0.962, "A", fontsize=24, fontweight="bold")
    fig.text(0.075, 0.458, "B", fontsize=24, fontweight="bold")
    fig.suptitle("All Benchmarks: AUPRC and EPR@100", fontsize=22, fontweight="bold", y=0.986)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Wilcoxon tests and heatmaps for fair benchmark baselines.")
    ap.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = _load_and_prepare(args.input_csv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_long = out_dir / "fair_benchmark_vs_baselines_long_renamed.csv"
    df.to_csv(normalized_long, index=False)

    methods = ["GRNAgent", "GRNFormer", "GNNLink"]
    metrics = ["aupr", "epr_at_100"]
    wilcoxon_df = _paired_wilcoxon(df, methods=methods, metrics=metrics)
    wilcoxon_path = out_dir / "wilcoxon_paired_grnagent_grnformer_gnnlink.csv"
    wilcoxon_df.to_csv(wilcoxon_path, index=False)

    heatmap_path = out_dir / "combined_auprc_epr100_heatmap_allbenchmarks.png"
    _plot_combined_heatmap(df, heatmap_path)

    print(f"Wrote {normalized_long}")
    print(f"Wrote {wilcoxon_path}")
    print(f"Wrote {heatmap_path}")


if __name__ == "__main__":
    main()
    # Use a print-friendly monotonic palette, but trim away the darkest low-end
    # tones so the many zero-valued cells remain visibly light.
    base = plt.get_cmap("cividis_r")
    cmap = LinearSegmentedColormap.from_list(
        "cividis_trimmed_r",
        base(np.linspace(0.18, 1.0, 256)),
    )
