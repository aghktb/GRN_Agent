#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/InferenceAblation/mHSC-GM_nonspecific_chipseq_tf1000/tf_eager/test_network_nomotif.csv"
DEFAULT_OUTDIR = ROOT / "artifacts/InferenceAblation/mHSC-GM_nonspecific_chipseq_tf1000"

def filter_novel_edges(input_csv: Path, output_csv: Path, min_p: float, max_p: float, random_n: int | None, seed: int) -> int:
    import random
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    novel = [r for r in rows if str(r.get("window_index", "")).strip() != "1" and min_p <= float(r.get("p_present", 0)) <= max_p]
    if random_n and len(novel) > random_n:
        random.seed(seed)
        novel = random.sample(novel, random_n)
    else:
        novel.sort(key=lambda r: float(r["p_present"]), reverse=True)
    if not novel:
        sys.exit("No edges passed filter.")
    if output_csv.exists():
        output_csv.unlink()
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(novel[0].keys()))
        w.writeheader()
        w.writerows(novel)
    return len(novel)

def run_validation(filtered_csv: Path, output_csv: Path, cell_type: str, limit: int | None) -> None:
    cmd = [sys.executable, str(ROOT / "scripts/run_literature_validation.py"), "--input", str(filtered_csv), "--output", str(output_csv), "--cell-type", cell_type, "--resume"]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[validate] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def make_scatter(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.ticker as ticker
    from matplotlib import colors
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "axes.labelweight": "bold", "axes.titleweight": "bold", "xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    min_p = df["p_present"].min()
    dist = 1.0 - min_p
    delta = dist / 10.0 if dist > 0 else 0.02
    x_min, x_max = min_p - delta, 1.0 + delta
    ax.axhspan(0.5, 1.05, color="#f1f2f6", alpha=0.5, zorder=0, label="High Literature Support")
    hex_list = ["#d5dbb6", "#e6ebd7","#f7fcf0", "#e0f3db","#d0e2ca", "#ccebc5","#bbdab4", "#a8ddb5", "#7bccc4", "#6abbb3", "#4eb3d3", "#3da2c2", "#2b8cbe", "#0868ac", "#084081", "#081d58", "#081000"]
    rev = list(reversed(hex_list))
    cmap = colors.LinearSegmentedColormap.from_list("custom_spectrum", rev)
    max_val = max(int(df["n_supporting"].max()), 1)
    norm = colors.Normalize(vmin=0, vmax=max_val)
    sizes = np.where(df["n_supporting"] > 0, 130, 130)
    edgecolors = np.where(df["n_supporting"] > 0, "black", "#bdc3c7")
    sc = ax.scatter(df["p_present"], df["lit_score"], c=df["n_supporting"], cmap=cmap, norm=norm, s=sizes, alpha=1.0, edgecolor=edgecolors, linewidth=0.7, zorder=3, clip_on=False)
    n_total = len(df)
    n_supported = (df["lit_score"] > 0).sum()
    perc_supported = (n_supported / n_total) * 100
    ax.set_title("Novel Edge Validation: Model vs. Literature", pad=15, fontsize=14)
    ax.text(0.5, 1.035, f"N = {n_total} novel edges  |  {perc_supported:.1f}% supported", transform=ax.transAxes, ha="center", fontsize=11, fontweight="normal", color="#2f3542")
    ax.set_xlabel("Model Prediction Confidence ($P_{present}$)", fontsize=12)
    ax.set_ylabel("Literature Validation Score", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle="--", alpha=0.3, zorder=1)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.set_label("Number of Supporting Papers", fontsize=11, labelpad=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")

def make_cumulative_discovery(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    df = df.sort_values("p_present", ascending=False).reset_index(drop=True)
    df["is_supported"] = (df["lit_score"] > 0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ranks = np.arange(1, len(df) + 1)
    cum_hits = df["is_supported"].cumsum()
    random_baseline = ranks * (df["is_supported"].sum() / len(df))
    ax.plot(ranks, cum_hits, lw=3.5, color="#084081", label="Model Discovery", zorder=3)
    ax.plot(ranks, random_baseline, lw=1.5, color="#969696", ls="--", label="Random Expectation", zorder=2)
    ax.fill_between(ranks, random_baseline, cum_hits, color="#0868ac", alpha=0.2, zorder=1)
    ax.set_title("Cumulative Discovery Curve", fontweight="bold", pad=20, fontsize=14)
    ax.set_xlabel("Ranked Predictions (High to Low Confidence)", fontsize=12)
    ax.set_ylabel("Cumulative Validated Edges", fontsize=12)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.grid(alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def make_binned_validation(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    df = df.sort_values("p_present", ascending=False).reset_index(drop=True)
    df["is_supported"] = (df["lit_score"] > 0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    n_bins = 5
    df["bin"] = pd.qcut(df.index, q=n_bins, labels=[f"Top {int((i+1)/n_bins*100)}%" for i in range(n_bins)])
    binned = df.groupby("bin", observed=True)["is_supported"].mean() * 100
    colors = ["#0868ac", "#43a2ca", "#7bccc4", "#bae4bc", "#f0f9e8"]
    bars = ax.bar(binned.index, binned.values, color=colors, edgecolor="#252525", alpha=1.0)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f"{height:.1f}%", ha="center", va="bottom", fontweight="bold", color="#252525", fontsize=11)
    #ax.set_title("Validation Rate by Confidence Tier", fontweight="bold", pad=20, fontsize=14)
    ax.set_ylabel("% Edges with Literature Support", fontsize=12)
    ax.set_ylim(0, max(binned.values + 5) * 1.1)
    ax.grid(axis="y", alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def make_confidence_separability(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    df["is_supported"] = (df["lit_score"] > 0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    validated = df[df["is_supported"] == 1]["p_present"]
    unsupported = df[df["is_supported"] == 0]["p_present"]
    if len(validated) > 0 and len(unsupported) > 0:
        parts = ax.violinplot([unsupported, validated], showmedians=True)
        colors = ["#f7fcf0", "#7bccc4"]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor("#252525")
            pc.set_alpha(1.0)
        for part in ["cbars", "cmins", "cmaxes", "cmedians"]:
            parts[part].set_edgecolor("#252525")
        stat, p_val = mannwhitneyu(validated, unsupported, alternative="greater")
        sig_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax.text(0.5, 0.9, f"Significance: {sig_text}", transform=ax.transAxes, ha="center", fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", edgecolor="#252525", alpha=0.9), fontsize=11)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Unsupported", "Validated"], fontweight="bold", fontsize=12)
    ax.set_ylabel("Model Confidence ($P_{present}$)", fontsize=12)
    ax.set_title("Confidence Separability", fontweight="bold", pad=20, fontsize=14)
    ax.grid(axis="y", alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def make_topn_table(df: pd.DataFrame, out_path: Path, top_n: int = 25) -> None:
    import textwrap
    supported = df[df["n_supporting"] > 0].copy()
    supported = supported.sort_values(["lit_score", "n_supporting", "p_present"], ascending=[False, False, False]).head(top_n)
    if supported.empty:
        print("[fig] no supported edges to show in top-N table; skipping")
        return
    cols = ["source_tf", "target_gene", "p_present", "lit_score", "n_supporting", "reg_type", "evidence_types", "pmids"]
    table = supported[cols].copy()
    table["p_present"] = table["p_present"].map(lambda x: f"{x:.3f}")
    table["lit_score"] = table["lit_score"].map(lambda x: f"{x:.3f}")
    table["evidence_types"] = table["evidence_types"].fillna("").map(lambda s: "\n".join(textwrap.wrap(str(s), width=45)))
    table["pmids"] = table["pmids"].fillna("").map(lambda s: "\n".join(textwrap.wrap(str(s).replace(", ", ",").replace(",", ", "), width=60)))
    table.columns = ["TF", "Target", "Conf.", "Score", "Papers", "Type", "Evidence", "PMIDs"]
    fig, ax = plt.subplots(figsize=(14, 0.5 * len(table) + 0.5))
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=list(table.columns), loc="upper center", cellLoc="left", colWidths=[0.08, 0.08, 0.07, 0.07, 0.07, 0.1, 0.27, 0.31])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.25, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#084081")
            cell.set_edgecolor("#084081")
        else:
            cell.set_edgecolor("#dfe6e9")
            if row % 2 == 0:
                cell.set_facecolor("#f8f9fa")
            cell.set_text_props(va='center')
    ax.set_title(f"Top {len(table)} Novel Predictions with Literature Support", y=1.0, pad=2, fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--cell-type", default="mHSC", dest="cell_type")
    ap.add_argument("--min-p", type=float, default=0.95)
    ap.add_argument("--max-p", type=float, default=1.0)
    ap.add_argument("--random-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--top-n", type=int, default=0, dest="top_n", help="limit figures to first N rows of lit CSV (0 = all)")
    ap.add_argument("--tag", default="", help="suffix before extension on all outputs, e.g. 'v2' -> novel_edges_filtered_v2.csv")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    def _tag(name: str) -> str:
        if not args.tag.strip():
            return name
        p = Path(name)
        return f"{p.stem}_{args.tag.strip()}{p.suffix}"

    filtered_csv = args.outdir / _tag("novel_edges_filtered.csv")
    lit_csv = args.outdir / _tag("novel_edges_lit_scores.csv")
    scatter_png = args.outdir / _tag("analysis_scatter.png")
    topn_png = args.outdir / _tag("analysis_topN.png")
    if not args.skip_validate:
        n = filter_novel_edges(args.input, filtered_csv, args.min_p, args.max_p, args.random_n, args.seed)
        print(f"[filter] kept {n} novel edges -> {filtered_csv}")
        run_validation(filtered_csv, lit_csv, args.cell_type, args.limit)
    elif not lit_csv.exists():
        sys.exit(f"--skip-validate set but {lit_csv} does not exist")
    df = pd.read_csv(lit_csv)
    df["p_present"] = pd.to_numeric(df["p_present"], errors="coerce")
    df["lit_score"] = pd.to_numeric(df["lit_score"], errors="coerce")
    df["n_supporting"] = pd.to_numeric(df["n_supporting"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["p_present", "lit_score"])
    if args.top_n > 0:
        df = df.head(args.top_n).reset_index(drop=True)
    make_scatter(df, scatter_png)
    make_cumulative_discovery(df, args.outdir / _tag("analysis_cumulative.png"))
    make_binned_validation(df, args.outdir / _tag("analysis_binned.png"))
    make_confidence_separability(df, args.outdir / _tag("analysis_violin.png"))
    make_topn_table(df, topn_png)
    print(f"\nDone.\n  CSV:      {lit_csv}\n  Scatter:  {scatter_png}\n  Analysis: {args.outdir}/analysis_*.png\n  Top-N:    {topn_png}")

if __name__ == "__main__":
    main()
