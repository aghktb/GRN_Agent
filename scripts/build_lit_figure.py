#!/usr/bin/env python3
"""
Filter novel high-confidence TF-EAGER predictions, run literature validation
on them, and build paper figures.

Pipeline:
  1. Filter test_network_nomotif.csv: drop window_index == 1 (in-training-data),
     drop p_present < 0.5. Keep top --limit by p_present.
  2. Run run_literature_validation.py on the filtered CSV (skip with --skip-validate
     if novel_edges_lit_scores.csv already exists).
  3. Build two figures:
       - novel_edges_scatter.png  — p_present vs lit_score, colored by n_supporting
       - novel_edges_topN.png     — top-N table of supported edges with evidence
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/InferenceAblation/mHSC-GM_nonspecific_chipseq_tf1000/tf_eager/test_network_nomotif.csv"
DEFAULT_OUTDIR = ROOT / "artifacts/InferenceAblation/mHSC-GM_nonspecific_chipseq_tf1000"


def filter_novel_edges(input_csv: Path, output_csv: Path, min_p: float) -> int:
    """Write the full set of novel high-confidence edges (no limit). Overwrites."""
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    novel = [r for r in rows
             if str(r.get("window_index", "")).strip() != "1"
             and float(r.get("p_present", 0)) >= min_p]
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


def run_validation(filtered_csv: Path, output_csv: Path,
                   cell_type: str, limit: int | None) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_literature_validation.py"),
        "--input", str(filtered_csv),
        "--output", str(output_csv),
        "--cell-type", cell_type,
        "--resume",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[validate] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def make_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """PhD-quality publication figure for model confidence vs literature validation."""
    import matplotlib.ticker as ticker
    
    # Use a clean, professional style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Calculate limits with user-specified delta logic
    min_p = df["p_present"].min()
    dist = 1.0 - min_p
    delta = dist / 10.0 if dist > 0 else 0.02
    x_min, x_max = min_p - delta, 1.0 + delta

    # Background: Shade the high-literature-confidence region (e.g., > 0.5)
    ax.axhspan(0.5, 1.05, color="#f1f2f6", alpha=0.5, zorder=0, label="High Literature Support")
    
    # Plotting points with overplotting management
    sc = ax.scatter(
        df["p_present"], df["lit_score"],
        c=df["n_supporting"], 
        cmap="magma",          # Perceptually uniform
        s=85, 
        alpha=0.7,             # Transparency for density perception
        edgecolor="white", 
        linewidth=0.8,
        zorder=3,
        clip_on=False
    )
    
    # Statistical Annotations
    n_total = len(df)
    n_supported = (df["lit_score"] > 0).sum()
    perc_supported = (n_supported / n_total) * 100
    
    # Professional labeling with stats in subtitle to avoid obscuring points
    ax.set_title("Novel Edge Validation: Model vs. Literature", 
                 pad=30, fontsize=13)
    ax.text(0.5, 1.035, f"N = {n_total} novel edges  |  {perc_supported:.1f}% with literature support", 
            transform=ax.transAxes, ha="center", fontsize=10, fontweight="normal", color="#2f3542")
    
    ax.set_xlabel("Model Prediction Confidence ($P_{present}$)", fontsize=11)
    ax.set_ylabel("Literature Validation Score", fontsize=11)
    
    # Axis Refinement
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.05, 1.05)
    
    # Custom Ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    
    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Grid
    ax.grid(axis="both", linestyle="--", alpha=0.3, zorder=1)
    
    # Colorbar styling
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.set_label("Number of Supporting Papers", fontsize=10, labelpad=10, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def make_topn_table(df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    supported = df[df["n_supporting"] > 0].copy()
    supported = supported.sort_values(
        ["lit_score", "n_supporting", "p_present"], ascending=[False, False, False]
    ).head(top_n)

    if supported.empty:
        print("[fig] no supported edges to show in top-N table; skipping")
        return

    cols = ["source_tf", "target_gene", "p_present", "lit_score",
            "n_supporting", "reg_type", "evidence_types"]
    table = supported[cols].copy()
    table["p_present"] = table["p_present"].map(lambda x: f"{x:.3f}")
    table["lit_score"] = table["lit_score"].map(lambda x: f"{x:.3f}")
    table["evidence_types"] = table["evidence_types"].fillna("").map(
        lambda s: (s[:55] + "...") if len(str(s)) > 55 else s
    )
    table.columns = ["TF", "Target", "Confidence", "Lit Score",
                     "Papers", "Reg Type", "Evidence"]

    # Figure height scales with number of rows
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(table) + 1.5))
    ax.axis("off")
    
    tbl = ax.table(
        cellText=table.values, 
        colLabels=list(table.columns),
        loc="center", 
        cellLoc="left",
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    
    # Apply professional styling
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            # Header
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2c3e50")
            cell.set_edgecolor("#2c3e50")
        else:
            # Data rows
            cell.set_edgecolor("#dfe6e9")
            if row % 2 == 0:
                cell.set_facecolor("#f8f9fa")
    
    ax.set_title(f"Top {len(table)} Novel Predictions with Literature Support",
                 pad=20, fontsize=14, fontweight="bold")
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--cell-type", default="mHSC", dest="cell_type")
    ap.add_argument("--min-p", type=float, default=0.95, dest="min_p")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max edges to lit-validate (top by p_present)")
    ap.add_argument("--skip-validate", action="store_true",
                    help="Reuse existing novel_edges_lit_scores.csv and only rebuild figures")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    filtered_csv = args.outdir / "novel_edges_filtered.csv"
    lit_csv = args.outdir / "novel_edges_lit_scores.csv"
    scatter_png = args.outdir / "novel_edges_scatter.png"
    topn_png = args.outdir / "novel_edges_topN.png"

    if not args.skip_validate:
        n = filter_novel_edges(args.input, filtered_csv, args.min_p)
        print(f"[filter] kept {n} novel edges -> {filtered_csv}")
        run_validation(filtered_csv, lit_csv, args.cell_type, args.limit)
    elif not lit_csv.exists():
        sys.exit(f"--skip-validate set but {lit_csv} does not exist")

    df = pd.read_csv(lit_csv)
    df["p_present"] = pd.to_numeric(df["p_present"], errors="coerce")
    df["lit_score"] = pd.to_numeric(df["lit_score"], errors="coerce")
    df["n_supporting"] = pd.to_numeric(df["n_supporting"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["p_present", "lit_score"])

    make_scatter(df, scatter_png)
    make_topn_table(df, topn_png)

    print(f"\nDone.\n  CSV:     {lit_csv}\n  Scatter: {scatter_png}\n  Top-N:   {topn_png}")


if __name__ == "__main__":
    main()
