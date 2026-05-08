#!/usr/bin/env python3
"""Plot delta heatmaps for TF-EAGER ablation summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = ("auprc", "auroc", "p_at_10")
ABLATIONS = ("functional_only", "single_stage")
AB_LABELS = {
    "functional_only": "Functional-only",
    "single_stage": "Single-stage",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def to_float(value: str) -> float:
    return float(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    regulatory_types = []
    values = {}
    for row in rows:
        reg_type = row["regulatory_type"]
        model = row["model"]
        if reg_type not in regulatory_types:
            regulatory_types.append(reg_type)
        values[(reg_type, model)] = {metric: to_float(row[metric]) for metric in METRICS}

    if "Overall" in regulatory_types:
        regulatory_types = [rt for rt in regulatory_types if rt != "Overall"] + ["Overall"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(10.5, 3.8), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r")

    for ax, metric in zip(axes, METRICS):
        delta = np.zeros((len(regulatory_types), len(ABLATIONS)), dtype=float)
        for i, reg_type in enumerate(regulatory_types):
            full = values[(reg_type, "full")][metric]
            for j, ablation in enumerate(ABLATIONS):
                delta[i, j] = values[(reg_type, ablation)][metric] - full
        vmax = np.max(np.abs(delta))
        vmax = max(vmax, 1e-6)
        im = ax.imshow(delta, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ABLATIONS)), [AB_LABELS[a] for a in ABLATIONS], rotation=20, ha="right")
        ax.set_yticks(range(len(regulatory_types)), regulatory_types)
        ax.set_title(f"Δ{metric.upper()} vs Full")
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                ax.text(j, i, f"{delta[i, j]:+.3f}", ha="center", va="center", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Delta", rotation=90)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"[plot-tf-eager-ablation-heatmap] wrote {args.out}")


if __name__ == "__main__":
    main()
