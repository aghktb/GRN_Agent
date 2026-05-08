#!/usr/bin/env python3
"""Evaluate EAGER ``network.csv`` + ``evidence_graphs.jsonl`` (binary) vs optional gold labels."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.eval.network_eval import evaluate_network_with_manifest, evaluate_network_vs_labels, write_eval_report
from grn_agent.pipeline.config import load_yaml_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="Optional YAML config; CLI args override config values")
    ap.add_argument("--evidence-jsonl", default="")
    ap.add_argument("--network-csv", default="", help="Decoded network CSV with p_present column")
    ap.add_argument(
        "--scored-csv",
        default="",
        help="All scored inference candidates with p_present column; overrides --network-csv when set",
    )
    ap.add_argument("--gold-edges", default="", help="Optional gold labels CSV/TSV")
    ap.add_argument("--split-manifest", default="", help="Optional split manifest")
    ap.add_argument("--strategy", default="")
    ap.add_argument("--fold-id", default="")
    ap.add_argument("--subset", default="", choices=["", "train", "val", "test"])
    ap.add_argument("--k-values", default="10,50,100")
    ap.add_argument("--negative-ratio", type=float, default=None, help="Sample negatives at ratio x positives")
    ap.add_argument(
        "--negative-ratios",
        default="",
        help="Comma-separated ratios (e.g. 1,2,5) for multi-run evaluation",
    )
    ap.add_argument(
        "--negative-repeats",
        type=int,
        default=None,
        help="Repeat negative sampling this many times and report mean/std metrics",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-report", default="")
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=None, help="Enable W&B logging for test metrics")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable W&B logging for test metrics")
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--wandb-run-name", default="")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config) if args.config.strip() else {}

    def _cfg(key: str, default):
        if key in cfg:
            return cfg[key]
        alt = key.replace("_", "-")
        if alt in cfg:
            return cfg[alt]
        return default

    evidence_jsonl = str(args.evidence_jsonl or _cfg("evidence_jsonl", ""))
    network_csv = str(args.network_csv or _cfg("network_csv", ""))
    scored_csv = str(args.scored_csv or _cfg("scored_csv", _cfg("scores_csv", "")))
    prediction_csv = scored_csv.strip() or network_csv.strip()
    prediction_kind = "all_scored_edges" if scored_csv.strip() else "decoded_network"
    gold_edges = str(args.gold_edges or _cfg("gold_edges", ""))
    split_manifest = str(args.split_manifest or _cfg("split_manifest", ""))
    strategy = str(args.strategy or _cfg("strategy", ""))
    fold_id = str(args.fold_id or _cfg("fold_id", ""))
    subset = str(args.subset or _cfg("subset", "test"))
    k_values = str(args.k_values or _cfg("k_values", "10,50,100"))
    negative_ratio = args.negative_ratio if args.negative_ratio is not None else _cfg("negative_ratio", None)
    negative_ratios = str(args.negative_ratios or _cfg("negative_ratios", ""))
    negative_repeats = int(args.negative_repeats if args.negative_repeats is not None else _cfg("negative_repeats", 1))
    seed = int(args.seed if args.seed is not None else _cfg("seed", 42))
    out_report = str(args.out_report or _cfg("out_report", ""))
    wandb_enabled = bool(args.wandb if args.wandb is not None else _cfg("wandb", False))
    wandb_project = str(args.wandb_project or _cfg("wandb_project", "grn-agent-eager"))
    wandb_run_name = str(args.wandb_run_name or _cfg("wandb_run_name", ""))

    if not evidence_jsonl.strip() or not prediction_csv.strip():
        raise SystemExit("Missing required args: --evidence-jsonl and --network-csv/--scored-csv (or set in --config)")

    ks = [int(x.strip()) for x in k_values.split(",") if x.strip()]
    if split_manifest.strip():
        if not strategy.strip() or not fold_id.strip():
            raise SystemExit("--strategy and --fold-id are required with --split-manifest")
        ratios: list[float] = []
        if negative_ratios.strip():
            ratios = [float(x.strip()) for x in negative_ratios.split(",") if x.strip()]
        elif negative_ratio is not None:
            ratios = [float(negative_ratio)]

        if ratios:
            by_ratio: dict[str, dict] = {}
            for r in ratios:
                by_ratio[str(r)] = evaluate_network_with_manifest(
                    prediction_csv,
                    evidence_jsonl,
                    split_manifest,
                    strategy=strategy.strip(),
                    fold_id=fold_id.strip(),
                    subset=subset.strip(),
                    gold_edges=(gold_edges.strip() or None),
                    k_values=ks,
                    negative_ratio=r,
                    negative_repeats=negative_repeats,
                    seed=seed,
                )
            report = {
                "strategy": strategy.strip(),
                "fold_id": fold_id.strip(),
                "subset": subset.strip(),
                "negative_ratios": ratios,
                "negative_repeats": negative_repeats,
                "prediction_kind": prediction_kind,
                "prediction_csv": prediction_csv,
                "results_by_ratio": by_ratio,
            }
        else:
            report = evaluate_network_with_manifest(
                prediction_csv,
                evidence_jsonl,
                split_manifest,
                strategy=strategy.strip(),
                fold_id=fold_id.strip(),
                subset=subset.strip(),
                gold_edges=(gold_edges.strip() or None),
                k_values=ks,
            )
            report["prediction_kind"] = prediction_kind
            report["prediction_csv"] = prediction_csv
    else:
        ratios: list[float] = []
        if negative_ratios.strip():
            ratios = [float(x.strip()) for x in negative_ratios.split(",") if x.strip()]
        elif negative_ratio is not None:
            ratios = [float(negative_ratio)]

        if ratios:
            by_ratio = {}
            for r in ratios:
                by_ratio[str(r)] = evaluate_network_vs_labels(
                    prediction_csv,
                    evidence_jsonl,
                    gold_edges=(gold_edges.strip() or None),
                    k_values=ks,
                    negative_ratio=r,
                    negative_repeats=negative_repeats,
                    seed=seed,
                )
            report = {
                "negative_ratios": ratios,
                "negative_repeats": negative_repeats,
                "prediction_kind": prediction_kind,
                "prediction_csv": prediction_csv,
                "results_by_ratio": by_ratio,
            }
        else:
            report = evaluate_network_vs_labels(
                prediction_csv,
                evidence_jsonl,
                gold_edges=(gold_edges.strip() or None),
                k_values=ks,
            )
            report["prediction_kind"] = prediction_kind
            report["prediction_csv"] = prediction_csv
    print(json.dumps(report, indent=2))
    if out_report:
        write_eval_report(out_report, report)
    if wandb_enabled:
        try:
            import wandb  # type: ignore

            run = wandb.init(
                project=wandb_project,
                name=(wandb_run_name.strip() or None),
                config={"mode": "evaluation"},
            )
            flat = {}
            for k, v in report.items():
                if isinstance(v, (int, float)):
                    flat[f"test/{k}"] = v
            run.log(flat)
            run.finish()
        except Exception as exc:
            print(f"[eval_grn_agent] wandb logging skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
