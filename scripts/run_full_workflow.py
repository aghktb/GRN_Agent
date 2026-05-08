#!/usr/bin/env python3
"""EAGER-only full workflow helper.

Runs pipeline and optional evaluation using binary p_present outputs.
Legacy MLP/LoRA workflow was moved to legacy/scripts/run_full_workflow.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Pipeline YAML config (must include scoring.checkpoint)")
    ap.add_argument("--gold-edges", default="", help="Optional gold edge CSV/TSV for eval")
    ap.add_argument("--out-report", default="", help="Optional JSON report path")
    args = ap.parse_args()

    _run([sys.executable, "-m", "grn_agent.pipeline.run", "--config", args.config])

    if args.gold_edges:
        cfg = Path(args.config)
        import yaml  # type: ignore

        c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        run_id = c.get("run_id")
        root = Path(c.get("artifact_root", "artifacts")) / str(run_id)
        scores = root / "exports/scored_edges.csv"
        eg = root / "evidence_graphs.jsonl"
        cmd = [
            sys.executable,
            "scripts/eval_grn_agent.py",
            "--scored-csv", str(scores),
            "--evidence-jsonl", str(eg),
            "--gold-edges", args.gold_edges,
        ]
        if args.out_report:
            cmd += ["--out-report", args.out_report]
        _run(cmd)


if __name__ == "__main__":
    main()
