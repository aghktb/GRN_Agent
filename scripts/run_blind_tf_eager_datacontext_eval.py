#!/usr/bin/env python3
"""Blind TF-EAGER inference/evaluation for DataContext directories."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.pipeline.config import load_yaml_config, save_yaml_config


DEFAULT_CELLS = {
    "mESC": {
        "species": "mouse",
        "cell_type": "embryonic stem cell ES-E14",
        "cell_line": "ES-E14",
        "cell_context": "embryonic stem cell ES-E14",
    },
    "mHSC-L": {
        "species": "mouse",
        "cell_type": "hematopoietic stem cell lymphoid",
        "cell_line": "HSC-L",
        "cell_context": "hematopoietic stem cell lymphoid HSC-L",
    },
}
DEFAULT_VARIANTS = [
    "nonspecific_chipseq_500",
    "nonspecific_chipseq_1000",
    "nonspecific_chipseq_tf500",
    "nonspecific_chipseq_tf1000",
    "specific_chipseq_500",
    "specific_chipseq_1000",
    "specific_chipseq_tf500",
    "specific_chipseq_tf1000",
    "string_500",
    "string_1000",
    "string_tf500",
    "string_tf1000",
]


def _run(cmd: list[str], stage: str) -> None:
    print(f"[blind-tf-eager] stage={stage} command={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _csv_list(value: Any, default: str = "") -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value or default).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parallel_slots(section: dict[str, Any], *, default_device: str = "", default_workers: int = 1) -> list[str]:
    devices = _csv_list(section.get("devices") or section.get("parallel_devices"), default=default_device)
    if devices:
        return devices
    try:
        workers = int(section.get("parallel_workers", default_workers))
    except (TypeError, ValueError):
        workers = default_workers
    return [f"worker-{i}" for i in range(max(1, workers))]


def _job_device(slot: str) -> str:
    return slot if slot.startswith("cuda") or slot.startswith("cpu") or slot.startswith("mps") else ""


def _run_parallel_jobs(jobs: list[dict[str, Any]], *, slots: list[str]) -> None:
    if not jobs:
        return
    slot_names = slots or ["worker-0"]
    running: list[dict[str, Any]] = []
    pending = list(jobs)

    def _launch(slot: str, job: dict[str, Any]) -> dict[str, Any]:
        log_file = tempfile.NamedTemporaryFile(prefix="blind_tf_eager_", suffix=".log", delete=False)
        log_path = Path(log_file.name)
        log_file.close()
        print(f"[blind-tf-eager] stage={job['stage']} slot={slot} command={' '.join(job['cmd'])}", flush=True)
        fp = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(job["cmd"], stdout=fp, stderr=subprocess.STDOUT, text=True)
        return {"slot": slot, "job": job, "proc": proc, "fp": fp, "log_path": log_path}

    while pending or running:
        used = {item["slot"] for item in running}
        for slot in slot_names:
            if not pending or slot in used:
                continue
            running.append(_launch(slot, pending.pop(0)))
        if not running:
            break
        time.sleep(0.5)
        still_running: list[dict[str, Any]] = []
        for item in running:
            ret = item["proc"].poll()
            if ret is None:
                still_running.append(item)
                continue
            item["fp"].close()
            text = item["log_path"].read_text(encoding="utf-8")
            if text:
                print(text, end="" if text.endswith("\n") else "\n", flush=True)
            if ret != 0:
                raise subprocess.CalledProcessError(ret, item["job"]["cmd"])
        running = still_running


def _flag_args(options: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in options.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if str(value).strip() == "":
            continue
        out.extend([flag, str(value)])
    return out


def _csv_arg(value: str, default: list[str]) -> list[str]:
    if not value.strip():
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name, {})
    return value if isinstance(value, dict) else {}


def _cfg_value(cfg: dict[str, Any], key: str, default: Any, *sections: str) -> Any:
    for section in sections:
        sec = _section(cfg, section)
        if key in sec:
            return sec[key]
    return cfg.get(key, default)


def _context_dirs(data_root: Path, cells: list[str], variants: list[str]) -> list[tuple[str, Path, dict[str, str]]]:
    out: list[tuple[str, Path, dict[str, str]]] = []
    for cell in cells:
        if cell not in DEFAULT_CELLS:
            raise SystemExit(f"Unknown cell {cell!r}; expected one of {sorted(DEFAULT_CELLS)}")
        for variant in variants:
            dataset_id = f"{cell}_{variant}"
            ctx_dir = data_root / dataset_id
            if ctx_dir.is_dir():
                out.append((dataset_id, ctx_dir, DEFAULT_CELLS[cell]))
            else:
                print(f"[blind-tf-eager] stage=discover:{dataset_id} skip=missing_dir path={ctx_dir}", flush=True)
    return out


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="", help="Optional YAML config for blind inference/evaluation")
    pre_args, _ = pre.parse_known_args()
    cfg = load_yaml_config(pre_args.config) if pre_args.config.strip() else {}

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--data-root", default=_cfg_value(cfg, "data_root", "Data/DataContext", "workflow"))
    ap.add_argument("--out-root", default=_cfg_value(cfg, "out_root", "artifacts/blind_tf_eager/mESC_mHSC-L", "workflow"))
    ap.add_argument(
        "--reuse-acquisition-root",
        default=_cfg_value(cfg, "reuse_acquisition_root", "", "workflow", "acquisition"),
        help="Optional existing blind workflow root whose per-context acquisition manifests should be reused",
    )
    ap.add_argument(
        "--checkpoint",
        default=_cfg_value(
            cfg,
            "checkpoint",
            "artifacts/multicontext_tf_eager/all_datacontext_contexts/tf_eager/tf_eager_bootstrap_v1.pt",
            "workflow",
        ),
    )
    ap.add_argument("--cells", default=_cfg_value(cfg, "cells", "mESC,mHSC-L", "workflow"))
    ap.add_argument("--variants", default=_cfg_value(cfg, "variants", "", "workflow"))
    ap.add_argument(
        "--skip-acquisition",
        action=argparse.BooleanOptionalAction,
        default=bool(_cfg_value(cfg, "skip_acquisition", False, "acquisition")),
    )
    ap.add_argument(
        "--force-acquisition",
        action=argparse.BooleanOptionalAction,
        default=bool(_cfg_value(cfg, "force_acquisition", False, "acquisition")),
    )
    ap.add_argument(
        "--skip-atac-search",
        action=argparse.BooleanOptionalAction,
        default=bool(_cfg_value(cfg, "skip_atac_search", False, "acquisition")),
    )
    ap.add_argument(
        "--skip-motif",
        action=argparse.BooleanOptionalAction,
        default=bool(_cfg_value(cfg, "skip_motif", False, "acquisition")),
    )
    ap.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=bool(_cfg_value(cfg, "strict", True, "acquisition")),
    )
    ap.add_argument(
        "--min-promoter-coverage",
        type=float,
        default=float(_cfg_value(cfg, "min_promoter_coverage", 0.6, "acquisition")),
    )
    ap.add_argument(
        "--coverage-denominator",
        choices=["rnaseq_expressed", "rnaseq_all"],
        default=str(_cfg_value(cfg, "coverage_denominator", "rnaseq_expressed", "acquisition")),
    )
    ap.add_argument(
        "--max-atac-candidates",
        type=int,
        default=int(_cfg_value(cfg, "max_atac_candidates", 5, "acquisition")),
    )
    ap.add_argument("--device", default=_cfg_value(cfg, "device", "cuda", "infer_tf_eager", "inference"))
    ap.add_argument(
        "--build-device",
        default=_cfg_value(
            cfg,
            "build_device",
            _cfg_value(cfg, "device", "cuda", "build_windows"),
            "build_windows",
            "scoring",
        ),
    )
    ap.add_argument("--threshold", type=float, default=float(_cfg_value(cfg, "threshold", 0.5, "infer_tf_eager", "inference")))
    ap.add_argument("--topk-per-tf", type=int, default=int(_cfg_value(cfg, "topk_per_tf", 100, "infer_tf_eager", "inference")))
    ap.add_argument(
        "--max-neighbors-per-tf",
        type=int,
        default=int(_cfg_value(cfg, "max_neighbors_per_tf", 200, "build_windows")),
        help="Training-style TF subgraph cap",
    )
    ap.add_argument("--subgraph-bootstraps", type=int, default=int(_cfg_value(cfg, "subgraph_bootstraps", 5, "build_windows")))
    ap.add_argument("--corr-threshold", type=float, default=float(_cfg_value(cfg, "corr_threshold", 0.25, "build_windows")))
    ap.add_argument(
        "--blind-exhaustive-all-pairs",
        action=argparse.BooleanOptionalAction,
        default=bool(_cfg_value(cfg, "blind_exhaustive_all_pairs", False, "build_windows")),
        help="In blind mode, exhaust each TF's candidate universe without replacement",
    )
    ap.add_argument(
        "--force-windows",
        action=argparse.BooleanOptionalAction,
        default=bool(_cfg_value(cfg, "force_windows", False, "build_windows")),
        help="Rebuild blind windows even when blind_windows.jsonl already exists",
    )
    ap.add_argument("--negative-ratios", default=_cfg_value(cfg, "negative_ratios", "1,2,5,10", "evaluation"))
    ap.add_argument("--negative-repeats", type=int, default=int(_cfg_value(cfg, "negative_repeats", 1, "evaluation")))
    ap.add_argument("--k-values", default=_cfg_value(cfg, "k_values", "10,50,100", "evaluation"))
    ap.add_argument("--seed", type=int, default=int(_cfg_value(cfg, "seed", 42, "workflow")))
    ap.add_argument("--force", action=argparse.BooleanOptionalAction, default=bool(_cfg_value(cfg, "force", False, "workflow")))
    args = ap.parse_args()

    py = sys.executable
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    reuse_acquisition_root = Path(args.reuse_acquisition_root).expanduser() if str(args.reuse_acquisition_root).strip() else None
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    cells = _csv_arg(args.cells, list(DEFAULT_CELLS))
    variants = _csv_arg(args.variants, DEFAULT_VARIANTS)
    build_slots = _parallel_slots(_section(cfg, "build_windows"), default_device=str(args.build_device), default_workers=1)
    infer_slots = _parallel_slots(_section(cfg, "infer_tf_eager"), default_device=str(args.device), default_workers=1)
    eval_slots = _parallel_slots(_section(cfg, "evaluation"), default_workers=1)
    contexts = _context_dirs(data_root, cells, variants)
    if not contexts:
        raise SystemExit("No matching DataContext directories found")

    build_jobs: list[dict[str, Any]] = []
    infer_jobs: list[dict[str, Any]] = []
    eval_jobs: list[dict[str, Any]] = []
    for i, (dataset_id, ctx_dir, meta) in enumerate(contexts):
        expr = ctx_dir / "ExpressionData.csv"
        tfs = ctx_dir / "TFs.csv"
        gold = ctx_dir / "refNetwork.csv"
        missing = [str(p) for p in (expr, tfs, gold) if not p.is_file()]
        if missing:
            print(f"[blind-tf-eager] stage=context:{dataset_id} skip=missing_files files={missing}", flush=True)
            continue

        run_dir = out_root / dataset_id
        acq_manifest = run_dir / "acquisition" / "multimodal_manifest.json"
        reuse_acq_manifest = (
            reuse_acquisition_root / dataset_id / "acquisition" / "multimodal_manifest.json"
            if reuse_acquisition_root is not None
            else None
        )
        windows = run_dir / "blind_windows.jsonl"
        scored = run_dir / "blind_scored_edges.csv"
        network = run_dir / "blind_network.csv"
        evidence = run_dir / "blind_flat_evidence.jsonl"
        report = run_dir / "blind_eval.json"

        if args.skip_acquisition:
            print(f"[blind-tf-eager] stage=acquisition:{dataset_id} skip=disabled", flush=True)
            multimodal_manifest = ""
        else:
            if (
                not args.force_acquisition
                and not acq_manifest.is_file()
                and reuse_acq_manifest is not None
                and reuse_acq_manifest.is_file()
            ):
                acq_manifest = reuse_acq_manifest
                print(
                    f"[blind-tf-eager] stage=acquisition:{dataset_id} skip=reuse_external path={acq_manifest}",
                    flush=True,
                )
            elif args.force_acquisition or not acq_manifest.is_file():
                acq_options = {
                    "expr": expr,
                    "species": meta["species"],
                    "tf_file": tfs,
                    "cell_type": meta["cell_type"],
                    "cell_line": meta.get("cell_line", ""),
                    "cell_context": meta.get("cell_context", ""),
                    "dataset_id": dataset_id,
                    "out_manifest": acq_manifest,
                    "genome": "hg38" if meta["species"] == "human" else "mm10",
                    "strict": bool(args.strict),
                    "min_promoter_coverage": float(args.min_promoter_coverage),
                    "coverage_denominator": args.coverage_denominator,
                    "max_atac_candidates": int(args.max_atac_candidates),
                    "skip_atac_search": bool(args.skip_atac_search),
                    "skip_motif": bool(args.skip_motif),
                }
                _run([py, "scripts/acquire_multimodal_data.py", *_flag_args(acq_options)], f"acquisition:{dataset_id}")
            else:
                print(f"[blind-tf-eager] stage=acquisition:{dataset_id} skip=reuse path={acq_manifest}", flush=True)
            multimodal_manifest = str(acq_manifest)

        build_cfg: dict[str, Any] = {
            "seed": int(args.seed) + i,
            "blind": True,
            "disable_priors": True,
            "use_ortholog_lookup": False,
            "tf_workers": int(_cfg_value(cfg, "tf_workers", 1, "build_windows")),
            "dataset": {
                "mode": "beeline_csv",
                "dataset_id": dataset_id,
                "species": meta["species"],
                "expression_path": str(expr),
                "tf_file": str(tfs),
                "modalities": ["scrna", "atac"] if multimodal_manifest else ["scrna"],
            },
            "cell_context": {"cell_type": meta["cell_type"]},
            "multimodal_manifest": multimodal_manifest,
            "candidates": {
                "mode": "tf_centered_window",
                "expression_transform": "arcsinh",
                "corr_threshold": float(args.corr_threshold),
                "train_window_neighbors": int(args.max_neighbors_per_tf),
                "train_subgraph_bootstraps": int(args.subgraph_bootstraps),
                "blind_ensure_coverage": True,
                "blind_exhaustive_all_pairs": bool(args.blind_exhaustive_all_pairs),
                "motif_score_threshold": 0.0,
                "accessibility_threshold": 0.0,
                "linkage_threshold": 0.0,
                "rescue_motif": not bool(args.skip_motif),
                "rescue_accessibility": not bool(args.skip_atac_search),
            },
            "scoring": {"device": args.build_device},
            "tf_eager": {"windows_jsonl": str(windows)},
        }
        build_slot = build_slots[len(build_jobs) % max(1, len(build_slots))]
        build_device = _job_device(build_slot) or str(args.build_device)
        if build_device:
            build_cfg["build_device"] = build_device
            build_cfg["scoring"] = {"device": build_device}
        build_cfg_path = run_dir / "build_blind.resolved.yml"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_yaml_config(build_cfg_path, build_cfg)
        if args.force_windows or not windows.is_file():
            build_jobs.append(
                {
                    "stage": f"build_blind:{dataset_id}",
                    "cmd": [py, "scripts/build_tf_eager_windows.py", "--config", str(build_cfg_path)],
                }
            )
        else:
            print(f"[blind-tf-eager] stage=build_blind:{dataset_id} skip=reuse path={windows}", flush=True)

        if args.force or not scored.is_file() or not evidence.is_file():
            infer_slot = infer_slots[len(infer_jobs) % max(1, len(infer_slots))]
            infer_device = _job_device(infer_slot) or str(args.device)
            infer_jobs.append(
                {
                    "stage": f"infer:{dataset_id}",
                    "cmd": [
                        py,
                        "scripts/infer_tf_eager.py",
                        "--windows-jsonl",
                        str(windows),
                        "--checkpoint",
                        str(checkpoint),
                        "--out-scored-csv",
                        str(scored),
                        "--out-network-csv",
                        str(network),
                        "--out-evidence-jsonl",
                        str(evidence),
                        "--threshold",
                        str(args.threshold),
                        "--topk-per-tf",
                        str(args.topk_per_tf),
                        "--device",
                        infer_device,
                    ],
                }
            )
        else:
            print(f"[blind-tf-eager] stage=infer:{dataset_id} skip=reuse path={scored}", flush=True)

        if args.force or not report.is_file():
            eval_jobs.append(
                {
                    "stage": f"evaluate:{dataset_id}",
                    "cmd": [
                        py,
                        "scripts/eval_grn_agent.py",
                        "--scored-csv",
                        str(scored),
                        "--evidence-jsonl",
                        str(evidence),
                        "--gold-edges",
                        str(gold),
                        "--k-values",
                        str(args.k_values),
                        "--negative-ratios",
                        str(args.negative_ratios),
                        "--negative-repeats",
                        str(args.negative_repeats),
                        "--out-report",
                        str(report),
                    ],
                }
            )
        else:
            print(f"[blind-tf-eager] stage=evaluate:{dataset_id} skip=reuse path={report}", flush=True)

    _run_parallel_jobs(build_jobs, slots=build_slots)
    _run_parallel_jobs(infer_jobs, slots=infer_slots)
    _run_parallel_jobs(eval_jobs, slots=eval_slots)


if __name__ == "__main__":
    main()
