#!/usr/bin/env python3
"""Build and train tf-eager from multiple DataContext datasets."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.pipeline.config import load_yaml_config, save_yaml_config


def _run_cmd(cmd: list[str], stage: str) -> None:
    print(f"[multicontext-tf-eager] stage={stage} command={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _run_cmd_allow_empty_split(cmd: list[str], stage: str) -> bool:
    print(f"[multicontext-tf-eager] stage={stage} command={' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode == 0:
        return True
    msg = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "No gold edges remained after applying expression node splits" in msg:
        print(f"[multicontext-tf-eager] stage={stage} skip=no_expression_tf_gold_edges", flush=True)
        return False
    proc.check_returncode()
    return False


def _run_cmd_allow_empty_windows(cmd: list[str], stage: str) -> bool:
    print(f"[multicontext-tf-eager] stage={stage} command={' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode == 0:
        return True
    msg = (proc.stdout or "") + "\n" + (proc.stderr or "")
    empty_markers = (
        "No requested split TFs are present in the expression matrix",
        "No TFs found for requested split/fold/subset",
    )
    if any(marker in msg for marker in empty_markers):
        print(f"[multicontext-tf-eager] stage={stage} skip=no_expression_tfs_for_subset", flush=True)
        return False
    proc.check_returncode()
    return False


def _device_strings(value: Any, default: str = "") -> list[str]:
    if isinstance(value, list):
        out = [str(x).strip() for x in value if str(x).strip()]
    else:
        s = str(value or default).strip()
        if not s:
            out = []
        else:
            out = [x.strip() for x in s.split(",") if x.strip()]
    return out


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parallel_slots(section: dict[str, Any], *, default_device: str = "", default_workers: int = 1) -> list[str]:
    devices = _device_strings(section.get("devices") or section.get("parallel_devices"), default=default_device)
    if devices:
        return devices
    workers = max(1, _int_value(section.get("parallel_workers"), default_workers))
    return [f"worker-{i}" for i in range(workers)]


def _job_device(slot: str) -> str:
    return slot if slot.startswith("cuda") or slot.startswith("cpu") or slot.startswith("mps") else ""


def _run_parallel_jobs(
    jobs: list[dict[str, Any]],
    *,
    slots: list[str],
    skip_markers: tuple[str, ...] = (),
    skip_message: str = "",
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    slot_names = slots or ["worker-0"]
    running: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    pending = list(jobs)

    def _launch(slot: str, job: dict[str, Any]) -> dict[str, Any]:
        log_file = tempfile.NamedTemporaryFile(prefix="multicontext_tf_eager_", suffix=".log", delete=False)
        log_path = Path(log_file.name)
        log_file.close()
        print(
            f"[multicontext-tf-eager] stage={job['stage']} slot={slot} command={' '.join(job['cmd'])}",
            flush=True,
        )
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
            if ret == 0:
                finished.append({**item["job"], "slot": item["slot"], "ok": True, "skipped": False})
            elif skip_markers and any(marker in text for marker in skip_markers):
                if skip_message:
                    print(f"[multicontext-tf-eager] stage={item['job']['stage']} skip={skip_message}", flush=True)
                finished.append({**item["job"], "slot": item["slot"], "ok": False, "skipped": True})
            else:
                raise subprocess.CalledProcessError(ret, item["job"]["cmd"])
        running = still_running
    return finished


def _dict_to_cli_args(section: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in section.items():
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                out.append(flag)
            continue
        out.extend([flag, str(v)])
    return out


def _bool(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    return bool(v)


def _seed(cfg: dict[str, Any]) -> int:
    wf = cfg.get("workflow", {}) if isinstance(cfg.get("workflow", {}), dict) else {}
    return int(wf.get("seed", cfg.get("seed", 42)))


def _workflow_dir(cfg: dict[str, Any]) -> Path:
    wf = cfg.get("workflow", {}) if isinstance(cfg.get("workflow", {}), dict) else {}
    wf_id = str(wf.get("id", "multicontext_tf_eager")).strip() or "multicontext_tf_eager"
    artifact_root = Path(str(wf.get("artifact_root", "artifacts"))).expanduser()
    out = artifact_root / wf_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _reuse_workflow_dir(cfg: dict[str, Any]) -> Path | None:
    wf = cfg.get("workflow", {}) if isinstance(cfg.get("workflow", {}), dict) else {}
    reuse_from = str(wf.get("reuse_from", "")).strip()
    if not reuse_from:
        return None
    artifact_root = Path(str(wf.get("artifact_root", "artifacts"))).expanduser()
    out = artifact_root / reuse_from
    return out if out.exists() else None


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _tf_eager_train_cmd(py: str, cfg_path: Path, train_cfg: dict[str, Any]) -> list[str]:
    distributed_cfg = (
        train_cfg.get("distributed", {}) if isinstance(train_cfg.get("distributed", {}), dict) else {}
    )
    nproc_per_node = int(distributed_cfg.get("nproc_per_node", 1) or 1)
    if nproc_per_node <= 1:
        return [py, "scripts/train_tf_eager.py", "--config", str(cfg_path)]
    cmd = [py, "-m", "torch.distributed.run"]
    if _bool(distributed_cfg.get("standalone"), default=True):
        cmd.append("--standalone")
    if distributed_cfg.get("master_port") not in (None, ""):
        cmd.extend(["--master_port", str(distributed_cfg["master_port"])])
    cmd.extend(["--nproc_per_node", str(nproc_per_node), "scripts/train_tf_eager.py", "--config", str(cfg_path)])
    if distributed_cfg.get("backend") not in (None, ""):
        cmd.extend(["--ddp-backend", str(distributed_cfg["backend"])])
    return cmd


def _context_cfg(master: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    common = master.get("common", {}) if isinstance(master.get("common", {}), dict) else {}
    return _merge(common, ctx)


def _expand_context_groups(master: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = master.get("contexts", [])
    if contexts is None:
        contexts = []
    if not isinstance(contexts, list):
        raise SystemExit("contexts must be a list when set")
    out: list[dict[str, Any]] = [dict(c) for c in contexts if isinstance(c, dict)]

    groups = master.get("context_groups", [])
    if groups is None:
        groups = []
    if not isinstance(groups, list):
        raise SystemExit("context_groups must be a list when set")
    for group in groups:
        if not isinstance(group, dict):
            raise SystemExit("each context_groups entry must be a mapping")
        base_dir = Path(str(group.get("base_dir", "Data/DataContext")))
        variants = group.get("variants", [])
        cells = group.get("cells", [])
        if not isinstance(variants, list) or not isinstance(cells, list):
            raise SystemExit("context_groups entries require list-valued cells and variants")
        for cell in cells:
            if not isinstance(cell, dict):
                raise SystemExit("context_groups.cells entries must be mappings")
            prefix = str(cell["prefix"])
            species = str(cell.get("species", group.get("species", "")))
            cell_type = str(cell.get("cell_type", prefix))
            for variant in variants:
                variant = str(variant)
                dataset_id = f"{prefix}_{variant}"
                ctx_dir = base_dir / dataset_id
                if not ctx_dir.is_dir():
                    if _bool(group.get("require_all", True), default=True):
                        raise SystemExit(f"Missing DataContext directory: {ctx_dir}")
                    continue
                out.append(
                    {
                        "name": dataset_id,
                        "dataset_id": dataset_id,
                        "species": species,
                        "cell_type": cell_type,
                        "expression_path": str(ctx_dir / "ExpressionData.csv"),
                        "tf_file": str(ctx_dir / "TFs.csv"),
                        "gold_edges": str(ctx_dir / "refNetwork.csv"),
                        "fold_id": f"{dataset_id}_f1",
                    }
                )
    return out


def _combine_jsonl(inputs: list[Path], out: Path, seed: int) -> int:
    records: list[str] = []
    for p in inputs:
        with p.open(encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    records.append(line)
    rng = random.Random(seed)
    rng.shuffle(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        fp.writelines(records)
    return len(records)


def _split_subset_counts(path: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    with path.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            subset = str(row.get("subset", "")).strip()
            if subset in counts:
                counts[subset] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument(
        "--reuse-acquisition",
        action="store_true",
        help="With --force-recompute, reuse existing acquisition manifests instead of rerunning acquisition.",
    )
    ap.add_argument(
        "--train-only",
        action="store_true",
        help="Skip per-context acquisition/split/window building and train directly from existing combined train/val windows.",
    )
    ap.add_argument(
        "--start-from",
        default="",
        help="When used with --force-recompute, reuse earlier contexts and start recomputing at the first context whose name matches or starts with this value.",
    )
    args = ap.parse_args()

    master = load_yaml_config(args.config)
    wf_dir = _workflow_dir(master)
    reuse_wf_dir = _reuse_workflow_dir(master)
    seed = _seed(master)
    py = sys.executable

    contexts = _expand_context_groups(master)
    if not contexts:
        raise SystemExit("contexts/context_groups must expand to at least one context")

    split_common = master.get("split", {}) if isinstance(master.get("split", {}), dict) else {}
    acq_common = master.get("acquisition", {}) if isinstance(master.get("acquisition", {}), dict) else {}
    build_common = master.get("build_windows", {}) if isinstance(master.get("build_windows", {}), dict) else {}
    train_cfg = master.get("train_tf_eager", {}) if isinstance(master.get("train_tf_eager", {}), dict) else {}
    infer_cfg = master.get("infer_tf_eager", {}) if isinstance(master.get("infer_tf_eager", {}), dict) else {}
    eval_cfg = master.get("evaluation", {}) if isinstance(master.get("evaluation", {}), dict) else {}
    scoring_cfg = master.get("scoring", {}) if isinstance(master.get("scoring", {}), dict) else {}
    default_build_device = str(master.get("build_device", scoring_cfg.get("device", "")))
    build_slots = _parallel_slots(build_common, default_device=default_build_device, default_workers=1)
    infer_slots = _parallel_slots(infer_cfg, default_device=str(infer_cfg.get("device", default_build_device)), default_workers=1)
    eval_slots = _parallel_slots(eval_cfg, default_workers=1)
    train_enabled = _bool(train_cfg.get("enabled"), default=True)
    train_inputs: list[Path] = []
    val_inputs: list[Path] = []
    context_outputs: list[dict[str, Any]] = []
    build_jobs: list[dict[str, Any]] = []
    build_job_meta: list[dict[str, Any]] = []
    subset_window_records: list[dict[str, Any]] = []
    context_output_records: list[dict[str, Any]] = []
    start_from = str(args.start_from or "").strip()
    recompute_active = not bool(start_from)
    combined = wf_dir / "tf_eager" / "combined_train_windows.jsonl"
    combined_val = wf_dir / "tf_eager" / "combined_val_windows.jsonl"

    if not args.train_only:
        for i, raw_ctx in enumerate(contexts):
            if not isinstance(raw_ctx, dict):
                raise SystemExit("each context entry must be a mapping")
            ctx = _context_cfg(master, raw_ctx)
            name = str(ctx["name"])
            ctx_dir = wf_dir / "contexts" / name
            ctx_dir.mkdir(parents=True, exist_ok=True)
            expression_path = str(ctx["expression_path"])
            tf_file = str(ctx["tf_file"])
            gold_edges = str(ctx["gold_edges"])
            fold_id = str(ctx.get("fold_id", f"{name}_f1"))
            species = str(ctx.get("species", ""))
            cell_type = str(ctx.get("cell_type", name))
            dataset_id = str(ctx.get("dataset_id", name))
            acq_manifest: Path | None = None
            if start_from and not recompute_active:
                recompute_active = name == start_from or name.startswith(start_from)
            force_recompute_ctx = bool(args.force_recompute) and recompute_active

            if _bool(acq_common.get("enabled"), default=False):
                acq_manifest = ctx_dir / "acquisition" / "multimodal_manifest.json"
                reuse_acq_manifest = (
                    reuse_wf_dir / "contexts" / name / "acquisition" / "multimodal_manifest.json"
                    if reuse_wf_dir is not None
                    else None
                )
                reuse_acq = _bool(acq_common.get("reuse_if_exists"), default=True)
                force_acquisition = force_recompute_ctx and not args.reuse_acquisition
                if not force_acquisition and reuse_acq and not acq_manifest.is_file() and reuse_acq_manifest is not None and reuse_acq_manifest.is_file():
                    acq_manifest = reuse_acq_manifest
                    print(
                        f"[multicontext-tf-eager] stage=acquisition:{name} skip=reuse_external path={acq_manifest}",
                        flush=True,
                    )
                elif force_acquisition or not (reuse_acq and acq_manifest.is_file()):
                    acq_cfg = {k: v for k, v in acq_common.items() if k not in {"enabled", "reuse_if_exists"}}
                    acq_cfg.update(
                        {
                            "expr": expression_path,
                            "tf_file": tf_file,
                            "species": species,
                            "cell_type": cell_type,
                            "dataset_id": dataset_id,
                            "out_manifest": str(acq_manifest),
                        }
                    )
                    if "genome" not in acq_cfg:
                        acq_cfg["genome"] = "hg38" if species == "human" else "mm10"
                    _run_cmd(
                        [py, "scripts/acquire_multimodal_data.py", *_dict_to_cli_args(acq_cfg)],
                        f"acquisition:{name}",
                    )
                else:
                    print(f"[multicontext-tf-eager] stage=acquisition:{name} skip=reuse path={acq_manifest}", flush=True)

            split_path = ctx_dir / "split_manifest.csv"
            reuse_split_path = (
                reuse_wf_dir / "contexts" / name / "split_manifest.csv"
                if reuse_wf_dir is not None
                else None
            )
            if force_recompute_ctx or not split_path.is_file():
                if (
                    not force_recompute_ctx
                    and reuse_split_path is not None
                    and reuse_split_path.is_file()
                ):
                    split_path = reuse_split_path
                    print(f"[multicontext-tf-eager] stage=split:{name} skip=reuse_external path={split_path}", flush=True)
                else:
                    split_resolved = {
                        **split_common,
                        "gold_edges": gold_edges,
                        "expression_path": expression_path,
                        "tf_file": tf_file,
                        "out": str(split_path),
                        "fold_id": fold_id,
                    }
                    split_cfg_path = ctx_dir / "split.resolved.yml"
                    save_yaml_config(split_cfg_path, split_resolved)
                    if not _run_cmd_allow_empty_split(
                        [py, "scripts/make_tf_holdout_split_manifest.py", "--config", str(split_cfg_path)],
                        f"split:{name}",
                    ):
                        continue
            else:
                split_resolved = {
                    "path": str(split_path),
                }
                print(f"[multicontext-tf-eager] stage=split:{name} skip=reuse path={split_path}", flush=True)
            subset_counts = _split_subset_counts(split_path)

            subset_windows: dict[str, Path] = {}
            requested_subsets = ("train", "val", "test") if train_enabled else ("test",)
            for subset in requested_subsets:
                if subset_counts.get(subset, 0) <= 0:
                    print(
                        f"[multicontext-tf-eager] stage=build_{subset}_windows:{name} skip=empty_split",
                        flush=True,
                    )
                    continue
                windows_path = ctx_dir / f"{subset}_windows.jsonl"
                subset_windows[subset] = windows_path
                if not (force_recompute_ctx or not windows_path.is_file()):
                    print(
                        f"[multicontext-tf-eager] stage=build_{subset}_windows:{name} skip=reuse path={windows_path}",
                        flush=True,
                    )
                    continue
                window_cfg = {
                    "seed": seed + i,
                    "disable_priors": bool(master.get("disable_priors", False)),
                    "use_ortholog_lookup": bool(master.get("use_ortholog_lookup", False)),
                    "tf_workers": int(build_common.get("tf_workers", 1)),
                    "dataset": {
                        "mode": "beeline_csv",
                        "dataset_id": dataset_id,
                        "species": species,
                        "expression_path": expression_path,
                        "tf_file": tf_file,
                        "modalities": list(ctx.get("modalities", ["scrna"])),
                    },
                    "cell_context": {"cell_type": cell_type},
                    "candidates": build_common.get("candidates", build_common),
                    "scoring": master.get("scoring", {}),
                    "multimodal_manifest": str(acq_manifest) if acq_manifest is not None else "",
                    "tf_eager": {
                        "gold_edges": gold_edges,
                        "split_manifest": str(split_path),
                        "strategy": str(ctx.get("strategy", split_common.get("strategy", "leave_one_tf_out"))),
                        "fold_id": fold_id,
                        "subset": subset,
                        "windows_jsonl": str(windows_path),
                    },
                }
                slot = build_slots[len(build_jobs) % max(1, len(build_slots))]
                build_device = _job_device(slot) or str(master.get("build_device", ""))
                if build_device:
                    window_cfg["build_device"] = build_device
                build_cfg_path = ctx_dir / f"build_{subset}_windows.resolved.yml"
                save_yaml_config(build_cfg_path, window_cfg)
                build_jobs.append(
                    {
                        "stage": f"build_{subset}_windows:{name}",
                        "cmd": [py, "scripts/build_tf_eager_windows.py", "--config", str(build_cfg_path)],
                    }
                )
                build_job_meta.append({"subset_windows": subset_windows, "subset": subset, "path": windows_path})
            if "train" in subset_windows and subset_windows["train"].is_file():
                subset_window_records.append({"subset": "train", "path": subset_windows["train"]})
            if "val" in subset_windows and subset_windows["val"].is_file():
                subset_window_records.append({"subset": "val", "path": subset_windows["val"]})
            if "test" in subset_windows:
                context_output_records.append(
                    {
                        "name": name,
                        "gold_edges": gold_edges,
                        "fold_id": fold_id,
                        "split_manifest": str(split_path),
                        "test_windows": subset_windows["test"],
                        "dir": ctx_dir,
                    }
                )

        build_results = _run_parallel_jobs(
            build_jobs,
            slots=build_slots,
            skip_markers=(
                "No requested split TFs are present in the expression matrix",
                "No TFs found for requested split/fold/subset",
            ),
            skip_message="no_expression_tfs_for_subset",
        )
        for result, meta in zip(build_results, build_job_meta):
            if result["skipped"]:
                meta["subset_windows"].pop(meta["subset"], None)

        for meta in build_job_meta:
            if meta["path"].is_file():
                if meta["subset"] in {"train", "val"}:
                    subset_window_records.append({"subset": meta["subset"], "path": meta["path"]})
                elif meta["subset"] == "test":
                    pass

        seen_paths: set[tuple[str, str]] = set()
        train_inputs = []
        val_inputs = []
        for item in subset_window_records:
            path = Path(item["path"])
            key = (item["subset"], str(path))
            if not path.is_file() or key in seen_paths:
                continue
            seen_paths.add(key)
            if item["subset"] == "train":
                train_inputs.append(path)
            elif item["subset"] == "val":
                val_inputs.append(path)

        context_outputs = [ctx for ctx in context_output_records if Path(ctx["test_windows"]).is_file()]

        if train_enabled:
            n_combined = _combine_jsonl(train_inputs, combined, seed)
            n_val_combined = _combine_jsonl(val_inputs, combined_val, seed + 1)
            print(f"[multicontext-tf-eager] stage=combine_train_windows windows={n_combined} path={combined}", flush=True)
            print(f"[multicontext-tf-eager] stage=combine_val_windows windows={n_val_combined} path={combined_val}", flush=True)
        else:
            print("[multicontext-tf-eager] stage=combine_train_windows skip=train_disabled", flush=True)
            print("[multicontext-tf-eager] stage=combine_val_windows skip=train_disabled", flush=True)
    else:
        if not combined.is_file() or not combined_val.is_file():
            raise SystemExit(
                f"--train-only requires existing combined windows:\n"
                f"  train={combined}\n"
                f"  val={combined_val}"
            )
        print(f"[multicontext-tf-eager] stage=combine_train_windows skip=train_only path={combined}", flush=True)
        print(f"[multicontext-tf-eager] stage=combine_val_windows skip=train_only path={combined_val}", flush=True)

    checkpoint_name = str(train_cfg.get("checkpoint_name", "tf_eager_bootstrap_v2.pt")).strip() or "tf_eager_bootstrap_v2.pt"
    checkpoint = wf_dir / "tf_eager" / checkpoint_name
    if train_enabled:
        train_resolved = {
            "seed": seed,
            "tf_eager": {
                "windows_jsonl": str(combined),
                "val_windows_jsonl": str(combined_val),
                "checkpoint": str(checkpoint),
                "train": {k: v for k, v in train_cfg.items() if k != "enabled"},
            },
        }
        train_cfg_path = wf_dir / "tf_eager_train.resolved.yml"
        save_yaml_config(train_cfg_path, train_resolved)
        _run_cmd(_tf_eager_train_cmd(py, train_cfg_path, train_cfg), "train_tf_eager")

    if _bool(infer_cfg.get("enabled"), default=True):
        infer_jobs: list[dict[str, Any]] = []
        eval_jobs: list[dict[str, Any]] = []
        output_suffix = str(infer_cfg.get("output_suffix", "") or "").strip()
        scored_name = str(infer_cfg.get("scored_filename", f"test_scored_edges{output_suffix}.csv")).strip() or f"test_scored_edges{output_suffix}.csv"
        network_name = str(infer_cfg.get("network_filename", f"test_network{output_suffix}.csv")).strip() or f"test_network{output_suffix}.csv"
        evidence_name = str(infer_cfg.get("evidence_filename", f"test_flat_evidence{output_suffix}.jsonl")).strip() or f"test_flat_evidence{output_suffix}.jsonl"
        infer_cfg_name = str(infer_cfg.get("resolved_config_name", f"infer_test{output_suffix}.resolved.yml")).strip() or f"infer_test{output_suffix}.resolved.yml"
        report_name = str(eval_cfg.get("report_filename", f"eval_test_by_ratio{output_suffix}.json")).strip() or f"eval_test_by_ratio{output_suffix}.json"
        eval_cfg_name = str(eval_cfg.get("resolved_config_name", f"evaluation{output_suffix}.resolved.yml")).strip() or f"evaluation{output_suffix}.resolved.yml"
        for ctx in context_outputs:
            out_dir = Path(ctx["dir"]) / "evaluation"
            scored_csv = out_dir / scored_name
            network_csv = out_dir / network_name
            evidence_jsonl = out_dir / evidence_name
            slot = infer_slots[len(infer_jobs) % max(1, len(infer_slots))]
            infer_device = _job_device(slot) or str(infer_cfg.get("device", master.get("build_device", "")))
            infer_resolved = {
                "scoring": master.get("scoring", {}),
                "tf_eager": {
                    "windows_jsonl": str(ctx["test_windows"]),
                    "checkpoint": str(checkpoint),
                    "infer": {
                        "scored_csv": str(scored_csv),
                        "network_csv": str(network_csv),
                        "evidence_jsonl": str(evidence_jsonl),
                        "device": infer_device,
                        **{k: v for k, v in infer_cfg.items() if k != "enabled"},
                    },
                },
            }
            infer_cfg_path = Path(ctx["dir"]) / infer_cfg_name
            save_yaml_config(infer_cfg_path, infer_resolved)
            infer_jobs.append(
                {
                    "stage": f"infer_test:{ctx['name']}",
                    "cmd": [py, "scripts/infer_tf_eager.py", "--config", str(infer_cfg_path)],
                }
            )

            if _bool(eval_cfg.get("enabled"), default=True):
                eval_resolved = {
                    "scored_csv": str(scored_csv),
                    "evidence_jsonl": str(evidence_jsonl),
                    "gold_edges": ctx["gold_edges"],
                    "split_manifest": ctx["split_manifest"],
                    "strategy": str(split_common.get("strategy", "leave_one_tf_out")),
                    "fold_id": ctx["fold_id"],
                    "subset": "test",
                    "out_report": str(out_dir / report_name),
                    **{k: v for k, v in eval_cfg.items() if k != "enabled"},
                }
                eval_cfg_path = Path(ctx["dir"]) / eval_cfg_name
                save_yaml_config(eval_cfg_path, eval_resolved)
                eval_jobs.append(
                    {
                        "stage": f"evaluation:{ctx['name']}",
                        "cmd": [py, "scripts/eval_grn_agent.py", "--config", str(eval_cfg_path)],
                    }
                )
        _run_parallel_jobs(infer_jobs, slots=infer_slots)
        _run_parallel_jobs(eval_jobs, slots=eval_slots)


if __name__ == "__main__":
    main()
