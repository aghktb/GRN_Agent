#!/usr/bin/env python3
"""Run acquisition -> training -> inference -> evaluation from one master config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.pipeline.config import load_yaml_config, save_yaml_config
from grn_agent.pipeline.run import run_pipeline


def _run_cmd(cmd: list[str], stage: str) -> None:
    print(f"[integrated] stage={stage} command={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _bool(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    return bool(v)


def _workflow_paths(cfg: dict[str, Any]) -> tuple[str, Path]:
    wf = cfg.get("workflow", {})
    wf_id = str(wf.get("id", "integrated_eager_workflow")).strip() or "integrated_eager_workflow"
    artifact_root = Path(str(wf.get("artifact_root", "artifacts"))).expanduser()
    workflow_dir = artifact_root / wf_id
    workflow_dir.mkdir(parents=True, exist_ok=True)
    return wf_id, workflow_dir


def _seed(cfg: dict[str, Any]) -> int:
    wf = cfg.get("workflow", {})
    return int(wf.get("seed", 42))


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Master workflow YAML config")
    ap.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute acquisition/phase1 even when cached artifacts already exist",
    )
    args = ap.parse_args()

    master = load_yaml_config(args.config)
    wf_id, wf_dir = _workflow_paths(master)
    wf_cfg = master.get("workflow", {})
    single_artifact_dir = _bool(wf_cfg.get("single_artifact_dir"), default=True)
    inference_evaluation_only = _bool(wf_cfg.get("inference_evaluation_only"), default=False)
    seed = _seed(master)
    py = sys.executable

    acq_manifest: str | None = None
    phase1_out: Path | None = None
    split_manifest: str | None = None
    build_pairs_out_prefix: str | None = None
    build_pairs_val_out_prefix: str | None = None
    train_ckpt: str | None = None
    reranker_path: str | None = None
    infer_out: Path | None = None
    phase1_expression_path: str | None = None

    def _stage_enabled(section: dict[str, Any], default: bool = True) -> bool:
        if inference_evaluation_only:
            return False
        return _bool(section.get("enabled"), default=default)

    # 1) Acquisition
    acq = master.get("acquisition", {})
    if _stage_enabled(acq, default=True):
        if not isinstance(acq, dict):
            raise SystemExit("acquisition section must be a mapping")
        acq_cfg = dict(acq)
        acq_cfg.pop("enabled", None)
        reuse_acq = _bool(acq_cfg.pop("reuse_if_exists", True), default=True) and (not args.force_recompute)
        if single_artifact_dir:
            acq_cfg["out_manifest"] = str(wf_dir / "acquisition" / "multimodal_manifest.json")
        if "out_manifest" not in acq_cfg:
            raise SystemExit("acquisition.out_manifest is required when acquisition is enabled")
        acq_manifest = str(acq_cfg["out_manifest"])
        if reuse_acq and Path(acq_manifest).is_file():
            print(f"[integrated] stage=acquisition skip=reuse path={acq_manifest}", flush=True)
        else:
            _run_cmd([py, "scripts/acquire_multimodal_data.py", *_dict_to_cli_args(acq_cfg)], "acquisition")
    else:
        acq_candidate = wf_dir / "acquisition" / "multimodal_manifest.json"
        if single_artifact_dir and acq_candidate.is_file():
            acq_manifest = str(acq_candidate)
        elif isinstance(acq, dict) and acq.get("out_manifest") and Path(str(acq["out_manifest"])).is_file():
            acq_manifest = str(acq["out_manifest"])

    # 2) Pipeline phase-1 (evidence graphs)
    p1 = master.get("pipeline_phase1", {})
    if not isinstance(p1, dict):
        raise SystemExit("pipeline_phase1 section must be a mapping")
    if _stage_enabled(p1, default=True):
        base_cfg_path = p1.get("config")
        if not base_cfg_path:
            raise SystemExit("pipeline_phase1.config is required")
        p1_cfg = load_yaml_config(str(base_cfg_path))
        ds_cfg = p1_cfg.get("dataset", {}) if isinstance(p1_cfg.get("dataset", {}), dict) else {}
        p1_expr = str(ds_cfg.get("expression_path", "")).strip()
        if p1_expr:
            phase1_expression_path = p1_expr
        p1_run_id = str(p1.get("run_id", p1_cfg.get("run_id", f"{wf_id}_phase1")))
        reuse_p1 = _bool(p1.get("reuse_if_exists", True), default=True) and (not args.force_recompute)
        p1_cfg["run_id"] = p1_run_id
        p1_cfg["stop_after"] = str(p1.get("stop_after", "evidence_graphs"))
        if isinstance(p1.get("candidates"), dict):
            p1_cfg["candidates"] = {**dict(p1_cfg.get("candidates", {})), **dict(p1["candidates"])}
        if acq_manifest:
            p1_cfg["multimodal_manifest"] = acq_manifest
        if "seed" not in p1_cfg:
            p1_cfg["seed"] = seed
        if single_artifact_dir:
            p1_cfg["artifact_root"] = str(wf_dir)
        p1_cfg_path = wf_dir / "pipeline_phase1.resolved.yml"
        save_yaml_config(p1_cfg_path, p1_cfg)
        p1_artifact_root = Path(str(p1_cfg.get("artifact_root", "artifacts"))).expanduser()
        p1_out_candidate = p1_artifact_root / p1_run_id
        p1_cached_evidence = p1_out_candidate / "evidence_graphs.jsonl"
        if reuse_p1 and p1_cached_evidence.is_file():
            print(f"[integrated] stage=pipeline_phase1 skip=reuse path={p1_out_candidate}", flush=True)
            phase1_out = p1_out_candidate
        else:
            phase1_out = run_pipeline(p1_cfg_path)
    elif single_artifact_dir and str(p1.get("run_id", "")).strip():
        phase1_candidate = wf_dir / str(p1["run_id"])
        if (phase1_candidate / "evidence_graphs.jsonl").is_file():
            phase1_out = phase1_candidate
        base_cfg_path = p1.get("config")
        if base_cfg_path:
            p1_cfg = load_yaml_config(str(base_cfg_path))
            ds_cfg = p1_cfg.get("dataset", {}) if isinstance(p1_cfg.get("dataset", {}), dict) else {}
            p1_expr = str(ds_cfg.get("expression_path", "")).strip()
            if p1_expr:
                phase1_expression_path = p1_expr
    if phase1_out is None and not inference_evaluation_only:
        raise SystemExit("phase-1 output directory is required")

    # 3) Split manifest
    split = master.get("split", {})
    if not isinstance(split, dict):
        raise SystemExit("split section must be a mapping")
    if _stage_enabled(split, default=True):
        split_cfg = dict(split)
        split_cfg.pop("enabled", None)
        split_cfg.setdefault("seed", seed)
        if phase1_expression_path and "expression_path" not in split_cfg and "expr" not in split_cfg:
            split_cfg["expression_path"] = phase1_expression_path
        dataset_cfg = master.get("dataset", {}) if isinstance(master.get("dataset", {}), dict) else {}
        if dataset_cfg.get("tf_file") and "tf_file" not in split_cfg:
            split_cfg["tf_file"] = dataset_cfg["tf_file"]
        if single_artifact_dir:
            split_cfg["out"] = str(wf_dir / "splits" / "split_manifest.csv")
        if "out" not in split_cfg:
            raise SystemExit("split.out is required")
        split_cfg_path = wf_dir / "split.resolved.yml"
        save_yaml_config(split_cfg_path, split_cfg)
        _run_cmd([py, "scripts/make_tf_holdout_split_manifest.py", "--config", str(split_cfg_path)], "split")
        split_manifest = str(split_cfg["out"])
    else:
        split_candidate = wf_dir / "splits" / "split_manifest.csv"
        if single_artifact_dir and split_candidate.is_file():
            split_manifest = str(split_candidate)
        elif split.get("out") and Path(str(split["out"])).is_file():
            split_manifest = str(split["out"])

    # 4) Build training pairs
    btp = master.get("build_pairs", {})
    if not isinstance(btp, dict):
        raise SystemExit("build_pairs section must be a mapping")
    if _stage_enabled(btp, default=True):
        if phase1_out is None:
            raise SystemExit("build_pairs requires phase-1 output directory")
        btp_cfg = dict(btp)
        btp_cfg.pop("enabled", None)
        build_val_pairs = _bool(btp_cfg.pop("build_val_pairs", True), default=True)
        btp_cfg.setdefault("seed", seed)
        btp_cfg.setdefault("evidence_jsonl", str(phase1_out / "evidence_graphs.jsonl"))
        if split_manifest and "split_manifest" not in btp_cfg:
            btp_cfg["split_manifest"] = split_manifest
        if single_artifact_dir:
            btp_cfg["out_prefix"] = str(wf_dir / "training" / "train")
        if "out_prefix" not in btp_cfg:
            raise SystemExit("build_pairs.out_prefix is required")
        build_pairs_out_prefix = str(btp_cfg["out_prefix"])
        btp_cfg_path = wf_dir / "build_pairs.resolved.yml"
        save_yaml_config(btp_cfg_path, btp_cfg)
        _run_cmd([py, "scripts/build_training_pairs.py", "--config", str(btp_cfg_path)], "build_pairs")
        if (
            build_val_pairs
            and split_manifest
            and btp_cfg.get("gold_edges")
            and btp_cfg.get("strategy")
            and btp_cfg.get("fold_id")
        ):
            val_cfg = dict(btp_cfg)
            val_cfg["subset"] = "val"
            val_cfg["out_prefix"] = str(btp.get("val_out_prefix", "")) or str(wf_dir / "training" / "val")
            build_pairs_val_out_prefix = str(val_cfg["out_prefix"])
            val_cfg_path = wf_dir / "build_pairs_val.resolved.yml"
            save_yaml_config(val_cfg_path, val_cfg)
            _run_cmd([py, "scripts/build_training_pairs.py", "--config", str(val_cfg_path)], "build_pairs_val")

    # 5) Train lightweight candidate reranker
    rr = master.get("train_reranker", {})
    if rr and not isinstance(rr, dict):
        raise SystemExit("train_reranker section must be a mapping")
    if isinstance(rr, dict) and _stage_enabled(rr, default=False):
        rr_cfg = dict(rr)
        rr_cfg.pop("enabled", None)
        rr_cfg.setdefault("seed", seed)
        if "graphs_jsonl" not in rr_cfg and build_pairs_out_prefix:
            rr_cfg["graphs_jsonl"] = f"{build_pairs_out_prefix}_graphs.jsonl"
        if "y_npz" not in rr_cfg and build_pairs_out_prefix:
            rr_cfg["y_npz"] = f"{build_pairs_out_prefix}_y.npz"
        if single_artifact_dir:
            rr_cfg["out"] = str(wf_dir / "training" / "candidate_reranker.pkl")
        if "out" not in rr_cfg:
            raise SystemExit("train_reranker.out is required")
        rr_cfg_path = wf_dir / "train_reranker.resolved.yml"
        save_yaml_config(rr_cfg_path, rr_cfg)
        _run_cmd([py, "scripts/train_candidate_reranker.py", "--config", str(rr_cfg_path)], "train_reranker")
        reranker_path = str(rr_cfg["out"])

    # 6) Train EAGER
    tr = master.get("train_eager", {})
    if not isinstance(tr, dict):
        raise SystemExit("train_eager section must be a mapping")
    if _stage_enabled(tr, default=True):
        tr_cfg = dict(tr)
        tr_cfg.pop("enabled", None)
        tr_cfg.setdefault("seed", seed)
        if "graphs_jsonl" not in tr_cfg and build_pairs_out_prefix:
            tr_cfg["graphs_jsonl"] = f"{build_pairs_out_prefix}_graphs.jsonl"
        if "y_npz" not in tr_cfg and build_pairs_out_prefix:
            tr_cfg["y_npz"] = f"{build_pairs_out_prefix}_y.npz"
        if "val_graphs_jsonl" not in tr_cfg and build_pairs_val_out_prefix:
            tr_cfg["val_graphs_jsonl"] = f"{build_pairs_val_out_prefix}_graphs.jsonl"
        if "val_y_npz" not in tr_cfg and build_pairs_val_out_prefix:
            tr_cfg["val_y_npz"] = f"{build_pairs_val_out_prefix}_y.npz"
        if single_artifact_dir:
            tr_cfg["out"] = str(wf_dir / "training" / "eager.pt")
        if "out" not in tr_cfg:
            raise SystemExit("train_eager.out is required")
        tr_cfg_path = wf_dir / "train_eager.resolved.yml"
        save_yaml_config(tr_cfg_path, tr_cfg)
        _run_cmd([py, "scripts/train_eager.py", "--config", str(tr_cfg_path)], "train_eager")
        train_ckpt = str(tr_cfg["out"])
    else:
        ckpt_candidate = wf_dir / "training" / "eager.pt"
        if single_artifact_dir and ckpt_candidate.is_file():
            train_ckpt = str(ckpt_candidate)
        elif tr.get("out") and Path(str(tr["out"])).is_file():
            train_ckpt = str(tr["out"])

    # 7) Pipeline phase-2 / inference
    inf = master.get("inference_phase2", {})
    if not isinstance(inf, dict):
        raise SystemExit("inference_phase2 section must be a mapping")
    if _bool(inf.get("enabled"), default=True) or inference_evaluation_only:
        base_cfg_path = inf.get("config")
        if not base_cfg_path:
            raise SystemExit("inference_phase2.config is required")
        inf_cfg = load_yaml_config(str(base_cfg_path))
        inf_run_id = str(inf.get("run_id", inf_cfg.get("run_id", f"{wf_id}_infer")))
        inf_cfg["run_id"] = inf_run_id
        if isinstance(p1.get("candidates"), dict):
            inf_cfg["candidates"] = {**dict(inf_cfg.get("candidates", {})), **dict(p1["candidates"])}
        if isinstance(inf.get("candidates"), dict):
            inf_cfg["candidates"] = {**dict(inf_cfg.get("candidates", {})), **dict(inf["candidates"])}
        if split_manifest:
            ev_for_filter = master.get("evaluation", {}) if isinstance(master.get("evaluation", {}), dict) else {}
            split_filter = dict(inf.get("inference_filter") or inf.get("split_filter") or {})
            split_filter.setdefault("split_manifest", split_manifest)
            split_filter.setdefault("strategy", ev_for_filter.get("strategy", "leave_one_tf_out"))
            split_filter.setdefault("fold_id", ev_for_filter.get("fold_id", master.get("split", {}).get("fold_id", "")))
            split_filter.setdefault("subset", ev_for_filter.get("subset", "test"))
            split_filter.setdefault("target_universe", "all_genes")
            inf_cfg["inference_filter"] = split_filter
        if reranker_path:
            cand_cfg = dict(inf_cfg.get("candidates", {}))
            cand_cfg["reranker_model_path"] = reranker_path
            inf_cfg["candidates"] = cand_cfg
        if acq_manifest:
            inf_cfg["multimodal_manifest"] = acq_manifest
        if train_ckpt or inf.get("checkpoint"):
            sc = dict(inf_cfg.get("scoring", {}))
            sc["checkpoint"] = str(inf.get("checkpoint", train_ckpt))
            if "device" in inf:
                sc["device"] = str(inf["device"])
            inf_cfg["scoring"] = sc
        if "seed" not in inf_cfg:
            inf_cfg["seed"] = seed
        if single_artifact_dir:
            inf_cfg["artifact_root"] = str(wf_dir)
        inf_cfg_path = wf_dir / "pipeline_phase2.resolved.yml"
        save_yaml_config(inf_cfg_path, inf_cfg)
        infer_out = run_pipeline(inf_cfg_path)

    # 8) Evaluation
    ev = master.get("evaluation", {})
    if not isinstance(ev, dict):
        raise SystemExit("evaluation section must be a mapping")
    if _bool(ev.get("enabled"), default=True) or inference_evaluation_only:
        ev_cfg = dict(ev)
        ev_cfg.pop("enabled", None)
        ev_cfg.setdefault("seed", seed)
        if infer_out is not None:
            ev_cfg.setdefault("evidence_jsonl", str(infer_out / "evidence_graphs.jsonl"))
            if _bool(ev_cfg.get("evaluate_all_scores"), default=False):
                ev_cfg.setdefault("scored_csv", str(infer_out / "exports" / "scored_edges.csv"))
            ev_cfg.setdefault("network_csv", str(infer_out / "exports" / "network.csv"))
        if split_manifest and "split_manifest" not in ev_cfg:
            ev_cfg["split_manifest"] = split_manifest
        if single_artifact_dir:
            ev_cfg["out_report"] = str(wf_dir / "evaluation" / "eval_test_by_ratio.json")
        if "evidence_jsonl" not in ev_cfg or ("network_csv" not in ev_cfg and "scored_csv" not in ev_cfg):
            raise SystemExit("evaluation requires evidence_jsonl and network_csv or scored_csv")
        ev_cfg_path = wf_dir / "evaluation.resolved.yml"
        save_yaml_config(ev_cfg_path, ev_cfg)
        _run_cmd([py, "scripts/eval_grn_agent.py", "--config", str(ev_cfg_path)], "evaluation")

    print(f"[integrated] completed workflow_id={wf_id} workspace={wf_dir}", flush=True)


if __name__ == "__main__":
    main()
