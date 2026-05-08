#!/usr/bin/env python3
"""Train tf-eager from TF-centered windows."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import distributed as dist
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset as TorchSubset

from grn_agent.models.tf_eager import TfEagerConfig, TfEagerWindowModel, stack_window_batches, window_record_to_batch
from grn_agent.models.tf_eager.window_batch import TfEagerWindowBatch
from grn_agent.pipeline.config import load_yaml_config


_MONITOR_MODES = {
    "val_auprc": "max",
    "val_loss": "min",
    "val_brier": "min",
}
_ATTENTION_BACKENDS = {"auto", "math", "memory_efficient", "flash"}


def _seed_worker(worker_id: int) -> None:
    _ = worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _load_windows(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                out.append(json.loads(line))
    return out


def _to_device(batch: TfEagerWindowBatch, device: str) -> TfEagerWindowBatch:
    return TfEagerWindowBatch(
        token_kind=batch.token_kind.to(device),
        x_value=batch.x_value.to(device),
        conf=batch.conf.to(device),
        token_target_pos=batch.token_target_pos.to(device),
        token_mask=batch.token_mask.to(device),
        modality=batch.modality.to(device),
        mech_mask=batch.mech_mask.to(device),
        func_mask=batch.func_mask.to(device),
        context_idx=batch.context_idx.to(device),
        tf_idx=batch.tf_idx.to(device),
        gene_idx=batch.gene_idx.to(device),
        gene_pos=batch.gene_pos.to(device),
        gene_mask=batch.gene_mask.to(device),
        labels=batch.labels.to(device),
        sample_weight=batch.sample_weight.to(device),
    )


def _model_config_from_sections(*sections: dict) -> TfEagerConfig:
    allowed = {f.name for f in dataclasses.fields(TfEagerConfig)}
    raw: dict[str, object] = {}
    for section in sections:
        if not isinstance(section, dict):
            continue
        model_section = section.get("model", {})
        if isinstance(model_section, dict):
            raw.update({k: v for k, v in model_section.items() if k in allowed})
        raw.update({k: v for k, v in section.items() if k in allowed})
    return TfEagerConfig(**raw)


def _needs_find_unused_parameters(model_cfg: TfEagerConfig) -> bool:
    if str(model_cfg.decoder_mode or "staged").strip().lower() != "staged":
        return True
    return False


class _CachedWindowDataset(Dataset):
    def __init__(self, records: list[dict], model_cfg: TfEagerConfig) -> None:
        self._batches = [
            window_record_to_batch(
                r,
                token_layout=model_cfg.token_layout,
                drop_token_kinds=model_cfg.drop_token_kinds,
                tf_vocab=model_cfg.tf_vocab,
                gene_vocab=model_cfg.gene_vocab,
                context_vocab=model_cfg.context_vocab,
            )
            for r in records
        ]

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> TfEagerWindowBatch:
        return self._batches[idx]


def _collate_window_batches(items: list[TfEagerWindowBatch]) -> TfEagerWindowBatch:
    return stack_window_batches(items)


def _metrics(scores: list[np.ndarray], labels: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    if not scores:
        return {"auprc": 0.0, "auroc": 0.0, "brier": 0.0}
    s = np.concatenate(scores)
    y = np.concatenate(labels).astype(np.int64)
    m = np.concatenate(masks).astype(bool)
    s = s[m]
    y = y[m]
    finite = np.isfinite(s) & np.isfinite(y)
    s = s[finite]
    y = y[finite]
    if len(y) == 0:
        return {"auprc": 0.0, "auroc": 0.0, "brier": 0.0}
    out = {"auprc": float(average_precision_score(y, s)) if len(set(y.tolist())) > 1 else float(y.mean())}
    try:
        out["auroc"] = float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else 0.0
    except ValueError:
        out["auroc"] = 0.0
    out["brier"] = float(np.mean((s - y.astype(np.float64)) ** 2))
    return out


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _is_main_process() -> bool:
    return _rank() == 0


def _print_main(msg: str) -> None:
    if _is_main_process():
        print(msg, flush=True)


def _setup_distributed(device: str, backend: str) -> tuple[str, int]:
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    resolved_device = device
    if world_size <= 1:
        return resolved_device, 1
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("WORLD_SIZE > 1 requires CUDA for TF-EAGER multi-GPU training")
        torch.cuda.set_device(local_rank)
        resolved_device = f"cuda:{local_rank}"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return resolved_device, world_size


def _teardown_distributed() -> None:
    if _is_distributed():
        dist.destroy_process_group()


def _all_reduce_pair(total: float, count: int, device: str) -> tuple[float, int]:
    if not _is_distributed():
        return total, count
    buf = torch.tensor([total, float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return float(buf[0].item()), int(buf[1].item())


def _gather_vector(chunks: list[np.ndarray]) -> np.ndarray:
    local = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float32)
    if not _is_distributed():
        return local
    gathered: list[np.ndarray | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    merged = [np.asarray(x).reshape(-1) for x in gathered if x is not None and np.asarray(x).size > 0]
    if not merged:
        return np.empty((0,), dtype=local.dtype if local.size > 0 else np.float32)
    return np.concatenate(merged)


def _model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    return {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}


def _load_model_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    base_model.load_state_dict(state)


def _monitor_value(metric: str, *, val_loss: float, val_auprc: float, val_brier: float) -> float:
    if metric == "val_loss":
        return float(val_loss)
    if metric == "val_brier":
        return float(val_brier)
    return float(val_auprc)


def _configure_attention_backend(backend: str, *, amp_enabled: bool) -> None:
    choice = str(backend or "auto").strip().lower()
    if choice not in _ATTENTION_BACKENDS:
        raise SystemExit(f"Unsupported attention_backend={backend!r}; use one of {sorted(_ATTENTION_BACKENDS)}")
    if not torch.cuda.is_available():
        return
    if not hasattr(torch.backends, "cuda"):
        return
    cuda_backends = torch.backends.cuda
    if not hasattr(cuda_backends, "enable_flash_sdp"):
        return
    if choice == "auto":
        return
    cuda_backends.enable_flash_sdp(choice == "flash")
    cuda_backends.enable_mem_efficient_sdp(choice == "memory_efficient")
    cuda_backends.enable_math_sdp(choice == "math")
    if amp_enabled:
        mode = choice
        print(f"[train_tf_eager] attention_backend={mode}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--windows-jsonl", default="")
    ap.add_argument("--val-windows-jsonl", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--early-stopping-patience", type=int, default=None)
    ap.add_argument("--early-stopping-min-delta", type=float, default=None)
    ap.add_argument("--early-stopping-metric", default="")
    ap.add_argument("--checkpoint-metric", default="")
    ap.add_argument("--auprc-tie-margin", type=float, default=None)
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=None)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--deterministic-warn-only", action="store_true", default=None)
    ap.add_argument("--attention-backend", default="")
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--wandb-run-name", default="")
    ap.add_argument("--ddp-backend", default="")
    ap.add_argument("--find-unused-parameters", dest="find_unused_parameters", action="store_true", default=None)
    ap.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config) if args.config.strip() else {}
    tf_cfg = cfg.get("tf_eager", {}) if isinstance(cfg.get("tf_eager", {}), dict) else {}
    tf_train_cfg = tf_cfg.get("train", {}) if isinstance(tf_cfg.get("train", {}), dict) else {}
    base_train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    scoring_cfg = cfg.get("scoring", {}) if isinstance(cfg.get("scoring", {}), dict) else {}
    distributed_cfg = tf_train_cfg.get("distributed", {}) if isinstance(tf_train_cfg.get("distributed", {}), dict) else {}

    def _cfg(k: str, default):
        cli_value = getattr(args, k)
        if cli_value not in (None, ""):
            return cli_value
        aliases = (k, k.replace("_", "-"))
        for section in (tf_train_cfg, tf_cfg, cfg, base_train_cfg, scoring_cfg):
            for key in aliases:
                if key in section:
                    return section[key]
        if k == "windows_jsonl":
            for key in ("windows_jsonl", "windows-jsonl"):
                if key in tf_cfg:
                    return tf_cfg[key]
        if k == "out":
            for key in ("checkpoint", "out"):
                if key in tf_cfg:
                    return tf_cfg[key]
        return default

    windows_jsonl = str(_cfg("windows_jsonl", ""))
    val_windows_jsonl = str(_cfg("val_windows_jsonl", ""))
    out_path = str(_cfg("out", ""))
    if not windows_jsonl or not out_path:
        raise SystemExit("Missing --windows-jsonl and --out")
    epochs = int(_cfg("epochs", 5))
    lr = float(_cfg("lr", 3e-4))
    weight_decay = float(_cfg("weight_decay", 1e-2))
    val_frac = float(_cfg("val_frac", 0.1))
    seed = int(_cfg("seed", 0))
    device = str(_cfg("device", "")) or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(_cfg("batch_size", 1))
    num_workers = int(_cfg("num_workers", 0))
    early_stopping_patience = int(_cfg("early_stopping_patience", 0) or 0)
    early_stopping_min_delta = float(_cfg("early_stopping_min_delta", 0.0) or 0.0)
    auprc_tie_margin = float(_cfg("auprc_tie_margin", 0.005) or 0.0)
    early_stopping_metric = str(_cfg("early_stopping_metric", "val_auprc") or "val_auprc").strip().lower()
    checkpoint_metric = str(_cfg("checkpoint_metric", early_stopping_metric) or early_stopping_metric).strip().lower()
    deterministic = bool(_cfg("deterministic", True))
    deterministic_warn_only = bool(_cfg("deterministic_warn_only", True))
    attention_backend = str(_cfg("attention_backend", "math" if deterministic else "auto") or ("math" if deterministic else "auto")).strip().lower()
    if early_stopping_metric not in _MONITOR_MODES:
        raise SystemExit(
            f"Unsupported early_stopping_metric={early_stopping_metric!r}; "
            f"use one of {sorted(_MONITOR_MODES)}"
        )
    if checkpoint_metric not in _MONITOR_MODES:
        raise SystemExit(
            f"Unsupported checkpoint_metric={checkpoint_metric!r}; "
            f"use one of {sorted(_MONITOR_MODES)}"
        )
    amp_enabled = str(device).startswith("cuda")
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    wandb_enabled = bool(_cfg("wandb", False))
    wandb_project = str(_cfg("wandb_project", "grn-agent-tf-eager"))
    wandb_run_name = str(_cfg("wandb_run_name", ""))
    ddp_backend = str(args.ddp_backend or distributed_cfg.get("backend") or ("nccl" if amp_enabled else "gloo")).strip()
    model_cfg = _model_config_from_sections(tf_cfg, tf_train_cfg, cfg)
    find_unused_parameters = _cfg("find_unused_parameters", None)
    if find_unused_parameters is None:
        find_unused_parameters = _needs_find_unused_parameters(model_cfg)
    find_unused_parameters = bool(find_unused_parameters)

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
    elif amp_enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device, world_size = _setup_distributed(device, ddp_backend)
    amp_enabled = str(device).startswith("cuda")
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    _configure_attention_backend(attention_backend, amp_enabled=amp_enabled)
    windows = _load_windows(windows_jsonl)
    val_windows = _load_windows(val_windows_jsonl) if val_windows_jsonl else []
    if not windows:
        raise SystemExit("No windows loaded")
    all_labels = []
    all_weights = []
    for rec in windows:
        for g in rec.get("genes", []):
            w = float(g.get("sample_weight", 1.0))
            if w <= 0.0:
                continue
            all_labels.append(int(g.get("label", 0)))
            all_weights.append(w)
    n_pos = float(sum(all_labels))
    n_neg = float(len(all_labels) - n_pos)
    pos_weight = (n_neg / n_pos) if n_pos > 0 and n_neg > 0 else 1.0

    idx = list(range(len(windows)))
    rng.shuffle(idx)
    if val_windows:
        train_idx = idx
        val_records = val_windows
        val_idx = list(range(len(val_records)))
    else:
        n_val = int(max(1, min(len(idx) - 1, round(val_frac * len(idx))))) if len(idx) > 1 else 0
        val_set = set(idx[:n_val])
        train_idx = [i for i in idx if i not in val_set]
        val_idx = [i for i in idx if i in val_set]
        val_records = windows

    train_dataset_full = _CachedWindowDataset(windows, model_cfg)
    val_dataset_full = _CachedWindowDataset(val_records, model_cfg)
    train_dataset = TorchSubset(train_dataset_full, train_idx)
    if world_size > 1:
        val_idx_rank = val_idx[_rank()::world_size]
        val_dataset = TorchSubset(val_dataset_full, val_idx_rank)
    else:
        val_dataset = TorchSubset(val_dataset_full, val_idx)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=_rank(), shuffle=True) if world_size > 1 else None
    )
    val_sampler = None
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + _rank())

    model = TfEagerWindowModel(model_cfg).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()] if amp_enabled else None,
            find_unused_parameters=find_unused_parameters,
        )
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    def _forward_loss(batch: TfEagerWindowBatch, *, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(batch)
            loss_mat = crit(logits, batch.labels) * batch.sample_weight * batch.gene_mask
            loss = loss_mat.sum() / torch.clamp((batch.sample_weight * batch.gene_mask).sum(), min=1.0)
        if torch.isfinite(logits).all() and torch.isfinite(loss):
            return logits, loss
        if amp_enabled:
            if training:
                print("[train_tf_eager] non-finite AMP batch detected; retrying in float32", flush=True)
            with autocast(device_type="cuda", enabled=False):
                logits = model(batch)
                loss_mat = crit(logits, batch.labels) * batch.sample_weight * batch.gene_mask
                loss = loss_mat.sum() / torch.clamp((batch.sample_weight * batch.gene_mask).sum(), min=1.0)
        return logits, loss

    wb = None
    if wandb_enabled and _is_main_process():
        try:
            import wandb  # type: ignore

            wb = wandb.init(
                project=wandb_project,
                name=(wandb_run_name.strip() or None),
                config={
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "device": device,
                    "world_size": world_size,
                    "amp_enabled": amp_enabled,
                    "pos_weight": pos_weight,
                    "model": dataclasses.asdict(model_cfg),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "auprc_tie_margin": auprc_tie_margin,
                    "early_stopping_metric": early_stopping_metric,
                    "checkpoint_metric": checkpoint_metric,
                    "deterministic": deterministic,
                    "deterministic_warn_only": deterministic_warn_only,
                    "attention_backend": attention_backend,
                    "find_unused_parameters": find_unused_parameters,
                },
            )
        except Exception as exc:
            _print_main(f"[train_tf_eager] wandb init failed: {exc}")
            wb = None

    best_state = None
    best_stop_value = float("-inf") if _MONITOR_MODES[early_stopping_metric] == "max" else float("inf")
    best_val_auprc = float("-inf")
    best_brier = float("inf")
    best_monitor = float("-inf") if _MONITOR_MODES[checkpoint_metric] == "max" else float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = 0
    try:
        for ep in range(epochs):
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(ep)
            train_loader = DataLoader(
                train_dataset,
                batch_size=max(1, batch_size),
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=max(0, num_workers),
                collate_fn=_collate_window_batches,
                pin_memory=amp_enabled,
                persistent_workers=False,
                worker_init_fn=_seed_worker if num_workers > 0 else None,
                generator=loader_generator,
            )
            total_loss = 0.0
            n_train = 0
            train_scores: list[np.ndarray] = []
            train_labels: list[np.ndarray] = []
            train_masks: list[np.ndarray] = []
            for batch_cpu in train_loader:
                batch = _to_device(batch_cpu, device)
                logits, loss = _forward_loss(batch, training=True)
                if not torch.isfinite(loss):
                    _print_main("[train_tf_eager] skipping batch with non-finite float32 fallback loss")
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total_loss += float(loss.item())
                n_train += int(batch.labels.shape[0])
                train_scores.append(torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1))
                train_labels.append(batch.labels.detach().float().cpu().numpy().reshape(-1))
                train_masks.append((batch.gene_mask * (batch.sample_weight > 0).float()).detach().float().cpu().numpy().reshape(-1))
            total_loss, n_train = _all_reduce_pair(total_loss, n_train, device)
            tr_m = _metrics(
                [_gather_vector(train_scores)],
                [_gather_vector(train_labels)],
                [_gather_vector(train_masks)],
            )

            model.eval()
            val_loss = 0.0
            val_n = 0
            val_scores: list[np.ndarray] = []
            val_labels: list[np.ndarray] = []
            val_masks: list[np.ndarray] = []
            val_loader = DataLoader(
                val_dataset,
                batch_size=max(1, batch_size),
                shuffle=False,
                sampler=val_sampler,
                num_workers=max(0, num_workers),
                collate_fn=_collate_window_batches,
                pin_memory=amp_enabled,
                persistent_workers=False,
                worker_init_fn=_seed_worker if num_workers > 0 else None,
                generator=loader_generator,
            )
            with torch.no_grad():
                for batch_cpu in val_loader:
                    batch = _to_device(batch_cpu, device)
                    logits, loss = _forward_loss(batch, training=False)
                    if not torch.isfinite(loss):
                        _print_main("[train_tf_eager] skipping validation batch with non-finite loss")
                        continue
                    val_loss += float(loss.item())
                    val_n += int(batch.labels.shape[0])
                    val_scores.append(torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1))
                    val_labels.append(batch.labels.detach().float().cpu().numpy().reshape(-1))
                    val_masks.append((batch.gene_mask * (batch.sample_weight > 0).float()).detach().float().cpu().numpy().reshape(-1))
            val_loss, val_n = _all_reduce_pair(val_loss, val_n, device)
            va_m = _metrics(
                [_gather_vector(val_scores)],
                [_gather_vector(val_labels)],
                [_gather_vector(val_masks)],
            )
            train_loss_epoch = total_loss / max(n_train, 1)
            val_loss_epoch = val_loss / max(val_n, 1)
            stop_value = _monitor_value(
                early_stopping_metric,
                val_loss=val_loss_epoch,
                val_auprc=va_m["auprc"],
                val_brier=va_m["brier"],
            )
            ckpt_value = _monitor_value(
                checkpoint_metric,
                val_loss=val_loss_epoch,
                val_auprc=va_m["auprc"],
                val_brier=va_m["brier"],
            )
            _print_main(
                f"epoch {ep + 1} train_loss={train_loss_epoch:.4f} "
                f"train_auprc={tr_m['auprc']:.4f} train_brier={tr_m['brier']:.4f} "
                f"val_loss={val_loss_epoch:.4f} val_auprc={va_m['auprc']:.4f} "
                f"val_brier={va_m['brier']:.4f}"
            )
            if wb is not None:
                wb.log(
                    {
                        "epoch": ep + 1,
                        "train/loss": train_loss_epoch,
                        "train/auprc": tr_m["auprc"],
                        "train/auroc": tr_m["auroc"],
                        "train/brier": tr_m["brier"],
                        "val/loss": val_loss_epoch,
                        "val/auprc": va_m["auprc"],
                        "val/auroc": va_m["auroc"],
                        "val/brier": va_m["brier"],
                        "train/windows": len(train_idx),
                        "val/windows": len(val_idx),
                        "train/lr": float(opt.param_groups[0]["lr"]),
                        "monitor/early_stopping_metric": stop_value,
                        "monitor/checkpoint_metric": ckpt_value,
                    }
                )
            improved_for_stopping = False
            if _MONITOR_MODES[early_stopping_metric] == "max":
                if stop_value > best_stop_value + early_stopping_min_delta:
                    best_stop_value = stop_value
                    improved_for_stopping = True
            else:
                if stop_value < best_stop_value - early_stopping_min_delta or best_stop_value == float("inf"):
                    best_stop_value = stop_value
                    improved_for_stopping = True

            if improved_for_stopping:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            improved_checkpoint = False
            if _MONITOR_MODES[checkpoint_metric] == "max":
                if ckpt_value > best_monitor + early_stopping_min_delta:
                    improved_checkpoint = True
                elif checkpoint_metric == "val_auprc" and ckpt_value >= best_monitor - auprc_tie_margin and va_m["brier"] < best_brier:
                    improved_checkpoint = True
            else:
                if ckpt_value < best_monitor - early_stopping_min_delta:
                    improved_checkpoint = True
                elif ckpt_value <= best_monitor + auprc_tie_margin and va_m["brier"] < best_brier:
                    improved_checkpoint = True

            if improved_checkpoint:
                best_monitor = ckpt_value
                best_val_auprc = va_m["auprc"]
                best_brier = va_m["brier"]
                best_epoch = ep + 1
                best_state = _model_state_dict(model)

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                stopped_epoch = ep + 1
                _print_main(
                    f"early_stopping stopped_epoch={stopped_epoch} best_epoch={best_epoch} "
                    f"monitor={early_stopping_metric} best_monitor={best_stop_value:.4f} "
                    f"checkpoint_metric={checkpoint_metric} best_val_brier={best_brier:.4f} "
                    f"patience={early_stopping_patience} min_delta={early_stopping_min_delta:g} "
                    f"auprc_tie_margin={auprc_tie_margin:g}"
                )
                break

        if best_state is not None:
            _load_model_state_dict(model, best_state)
        if _is_main_process():
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": _model_state_dict(model),
                    "config": dataclasses.asdict(model_cfg),
                    "window_size": 100,
                    "pos_weight": pos_weight,
                    "best_val_auprc": best_val_auprc,
                    "best_val_brier": best_brier,
                    "best_early_stopping_metric": best_stop_value,
                    "best_checkpoint_metric": best_monitor,
                    "best_epoch": best_epoch,
                    "stopped_epoch": stopped_epoch,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "auprc_tie_margin": auprc_tie_margin,
                    "early_stopping_metric": early_stopping_metric,
                    "checkpoint_metric": checkpoint_metric,
                    "deterministic": deterministic,
                    "deterministic_warn_only": deterministic_warn_only,
                    "attention_backend": attention_backend,
                    "find_unused_parameters": find_unused_parameters,
                    "amp_enabled": amp_enabled,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "world_size": world_size,
                },
                out_path,
            )
            if wb is not None:
                wb.log(
                    {
                        "checkpoint/best_epoch": best_epoch,
                        "checkpoint/best_val_auprc": best_val_auprc,
                        "checkpoint/best_val_brier": best_brier,
                        "checkpoint/best_metric_value": best_monitor,
                    }
                )
                wb.summary["checkpoint/best_epoch"] = best_epoch
                wb.summary["checkpoint/best_val_auprc"] = best_val_auprc
                wb.summary["checkpoint/best_val_brier"] = best_brier
                wb.summary["checkpoint/best_metric_value"] = best_monitor
                wb.finish()
            _print_main(f"Saved {out_path}")
    finally:
        _teardown_distributed()


if __name__ == "__main__":
    main()
