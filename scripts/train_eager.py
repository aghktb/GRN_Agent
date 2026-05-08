#!/usr/bin/env python3
"""Train EAGER (binary BCE) from aligned evidence graph JSONL + y npz."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch import optim
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from grn_agent.models.eager.checkpoint import save_eager_checkpoint
from grn_agent.models.eager.eager_model import EagerRegulator
from grn_agent.models.eager.graph_batch import EagerGraphBatch, evidence_graph_to_batch
from grn_agent.pipeline.config import load_yaml_config
from grn_agent.schemas import EvalTrack, EvidenceGraph


def _load_graphs(path: Path) -> list[EvidenceGraph]:
    g: list[EvidenceGraph] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            g.append(EvidenceGraph.model_validate(json.loads(line)))
    return g


def _modality_dropout(batch, p_drop: float, rng: random.Random) -> object:
    """Randomly clear acc/motif bits and zero corresponding nodes (simplified: mask modalities)."""
    if p_drop <= 0 or rng.random() > p_drop:
        return batch
    from grn_agent.models.eager.graph_batch import EagerGraphBatch

    m = batch.modality.clone()
    if rng.random() < 0.5:
        m[:, 1] = 0.0
    if rng.random() < 0.5:
        m[:, 2] = 0.0
    if rng.random() < 0.5:
        m[:, 3] = 0.0
    return EagerGraphBatch(
        node_kind=batch.node_kind,
        x_value=batch.x_value,
        conf=batch.conf,
        edge_index=batch.edge_index,
        edge_type=batch.edge_type,
        node_mask=batch.node_mask,
        modality=m,
        mech_mask=batch.mech_mask,
        func_mask=batch.func_mask,
        context_idx=batch.context_idx,
        tf_idx=batch.tf_idx,
        gene_idx=batch.gene_idx,
    )


def _to_device(batch: EagerGraphBatch, device: str) -> EagerGraphBatch:
    return EagerGraphBatch(
        node_kind=batch.node_kind.to(device),
        x_value=batch.x_value.to(device),
        conf=batch.conf.to(device),
        edge_index=batch.edge_index.to(device),
        edge_type=batch.edge_type.to(device),
        node_mask=batch.node_mask.to(device),
        modality=batch.modality.to(device),
        mech_mask=batch.mech_mask.to(device),
        func_mask=batch.func_mask.to(device),
        context_idx=batch.context_idx.to(device),
        tf_idx=batch.tf_idx.to(device),
        gene_idx=batch.gene_idx.to(device),
    )


def _build_binary_metrics(device: str) -> dict[str, object]:
    return {
        "accuracy": BinaryAccuracy().to(device),
        "precision": BinaryPrecision().to(device),
        "recall": BinaryRecall().to(device),
        "f1": BinaryF1Score().to(device),
        "auroc": BinaryAUROC().to(device),
        "auprc": BinaryAveragePrecision().to(device),
    }


def _compute_epoch_metrics(logits: list[torch.Tensor], labels: list[torch.Tensor], device: str) -> dict[str, float]:
    if not logits:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
            "auprc": 0.0,
        }
    y_hat = torch.sigmoid(torch.cat(logits)).to(device)
    y_true = torch.cat(labels).to(device).long()
    metrics = _build_binary_metrics(device)
    out: dict[str, float] = {}
    for k, m in metrics.items():
        out[k] = float(m(y_hat, y_true).item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="Optional YAML config; CLI args override config values")
    ap.add_argument("--graphs-jsonl", default="")
    ap.add_argument("--y-npz", default="", help="npz with array y of shape (N,)")
    ap.add_argument(
        "--val-graphs-jsonl",
        default="",
        help="Optional validation graphs JSONL. When set with --val-y-npz, uses explicit validation set (manifest val) instead of random val split.",
    )
    ap.add_argument(
        "--val-y-npz",
        default="",
        help="Optional validation labels npz with array y (and optional sample_weight). Requires --val-graphs-jsonl.",
    )
    ap.add_argument("--out", default="", help="Output .pt checkpoint")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument(
        "--scheduler",
        default="none",
        choices=["none", "step", "cosine", "plateau"],
        help="Learning-rate scheduler type",
    )
    ap.add_argument("--step-size", type=int, default=None, help="StepLR step_size")
    ap.add_argument("--gamma", type=float, default=None, help="StepLR/Plateau decay factor")
    ap.add_argument("--min-lr", type=float, default=None, help="Minimum LR for cosine scheduler")
    ap.add_argument("--plateau-patience", type=int, default=None, help="ReduceLROnPlateau patience")
    ap.add_argument("--pos-weight", type=float, default=None, help="BCE pos_weight for class imbalance")
    ap.add_argument("--modality-dropout", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay")
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--early-stopping-patience", type=int, default=None)
    ap.add_argument("--early-stopping-min-delta", type=float, default=None)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=None, help="Enable Weights & Biases logging")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging")
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

    graphs_jsonl = str(args.graphs_jsonl or _cfg("graphs_jsonl", ""))
    y_npz = str(args.y_npz or _cfg("y_npz", ""))
    val_graphs_jsonl = str(args.val_graphs_jsonl or _cfg("val_graphs_jsonl", ""))
    val_y_npz = str(args.val_y_npz or _cfg("val_y_npz", ""))
    out_path = str(args.out or _cfg("out", ""))
    if not graphs_jsonl.strip() or not y_npz.strip() or not out_path.strip():
        raise SystemExit("Missing required args: --graphs-jsonl, --y-npz, --out (or set in --config)")
    has_val_graphs = bool(val_graphs_jsonl.strip())
    has_val_y = bool(val_y_npz.strip())
    if has_val_graphs != has_val_y:
        raise SystemExit("--val-graphs-jsonl and --val-y-npz must be provided together")
    use_explicit_val = has_val_graphs and has_val_y

    epochs = int(args.epochs if args.epochs is not None else _cfg("epochs", 5))
    lr = float(args.lr if args.lr is not None else _cfg("lr", 3e-4))
    scheduler_name = str(args.scheduler or _cfg("scheduler", "none")).strip().lower()
    step_size = int(args.step_size if args.step_size is not None else _cfg("step_size", 10))
    gamma = float(args.gamma if args.gamma is not None else _cfg("gamma", 0.5))
    min_lr = float(args.min_lr if args.min_lr is not None else _cfg("min_lr", 1e-6))
    plateau_patience = int(
        args.plateau_patience if args.plateau_patience is not None else _cfg("plateau_patience", 5)
    )
    pos_weight_arg = args.pos_weight if args.pos_weight is not None else _cfg("pos_weight", None)
    modality_dropout = float(args.modality_dropout if args.modality_dropout is not None else _cfg("modality_dropout", 0.0))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else _cfg("weight_decay", 1e-2))
    seed = int(args.seed if args.seed is not None else _cfg("seed", 0))
    val_frac = float(args.val_frac if args.val_frac is not None else _cfg("val_frac", 0.2))
    early_stopping_patience = int(
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else _cfg("early_stopping_patience", 0)
    )
    early_stopping_min_delta = float(
        args.early_stopping_min_delta
        if args.early_stopping_min_delta is not None
        else _cfg("early_stopping_min_delta", 0.0)
    )
    device = str(args.device or _cfg("device", "")) or ("cuda" if torch.cuda.is_available() else "cpu")
    wandb_enabled = bool(args.wandb if args.wandb is not None else _cfg("wandb", False))
    wandb_project = str(args.wandb_project or _cfg("wandb_project", "grn-agent-eager"))
    wandb_run_name = str(args.wandb_run_name or _cfg("wandb_run_name", ""))

    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs = _load_graphs(Path(graphs_jsonl))
    y_data = np.load(y_npz, allow_pickle=False)
    y_raw = y_data["y"]
    y = y_raw.astype(np.float32)
    sample_weight = (
        y_data["sample_weight"].astype(np.float32)
        if "sample_weight" in y_data.files
        else np.ones_like(y, dtype=np.float32)
    )
    if len(graphs) != len(y):
        raise SystemExit(f"graphs ({len(graphs)}) and y ({len(y)}) length mismatch")
    if len(sample_weight) != len(y):
        raise SystemExit(f"sample_weight ({len(sample_weight)}) and y ({len(y)}) length mismatch")
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    pos_weight = pos_weight_arg
    if pos_weight is None and n_neg > 0 and n_pos > 0:
        pos_weight = n_neg / n_pos
    elif pos_weight is None:
        pos_weight = 1.0
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")

    model = EagerRegulator()
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if scheduler_name == "step":
        scheduler = StepLR(opt, step_size=max(1, step_size), gamma=gamma)
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=min_lr)
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(opt, mode="min", factor=gamma, patience=max(1, plateau_patience))
    train_idx = list(range(len(graphs)))
    val_order: list[int] = []
    val_graphs: list[EvidenceGraph] = []
    val_y: np.ndarray | None = None
    val_sample_weight: np.ndarray | None = None
    validation_mode = "manifest" if use_explicit_val else "random_holdout"
    if use_explicit_val:
        val_graphs = _load_graphs(Path(val_graphs_jsonl))
        val_data = np.load(val_y_npz, allow_pickle=False)
        val_y_raw = val_data["y"]
        val_y = val_y_raw.astype(np.float32)
        val_sample_weight = (
            val_data["sample_weight"].astype(np.float32)
            if "sample_weight" in val_data.files
            else np.ones_like(val_y, dtype=np.float32)
        )
        if len(val_graphs) != len(val_y):
            raise SystemExit(f"val graphs ({len(val_graphs)}) and val y ({len(val_y)}) length mismatch")
        if len(val_sample_weight) != len(val_y):
            raise SystemExit(f"val sample_weight ({len(val_sample_weight)}) and val y ({len(val_y)}) length mismatch")
    else:
        n = len(graphs)
        idx = list(range(n))
        rng.shuffle(idx)
        n_val = int(max(1, min(n - 1, round(val_frac * n)))) if n > 1 else 0
        val_idx = set(idx[:n_val])
        train_idx = [i for i in idx if i not in val_idx]
        val_order = [i for i in idx if i in val_idx]

    wb = None
    if wandb_enabled:
        try:
            import wandb  # type: ignore

            wb = wandb.init(
                project=wandb_project,
                name=(wandb_run_name.strip() or None),
                config={
                    "epochs": epochs,
                    "lr": lr,
                    "scheduler": scheduler_name,
                    "step_size": step_size,
                    "gamma": gamma,
                    "min_lr": min_lr,
                    "plateau_patience": plateau_patience,
                    "pos_weight": pos_weight,
                    "modality_dropout": modality_dropout,
                    "weight_decay": weight_decay,
                    "val_frac": val_frac,
                    "validation_mode": validation_mode,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "n_train": len(train_idx),
                    "n_val": (len(val_graphs) if use_explicit_val else len(val_order)),
                },
            )
        except Exception as exc:
            print(f"[train_eager] wandb init failed: {exc}", flush=True)
            wb = None

    best_val_auprc = float("-inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = 0
    for ep in range(epochs):
        model.train()
        order = list(train_idx)
        rng.shuffle(order)
        train_loss_tot = 0.0
        train_n = 0
        train_logits: list[torch.Tensor] = []
        train_labels: list[torch.Tensor] = []
        for i in order:
            eg = graphs[i]
            yv = torch.tensor([y[i]], device=device, dtype=torch.float32)
            batch = evidence_graph_to_batch(eg, EvalTrack.NO_LITERATURE, literature_in_graph=False)
            batch = _modality_dropout(batch, modality_dropout, rng)
            batch = _to_device(batch, device)
            logit = model(batch)
            wv = torch.tensor([sample_weight[i]], device=device, dtype=torch.float32)
            loss = (crit(logit, yv) * wv).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_tot += float(loss.item())
            train_n += 1
            train_logits.append(logit.detach())
            train_labels.append(yv.detach())

        train_metrics = _compute_epoch_metrics(train_logits, train_labels, device)
        train_loss = train_loss_tot / max(train_n, 1)

        model.eval()
        val_loss_tot = 0.0
        val_n = 0
        val_logits: list[torch.Tensor] = []
        val_labels: list[torch.Tensor] = []
        with torch.no_grad():
            if use_explicit_val:
                assert val_y is not None
                assert val_sample_weight is not None
                for i, eg in enumerate(val_graphs):
                    yv = torch.tensor([val_y[i]], device=device, dtype=torch.float32)
                    batch = evidence_graph_to_batch(eg, EvalTrack.NO_LITERATURE, literature_in_graph=False)
                    batch = _to_device(batch, device)
                    logit = model(batch)
                    wv = torch.tensor([val_sample_weight[i]], device=device, dtype=torch.float32)
                    loss = (crit(logit, yv) * wv).mean()
                    val_loss_tot += float(loss.item())
                    val_n += 1
                    val_logits.append(logit)
                    val_labels.append(yv)
            else:
                for i in val_order:
                    eg = graphs[i]
                    yv = torch.tensor([y[i]], device=device, dtype=torch.float32)
                    batch = evidence_graph_to_batch(eg, EvalTrack.NO_LITERATURE, literature_in_graph=False)
                    batch = _to_device(batch, device)
                    logit = model(batch)
                    wv = torch.tensor([sample_weight[i]], device=device, dtype=torch.float32)
                    loss = (crit(logit, yv) * wv).mean()
                    val_loss_tot += float(loss.item())
                    val_n += 1
                    val_logits.append(logit)
                    val_labels.append(yv)
        val_metrics = _compute_epoch_metrics(val_logits, val_labels, device)
        val_loss = val_loss_tot / max(val_n, 1) if val_n > 0 else 0.0

        line = (
            f"epoch {ep+1} "
            f"train_loss={train_loss:.4f} train_auprc={train_metrics['auprc']:.4f} "
            f"val_loss={val_loss:.4f} val_auprc={val_metrics['auprc']:.4f}"
        )
        print(line, flush=True)
        val_auprc = float(val_metrics["auprc"])
        if val_n > 0 and val_auprc > best_val_auprc + early_stopping_min_delta:
            best_val_auprc = val_auprc
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        elif val_n > 0:
            epochs_without_improvement += 1

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_loss if val_n > 0 else train_loss)
            else:
                scheduler.step()
        lr_now = float(opt.param_groups[0]["lr"])
        if wb is not None:
            wb.log(
                {
                    "epoch": ep + 1,
                    "train/lr": lr_now,
                    "train/loss": train_loss,
                    "train/accuracy": train_metrics["accuracy"],
                    "train/precision": train_metrics["precision"],
                    "train/recall": train_metrics["recall"],
                    "train/f1": train_metrics["f1"],
                    "train/auroc": train_metrics["auroc"],
                    "train/auprc": train_metrics["auprc"],
                    "val/loss": val_loss,
                    "val/accuracy": val_metrics["accuracy"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/f1": val_metrics["f1"],
                    "val/auroc": val_metrics["auroc"],
                    "val/auprc": val_metrics["auprc"],
                }
            )

        if val_n > 0 and early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stopped_epoch = ep + 1
            print(
                f"early_stopping stopped_epoch={stopped_epoch} best_epoch={best_epoch} "
                f"best_val_auprc={best_val_auprc:.4f} patience={early_stopping_patience} "
                f"min_delta={early_stopping_min_delta:g}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_eager_checkpoint(
        out_path,
        model,
        extra={
            "pos_weight": pos_weight,
            "n": len(y),
            "n_train": len(train_idx),
            "n_val": (len(val_graphs) if use_explicit_val else len(val_order)),
            "validation_mode": validation_mode,
            "best_val_auprc": best_val_auprc,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        },
    )
    print(f"Saved {out_path}", flush=True)
    if wb is not None:
        wb.log(
            {
                "checkpoint_path": str(Path(out_path).resolve()),
                "checkpoint/best_epoch": best_epoch,
                "checkpoint/best_val_auprc": best_val_auprc,
                "checkpoint/stopped_epoch": stopped_epoch,
            }
        )
        wb.summary["checkpoint_path"] = str(Path(out_path).resolve())
        wb.summary["checkpoint/best_epoch"] = best_epoch
        wb.summary["checkpoint/best_val_auprc"] = best_val_auprc
        wb.summary["checkpoint/stopped_epoch"] = stopped_epoch
        wb.finish()


if __name__ == "__main__":
    main()
