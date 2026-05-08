"""Load/save EAGER checkpoints and create minimal weights for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from grn_agent.models.eager.eager_model import EagerRegulator, EagerRegulatorConfig


def save_eager_checkpoint(path: str | Path, model: EagerRegulator, extra: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "d_model": model.cfg.d_model,
        "n_mpnn_layers": model.cfg.n_mpnn_layers,
        "num_edge_types": model.cfg.num_edge_types,
        "n_heads": model.cfg.n_heads,
        "dropout": model.cfg.dropout,
        "tf_vocab": model.cfg.tf_vocab,
        "gene_vocab": model.cfg.gene_vocab,
        "context_vocab": model.cfg.context_vocab,
    }
    payload: dict[str, Any] = {"state_dict": model.state_dict(), "config": cfg, "version": 1}
    if extra:
        payload["extra"] = extra
    torch.save(payload, p)
    p.with_suffix(".json").write_text(json.dumps({"config": cfg}, indent=2), encoding="utf-8")


def load_eager_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> EagerRegulator:
    p = Path(path)
    raw = torch.load(p, map_location=map_location, weights_only=False)
    cfg_d = raw.get("config") or {}
    cfg = EagerRegulatorConfig(**{k: v for k, v in cfg_d.items() if k in EagerRegulatorConfig.__dataclass_fields__})
    m = EagerRegulator(cfg)
    m.load_state_dict(raw["state_dict"], strict=True)
    m.eval()
    return m


def save_minimal_eager_for_tests(path: str | Path) -> None:
    m = EagerRegulator()
    save_eager_checkpoint(path, m)
