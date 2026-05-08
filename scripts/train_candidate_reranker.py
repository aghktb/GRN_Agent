#!/usr/bin/env python3
"""Train a lightweight candidate reranker with predict_proba for TF-neighborhood candidates."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier

from grn_agent.pipeline.config import load_yaml_config
from grn_agent.schemas import EvidenceGraph


def _cfg_get(cfg: dict, key: str, default):
    if key in cfg:
        return cfg[key]
    alt = key.replace("_", "-")
    return cfg.get(alt, default)


def _as_float(x: object, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _row_from_graph(g: EvidenceGraph) -> list[float]:
    e = g.evidence or {}
    corr = _as_float(e.get("correlation"), 0.0)
    motif = e.get("motif_present")
    acc = _as_float(e.get("accessibility"), 0.0)
    prior = _as_float(e.get("ensemble_prior"), 0.0)
    in_module = 1.0 if bool(e.get("in_same_module", False)) else 0.0
    shared = _as_float(e.get("shared_neighbors"), 0.0)
    return [
        abs(corr),
        corr,
        1.0 if motif is True else 0.0,
        acc,
        prior,
        in_module,
        min(shared, 20.0) / 20.0,
    ]


def _load_graphs(path: Path) -> list[EvidenceGraph]:
    out: list[EvidenceGraph] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                out.append(EvidenceGraph.model_validate(json.loads(line)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="Optional YAML config; CLI args override config values")
    ap.add_argument("--graphs-jsonl", default="")
    ap.add_argument("--y-npz", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config) if args.config.strip() else {}
    graphs_jsonl = str(args.graphs_jsonl or _cfg_get(cfg, "graphs_jsonl", ""))
    y_npz = str(args.y_npz or _cfg_get(cfg, "y_npz", ""))
    out_path = str(args.out or _cfg_get(cfg, "out", ""))
    seed = int(args.seed if args.seed is not None else _cfg_get(cfg, "seed", 42))
    n_estimators = int(args.n_estimators if args.n_estimators is not None else _cfg_get(cfg, "n_estimators", 100))
    lr = float(args.learning_rate if args.learning_rate is not None else _cfg_get(cfg, "learning_rate", 0.05))
    max_depth = int(args.max_depth if args.max_depth is not None else _cfg_get(cfg, "max_depth", 2))

    if not graphs_jsonl or not y_npz or not out_path:
        raise SystemExit("Missing required args: --graphs-jsonl, --y-npz, --out (or set in --config)")

    graphs = _load_graphs(Path(graphs_jsonl))
    y = np.load(y_npz, allow_pickle=False)["y"].astype(np.int64)
    if len(graphs) != len(y):
        raise SystemExit(f"graphs ({len(graphs)}) and y ({len(y)}) length mismatch")
    x = np.asarray([_row_from_graph(g) for g in graphs], dtype=np.float32)

    if len(set(y.tolist())) < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            random_state=seed,
        )
    model.fit(x, y)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("wb") as fp:
        pickle.dump(model, fp)
    print(f"Saved candidate reranker to {out_path} with X={x.shape} pos={int(y.sum())} neg={int((y == 0).sum())}")


if __name__ == "__main__":
    main()
