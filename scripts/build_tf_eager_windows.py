#!/usr/bin/env python3
"""Build split-specific TF-centered evidence windows for tf-eager.

This builder intentionally does not read per-edge EvidenceGraph JSONL. It builds
one compact evidence graph per TF-window directly from expression and optional
multimodal indexes, so the expensive per-edge graph assembly path is bypassed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.agents import priors as prior_features
from grn_agent.agents.ingest import ingest_from_beeline_csv, ingest_from_npy
from grn_agent.agents.multimodal_loader import MultimodalFeatureLoader
from grn_agent.io import load_split_manifest
from grn_agent.io.gold_edges import load_gold_edge_labels
from grn_agent.models.tf_eager.window_batch import TF_EAGER_WINDOW_SIZE
from grn_agent.pipeline.config import load_yaml_config
from grn_agent.schemas import PriorBundle, SplitStrategy, SplitSubset


_MAX_SHARED_NEIGHBORS = 20
_SHARED_NEIGHBOR_CORR_THRESHOLD = 0.3
_DEFAULT_NEGATIVE_RATIO = 5
_DEFAULT_EXPR_WEAK_PERCENTILE = 0.30
_DEFAULT_EXPR_PROBABLE_PERCENTILE = 0.60
_PROBABLE_NEGATIVE_WEIGHT = 0.5
_DEFAULT_TF_SUBGRAPH_SIZE = 200
_DEFAULT_CORR_THRESHOLD = 0.25
_DEFAULT_TRAIN_SUBGRAPH_BOOTSTRAPS = 50


@dataclass(frozen=True)
class LoadedExpression:
    expression: np.ndarray
    gene_symbols: list[str]
    dataset_id: str
    species: str


@dataclass
class ContextStats:
    context: dict[str, Any]
    gene_symbols: list[str]
    gene_to_idx: dict[str, int]
    module_genes: list[str]
    module_set: set[str]
    module_idxs: np.ndarray
    sub: np.ndarray
    z_sub: np.ndarray
    denom: float
    global_mean: np.ndarray
    global_std: np.ndarray
    ctx_mean: np.ndarray
    ctx_dropout: np.ndarray
    torch_device: str | None = None
    z_sub_t: torch.Tensor | None = None
    module_idxs_t: torch.Tensor | None = None


@dataclass
class TfBuildResult:
    grouped: dict[tuple[str, str, int], OrderedDict[str, dict[str, Any]]]
    skipped_global_positive: int
    n_candidates_before_filter: int


def _pair_key(tf: str, target: str) -> tuple[str, str]:
    return (str(tf).strip().upper(), str(target).strip().upper())


def _clean_symbol(v: object) -> str:
    return str(v).strip().upper()


def _as_float(v: object, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        f = float(v)
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _as_bool(v: object, default: bool | None = None) -> bool | None:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    return default


def _cfg_get(args: argparse.Namespace, cfg: dict[str, Any], attr: str, default: Any = None) -> Any:
    cli_value = getattr(args, attr, None)
    if cli_value not in (None, ""):
        return cli_value
    tf_cfg = cfg.get("tf_eager", {}) if isinstance(cfg.get("tf_eager", {}), dict) else {}
    tf_build_cfg = tf_cfg.get("build", {}) if isinstance(tf_cfg.get("build", {}), dict) else {}
    for section in (tf_build_cfg, tf_cfg):
        if attr in section:
            return section[attr]
        dashed = attr.replace("_", "-")
        if dashed in section:
            return section[dashed]
    if attr in cfg:
        return cfg[attr]
    dashed = attr.replace("_", "-")
    if dashed in cfg:
        return cfg[dashed]
    return default


def _nested_get(cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _read_symbol_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Symbol file not found: {p}")
    out: list[str] = []
    with p.open(encoding="utf-8") as fp:
        for line in fp:
            token = line.strip().split(",")[0].split("\t")[0].strip()
            if token:
                out.append(_clean_symbol(token))
    return out


def _symbols_from_arg(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [_clean_symbol(x) for x in value.replace("\n", ",").split(",") if str(x).strip()]
    return [_clean_symbol(x) for x in list(value) if str(x).strip()]


def _load_expression(args: argparse.Namespace, cfg: dict[str, Any]) -> LoadedExpression:
    ds_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
    mode = str(getattr(args, "dataset_mode") or ds_cfg.get("mode") or "").strip().lower()
    expression_path = str(getattr(args, "expression_path") or ds_cfg.get("expression_path") or "").strip()
    dataset_id = str(getattr(args, "dataset_id") or ds_cfg.get("dataset_id") or "TF_EAGER_DS")
    species = str(getattr(args, "species") or ds_cfg.get("species") or "unknown")

    if not expression_path:
        raise SystemExit("Missing expression input: pass --expression-path or dataset.expression_path in --config")
    if not mode:
        suffix = Path(expression_path).suffix.lower()
        mode = "npy" if suffix == ".npy" else "beeline_csv"

    if mode == "beeline_csv":
        dataset, expression, gene_symbols = ingest_from_beeline_csv(
            dataset_id=dataset_id,
            species=species,
            path=expression_path,
            modalities=list(ds_cfg.get("modalities", ["scrna"])),
        )
        return LoadedExpression(
            expression=np.asarray(expression, dtype=np.float64),
            gene_symbols=[_clean_symbol(g) for g in gene_symbols],
            dataset_id=dataset.dataset_id,
            species=dataset.species or species,
        )

    if mode == "npy":
        gene_symbols = _symbols_from_arg(getattr(args, "gene_symbols", None) or ds_cfg.get("gene_symbols"))
        gene_symbols_file = getattr(args, "gene_symbols_file", "") or ds_cfg.get("gene_symbols_file") or ds_cfg.get("genes_file")
        if gene_symbols_file:
            gene_symbols = _read_symbol_file(gene_symbols_file)
        if not gene_symbols:
            raise SystemExit("dataset.mode=npy requires --gene-symbols, --gene-symbols-file, or dataset.gene_symbols")
        dataset, expression = ingest_from_npy(
            dataset_id=dataset_id,
            species=species,
            path=expression_path,
            gene_symbols=gene_symbols,
            modalities=list(ds_cfg.get("modalities", ["scrna"])),
        )
        return LoadedExpression(
            expression=np.asarray(expression, dtype=np.float64),
            gene_symbols=[_clean_symbol(g) for g in gene_symbols],
            dataset_id=dataset.dataset_id,
            species=dataset.species or species,
        )

    raise SystemExit(f"Unsupported --dataset-mode {mode!r}; use beeline_csv or npy")


def _load_tf_list(args: argparse.Namespace, cfg: dict[str, Any], gene_symbols: list[str], subset_tfs: set[str]) -> list[str]:
    ds_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
    tf_list = _symbols_from_arg(getattr(args, "tf_list", None) or ds_cfg.get("tf_list"))
    tf_file = getattr(args, "tf_file", "") or ds_cfg.get("tf_file")
    if tf_file:
        tf_list = _read_symbol_file(tf_file)
    if not tf_list:
        tf_list = sorted(subset_tfs)
    gene_set = set(gene_symbols)
    return [tf for tf in tf_list if tf in subset_tfs and tf in gene_set]


def _load_blind_tf_list(args: argparse.Namespace, cfg: dict[str, Any], gene_symbols: list[str]) -> list[str]:
    ds_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
    tf_list = _symbols_from_arg(getattr(args, "tf_list", None) or ds_cfg.get("tf_list"))
    tf_file = getattr(args, "tf_file", "") or ds_cfg.get("tf_file")
    if tf_file:
        tf_list = _read_symbol_file(tf_file)
    gene_set = set(gene_symbols)
    if not tf_list:
        return []
    return [tf for tf in tf_list if tf in gene_set]


def _build_context_stats(
    loaded: LoadedExpression,
    tf_list: list[str],
    *,
    cell_type: str,
    module_size: int,
    expression_transform: str,
) -> ContextStats:
    expression = loaded.expression.astype(np.float64, copy=False)
    if expression_transform == "arcsinh":
        expression = np.arcsinh(expression)
    elif expression_transform in {"none", ""}:
        pass
    else:
        raise SystemExit(f"Unsupported expression_transform={expression_transform!r}; use arcsinh or none")
    gene_symbols = loaded.gene_symbols
    n_cells, n_genes = expression.shape
    if len(gene_symbols) != n_genes:
        raise SystemExit(f"gene symbol count {len(gene_symbols)} does not match expression columns {n_genes}")

    if module_size and module_size > 0 and module_size < n_genes:
        var = expression.var(axis=0)
        top_idx = np.argsort(-var)[:module_size]
        module_genes = [gene_symbols[int(i)] for i in top_idx]
        for tf in tf_list:
            if tf not in module_genes:
                module_genes.append(tf)
    else:
        module_genes = list(gene_symbols)

    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    module_idxs = np.asarray([gene_to_idx[g] for g in module_genes if g in gene_to_idx], dtype=np.int64)
    sub = expression.astype(np.float64, copy=False)
    sub_means = sub.mean(axis=0, keepdims=True)
    sub_stds = sub.std(axis=0, keepdims=True)
    z_sub = (sub - sub_means) / (sub_stds + 1e-8)
    context = {
        "context_id": f"{loaded.species}_global_ctx",
        "cell_type": cell_type,
        "n_cells": int(n_cells),
        "n_module_genes": int(len(module_genes)),
        "species": loaded.species,
        "dataset_id": loaded.dataset_id,
    }
    return ContextStats(
        context=context,
        gene_symbols=gene_symbols,
        gene_to_idx=gene_to_idx,
        module_genes=module_genes,
        module_set=set(module_genes),
        module_idxs=module_idxs,
        sub=sub,
        z_sub=z_sub,
        denom=max(float(z_sub.shape[0]), 1.0),
        global_mean=expression.mean(axis=0),
        global_std=expression.std(axis=0),
        ctx_mean=sub.mean(axis=0),
        ctx_dropout=(sub == 0).mean(axis=0),
    )


def _resolve_torch_device(device: str | None) -> str | None:
    dev = str(device or "").strip()
    if not dev:
        return None
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[tf-eager-build] build_device=cuda requested but CUDA is unavailable; using CPU numpy", flush=True)
        return None
    return dev


def _attach_torch_context(stats: ContextStats, device: str | None) -> ContextStats:
    dev = _resolve_torch_device(device)
    if dev is None:
        return stats
    stats.torch_device = dev
    stats.z_sub_t = torch.as_tensor(stats.z_sub, dtype=torch.float32, device=dev)
    stats.module_idxs_t = torch.as_tensor(stats.module_idxs, dtype=torch.long, device=dev)
    print(f"[tf-eager-build] correlation backend=torch device={dev}", flush=True)
    return stats


def _corr_vector(stats: ContextStats, tf: str) -> np.ndarray | None:
    ti = stats.gene_to_idx.get(tf)
    if ti is None or stats.z_sub.shape[0] < 2:
        return None
    if stats.z_sub_t is not None:
        corr_t = (stats.z_sub_t[:, int(ti)].T @ stats.z_sub_t) / float(stats.denom)
        return corr_t.detach().cpu().numpy()
    return (stats.z_sub[:, ti].T @ stats.z_sub) / stats.denom


def _shared_neighbors(stats: ContextStats, tf_idx: int, gene_idx: int, corr_tf: np.ndarray) -> tuple[int, list[str]]:
    if stats.z_sub.shape[0] < 2 or stats.z_sub.shape[1] == 0:
        return 0, []
    if stats.z_sub_t is not None:
        corr_tf_t = torch.as_tensor(corr_tf, dtype=torch.float32, device=stats.z_sub_t.device)
        corr_gene_t = (stats.z_sub_t[:, int(gene_idx)].T @ stats.z_sub_t) / float(stats.denom)
        mask_t = (corr_tf_t.abs() >= _SHARED_NEIGHBOR_CORR_THRESHOLD) & (
            corr_gene_t.abs() >= _SHARED_NEIGHBOR_CORR_THRESHOLD
        )
        mask_t[int(tf_idx)] = False
        mask_t[int(gene_idx)] = False
        idxs_t = torch.nonzero(mask_t, as_tuple=False).flatten()
        if idxs_t.numel() == 0:
            return 0, []
        joint_t = corr_tf_t[idxs_t].abs() * corr_gene_t[idxs_t].abs()
        k = min(_MAX_SHARED_NEIGHBORS, int(idxs_t.numel()))
        picked_t = idxs_t[torch.topk(joint_t, k=k, largest=True).indices]
        picked = picked_t.detach().cpu().numpy()
        return int(len(picked)), [stats.gene_symbols[int(i)] for i in picked]
    corr_gene = (stats.z_sub[:, gene_idx].T @ stats.z_sub) / stats.denom
    mask = (np.abs(corr_tf) >= _SHARED_NEIGHBOR_CORR_THRESHOLD) & (
        np.abs(corr_gene) >= _SHARED_NEIGHBOR_CORR_THRESHOLD
    )
    mask[int(tf_idx)] = False
    mask[int(gene_idx)] = False
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return 0, []
    joint = np.abs(corr_tf[idxs]) * np.abs(corr_gene[idxs])
    picked = idxs[np.argsort(-joint)[:_MAX_SHARED_NEIGHBORS]]
    return int(len(picked)), [stats.gene_symbols[int(i)] for i in picked]


def _corr_vector_for_idx(stats: ContextStats, idx: int) -> np.ndarray:
    if stats.z_sub_t is not None:
        corr_t = (stats.z_sub_t[:, int(idx)].T @ stats.z_sub_t) / float(stats.denom)
        return corr_t.detach().cpu().numpy()
    return (stats.z_sub[:, int(idx)].T @ stats.z_sub) / stats.denom


def _expand_coexpression_hops(
    stats: ContextStats,
    *,
    frontier: list[int],
    selected: set[int],
    excluded: set[int],
    corr_threshold: float,
    target_count: int,
) -> list[int]:
    """Expand a thresholded co-expression graph without lowering the threshold."""
    out: list[int] = []
    module_set = {int(i) for i in stats.module_idxs.tolist()} if stats.module_idxs.size else set(range(len(stats.gene_symbols)))
    current = [int(i) for i in frontier if int(i) in module_set]
    seen_frontier: set[int] = set(current)
    while len(selected) + len(out) < target_count and current:
        candidates: dict[int, float] = {}
        next_frontier: list[int] = []
        for idx in current:
            corr_i = _corr_vector_for_idx(stats, idx)
            neighbor_idxs = np.where(np.abs(corr_i) >= corr_threshold)[0]
            for nb in neighbor_idxs:
                nb_i = int(nb)
                if nb_i in excluded or nb_i in selected or nb_i in out or nb_i not in module_set:
                    continue
                candidates[nb_i] = max(candidates.get(nb_i, 0.0), abs(float(corr_i[nb_i])))
        for nb_i, _score in sorted(candidates.items(), key=lambda kv: (-kv[1], stats.gene_symbols[kv[0]])):
            out.append(nb_i)
            next_frontier.append(nb_i)
            if len(selected) + len(out) >= target_count:
                break
        current = [i for i in next_frontier if i not in seen_frontier]
        seen_frontier.update(current)
    return out


def _ranked_gene_indices(
    stats: ContextStats,
    tf: str,
    corr: np.ndarray,
    *,
    topk_corr: int,
    bottomk_corr: int,
    corr_threshold: float,
) -> tuple[list[int], list[int]]:
    ti = stats.gene_to_idx[tf]
    if stats.z_sub_t is not None:
        corr_t = torch.as_tensor(corr, dtype=torch.float32, device=stats.z_sub_t.device)
        pool_t = stats.module_idxs_t if stats.module_idxs_t is not None and stats.module_idxs_t.numel() else torch.arange(
            len(stats.gene_symbols), dtype=torch.long, device=stats.z_sub_t.device
        )
        pool_t = pool_t[pool_t != int(ti)]
        abs_pool = corr_t[pool_t].abs()
        if corr_threshold > 0:
            keep = abs_pool >= float(corr_threshold)
            high_pool_t = pool_t[keep]
            abs_high = abs_pool[keep]
        else:
            high_pool_t = pool_t
            abs_high = abs_pool
        if high_pool_t.numel():
            k_high = min(int(topk_corr), int(high_pool_t.numel()))
            high = high_pool_t[torch.topk(abs_high, k=k_high, largest=True).indices].detach().cpu().tolist()
        else:
            high = []
        if pool_t.numel():
            k_low = min(int(bottomk_corr), int(pool_t.numel()))
            low = pool_t[torch.topk(abs_pool, k=k_low, largest=False).indices].detach().cpu().tolist()
        else:
            low = []
        return [int(i) for i in high], [int(i) for i in low]
    pool = stats.module_idxs if stats.module_idxs.size else np.arange(len(stats.gene_symbols), dtype=np.int64)
    pool = pool[pool != ti]
    if corr_threshold > 0:
        high_pool = pool[np.abs(corr[pool]) >= corr_threshold]
    else:
        high_pool = pool
    high_order = high_pool[np.argsort(-np.abs(corr[high_pool]))] if high_pool.size else np.asarray([], dtype=np.int64)
    low_order = pool[np.argsort(np.abs(corr[pool]))] if pool.size else np.asarray([], dtype=np.int64)
    return [int(i) for i in high_order[:topk_corr]], [int(i) for i in low_order[:bottomk_corr]]


def _motif_payload(loader: MultimodalFeatureLoader | None, tf: str, gene: str) -> dict[str, Any]:
    if loader is None or not loader.has_motif:
        return {"motif_present": None, "motif_score": None, "n_motif_regions": None}
    motif = loader.get_motif_features(tf, gene)
    if motif is None:
        return {"motif_present": False, "motif_score": 0.0, "n_motif_regions": 0}
    return {
        "motif_present": _as_bool(motif.motif_present, False),
        "motif_score": _as_float(motif.motif_score, 0.0),
        "n_motif_regions": int(motif.n_supporting_regions or 0),
    }


def _accessibility_payload(loader: MultimodalFeatureLoader | None, gene: str) -> dict[str, Any]:
    if loader is None or not loader.has_atac:
        return {"peak_accessibility": None, "celltype_specificity": None}
    atac = loader.get_atac_features(gene)
    if atac is None:
        return {"peak_accessibility": 0.0, "celltype_specificity": None}
    return {
        "peak_accessibility": _as_float(atac.peak_accessibility, 0.0),
        "celltype_specificity": _as_float(atac.celltype_specificity, None),
    }


def _linkage_payload(loader: MultimodalFeatureLoader | None, gene: str) -> dict[str, Any]:
    if loader is None or not loader.has_atac:
        return {"peak_to_gene_linked": None, "linkage_score": None}
    atac = loader.get_atac_features(gene)
    if atac is None:
        return {"peak_to_gene_linked": False, "linkage_score": 0.0}
    linked = _as_bool(atac.peak_to_gene_linked, False)
    score = _as_float(atac.peak_accessibility, 0.0)
    return {"peak_to_gene_linked": linked, "linkage_score": score}


def _prior_payload(bundle: PriorBundle | None) -> dict[str, Any]:
    if bundle is None:
        bundle = PriorBundle(ensemble_prior=0.0)
    return {
        "p_grnboost": _as_float(bundle.p_grnboost, None),
        "p_genie3": _as_float(bundle.p_genie3, None),
        "p_pidc": _as_float(bundle.p_pidc, None),
        "scenic_regulon_support": _as_float(bundle.scenic_regulon_support, None),
        "bootstrap_stability": _as_float(bundle.bootstrap_stability, None),
        "ensemble_prior": _as_float(bundle.ensemble_prior, 0.0),
    }


def _empty_orthology_payload() -> dict[str, Any]:
    return {
        "ortholog_support": None,
        "ortholog_confidence": None,
        "supporting_species": [],
        "conserved_in_human": None,
        "conserved_in_mouse": None,
    }


def _lookup_orthology(
    tf: str,
    gene: str,
    *,
    species: str,
    enabled: bool,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    if not enabled:
        return _empty_orthology_payload()
    pair = _pair_key(tf, gene)
    if pair in cache:
        return cache[pair]
    try:
        from grn_agent.agents.ortholog_client import get_ortholog_info

        payload = get_ortholog_info(tf, gene, source_species=species)
        out = {
            "ortholog_support": _as_float(payload.get("ortholog_support"), None),
            "ortholog_confidence": payload.get("ortholog_confidence"),
            "supporting_species": list(payload.get("supporting_species", []) or []),
            "conserved_in_human": _as_bool(payload.get("conserved_in_human"), None),
            "conserved_in_mouse": _as_bool(payload.get("conserved_in_mouse"), None),
        }
    except Exception:
        out = _empty_orthology_payload()
    cache[pair] = out
    return out


def _gene_evidence(
    stats: ContextStats,
    tf: str,
    gene: str,
    corr: np.ndarray,
    *,
    label: int,
    sample_weight: float,
    candidate_bucket: str,
    loader: MultimodalFeatureLoader | None,
    prior: PriorBundle | None,
    orthology: dict[str, Any],
    ctx_mean: np.ndarray | None = None,
    global_mean: np.ndarray | None = None,
    global_std: np.ndarray | None = None,
    ctx_dropout: np.ndarray | None = None,
) -> dict[str, Any]:
    ti = stats.gene_to_idx[tf]
    gi = stats.gene_to_idx[gene]
    ctx_mean_arr = ctx_mean if ctx_mean is not None else stats.ctx_mean
    global_mean_arr = global_mean if global_mean is not None else stats.global_mean
    global_std_arr = global_std if global_std is not None else stats.global_std
    ctx_dropout_arr = ctx_dropout if ctx_dropout is not None else stats.ctx_dropout
    r = float(corr[gi])
    z_t = float((ctx_mean_arr[ti] - global_mean_arr[ti]) / (global_std_arr[ti] + 1e-8))
    z_g = float((ctx_mean_arr[gi] - global_mean_arr[gi]) / (global_std_arr[gi] + 1e-8))
    n_shared, shared_names = _shared_neighbors(stats, ti, gi, corr)
    motif = _motif_payload(loader, tf, gene)
    acc = _accessibility_payload(loader, gene)
    link = _linkage_payload(loader, gene)
    prior_payload = _prior_payload(prior)
    orth_payload = orthology or _empty_orthology_payload()
    return {
        "target_gene": gene,
        "label": int(label),
        "sample_weight": float(sample_weight),
        "candidate_bucket": candidate_bucket,
        "evidence": {
            "correlation": r,
            "in_same_module": gene in stats.module_set and tf in stats.module_set,
            "z_t": z_t,
            "z_g": z_g,
            "activity_t": r,
            "motif_present": motif["motif_present"],
            "accessibility": acc["peak_accessibility"],
            "ensemble_prior": prior_payload["ensemble_prior"],
        },
        "expression": {
            "z_t": z_t,
            "z_g": z_g,
            "activity_t": r,
            "mean_expr_t": float(ctx_mean_arr[ti]),
            "mean_expr_g": float(ctx_mean_arr[gi]),
            "dropout_t": float(ctx_dropout_arr[ti]),
            "dropout_g": float(ctx_dropout_arr[gi]),
        },
        "network": {
            "pearson_r": r,
            "partial_corr": None,
            "in_same_module": gene in stats.module_set and tf in stats.module_set,
            "k_hop_distance": 1 if gene in stats.module_set and tf in stats.module_set else 2,
            "shared_neighbors": n_shared,
            "shared_neighbor_names": shared_names,
        },
        "motif": motif,
        "accessibility": acc,
        "linkage": link,
        "prior": prior_payload,
        "orthology": orth_payload,
        "literature": {},
    }


def _window_evidence_graph(tf: str, context: dict[str, Any], window_index: int, genes: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {"node_id": "tf", "node_type": "tf", "label": tf, "payload": {"symbol": tf}},
        {"node_id": "ctx", "node_type": "context", "label": context["context_id"], "payload": context},
    ]
    relations: list[dict[str, Any]] = [{"src_id": "tf", "dst_id": "ctx", "relation": "in_context"}]
    for i, gene in enumerate(genes):
        gid = f"gene_{i:03d}"
        target = gene["target_gene"]
        evidence = gene.get("evidence", {})
        nodes.append(
            {
                "node_id": gid,
                "node_type": "candidate_gene",
                "label": target,
                "payload": {
                    "target_gene": target,
                    "candidate_bucket": gene.get("candidate_bucket"),
                    "label": int(gene.get("label", 0)),
                    "sample_weight": float(gene.get("sample_weight", 1.0)),
                    "features": {
                        "evidence": evidence,
                        "expression": gene.get("expression", {}),
                        "network": gene.get("network", {}),
                        "motif": gene.get("motif", {}),
                        "accessibility": gene.get("accessibility", {}),
                        "linkage": gene.get("linkage", {}),
                        "prior": gene.get("prior", {}),
                        "orthology": gene.get("orthology", {}),
                        "literature": gene.get("literature", {}),
                    },
                },
            }
        )
        relations.append(
            {
                "src_id": "tf",
                "dst_id": gid,
                "relation": "candidate_regulates",
                "payload": {
                    "correlation": evidence.get("correlation"),
                    "motif_present": evidence.get("motif_present"),
                    "accessibility": evidence.get("accessibility"),
                },
            }
        )
    modality_mask = {
        "expr": any(g.get("network", {}).get("pearson_r") is not None for g in genes),
        "acc": any(g.get("accessibility", {}).get("peak_accessibility") is not None for g in genes),
        "motif": any(g.get("motif", {}).get("motif_present") is not None for g in genes),
        "link": any(g.get("linkage", {}).get("peak_to_gene_linked") is not None for g in genes),
    }
    return {
        "schema": "tf_eager_window_evidence_graph_v1",
        "graph_type": "tf_centered_window",
        "source_tf": tf,
        "context_id": context["context_id"],
        "window_index": int(window_index),
        "window_size": TF_EAGER_WINDOW_SIZE,
        "nodes": nodes,
        "relations": relations,
        "modality_mask": modality_mask,
    }


def _add_candidate(targets: OrderedDict[str, str], gene: str, bucket: str) -> None:
    if gene and gene not in targets:
        targets[gene] = bucket


def _candidate_targets_for_tf(
    stats: ContextStats,
    tf: str,
    corr: np.ndarray,
    *,
    allowed_positive: set[tuple[str, str]],
    positive_global: set[tuple[str, str]],
    loader: MultimodalFeatureLoader | None,
    topk_corr: int,
    bottomk_corr: int,
    corr_threshold: float,
    rescue_accessibility: bool,
    rescue_motif: bool,
    rescue_max_per_tf: int,
    max_edges_per_tf: int,
) -> tuple[OrderedDict[str, str], int]:
    targets: OrderedDict[str, str] = OrderedDict()
    skipped_global_positive = 0

    for pos_tf, pos_gene in sorted(allowed_positive):
        if pos_tf == tf and pos_gene in stats.gene_to_idx and pos_gene != tf:
            _add_candidate(targets, pos_gene, "split_positive")

    high_idxs, low_idxs = _ranked_gene_indices(
        stats,
        tf,
        corr,
        topk_corr=topk_corr,
        bottomk_corr=bottomk_corr,
        corr_threshold=corr_threshold,
    )
    for gi in high_idxs:
        _add_candidate(targets, stats.gene_symbols[gi], "expression_high_abs_corr")
    for gi in low_idxs:
        _add_candidate(targets, stats.gene_symbols[gi], "expression_low_abs_corr_background")

    if loader is not None and rescue_motif and loader.has_motif:
        for gene in sorted(loader.motif_targets_for_tf(tf)):
            if gene in stats.gene_to_idx and gene != tf:
                _add_candidate(targets, gene, "motif_rescue")

    if loader is not None and rescue_accessibility and loader.has_atac:
        rescued = 0
        for gene in sorted(loader.accessible_genes() & stats.module_set):
            if gene in stats.gene_to_idx and gene != tf:
                _add_candidate(targets, gene, "accessibility_rescue")
                rescued += 1
                if rescued >= rescue_max_per_tf:
                    break

    filtered: OrderedDict[str, str] = OrderedDict()
    for gene, bucket in targets.items():
        pair = (tf, gene)
        if pair in positive_global and pair not in allowed_positive:
            skipped_global_positive += 1
            continue
        filtered[gene] = bucket
        if max_edges_per_tf > 0 and len(filtered) >= max_edges_per_tf:
            break
    return filtered, skipped_global_positive


def _full_neighborhood_targets_for_tf(
    stats: ContextStats,
    tf: str,
    corr: np.ndarray,
    *,
    allowed_positive: set[tuple[str, str]],
    positive_global: set[tuple[str, str]],
    loader: MultimodalFeatureLoader | None,
    corr_threshold: float,
    max_count: int,
    rng: np.random.Generator,
    rescue_accessibility: bool,
    rescue_motif: bool,
) -> tuple[OrderedDict[str, str], int]:
    targets: OrderedDict[str, str] = OrderedDict()
    skipped_global_positive = 0

    ti = stats.gene_to_idx[tf]
    pool = stats.module_idxs if stats.module_idxs.size else np.arange(len(stats.gene_symbols), dtype=np.int64)
    pool = pool[pool != ti]
    abs_corr = np.abs(corr[pool])
    order = pool[np.argsort(-abs_corr)] if pool.size else np.asarray([], dtype=np.int64)
    strong = [int(i) for i in order if abs(float(corr[int(i)])) >= corr_threshold]
    weak = [int(i) for i in pool[np.argsort(abs_corr)] if abs(float(corr[int(i)])) < corr_threshold]

    for gi in strong:
        _add_candidate(targets, stats.gene_symbols[gi], "coexpression_neighbor")

    if loader is not None and rescue_motif and loader.has_motif:
        for gene in sorted(loader.motif_targets_for_tf(tf)):
            if gene in stats.gene_to_idx and gene != tf:
                _add_candidate(targets, gene, "motif_rescue")

    if loader is not None and rescue_accessibility and loader.has_atac:
        for gene in sorted(loader.accessible_genes() & stats.module_set):
            if gene in stats.gene_to_idx and gene != tf:
                _add_candidate(targets, gene, "accessibility_rescue")

    for gi in weak:
        _add_candidate(targets, stats.gene_symbols[gi], "low_coexpression_background")

    filtered: OrderedDict[str, str] = OrderedDict()
    for gene, bucket in targets.items():
        pair = (tf, gene)
        if pair in positive_global and pair not in allowed_positive:
            skipped_global_positive += 1
            continue
        filtered[gene] = bucket
    if max_count > 0 and len(filtered) > max_count:
        filtered = _cap_tf_subgraph(
            filtered,
            rng=rng,
            max_count=max_count,
        )
    return filtered, skipped_global_positive


def _cap_tf_subgraph(
    targets: OrderedDict[str, str],
    *,
    rng: np.random.Generator,
    max_count: int,
) -> OrderedDict[str, str]:
    if max_count <= 0 or len(targets) <= max_count:
        return targets
    out: OrderedDict[str, str] = OrderedDict()
    items = list(targets.items())
    picked = rng.choice(len(items), size=min(max_count, len(items)), replace=False)
    for idx in picked.tolist():
        gene, bucket = items[int(idx)]
        out[gene] = bucket
    return out


def _without_replacement_tf_subgraph_chunks(
    targets: OrderedDict[str, str],
    *,
    rng: np.random.Generator,
    max_count: int,
    n_chunks: int,
) -> list[OrderedDict[str, str]]:
    if not targets:
        return []
    if max_count <= 0:
        return [targets]
    items = list(targets.items())
    order = rng.permutation(len(items)).tolist()
    chunks: list[OrderedDict[str, str]] = []
    offset = 0
    for _ in range(max(1, int(n_chunks))):
        if offset >= len(order):
            break
        picked = order[offset : offset + max_count]
        chunks.append(OrderedDict((items[int(i)][0], items[int(i)][1]) for i in picked))
        offset += max_count
    return chunks


def _subgraph_normalized_arrays(
    stats: ContextStats,
    tf: str,
    targets: OrderedDict[str, str],
    base_corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    genes = [tf] + [gene for gene in targets if gene in stats.gene_to_idx and gene != tf]
    idxs = [stats.gene_to_idx[g] for g in genes]
    mat = stats.sub[:, idxs].astype(np.float64, copy=False)
    row_mean = mat.mean(axis=1, keepdims=True)
    row_std = mat.std(axis=1, keepdims=True)
    row_z = (mat - row_mean) / (row_std + 1e-8)

    col_mean = row_z.mean(axis=0, keepdims=True)
    col_std = row_z.std(axis=0, keepdims=True)
    col_z = (row_z - col_mean) / (col_std + 1e-8)
    local_corr = (col_z[:, 0].T @ col_z) / max(float(col_z.shape[0]), 1.0)

    corr = np.asarray(base_corr, dtype=np.float64).copy()
    ctx_mean = np.asarray(stats.ctx_mean, dtype=np.float64).copy()
    global_mean = np.asarray(stats.global_mean, dtype=np.float64).copy()
    global_std = np.asarray(stats.global_std, dtype=np.float64).copy()
    ctx_dropout = np.asarray(stats.ctx_dropout, dtype=np.float64).copy()
    local_means = row_z.mean(axis=0)
    local_stds = row_z.std(axis=0)
    local_dropout = (mat == 0).mean(axis=0)
    for local_i, global_i in enumerate(idxs):
        corr[global_i] = float(local_corr[local_i])
        ctx_mean[global_i] = float(local_means[local_i])
        global_mean[global_i] = 0.0
        global_std[global_i] = float(local_stds[local_i]) if float(local_stds[local_i]) > 1e-8 else 1.0
        ctx_dropout[global_i] = float(local_dropout[local_i])
    return corr, ctx_mean, global_mean, global_std, ctx_dropout


def _available_modalities(genes: list[dict[str, Any]]) -> set[str]:
    available = {"expr"}
    if any(g.get("motif", {}).get("motif_present") is not None for g in genes):
        available.add("motif")
    if any(g.get("accessibility", {}).get("peak_accessibility") is not None for g in genes):
        available.add("acc")
    if any(g.get("linkage", {}).get("peak_to_gene_linked") is not None for g in genes):
        available.add("link")
    return available


def _as_present_bool(v: object) -> bool:
    return v is True or str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _negative_class_for_gene(
    gene: dict[str, Any],
    *,
    available: set[str],
    corr_threshold: float,
    motif_score_threshold: float,
    acc_threshold: float,
) -> tuple[str, float]:
    corr = abs(_as_float((gene.get("evidence") or {}).get("correlation"), 0.0) or 0.0)
    low_expr = corr < corr_threshold

    low_motif = True
    if "motif" in available:
        motif = gene.get("motif") or {}
        motif_present = _as_present_bool(motif.get("motif_present"))
        motif_score = _as_float(motif.get("motif_score"), 0.0) or 0.0
        low_motif = (not motif_present) or motif_score <= motif_score_threshold

    low_acc = True
    if "acc" in available:
        acc = _as_float((gene.get("accessibility") or {}).get("peak_accessibility"), 0.0) or 0.0
        low_acc = acc <= acc_threshold

    has_acc = "acc" in available
    has_motif = "motif" in available
    if has_acc and has_motif:
        reliable = low_expr and low_acc and low_motif
        partial = low_expr and (low_acc or low_motif)
    elif has_acc:
        reliable = low_expr and low_acc
        partial = low_expr or low_acc
    else:
        reliable = low_expr
        partial = low_expr

    if reliable:
        return "reliable_negative", 1.0
    if partial:
        return "probable_negative", _PROBABLE_NEGATIVE_WEIGHT
    return "ambiguous", 0.0


def _label_and_sample_tf_genes(
    genes: OrderedDict[str, dict[str, Any]],
    *,
    tf: str,
    allowed_positive: set[tuple[str, str]],
    positive_global: set[tuple[str, str]],
    rng: np.random.Generator,
    negative_ratio: int,
    corr_threshold: float,
    motif_score_threshold: float,
    acc_threshold: float,
) -> list[dict[str, Any]]:
    gene_list = list(genes.values())
    if not gene_list:
        return []
    available = _available_modalities(gene_list)

    positives: list[dict[str, Any]] = []
    reliable: list[dict[str, Any]] = []
    probable: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for gene, rec in genes.items():
        pair = (tf, gene)
        out = dict(rec)
        out["available_modalities"] = sorted(available)
        if pair in allowed_positive:
            out["label"] = 1
            out["sample_weight"] = 1.0
            out["negative_class"] = "positive"
            positives.append(out)
            continue
        if pair in positive_global:
            # Defensive leakage guard: global positives outside this split were
            # already filtered, but never train them as negatives if one remains.
            out["label"] = 0
            out["sample_weight"] = 0.0
            out["negative_class"] = "ambiguous_global_positive"
            ambiguous.append(out)
            continue

        negative_class, sample_weight = _negative_class_for_gene(
            out,
            available=available,
            corr_threshold=corr_threshold,
            motif_score_threshold=motif_score_threshold,
            acc_threshold=acc_threshold,
        )
        out["negative_class"] = negative_class
        out["sample_weight"] = sample_weight
        if negative_class == "reliable_negative":
            out["label"] = 0
            reliable.append(out)
        elif negative_class == "probable_negative":
            out["label"] = 0
            probable.append(out)
        else:
            out["label"] = 0
            ambiguous.append(out)

    target_negatives = max(0, int(negative_ratio)) * len(positives)
    selected: list[dict[str, Any]] = list(positives)
    if target_negatives > 0:
        rng.shuffle(reliable)
        rng.shuffle(probable)
        selected.extend(reliable[:target_negatives])
        remaining = max(0, target_negatives - len(reliable))
        if remaining:
            selected.extend(probable[:remaining])
    else:
        selected.extend(reliable)
        selected.extend(probable)

    # Keep ambiguous candidates in the window with mask/sample_weight=0, so they
    # can be scored at inference but do not become training negatives.
    selected_keys = {str(g["target_gene"]).strip().upper() for g in selected}
    selected.extend([g for g in ambiguous if str(g["target_gene"]).strip().upper() not in selected_keys])
    # Target-position binding must not see label-derived ordering.
    rng.shuffle(selected)
    return selected


def _should_add_coverage_window(
    *,
    subset: str,
    blind: bool,
    blind_ensure_coverage: bool,
    eval_ensure_coverage: bool,
) -> bool:
    if blind:
        return blind_ensure_coverage
    return subset == "test" and eval_ensure_coverage


def _subgraph_bootstraps_for_subset(
    *,
    subset: str,
    blind: bool,
    train_subgraph_bootstraps: int,
    val_subgraph_bootstraps: int,
    test_subgraph_bootstraps: int,
) -> int:
    if blind:
        return max(1, int(train_subgraph_bootstraps))
    if subset == "train":
        return max(1, int(train_subgraph_bootstraps))
    if subset == "val":
        return max(1, int(val_subgraph_bootstraps))
    if subset == "test":
        return max(1, int(test_subgraph_bootstraps))
    return 1


def _use_without_replacement_chunks(*, subset: str, blind: bool) -> bool:
    return bool(blind or subset in {"train", "val", "test"})


def _stable_tf_seed(base_seed: int, tf: str) -> int:
    payload = f"{int(base_seed)}::{str(tf).strip().upper()}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=False)


def _build_tf_records(
    *,
    tf: str,
    stats: ContextStats,
    train_mask: np.ndarray,
    loaded: LoadedExpression,
    allowed_positive: set[tuple[str, str]],
    positive_global: set[tuple[str, str]],
    loader: MultimodalFeatureLoader | None,
    fold_id: str,
    seed: int,
    subset: str,
    blind: bool,
    train_subgraph_bootstraps: int,
    val_subgraph_bootstraps: int,
    test_subgraph_bootstraps: int,
    train_window_neighbors: int,
    corr_threshold: float,
    max_edges_per_tf: int,
    negative_ratio: int,
    motif_score_threshold: float,
    acc_threshold: float,
    rescue_accessibility: bool,
    rescue_motif: bool,
    prior_bootstrap: int,
    prior_device: str | None,
    disable_priors: bool,
    use_ortholog_lookup: bool,
    blind_ensure_coverage: bool,
    eval_ensure_coverage: bool,
    blind_exhaustive_all_pairs: bool,
) -> TfBuildResult:
    corr = _corr_vector(stats, tf)
    if corr is None:
        return TfBuildResult(grouped={}, skipped_global_positive=0, n_candidates_before_filter=0)

    rng = np.random.default_rng(_stable_tf_seed(seed, tf))
    orthology_cache: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, int], OrderedDict[str, dict[str, Any]]] = defaultdict(OrderedDict)
    skipped_global_positive = 0
    n_candidates_before_filter = 0

    def _add_window_records(
        *,
        targets: OrderedDict[str, str],
        boot_idx: int,
    ) -> None:
        sub_corr, sub_ctx_mean, sub_global_mean, sub_global_std, sub_dropout = _subgraph_normalized_arrays(
            stats, tf, targets, corr
        )
        group_key = (stats.context["context_id"], tf, boot_idx)
        pair_list = [(tf, gene) for gene in targets]
        if disable_priors:
            prior_map: dict[tuple[str, str], PriorBundle] = {}
        else:
            prior_map = prior_features.compute_priors_for_pairs(
                loaded.expression,
                train_mask,
                loaded.gene_symbols,
                pair_list,
                split_id=fold_id,
                n_bootstrap=max(prior_bootstrap, 0),
                seed=seed + boot_idx,
                device=prior_device,
            )
        raw_gene_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for gene, bucket in targets.items():
            pair = (tf, gene)
            raw_gene_records[gene] = _gene_evidence(
                stats,
                tf,
                gene,
                sub_corr,
                label=1 if pair in allowed_positive else 0,
                sample_weight=1.0,
                candidate_bucket=bucket,
                loader=loader,
                prior=prior_map.get(pair),
                orthology=_lookup_orthology(
                    tf,
                    gene,
                    species=loaded.species,
                    enabled=use_ortholog_lookup,
                    cache=orthology_cache,
                ),
                ctx_mean=sub_ctx_mean,
                global_mean=sub_global_mean,
                global_std=sub_global_std,
                ctx_dropout=sub_dropout,
            )
        sampled_gene_records = _label_and_sample_tf_genes(
            raw_gene_records,
            tf=tf,
            allowed_positive=allowed_positive,
            positive_global=positive_global,
            rng=rng,
            negative_ratio=negative_ratio,
            corr_threshold=max(corr_threshold, 0.0),
            motif_score_threshold=motif_score_threshold,
            acc_threshold=acc_threshold,
        )
        for rec in sampled_gene_records:
            grouped[group_key][str(rec["target_gene"]).strip().upper()] = rec

    n_boot = _subgraph_bootstraps_for_subset(
        subset=subset,
        blind=blind,
        train_subgraph_bootstraps=train_subgraph_bootstraps,
        val_subgraph_bootstraps=val_subgraph_bootstraps,
        test_subgraph_bootstraps=test_subgraph_bootstraps,
    )
    full_targets: OrderedDict[str, str] | None = None
    if _use_without_replacement_chunks(subset=subset, blind=blind):
        full_targets, skipped = _full_neighborhood_targets_for_tf(
            stats,
            tf,
            corr,
            allowed_positive=allowed_positive,
            positive_global=positive_global,
            loader=loader,
            corr_threshold=max(corr_threshold, 0.0),
            max_count=0,
            rng=rng,
            rescue_accessibility=rescue_accessibility,
            rescue_motif=rescue_motif,
        )
        skipped_global_positive += skipped
        n_chunks = n_boot
        if blind and blind_exhaustive_all_pairs and train_window_neighbors > 0:
            n_chunks = max(1, int(math.ceil(len(full_targets) / float(train_window_neighbors))))
        for boot_idx, targets in enumerate(
            _without_replacement_tf_subgraph_chunks(
                full_targets,
                rng=rng,
                max_count=max(train_window_neighbors, 0),
                n_chunks=n_chunks,
            )
        ):
            n_candidates_before_filter += len(targets)
            _add_window_records(targets=targets, boot_idx=boot_idx)
    else:
        for boot_idx in range(n_boot):
            targets, skipped = _full_neighborhood_targets_for_tf(
                stats,
                tf,
                corr,
                allowed_positive=allowed_positive,
                positive_global=positive_global,
                loader=loader,
                corr_threshold=max(corr_threshold, 0.0),
                max_count=max(train_window_neighbors, 0),
                rng=rng,
                rescue_accessibility=rescue_accessibility,
                rescue_motif=rescue_motif,
            )
            skipped_global_positive += skipped
            n_candidates_before_filter += len(targets)
            _add_window_records(targets=targets, boot_idx=boot_idx)

    should_add_coverage = _should_add_coverage_window(
        subset=subset,
        blind=blind,
        blind_ensure_coverage=blind_ensure_coverage,
        eval_ensure_coverage=eval_ensure_coverage,
    )
    if should_add_coverage and train_window_neighbors > 0:
        if full_targets is not None:
            all_targets = full_targets
        else:
            all_targets, skipped = _full_neighborhood_targets_for_tf(
                stats,
                tf,
                corr,
                allowed_positive=allowed_positive,
                positive_global=positive_global,
                loader=loader,
                corr_threshold=max(corr_threshold, 0.0),
                max_count=0,
                rng=rng,
                rescue_accessibility=rescue_accessibility,
                rescue_motif=rescue_motif,
            )
            skipped_global_positive += skipped
        seen = {
            gene
            for (context_id, seen_tf, _), target_map in grouped.items()
            if context_id == stats.context["context_id"] and seen_tf == tf
            for gene in target_map
        }
        missing = OrderedDict((gene, bucket) for gene, bucket in all_targets.items() if gene not in seen)
        if missing:
            items = list(missing.items())
            chunk_size = max(1, int(train_window_neighbors))
            for offset in range(0, len(items), chunk_size):
                coverage_targets = OrderedDict(items[offset : offset + chunk_size])
                n_candidates_before_filter += len(coverage_targets)
                _add_window_records(
                    targets=coverage_targets,
                    boot_idx=n_boot + (offset // chunk_size),
                )

    if max_edges_per_tf > 0:
        for group_key, target_map in grouped.items():
            if len(target_map) <= max_edges_per_tf:
                continue
            trimmed = OrderedDict()
            for gene, rec in list(target_map.items())[:max_edges_per_tf]:
                trimmed[gene] = rec
            grouped[group_key] = trimmed

    return TfBuildResult(
        grouped=dict(grouped),
        skipped_global_positive=skipped_global_positive,
        n_candidates_before_filter=n_candidates_before_filter,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="Optional YAML config with dataset/multimodal settings")
    ap.add_argument("--expression-path", default="")
    ap.add_argument("--dataset-mode", default="", choices=["", "beeline_csv", "npy"])
    ap.add_argument("--dataset-id", default="")
    ap.add_argument("--species", default="")
    ap.add_argument("--gene-symbols", default="", help="Comma-separated gene symbols for --dataset-mode npy")
    ap.add_argument("--gene-symbols-file", default="")
    ap.add_argument("--expression-transform", default="", choices=["", "arcsinh", "none"])
    ap.add_argument("--tf-list", default="", help="Comma-separated TF symbols; defaults to split subset TFs")
    ap.add_argument("--tf-file", default="")
    ap.add_argument("--multimodal-manifest", default="")
    ap.add_argument("--gold-edges", default="")
    ap.add_argument("--split-manifest", default="")
    ap.add_argument("--strategy", default="")
    ap.add_argument("--fold-id", default="")
    ap.add_argument("--subset", default="", choices=["", "train", "val", "test"])
    ap.add_argument("--blind", action="store_true", help="Build unlabeled inference windows without gold/split inputs")
    ap.add_argument("--out-jsonl", default="")
    ap.add_argument("--cell-type", default="")
    ap.add_argument("--module-size", type=int, default=None, help="0 means all genes")
    ap.add_argument("--topk-corr", type=int, default=None)
    ap.add_argument("--bottomk-corr", type=int, default=None)
    ap.add_argument("--corr-threshold", type=float, default=None)
    ap.add_argument("--max-edges-per-tf", type=int, default=None)
    ap.add_argument("--negative-ratio", type=int, default=None)
    ap.add_argument("--train-window-neighbors", type=int, default=None)
    ap.add_argument("--train-subgraph-bootstraps", type=int, default=None)
    ap.add_argument("--val-subgraph-bootstraps", type=int, default=None)
    ap.add_argument("--test-subgraph-bootstraps", type=int, default=None)
    ap.add_argument("--train-include-positives", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--expr-weak-percentile", type=float, default=None)
    ap.add_argument("--expr-probable-percentile", type=float, default=None)
    ap.add_argument("--motif-score-threshold", type=float, default=None)
    ap.add_argument("--accessibility-threshold", type=float, default=None)
    ap.add_argument("--linkage-threshold", type=float, default=None)
    ap.add_argument("--rescue-max-per-tf", type=int, default=None)
    ap.add_argument("--disable-priors", action="store_true", default=None)
    ap.add_argument("--prior-train-frac", type=float, default=None)
    ap.add_argument("--prior-bootstrap", type=int, default=None)
    ap.add_argument("--prior-device", default="")
    ap.add_argument("--build-device", default="", help="Device for correlation/ranking backend; defaults to scoring.device")
    ap.add_argument("--tf-workers", type=int, default=None, help="Parallel TF worker threads for window construction")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--use-ortholog-lookup", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--no-rescue-motif", action="store_true")
    ap.add_argument("--no-rescue-accessibility", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config) if args.config.strip() else {}
    gold_edges = str(_cfg_get(args, cfg, "gold_edges", _nested_get(cfg, "split", "gold_edges", default="")))
    split_manifest = str(_cfg_get(args, cfg, "split_manifest", _nested_get(cfg, "split", "out", default="")))
    strategy = str(_cfg_get(args, cfg, "strategy", "leave_one_tf_out"))
    fold_id = str(_cfg_get(args, cfg, "fold_id", _nested_get(cfg, "split", "fold_id", default="")))
    subset = str(_cfg_get(args, cfg, "subset", "train"))
    out_jsonl = str(_cfg_get(args, cfg, "out_jsonl", _cfg_get(args, cfg, "windows_jsonl", "")))
    blind = bool(args.blind or cfg.get("blind", False))
    if not out_jsonl:
        raise SystemExit("Missing required output: out_jsonl/windows_jsonl")

    if blind:
        allowed_positive: set[tuple[str, str]] = set()
        positive_global: set[tuple[str, str]] = set()
        subset_tfs: set[str] = set()
        subset = "blind"
    else:
        if not all([gold_edges, split_manifest, fold_id, out_jsonl]):
            raise SystemExit("Missing required inputs: gold_edges, split_manifest, fold_id, out_jsonl")

        gold = load_gold_edge_labels(gold_edges)
        positive_global = {k for k, v in gold.items() if int(v) == 1}
        manifest = load_split_manifest(split_manifest)
        strategy_e = SplitStrategy(strategy)
        subset_e = SplitSubset(subset)
        allowed_positive = {
            _pair_key(r.source_tf, r.target_gene)
            for r in manifest.rows
            if r.split_name == strategy_e and r.fold_id == fold_id and r.subset == subset_e
        }
        subset_tfs = {tf for tf, _ in allowed_positive}
        if not subset_tfs:
            raise SystemExit("No TFs found for requested split/fold/subset")

    loaded = _load_expression(args, cfg)
    tf_list = _load_blind_tf_list(args, cfg, loaded.gene_symbols) if blind else _load_tf_list(args, cfg, loaded.gene_symbols, subset_tfs)
    if not tf_list:
        raise SystemExit(
            "No requested TFs are present in the expression matrix"
            if blind
            else "No requested split TFs are present in the expression matrix"
        )

    mm_manifest = str(getattr(args, "multimodal_manifest") or cfg.get("multimodal_manifest") or "").strip()
    loader: MultimodalFeatureLoader | None = None
    if mm_manifest:
        loader = MultimodalFeatureLoader(mm_manifest)
        loader.load()

    cand_cfg = cfg.get("candidates", {}) if isinstance(cfg.get("candidates", {}), dict) else {}
    cell_type = str(_cfg_get(args, cfg, "cell_type", _nested_get(cfg, "cell_context", "cell_type", default="unknown")))
    module_size = int(_cfg_get(args, cfg, "module_size", 0))
    seed = int(_cfg_get(args, cfg, "seed", cfg.get("seed", 0)))
    rng = np.random.default_rng(seed)
    prior_frac = float(np.clip(float(_cfg_get(args, cfg, "prior_train_frac", _nested_get(cfg, "train", "frac", default=0.8))), 0.0, 1.0))
    train_mask = np.zeros(loaded.expression.shape[0], dtype=bool)
    if prior_frac >= 1.0:
        train_mask[:] = True
    else:
        n_train = max(1, int(round(loaded.expression.shape[0] * prior_frac)))
        train_mask[rng.permutation(loaded.expression.shape[0])[:n_train]] = True
    use_ortholog_lookup = bool(cfg.get("use_ortholog_lookup", False) if args.use_ortholog_lookup is None else args.use_ortholog_lookup)
    disable_priors = bool(_cfg_get(args, cfg, "disable_priors", cfg.get("disable_priors", False)))
    topk_corr = int(_cfg_get(args, cfg, "topk_corr", cand_cfg.get("topk_corr", 700)))
    bottomk_corr = int(_cfg_get(args, cfg, "bottomk_corr", cand_cfg.get("bottomk_corr", 300)))
    expression_transform = str(
        _cfg_get(args, cfg, "expression_transform", cand_cfg.get("expression_transform", "arcsinh"))
    ).strip()
    corr_threshold = float(
        _cfg_get(args, cfg, "corr_threshold", cand_cfg.get("corr_threshold", cand_cfg.get("min_pearson", _DEFAULT_CORR_THRESHOLD)))
    )
    max_edges_per_tf = int(_cfg_get(args, cfg, "max_edges_per_tf", cand_cfg.get("max_edges_per_tf", 1000)))
    negative_ratio = int(_cfg_get(args, cfg, "negative_ratio", cand_cfg.get("negative_ratio", _DEFAULT_NEGATIVE_RATIO)))
    train_window_neighbors = int(
        _cfg_get(args, cfg, "train_window_neighbors", cand_cfg.get("train_window_neighbors", _DEFAULT_TF_SUBGRAPH_SIZE))
    )
    train_subgraph_bootstraps = int(
        _cfg_get(
            args,
            cfg,
            "train_subgraph_bootstraps",
            cand_cfg.get("train_subgraph_bootstraps", _DEFAULT_TRAIN_SUBGRAPH_BOOTSTRAPS),
        )
    )
    val_subgraph_bootstraps = int(
        _cfg_get(
            args,
            cfg,
            "val_subgraph_bootstraps",
            cand_cfg.get("val_subgraph_bootstraps", train_subgraph_bootstraps),
        )
    )
    test_subgraph_bootstraps = int(
        _cfg_get(
            args,
            cfg,
            "test_subgraph_bootstraps",
            cand_cfg.get("test_subgraph_bootstraps", 1),
        )
    )
    train_include_positives = bool(
        cand_cfg.get("train_include_positives", True)
        if args.train_include_positives is None
        else args.train_include_positives
    )
    blind_ensure_coverage = bool(cand_cfg.get("blind_ensure_coverage", True))
    eval_ensure_coverage = bool(cand_cfg.get("eval_ensure_coverage", True))
    blind_exhaustive_all_pairs = bool(cand_cfg.get("blind_exhaustive_all_pairs", False))
    expr_weak_percentile = float(
        _cfg_get(args, cfg, "expr_weak_percentile", cand_cfg.get("expr_weak_percentile", _DEFAULT_EXPR_WEAK_PERCENTILE))
    )
    expr_probable_percentile = float(
        _cfg_get(
            args,
            cfg,
            "expr_probable_percentile",
            cand_cfg.get("expr_probable_percentile", _DEFAULT_EXPR_PROBABLE_PERCENTILE),
        )
    )
    motif_score_threshold = float(_cfg_get(args, cfg, "motif_score_threshold", cand_cfg.get("motif_score_threshold", 0.0)))
    acc_threshold = float(_cfg_get(args, cfg, "accessibility_threshold", cand_cfg.get("accessibility_threshold", 0.0)))
    link_threshold = float(_cfg_get(args, cfg, "linkage_threshold", cand_cfg.get("linkage_threshold", 0.0)))
    rescue_max_per_tf = int(_cfg_get(args, cfg, "rescue_max_per_tf", cand_cfg.get("rescue_max_per_tf", 100)))
    rescue_motif = bool(_cfg_get(args, cfg, "rescue_motif", cand_cfg.get("rescue_motif", True))) and not bool(args.no_rescue_motif)
    rescue_accessibility = bool(
        _cfg_get(args, cfg, "rescue_accessibility", cand_cfg.get("rescue_accessibility", True))
    ) and not bool(args.no_rescue_accessibility)
    prior_bootstrap = int(_cfg_get(args, cfg, "prior_bootstrap", 8))
    prior_device = str(_cfg_get(args, cfg, "prior_device", _nested_get(cfg, "scoring", "device", default=""))).strip() or None
    build_device = str(_cfg_get(args, cfg, "build_device", _nested_get(cfg, "scoring", "device", default=""))).strip() or None
    tf_workers = int(_cfg_get(args, cfg, "tf_workers", min(8, os.cpu_count() or 1)))
    stats = _attach_torch_context(
        _build_context_stats(
            loaded,
            tf_list,
            cell_type=cell_type,
            module_size=module_size,
            expression_transform=expression_transform,
        ),
        build_device,
    )
    if build_device and str(build_device).strip().lower().startswith("cuda") and tf_workers > 1:
        print("[tf-eager-build] build_device uses CUDA; forcing tf_workers=1 to avoid GPU contention", flush=True)
        tf_workers = 1

    grouped: dict[tuple[str, str, int], OrderedDict[str, dict[str, Any]]] = defaultdict(OrderedDict)
    skipped_global_positive = 0
    n_candidates_before_filter = 0
    tf_workers = max(1, min(tf_workers, len(tf_list)))
    if tf_workers > 1:
        print(f"[tf-eager-build] parallel TF workers={tf_workers}", flush=True)
        with ThreadPoolExecutor(max_workers=tf_workers) as pool:
            futures = {
                pool.submit(
                    _build_tf_records,
                    tf=tf,
                    stats=stats,
                    train_mask=train_mask,
                    loaded=loaded,
                    allowed_positive=allowed_positive,
                    positive_global=positive_global,
                    loader=loader,
                    fold_id=fold_id,
                    seed=seed,
                    subset=subset,
                    blind=blind,
                    train_subgraph_bootstraps=train_subgraph_bootstraps,
                    val_subgraph_bootstraps=val_subgraph_bootstraps,
                    test_subgraph_bootstraps=test_subgraph_bootstraps,
                    train_window_neighbors=train_window_neighbors,
                    corr_threshold=corr_threshold,
                    max_edges_per_tf=max_edges_per_tf,
                    negative_ratio=negative_ratio,
                    motif_score_threshold=motif_score_threshold,
                    acc_threshold=acc_threshold,
                    rescue_accessibility=rescue_accessibility,
                    rescue_motif=rescue_motif,
                    prior_bootstrap=prior_bootstrap,
                    prior_device=prior_device,
                    disable_priors=disable_priors,
                    use_ortholog_lookup=use_ortholog_lookup,
                    blind_ensure_coverage=blind_ensure_coverage,
                    eval_ensure_coverage=eval_ensure_coverage,
                    blind_exhaustive_all_pairs=blind_exhaustive_all_pairs,
                ): tf
                for tf in tf_list
            }
            for future in as_completed(futures):
                result = future.result()
                skipped_global_positive += result.skipped_global_positive
                n_candidates_before_filter += result.n_candidates_before_filter
                for group_key, target_map in result.grouped.items():
                    grouped[group_key] = target_map
    else:
        for tf in tf_list:
            result = _build_tf_records(
                tf=tf,
                stats=stats,
                train_mask=train_mask,
                loaded=loaded,
                allowed_positive=allowed_positive,
                positive_global=positive_global,
                loader=loader,
                fold_id=fold_id,
                seed=seed,
                subset=subset,
                blind=blind,
                train_subgraph_bootstraps=train_subgraph_bootstraps,
                val_subgraph_bootstraps=val_subgraph_bootstraps,
                test_subgraph_bootstraps=test_subgraph_bootstraps,
                train_window_neighbors=train_window_neighbors,
                corr_threshold=corr_threshold,
                max_edges_per_tf=max_edges_per_tf,
                negative_ratio=negative_ratio,
                motif_score_threshold=motif_score_threshold,
                acc_threshold=acc_threshold,
                rescue_accessibility=rescue_accessibility,
                rescue_motif=rescue_motif,
                prior_bootstrap=prior_bootstrap,
                prior_device=prior_device,
                disable_priors=disable_priors,
                use_ortholog_lookup=use_ortholog_lookup,
                blind_ensure_coverage=blind_ensure_coverage,
                eval_ensure_coverage=eval_ensure_coverage,
                blind_exhaustive_all_pairs=blind_exhaustive_all_pairs,
            )
            skipped_global_positive += result.skipped_global_positive
            n_candidates_before_filter += result.n_candidates_before_filter
            for group_key, target_map in result.grouped.items():
                grouped[group_key] = target_map

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_windows = 0
    n_edges = 0
    n_pos = 0
    n_weighted = 0
    bucket_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)
    with out_path.open("w", encoding="utf-8") as fp:
        for (context_id, tf, boot_idx), target_map in sorted(grouped.items()):
            genes = list(target_map.values())
            for start in range(0, len(genes), TF_EAGER_WINDOW_SIZE):
                chunk = genes[start : start + TF_EAGER_WINDOW_SIZE]
                if not chunk:
                    continue
                window_index = start // TF_EAGER_WINDOW_SIZE
                for g in chunk:
                    bucket_counts[str(g.get("candidate_bucket", "unknown"))] += 1
                    class_counts[str(g.get("negative_class", "unknown"))] += 1
                context = {**stats.context, "subgraph_sample_index": int(boot_idx)}
                rec = {
                    "schema": "tf_eager_window_v1",
                    "source_tf": tf,
                    "context": context,
                    "window_index": window_index,
                    "window_size": TF_EAGER_WINDOW_SIZE,
                    "genes": chunk,
                    "evidence_graph": _window_evidence_graph(tf, context, window_index, chunk),
                }
                fp.write(json.dumps(rec) + "\n")
                n_windows += 1
                n_edges += len(chunk)
                n_pos += sum(int(g["label"]) for g in chunk)
                n_weighted += sum(1 for g in chunk if float(g.get("sample_weight", 0.0)) > 0.0)

    print(
        f"Wrote {out_path} windows={n_windows} edges={n_edges} positives={n_pos} "
        f"train_weighted_edges={n_weighted} negatives={n_weighted - n_pos} "
        f"candidate_targets={n_candidates_before_filter} skipped_global_positive={skipped_global_positive} "
        f"buckets={dict(sorted(bucket_counts.items()))} classes={dict(sorted(class_counts.items()))}"
    )


if __name__ == "__main__":
    main()
