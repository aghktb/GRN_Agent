"""
End-to-end GRNAgent pipeline (dry-run and real runs).

Usage: python -m grn_agent.pipeline.run --config conf/dry_run.yml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from grn_agent.agents import (
    calibrate,
    candidates,
    cell_context,
    decode,
    evidence_graph,
    export,
    harmonize,
    ingest,
    literature,
    priors,
    score_eager,
)
from grn_agent.agents.features import extract_features_for_edge
from grn_agent.agents.multimodal_loader import MultimodalFeatureLoader
from grn_agent.io.artifact_store import save_json
from grn_agent.schemas import EvidenceGraph, PriorBundle, RunManifest, ScoredEdge
from grn_agent.schemas.export_schema import export_json_schemas

from .config import load_yaml_config, parse_eval_track


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:12]
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_evidence_device(cfg: dict[str, Any], cand_cfg: dict[str, Any], sc_cfg: dict[str, Any]) -> str | None:
    ev_cfg = cfg.get("evidence", {})
    requested = ev_cfg.get("device") or cand_cfg.get("device") or cfg.get("device")
    if requested is None and str(sc_cfg.get("device", "")).strip().lower().startswith("cuda"):
        requested = sc_cfg.get("device")
    if requested is None:
        return None
    dev = str(requested).strip().lower()
    if dev in {"auto", ""}:
        return "cuda" if _cuda_available() else "cpu"
    if dev in {"gpu", "torch_cuda"}:
        return "cuda"
    return dev


def _evidence_backend_label(device: str | None) -> str:
    dev = str(device or "").strip().lower()
    if (dev == "cuda" or dev.startswith("cuda:")) and _cuda_available():
        return f"torch({dev})"
    if dev == "cuda" or dev.startswith("cuda:"):
        return "numpy(cpu; cuda unavailable)"
    return "numpy(cpu)"


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[pipeline] {message}", flush=True)


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _pick_col(df: Any, *aliases: str, required: bool = True) -> str | None:
    cmap = {_normalize_col(c): c for c in df.columns}
    for name in aliases:
        key = _normalize_col(name)
        if key in cmap:
            return cmap[key]
    if required:
        raise ValueError(f"Missing required column one of {aliases}; got {list(df.columns)}")
    return None


def _load_inference_filter(cfg: dict[str, Any]) -> dict[str, Any] | None:
    raw = cfg.get("inference_filter") or cfg.get("split_filter") or {}
    if not isinstance(raw, dict) or not raw:
        return None
    path = str(raw.get("split_manifest", "")).strip()
    strategy = str(raw.get("strategy", "leave_one_tf_out")).strip()
    fold_id = str(raw.get("fold_id", "")).strip()
    subset = str(raw.get("subset", "test")).strip()
    if not path:
        return None
    if not fold_id:
        raise ValueError("inference_filter.fold_id is required when inference_filter.split_manifest is set")
    import pandas as _pd

    p = Path(path).expanduser()
    sep = "\t" if p.suffix.lower() in (".tsv", ".tab") else ","
    header = _pd.read_csv(p, sep=sep, nrows=0)
    c_split = _pick_col(header, "split_name", "strategy")
    c_fold = _pick_col(header, "fold_id", "fold")
    c_subset = _pick_col(header, "subset")
    c_tf = _pick_col(header, "source_tf", "tf", "source")
    c_tg = _pick_col(header, "target_gene", "target", "gene")
    df = _pd.read_csv(p, sep=sep, dtype=str, usecols=[c_split, c_fold, c_subset, c_tf, c_tg])
    mask = (
        df[c_split].astype(str).str.strip().eq(strategy)
        & df[c_fold].astype(str).str.strip().eq(fold_id)
        & df[c_subset].astype(str).str.strip().eq(subset)
    )
    sub_df = df.loc[mask]
    pairs = {
        (str(r[c_tf]).strip().upper(), str(r[c_tg]).strip().upper())
        for _, r in sub_df.iterrows()
        if str(r[c_tf]).strip() and str(r[c_tg]).strip()
    }
    if not pairs:
        raise ValueError(
            f"inference_filter matched no pairs: split_manifest={p}, strategy={strategy}, "
            f"fold_id={fold_id}, subset={subset}"
        )
    return {
        "split_manifest": str(p),
        "strategy": strategy,
        "fold_id": fold_id,
        "subset": subset,
        "target_universe": str(raw.get("target_universe", "all_genes")).strip().lower(),
        "pairs": pairs,
        "tfs": {tf for tf, _ in pairs},
        "targets": {g for _, g in pairs},
    }


def run_pipeline(config_path: str | Path) -> Path:
    cfg = load_yaml_config(config_path)
    verbose = bool(cfg.get("verbose", False))
    ev_cfg = cfg.get("evidence", {})
    progress = bool(cfg.get("progress", cfg.get("log_progress", True)))
    if isinstance(ev_cfg, dict) and "log_progress" in ev_cfg:
        progress = bool(ev_cfg.get("log_progress"))
    seed = int(cfg.get("seed", 0))
    run_id = str(cfg.get("run_id", uuid.uuid4().hex[:8]))
    artifact_root = Path(str(cfg.get("artifact_root", "artifacts")))
    out_dir = artifact_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    wb = None
    wb_cfg = cfg.get("wandb", {})
    if bool(wb_cfg.get("enabled", False)):
        try:
            import wandb  # type: ignore

            wb = wandb.init(
                project=str(wb_cfg.get("project", "grn-agent-eager")),
                name=(str(wb_cfg.get("run_name", "")).strip() or run_id),
                config={
                    "run_id": run_id,
                    "eval_track": str(cfg.get("eval_track", "track1_no_literature")),
                    "config_path": str(config_path),
                },
            )
        except Exception as exc:
            if verbose:
                print(f"[pipeline] wandb disabled ({exc})", flush=True)
            wb = None

    export_schemas_to = cfg.get("export_json_schemas_to")
    if export_schemas_to:
        export_json_schemas(Path(export_schemas_to))

    eval_track = parse_eval_track(str(cfg.get("eval_track", "track1_no_literature")))
    disable_priors = bool(cfg.get("disable_priors", False))
    if eval_track.value == "track1_no_literature":
        disable_priors = True
    lit_cutoff_year = cfg.get("literature", {}).get("time_cutoff_year", 2020)
    split_id = str(cfg.get("split_id", "default"))
    train_frac = float(cfg.get("train", {}).get("frac", 0.8))

    ds_cfg = cfg.get("dataset", {})
    mode = str(ds_cfg.get("mode", "synthetic"))
    n_cells = int(ds_cfg.get("n_cells", 100))
    n_genes = int(ds_cfg.get("n_genes", 50))
    rng = np.random.default_rng(seed)
    gene_symbols = [(str(ds_cfg.get("gene_prefix", "G")) + str(i)).upper() for i in range(n_genes)]
    tf_list = [str(t).strip().upper() for t in list(ds_cfg.get("tf_list", gene_symbols[:5]))]

    if mode == "synthetic":
        if verbose:
            print("[pipeline] ingest: synthetic dataset", flush=True)
        dataset, expression = ingest.ingest_from_synthetic(
            dataset_id=str(ds_cfg.get("dataset_id", "DS_SYN")),
            species=str(ds_cfg.get("species", "human")),
            n_cells=n_cells,
            n_genes=n_genes,
            gene_symbols=gene_symbols,
            seed=seed,
            modalities=list(ds_cfg.get("modalities", ["scrna"])),
        )
    elif mode == "npy":
        if verbose:
            print(f"[pipeline] ingest: npy from {ds_cfg['expression_path']}", flush=True)
        path = str(ds_cfg["expression_path"])
        dataset, expression = ingest.ingest_from_npy(
            dataset_id=str(ds_cfg.get("dataset_id", "DS_NPY")),
            species=str(ds_cfg.get("species", "human")),
            path=path,
            gene_symbols=list(ds_cfg["gene_symbols"]),
            modalities=list(ds_cfg.get("modalities", ["scrna"])),
        )
        tf_list = [str(t).strip().upper() for t in list(ds_cfg.get("tf_list", gene_symbols[:5]))]
    elif mode == "beeline_csv":
        if verbose:
            print(f"[pipeline] ingest: beeline_csv from {ds_cfg['expression_path']}", flush=True)
        import pandas as _pd

        csv_path = str(ds_cfg["expression_path"])
        dataset, expression, gene_symbols = ingest.ingest_from_beeline_csv(
            dataset_id=str(ds_cfg.get("dataset_id", "DS_BEELINE")),
            species=str(ds_cfg.get("species", "mouse")),
            path=csv_path,
            modalities=list(ds_cfg.get("modalities", ["scrna"])),
        )
        # Load TF list: can be explicit list, path to TF file, or auto-derive from gene_symbols
        tf_cfg = ds_cfg.get("tf_list")
        tf_file = ds_cfg.get("tf_file")
        if tf_file and Path(tf_file).is_file():
            tf_df = _pd.read_csv(tf_file)
            col = tf_df.columns[0]
            all_tfs = {str(t).strip().upper() for t in tf_df[col].tolist()}
            tf_list = [str(g).strip().upper() for g in gene_symbols if str(g).strip().upper() in all_tfs]
        elif tf_cfg:
            tf_list = [str(t).strip().upper() for t in list(tf_cfg)]
        else:
            tf_list = [str(g).strip().upper() for g in gene_symbols[:5]]
    else:
        raise ValueError(f"Unknown dataset.mode: {mode}")

    inference_filter = _load_inference_filter(cfg)
    if inference_filter is not None:
        subset_tfs = set(inference_filter["tfs"])
        tf_list = [tf for tf in tf_list if tf in subset_tfs]
        if not tf_list:
            raise ValueError(
                "inference_filter selected TFs that are absent from the dataset TF list: "
                f"selected={len(subset_tfs)}"
            )
        _progress(
            progress or verbose,
            f"inference_filter: subset={inference_filter['subset']} fold={inference_filter['fold_id']} "
            f"heldout_tfs={len(tf_list)} manifest_pairs={len(inference_filter['pairs'])} "
            f"target_universe={inference_filter['target_universe']}",
        )

    manifest = RunManifest(
        run_id=run_id,
        dataset_id=dataset.dataset_id,
        split_id=split_id,
        eval_track=eval_track,
        git_sha=_git_sha(),
        dependency_versions=deps_placeholder(),
        seed=seed,
        artifact_dir=str(out_dir.resolve()),
    )

    dataset = harmonize.harmonize_genes(dataset, dataset.species)
    if verbose:
        print(
            f"[pipeline] harmonize: {len(gene_symbols)} genes, {expression.shape[0]} cells, {len(tf_list)} candidate TFs",
            flush=True,
        )
    save_json(out_dir / "01_dataset.json", dataset)
    np.save(out_dir / "01_expression.npy", expression)
    manifest.stages_completed.append("ingest")
    manifest.stages_completed.append("harmonize")

    use_scanpy = bool(cfg.get("cell_context", {}).get("use_scanpy", False))
    configured_cell_type = cfg.get("cell_context", {}).get("cell_type")
    configured_cell_type = str(configured_cell_type).strip() if configured_cell_type else None
    if use_scanpy:
        contexts = cell_context.try_contexts_from_scanpy(
            dataset,
            expression,
            gene_symbols,
            tf_list,
            default_cell_type=configured_cell_type,
        )
    else:
        contexts = cell_context.default_contexts_from_dataset(
            dataset,
            expression,
            gene_symbols,
            tf_list,
            default_cell_type=configured_cell_type,
            seed=seed,
        )
    if configured_cell_type:
        for c in contexts:
            if c.cell_type in (None, "", "unknown"):
                c.cell_type = configured_cell_type
    if inference_filter is not None:
        gene_set = {str(g).strip().upper() for g in gene_symbols}
        subset_tfs = set(inference_filter["tfs"])
        target_universe = str(inference_filter["target_universe"])
        if target_universe in {"all", "all_genes", "expression_genes"}:
            filter_module = [str(g).strip().upper() for g in gene_symbols]
        elif target_universe in {"split_targets", "manifest_targets", "positive_targets"}:
            keep = (set(inference_filter["targets"]) | subset_tfs) & gene_set
            filter_module = [g for g in gene_symbols if str(g).strip().upper() in keep]
        elif target_universe in {"context", "context_module"}:
            filter_module = []
        elif target_universe in {"split_pairs", "manifest_pairs"}:
            keep = (set(inference_filter["targets"]) | subset_tfs) & gene_set
            filter_module = [g for g in gene_symbols if str(g).strip().upper() in keep]
        else:
            raise ValueError(
                "inference_filter.target_universe must be one of all_genes, split_targets, "
                "split_pairs, or context"
            )
        filtered_contexts = []
        for c in contexts:
            c.candidate_tfs = [tf for tf in c.candidate_tfs if str(tf).strip().upper() in subset_tfs]
            if target_universe not in {"context", "context_module"}:
                c.module_genes = filter_module
            if c.candidate_tfs:
                filtered_contexts.append(c)
        contexts = filtered_contexts
        if not contexts:
            raise ValueError("inference_filter removed all contexts; no heldout TFs remain for inference")
        manifest.extra["inference_filter"] = {
            "split_manifest": inference_filter["split_manifest"],
            "strategy": inference_filter["strategy"],
            "fold_id": inference_filter["fold_id"],
            "subset": inference_filter["subset"],
            "target_universe": target_universe,
            "n_manifest_pairs": len(inference_filter["pairs"]),
            "n_heldout_tfs": len(subset_tfs),
            "n_target_genes": len(filter_module) if filter_module else None,
        }
    if verbose:
        print(f"[pipeline] cell_context: built {len(contexts)} context(s)", flush=True)
    save_json(out_dir / "02_contexts.json", {"contexts": [c.model_dump() for c in contexts]})
    manifest.stages_completed.append("cell_context")

    n = expression.shape[0]
    train_mask = np.zeros(n, dtype=bool)
    train_idx = rng.permutation(n)[: int(n * train_frac)]
    train_mask[train_idx] = True
    np.save(out_dir / "train_mask.npy", train_mask)

    stop_after = str(cfg.get("stop_after", "full")).strip().lower()
    if stop_after not in frozenset({"full", "evidence_graphs"}):
        raise ValueError("stop_after must be 'full' or 'evidence_graphs'")

    all_edges: list[ScoredEdge] = []
    evidence_graphs: list[EvidenceGraph] = []

    cand_cfg = cfg.get("candidates", {})
    cand_mode = str(cand_cfg.get("mode", "full_features")).strip().lower()
    min_pearson = float(cand_cfg.get("min_pearson", 0.01))
    max_per_tf = int(cand_cfg.get("max_edges_per_tf", 25))

    sc_cfg = cfg.get("scoring", {})
    eager_ckpt = sc_cfg.get("checkpoint") or cfg.get("eager_checkpoint")
    if eager_ckpt:
        eager_ckpt = str(Path(eager_ckpt).expanduser())
        if not Path(eager_ckpt).is_file():
            eager_ckpt = None
    else:
        eager_ckpt = None
    eager_device = sc_cfg.get("device")
    evidence_device = _resolve_evidence_device(cfg, cand_cfg, sc_cfg)
    _progress(
        progress or verbose,
        f"evidence setup: numeric_backend={_evidence_backend_label(evidence_device)}, "
        f"graph_assembly=python(cpu), scoring_device={eager_device or 'auto'}",
    )

    temp = float(cfg.get("calibration", {}).get("temperature", 1.0))
    d_cfg = cfg.get("decode", {})
    max_k = int(d_cfg.get("max_regulators_per_target", 5))
    min_conf = float(d_cfg.get("min_confidence", 0.0))
    eager_model = None
    if stop_after == "full":
        if eager_ckpt is None:
            raise ValueError(
                "EAGER scoring requires scoring.checkpoint (or eager_checkpoint) pointing to a .pt file. "
                "Generate one with scripts/train_eager.py or grn_agent.models.eager.save_minimal_eager_for_tests"
            )
        import torch

        from grn_agent.models.eager.checkpoint import load_eager_checkpoint

        resolved_eager_device = str(eager_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if resolved_eager_device.startswith("cuda") and not torch.cuda.is_available():
            if verbose:
                print("[pipeline] scoring_device=cuda requested but CUDA is unavailable; falling back to cpu", flush=True)
            resolved_eager_device = "cpu"
        eager_model = load_eager_checkpoint(eager_ckpt, map_location=resolved_eager_device).to(resolved_eager_device)
        eager_model.eval()
        if verbose:
            print(f"[pipeline] EAGER checkpoint loaded once on {resolved_eager_device}", flush=True)

    # ── Multimodal feature loader (motif + ATAC from acquisition manifest) ────
    _mm_loader: MultimodalFeatureLoader | None = None
    mm_manifest_path = cfg.get("multimodal_manifest")
    if mm_manifest_path:
        from pathlib import Path as _Path
        _mm_loader = MultimodalFeatureLoader(str(_Path(mm_manifest_path).expanduser()))
        _mm_loader.load()
        qc = _mm_loader.qc_summary()
        if verbose:
            print(
                f"[pipeline] multimodal: {qc['motif_pairs_loaded']} motif pairs "
                f"({qc['motif_pairs_with_hit']} with hit), "
                f"{qc['atac_genes_loaded']} ATAC gene profiles",
                flush=True,
            )
    else:
        if verbose:
            print("[pipeline] multimodal: no manifest configured — motif/ATAC features disabled", flush=True)

    # Warm the ortholog cache once for ALL unique gene symbols before the per-edge loop.
    # This avoids N*M individual API calls and batches them through mygene.io.
    _use_orthologs = bool(cfg.get("use_ortholog_lookup", True))
    if _use_orthologs:
        try:
            from grn_agent.agents.ortholog_client import prefetch_orthologs
            _all_tfs = list({tf for ctx in contexts for tf in ctx.candidate_tfs})
            _all_genes = list({g for ctx in contexts for g in ctx.module_genes})
            _unique_symbols = list(set(_all_tfs + _all_genes))
            _species = dataset.species or "mouse"
            if verbose:
                print(f"[pipeline] orthologs: prefetching for {len(_unique_symbols)} unique symbols (species={_species})", flush=True)
            prefetch_orthologs(_unique_symbols, source_species=_species)
        except Exception as _oe:
            if verbose:
                print(f"[pipeline] orthologs: prefetch skipped ({_oe})", flush=True)
            _use_orthologs = False

    gene_to_idx = {str(g).strip().upper(): i for i, g in enumerate(gene_symbols)}
    global_mean = expression.mean(axis=0)
    global_std = expression.std(axis=0)
    evidence_t0 = time.perf_counter()
    _progress(
        progress,
        f"evidence_graphs: start contexts={len(contexts)}, mode={cand_mode}, "
        f"min_pearson={min_pearson}, max_edges_per_tf={max_per_tf}",
    )

    for ctx_i, ctx in enumerate(contexts, 1):
        ctx_t0 = time.perf_counter()
        _progress(
            progress,
            f"context {ctx_i}/{len(contexts)} {ctx.context_id}: start "
            f"cells={len(ctx.cell_indices)}, module_genes={len(ctx.module_genes)}, "
            f"candidate_tfs={len(ctx.candidate_tfs)}",
        )
        ctx_idx = np.asarray(ctx.cell_indices, dtype=np.int64)
        sub = expression[ctx_idx, :].astype(np.float64)
        sub_means = sub.mean(axis=0, keepdims=True)
        sub_stds = sub.std(axis=0, keepdims=True)
        z_sub = (sub - sub_means) / (sub_stds + 1e-8)
        feature_precomputed = {
            "sub": sub,
            "z_sub": z_sub,
            "denom": max(float(z_sub.shape[0] - 1), 1.0),
            "gene_to_idx": gene_to_idx,
            "global_mean": global_mean,
            "global_std": global_std,
            "ctx_mean": sub.mean(axis=0),
            "ctx_dropout": (sub == 0).mean(axis=0),
            "module_set": {str(g).strip().upper() for g in ctx.module_genes},
        }
        feats_map: dict[tuple[str, str], Any] = {}
        cand_t0 = time.perf_counter()
        _progress(progress, f"context {ctx.context_id}: candidate generation start")
        if cand_mode in {"tf_neighborhood", "tf_neighborhood_rerank", "neighborhood"}:
            cands = candidates.generate_tf_neighborhood_candidates(
                expression,
                gene_symbols,
                ctx,
                multimodal_loader=_mm_loader,
                train_mask=train_mask,
                topk_corr=int(cand_cfg.get("topk_corr", 200)),
                topk_prior=int(cand_cfg.get("topk_prior", 100)),
                corr_threshold=float(cand_cfg.get("corr_threshold", min_pearson)),
                rescue_motif=bool(cand_cfg.get("rescue_motif", True)),
                rescue_accessibility=bool(cand_cfg.get("rescue_accessibility", True)),
                rescue_prior=bool(cand_cfg.get("rescue_prior", True)),
                rescue_max_per_tf=int(cand_cfg.get("rescue_max_per_tf", 100)),
                max_edges_per_tf=max_per_tf,
                reranker_model_path=(str(cand_cfg.get("reranker_model_path")) if cand_cfg.get("reranker_model_path") else None),
                device=evidence_device,
            )
        else:
            for tf in ctx.candidate_tfs:
                for g in ctx.module_genes:
                    if g == tf:
                        continue
                    f = extract_features_for_edge(
                        expression, gene_symbols, ctx, tf, g,
                        use_ortholog_lookup=_use_orthologs,
                        multimodal_loader=_mm_loader,
                        precomputed=feature_precomputed,
                    )
                    feats_map[(tf, g)] = f
            cands = candidates.generate_candidates(ctx, feats_map, min_pearson=min_pearson, max_edges_per_tf=max_per_tf)
        if not cands:
            cands = candidates.quick_candidates_from_module(ctx, max_per_tf=max_per_tf)
        if inference_filter is not None and str(inference_filter["target_universe"]) in {"split_pairs", "manifest_pairs"}:
            allowed_pairs = set(inference_filter["pairs"])
            cands = [
                ce for ce in cands
                if (str(ce.source_tf).strip().upper(), str(ce.target_gene).strip().upper()) in allowed_pairs
            ]
        _progress(
            progress or verbose,
            f"context {ctx.context_id}: candidate generation done edges={len(cands)} "
            f"elapsed={time.perf_counter() - cand_t0:.2f}s",
        )
        if disable_priors:
            _progress(progress, f"context {ctx.context_id}: priors skipped")
            prior_map: dict[tuple[str, str], PriorBundle] = {}
        else:
            prior_pairs = [(ce.source_tf, ce.target_gene) for ce in cands]
            prior_t0 = time.perf_counter()
            _progress(
                progress,
                f"context {ctx.context_id}: prior batch start edges={len(prior_pairs)} "
                f"backend={_evidence_backend_label(evidence_device)}",
            )
            prior_map = priors.compute_priors_for_pairs(
                expression,
                train_mask,
                gene_symbols,
                prior_pairs,
                split_id=split_id,
                seed=seed,
                device=evidence_device,
            )
            _progress(
                progress or verbose,
                f"context {ctx.context_id}: prior batch done bundles={len(prior_map)} "
                f"elapsed={time.perf_counter() - prior_t0:.2f}s",
            )
        ctx_egs: list[EvidenceGraph] = []
        graph_t0 = time.perf_counter()
        _progress(progress, f"context {ctx.context_id}: evidence graph assembly start edges={len(cands)}")
        for n_built, ce in enumerate(cands, 1):
            f = feats_map.get((ce.source_tf, ce.target_gene))
            if f is None:
                f = extract_features_for_edge(
                    expression, gene_symbols, ctx, ce.source_tf, ce.target_gene,
                    use_ortholog_lookup=_use_orthologs,
                    multimodal_loader=_mm_loader,
                    precomputed=feature_precomputed,
                )
            if disable_priors:
                pr = PriorBundle(ensemble_prior=0.0)
            else:
                pr = prior_map.get(
                    (str(ce.source_tf).strip().upper(), str(ce.target_gene).strip().upper()),
                    PriorBundle(ensemble_prior=0.0),
                )
            lit = literature.literature_features_for_track(
                eval_track,
                ce.source_tf,
                ce.target_gene,
                time_cutoff_year=(int(lit_cutoff_year) if lit_cutoff_year is not None else None),
            )
            eg = evidence_graph.build_evidence_graph(ctx, ce, f, pr, eval_track, literature_payload=lit)
            ctx_egs.append(eg)
            if (progress or verbose) and (n_built % 500 == 0 or n_built == len(cands)):
                _progress(
                    True,
                    f"context {ctx.context_id}: evidence graph assembly "
                    f"{n_built}/{len(cands)} elapsed={time.perf_counter() - graph_t0:.2f}s",
                )

        evidence_graphs.extend(ctx_egs)
        _progress(
            progress,
            f"context {ctx.context_id}: evidence graph assembly done graphs={len(ctx_egs)} "
            f"context_elapsed={time.perf_counter() - ctx_t0:.2f}s",
        )

        if stop_after == "full":
            score_t0 = time.perf_counter()
            _progress(progress or verbose, f"context {ctx.context_id}: EAGER scoring start edges={len(ctx_egs)}")
            for n_scored, eg in enumerate(ctx_egs, 1):
                se = score_eager.score_evidence_graph(
                    eg,
                    eval_track,
                    model=eager_model,
                )
                all_edges.append(
                    se
                )
                if (progress or verbose) and (n_scored % 500 == 0 or n_scored == len(ctx_egs)):
                    _progress(
                        True,
                        f"context {ctx.context_id}: EAGER scoring {n_scored}/{len(ctx_egs)} "
                        f"elapsed={time.perf_counter() - score_t0:.2f}s",
                    )
            if wb is not None and ctx_egs:
                ctx_scores = [e.p_present for e in all_edges[-len(ctx_egs) :]]
                wb.log(
                    {
                        "inference/context_edges": len(ctx_egs),
                        "inference/context_mean_p_present": float(np.mean(ctx_scores)),
                        "inference/context_max_p_present": float(np.max(ctx_scores)),
                        "inference/context_min_p_present": float(np.min(ctx_scores)),
                    }
                )

    # write jsonl
    eg_path = out_dir / "evidence_graphs.jsonl"
    write_t0 = time.perf_counter()
    _progress(progress, f"evidence_graphs: writing {len(evidence_graphs)} rows to {eg_path}")
    with eg_path.open("w", encoding="utf-8") as fp:
        for eg in evidence_graphs:
            fp.write(json.dumps(eg.model_dump(mode="json")) + "\n")
    _progress(
        progress or verbose,
        f"evidence_graphs: wrote {len(evidence_graphs)} rows to {eg_path} "
        f"write_elapsed={time.perf_counter() - write_t0:.2f}s total_elapsed={time.perf_counter() - evidence_t0:.2f}s",
    )

    manifest.stages_completed.extend(["features", "priors", "candidates", "evidence_graphs"])
    if stop_after == "evidence_graphs":
        manifest.stages_completed.append("stop_after_evidence_graphs")
        save_json(out_dir / "run_manifest.json", manifest)
        if wb is not None:
            wb.log(
                {
                    "evidence/num_graphs": len(evidence_graphs),
                    "artifacts/output_dir": str(out_dir),
                }
            )
            wb.finish()
        if verbose:
            print("[pipeline] stop_after=evidence_graphs complete", flush=True)
        return out_dir

    manifest.stages_completed.append("score")
    calibrated = calibrate.calibrate_edges_temperature(all_edges, temperature=temp)
    manifest.stages_completed.append("calibrate")

    export.export_scored_edges_csv(calibrated, out_dir / "exports" / "scored_edges.csv")

    ctx0 = contexts[0].context_id if contexts else "global"
    net = decode.decode_grn(ctx0, calibrated, max_regulators_per_target=max_k, min_confidence=min_conf)
    if dataset.species:
        net.graph_metadata.species = dataset.species
    if contexts:
        net.graph_metadata.cell_type = contexts[0].cell_type
    manifest.stages_completed.append("decode")

    export.export_network_bundle(net, out_dir / "exports")
    save_json(out_dir / "run_manifest.json", manifest)
    manifest.stages_completed.append("export")
    if wb is not None:
        wb.log(
            {
                "inference/total_edges_scored": len(all_edges),
                "decode/final_edges": len(net.edges),
                "artifacts/output_dir": str(out_dir),
            }
        )
        wb.finish()
    if verbose:
        print(f"[pipeline] export: wrote outputs to {out_dir / 'exports'}", flush=True)

    return out_dir


def deps_placeholder() -> dict[str, str]:
    import sys

    return {"python": sys.version.split()[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description="GRNAgent pipeline runner")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    out = run_pipeline(args.config)
    print(f"Done. Artifacts: {out}")


if __name__ == "__main__":
    main()
