from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from grn_agent.schemas import CandidateEdge, CellContext, FeatureBundle

if TYPE_CHECKING:
    from grn_agent.agents.multimodal_loader import MultimodalFeatureLoader


def generate_candidates(
    ctx: CellContext,
    features_by_pair: dict[tuple[str, str], FeatureBundle],
    min_pearson: float = 0.05,
    require_same_module: bool = False,
    max_edges_per_tf: int = 20,
) -> list[CandidateEdge]:
    """Rule-based candidate TF→target edges (SDD §3.6)."""
    cands: list[CandidateEdge] = []
    for tf in ctx.candidate_tfs:
        scored: list[tuple[str, float]] = []
        for g in ctx.module_genes:
            if g == tf:
                continue
            fb = features_by_pair.get((tf, g))
            if fb is None:
                continue
            r = fb.network.pearson_r
            if r is None:
                continue
            if abs(r) < min_pearson:
                continue
            if require_same_module and not (fb.network.in_same_module):
                continue
            scored.append((g, abs(r)))
        scored.sort(key=lambda x: -x[1])
        for tgt, _ in scored[:max_edges_per_tf]:
            cands.append(CandidateEdge(source_tf=tf, target_gene=tgt, context_id=ctx.context_id))
    return cands


def quick_candidates_from_module(ctx: CellContext, max_per_tf: int = 15) -> list[CandidateEdge]:
    """Fallback when features are not precomputed: all TF×gene in module."""
    out: list[CandidateEdge] = []
    for tf in ctx.candidate_tfs:
        n = 0
        for g in ctx.module_genes:
            if g == tf:
                continue
            out.append(CandidateEdge(source_tf=tf, target_gene=g, context_id=ctx.context_id))
            n += 1
            if n >= max_per_tf:
                break
    return out


def generate_tf_neighborhood_candidates(
    expression: np.ndarray,
    gene_symbols: list[str],
    ctx: CellContext,
    *,
    multimodal_loader: "MultimodalFeatureLoader | None" = None,
    train_mask: np.ndarray | None = None,
    topk_corr: int = 200,
    topk_prior: int = 100,
    corr_threshold: float = 0.05,
    rescue_motif: bool = True,
    rescue_accessibility: bool = True,
    rescue_prior: bool = True,
    rescue_max_per_tf: int = 100,
    max_edges_per_tf: int = 50,
    reranker_model_path: str | None = None,
    device: str | None = None,
) -> list[CandidateEdge]:
    """
    Scalable candidate generation:
      1) TF-centered co-expression neighborhood
      2) motif/accessibility/prior rescue union
      3) optional lightweight supervised reranker
      4) top-K per TF/context
    """
    gene_u = [str(g).strip().upper() for g in gene_symbols]
    gene_to_idx = {g: i for i, g in enumerate(gene_u)}
    module_genes = [str(g).strip().upper() for g in ctx.module_genes if str(g).strip()]
    module_set = set(module_genes)
    module_idxs = np.asarray([gene_to_idx[g] for g in module_genes if g in gene_to_idx], dtype=np.int64)
    idx = ctx.cell_indices
    sub = expression[idx, :].astype(np.float64)
    if sub.shape[0] < 2 or sub.shape[1] == 0:
        return quick_candidates_from_module(ctx, max_per_tf=max_edges_per_tf)

    # Z-score columns once and use matrix products for all TF-neighborhood correlations.
    means = sub.mean(axis=0, keepdims=True)
    stds = sub.std(axis=0, keepdims=True)
    z = (sub - means) / (stds + 1e-8)
    denom = max(float(z.shape[0] - 1), 1.0)

    # Prior-neighborhood uses train-split rows if available; fallback to context rows.
    if train_mask is not None and len(train_mask) == expression.shape[0]:
        train_idx = [i for i in idx if bool(train_mask[int(i)])]
    else:
        train_idx = []
    prior_sub = expression[train_idx, :].astype(np.float64) if len(train_idx) >= 2 else sub
    prior_means = prior_sub.mean(axis=0, keepdims=True)
    prior_stds = prior_sub.std(axis=0, keepdims=True)
    prior_z = (prior_sub - prior_means) / (prior_stds + 1e-8)
    prior_denom = max(float(prior_z.shape[0] - 1), 1.0)

    accessible = multimodal_loader.accessible_genes() if (multimodal_loader and rescue_accessibility) else set()
    model = _load_reranker(reranker_model_path)
    torch_state = _maybe_torch_state(device, z, prior_z)

    out: list[CandidateEdge] = []
    for tf in [str(t).strip().upper() for t in ctx.candidate_tfs if str(t).strip()]:
        ti = gene_to_idx.get(tf)
        if ti is None:
            continue
        corr = _corr_vector(z, ti, denom, torch_state=torch_state, which="ctx")
        scored: dict[str, dict[str, float]] = {}

        # Co-expression neighborhood.
        if module_idxs.size:
            order = module_idxs[np.argsort(-np.abs(corr[module_idxs]))]
        else:
            order = np.argsort(-np.abs(corr))
        n_corr = 0
        for gi in order:
            g = gene_u[int(gi)]
            if g == tf or g not in module_set:
                continue
            r = float(corr[int(gi)])
            if abs(r) < corr_threshold:
                continue
            scored[g] = {"corr": r, "motif": 0.0, "accessibility": 0.0, "rescue": 0.0}
            n_corr += 1
            if n_corr >= topk_corr:
                break

        # Motif/accessibility rescue: retain mechanistic candidates even if correlation is weak.
        if multimodal_loader and rescue_motif:
            for g in sorted(multimodal_loader.motif_targets_for_tf(tf)):
                if g == tf or g not in module_set:
                    continue
                gi = gene_to_idx.get(g)
                if gi is None:
                    continue
                row = scored.setdefault(
                    g,
                    {"corr": float(corr[gi]), "motif": 0.0, "accessibility": 0.0, "rescue": 1.0},
                )
                row["motif"] = 1.0

        if multimodal_loader and rescue_accessibility:
            rescued = 0
            for g in sorted(accessible & module_set):
                if g == tf:
                    continue
                gi = gene_to_idx.get(g)
                if gi is None:
                    continue
                row = scored.setdefault(
                    g,
                    {"corr": float(corr[gi]), "motif": 0.0, "accessibility": 0.0, "rescue": 1.0},
                )
                row["accessibility"] = 1.0
                rescued += 1
                if rescued >= rescue_max_per_tf:
                    break

        # Prior-neighborhood rescue (train-split proxy prior ranking).
        if rescue_prior and topk_prior > 0:
            prior_corr = _corr_vector(prior_z, ti, prior_denom, torch_state=torch_state, which="prior")
            if module_idxs.size:
                order_prior = module_idxs[np.argsort(-np.abs(prior_corr[module_idxs]))]
            else:
                order_prior = np.argsort(-np.abs(prior_corr))
            n_prior = 0
            for gi in order_prior:
                g = gene_u[int(gi)]
                if g == tf or g not in module_set:
                    continue
                prior_score = float(np.clip(abs(float(prior_corr[int(gi)])), 0.0, 1.0))
                row = scored.setdefault(
                    g,
                    {
                        "corr": float(corr[int(gi)]),
                        "motif": 0.0,
                        "accessibility": 0.0,
                        "prior": 0.0,
                        "rescue": 1.0,
                    },
                )
                row["prior"] = max(float(row.get("prior", 0.0)), prior_score)
                n_prior += 1
                if n_prior >= topk_prior:
                    break

        ranked = []
        for g, row in scored.items():
            x = np.array([[
                abs(row["corr"]),
                row["corr"],
                row["motif"],
                row["accessibility"],
                row.get("prior", 0.0),
                0.0,  # in_same_module unavailable until full features are extracted
                0.0,  # shared-neighbor support unavailable until full features are extracted
            ]])
            if model is not None and hasattr(model, "predict_proba"):
                try:
                    score = float(model.predict_proba(x)[0, 1])
                except Exception:
                    score = _heuristic_rerank_score(row)
            else:
                score = _heuristic_rerank_score(row)
            ranked.append((g, score))
        ranked.sort(key=lambda x: -x[1])
        for g, _ in ranked[:max_edges_per_tf]:
            out.append(CandidateEdge(source_tf=tf, target_gene=g, context_id=ctx.context_id))
    return out


def _heuristic_rerank_score(row: dict[str, float]) -> float:
    return (
        0.55 * abs(float(row.get("corr", 0.0)))
        + 0.20 * float(row.get("motif", 0.0))
        + 0.10 * float(row.get("accessibility", 0.0))
        + 0.10 * float(row.get("prior", 0.0))
        + 0.05 * float(row.get("rescue", 0.0))
    )


def _load_reranker(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    with p.open("rb") as fp:
        return pickle.load(fp)


def _corr_vector(
    z: np.ndarray,
    ti: int,
    denom: float,
    *,
    torch_state: dict[str, object] | None,
    which: str,
) -> np.ndarray:
    if torch_state is None:
        return (z[:, ti].T @ z) / denom
    try:
        z_t = torch_state["z_t"] if which == "ctx" else torch_state["prior_z_t"]
        corr_t = (z_t[:, int(ti)].T @ z_t) / denom
        return corr_t.detach().cpu().numpy()
    except Exception:
        return (z[:, ti].T @ z) / denom


def _maybe_torch_state(device: str | None, z: np.ndarray, prior_z: np.ndarray) -> dict[str, object] | None:
    dev = str(device or "").strip().lower()
    if dev not in {"cuda", "gpu", "torch_cuda"} and not dev.startswith("cuda:"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        cuda_dev = torch.device(dev if dev.startswith("cuda:") else "cuda")
        return {
            "z_t": torch.as_tensor(z, dtype=torch.float64, device=cuda_dev),
            "prior_z_t": torch.as_tensor(prior_z, dtype=torch.float64, device=cuda_dev),
        }
    except Exception:
        return None
