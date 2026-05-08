from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from grn_agent.schemas import (
    CellContext,
    ExpressionFeatures,
    FeatureBundle,
    NetworkFeatures,
    OrthologyFeatures,
)

if TYPE_CHECKING:
    from grn_agent.agents.multimodal_loader import MultimodalFeatureLoader

log = logging.getLogger(__name__)

_MAX_SHARED_NEIGHBORS = 20


def _gene_index(gene_symbols: list[str], g: str) -> int:
    g_u = str(g).strip().upper()
    try:
        return gene_symbols.index(g_u)
    except ValueError:
        for i, sym in enumerate(gene_symbols):
            if str(sym).strip().upper() == g_u:
                return i
        return -1


def _compute_shared_neighbors(
    z_sub: np.ndarray,
    gene_symbols: list[str],
    ti: int,
    gi: int,
    corr_vec_ti: np.ndarray | None = None,
    corr_vec_gi: np.ndarray | None = None,
    denom: float | None = None,
    corr_threshold: float = 0.3,
    max_neighbors: int = _MAX_SHARED_NEIGHBORS,
) -> tuple[int, list[str]]:
    """Genes significantly co-expressed with *both* TF and target in this context."""
    if z_sub.shape[0] < 2 or z_sub.shape[1] == 0:
        return 0, []
    if denom is None:
        denom = max(float(z_sub.shape[0] - 1), 1.0)
    if corr_vec_ti is None:
        corr_vec_ti = (z_sub[:, ti].T @ z_sub) / denom
    if corr_vec_gi is None:
        corr_vec_gi = (z_sub[:, gi].T @ z_sub) / denom
    mask = (np.abs(corr_vec_ti) >= corr_threshold) & (np.abs(corr_vec_gi) >= corr_threshold)
    mask[int(ti)] = False
    mask[int(gi)] = False
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return 0, []
    # Keep most jointly-supported neighbors first.
    joint = np.abs(corr_vec_ti[idxs]) * np.abs(corr_vec_gi[idxs])
    order = np.argsort(-joint)
    picked = idxs[order[:max_neighbors]]
    names = [gene_symbols[int(j)] for j in picked]
    return len(names), names


def extract_features_for_edge(
    expression: np.ndarray,
    gene_symbols: list[str],
    ctx: CellContext,
    source_tf: str,
    target_gene: str,
    use_ortholog_lookup: bool = True,
    multimodal_loader: "MultimodalFeatureLoader | None" = None,
    precomputed: dict[str, Any] | None = None,
) -> FeatureBundle:
    """
    Algorithm 8: expression + network features on context cells.

    Computes:
      expression: z_t, z_g, activity(t), mean_expr, dropout_rate
      network: corr(t,g), shared_neighbors, in_same_module, k_hop_distance
      go_context: stub (populated downstream if GO annotations available)
      orthology: stub (populated downstream if ortholog DB available)
    """
    source_tf = str(source_tf).strip().upper()
    target_gene = str(target_gene).strip().upper()
    if precomputed is not None:
        sub = precomputed["sub"]
        z_sub = precomputed["z_sub"]
        denom = float(precomputed["denom"])
        ti = int(precomputed["gene_to_idx"].get(source_tf, -1))
        gi = int(precomputed["gene_to_idx"].get(target_gene, -1))
        global_mean = precomputed["global_mean"]
        global_std = precomputed["global_std"]
        ctx_mean = precomputed["ctx_mean"]
        ctx_dropout = precomputed["ctx_dropout"]
        module_set = precomputed["module_set"]
    else:
        idx = ctx.cell_indices
        sub = expression[idx, :]
        means = sub.mean(axis=0, keepdims=True)
        stds = sub.std(axis=0, keepdims=True)
        z_sub = (sub - means) / (stds + 1e-8)
        denom = max(float(z_sub.shape[0] - 1), 1.0)
        ti = _gene_index(gene_symbols, source_tf)
        gi = _gene_index(gene_symbols, target_gene)
        global_mean = expression.mean(axis=0)
        global_std = expression.std(axis=0)
        ctx_mean = sub.mean(axis=0)
        ctx_dropout = (sub == 0).mean(axis=0)
        module_set = set(ctx.module_genes)

    expr_ft = ExpressionFeatures()
    net_ft = NetworkFeatures()

    if ti >= 0:
        expr_ft.tf_mean_expr = float(ctx_mean[ti])
        expr_ft.tf_zscore = float((expr_ft.tf_mean_expr - float(global_mean[ti])) / (float(global_std[ti]) + 1e-8))
        expr_ft.tf_dropout_rate = float(ctx_dropout[ti])

    if gi >= 0:
        expr_ft.target_mean_expr = float(ctx_mean[gi])
        expr_ft.target_zscore = float((expr_ft.target_mean_expr - float(global_mean[gi])) / (float(global_std[gi]) + 1e-8))
        expr_ft.target_dropout_rate = float(ctx_dropout[gi])

    if ti >= 0 and gi >= 0:
        corr_ti = (z_sub[:, ti].T @ z_sub) / denom
        corr_gi = (z_sub[:, gi].T @ z_sub) / denom
        r = float(corr_ti[gi])
        expr_ft.tf_activity_proxy = r
        net_ft.pearson_r = r

        net_ft.in_same_module = source_tf in module_set and target_gene in module_set
        net_ft.k_hop_distance = 1 if net_ft.in_same_module else 2

        n_shared, shared_names = _compute_shared_neighbors(
            z_sub,
            gene_symbols,
            ti,
            gi,
            corr_vec_ti=corr_ti,
            corr_vec_gi=corr_gi,
            denom=denom,
        )
        net_ft.shared_neighbors = n_shared
        net_ft.shared_neighbor_names = shared_names

    orthology_ft = OrthologyFeatures()
    if use_ortholog_lookup:
        species = ctx.metadata.get("species", "mouse")
        try:
            from grn_agent.agents.ortholog_client import get_ortholog_info
            orth_data = get_ortholog_info(source_tf, target_gene, source_species=species)
            orthology_ft = OrthologyFeatures(**orth_data)
        except Exception as exc:
            log.debug("Ortholog lookup failed for %s->%s: %s", source_tf, target_gene, exc)

    # ── Multimodal injection: motif + ATAC from acquisition manifest ──────────
    motif_ft = None
    atac_ft = None
    if multimodal_loader is not None:
        motif_ft = multimodal_loader.get_motif_features(source_tf, target_gene)
        atac_ft = multimodal_loader.get_atac_features(target_gene)

    return FeatureBundle(
        context_id=ctx.context_id,
        source_tf=source_tf,
        target_gene=target_gene,
        expression=expr_ft,
        network=net_ft,
        motif=motif_ft,
        atac=atac_ft,
        orthology=orthology_ft,
    )
