from __future__ import annotations

import numpy as np

from grn_agent.schemas import CellContext, Dataset


def build_single_cell_context(
    dataset: Dataset,
    expression: np.ndarray,
    context_id: str,
    cell_type: str | None,
    cell_indices: list[int],
    module_genes: list[str],
    candidate_tfs: list[str],
) -> CellContext:
    meta = {"n_cells": len(cell_indices)}
    if dataset.species:
        meta["species"] = dataset.species
    return CellContext(
        context_id=context_id,
        cell_type=cell_type,
        module_genes=module_genes,
        candidate_tfs=candidate_tfs,
        cell_indices=cell_indices,
        metadata=meta,
    )


def default_contexts_from_dataset(
    dataset: Dataset,
    expression: np.ndarray,
    gene_symbols: list[str],
    tf_list: list[str],
    default_cell_type: str | None = None,
    module_size: int = 40,
    seed: int = 0,
) -> list[CellContext]:
    """
    Simple default: one global context using all cells; module = top variance genes + TF set.
    """
    rng = np.random.default_rng(seed)
    n_cells, n_genes = expression.shape
    assert len(gene_symbols) == n_genes
    var = expression.var(axis=0)
    top_idx = np.argsort(-var)[: module_size + len(tf_list)]
    top_genes = [gene_symbols[i] for i in top_idx]
    module = list(dict.fromkeys(top_genes))
    tfs = [t for t in tf_list if t in gene_symbols]
    if not tfs:
        tfs = [gene_symbols[i] for i in rng.choice(n_genes, size=min(5, n_genes), replace=False)]
    return [
        build_single_cell_context(
            dataset,
            expression,
            context_id=f"{dataset.species}_global_ctx",
            cell_type=(default_cell_type or "unknown"),
            cell_indices=list(range(n_cells)),
            module_genes=module,
            candidate_tfs=tfs,
        )
    ]


def try_contexts_from_scanpy(
    dataset: Dataset,
    expression: np.ndarray,
    gene_symbols: list[str],
    tf_list: list[str],
    default_cell_type: str | None = None,
    resolution: float = 0.5,
) -> list[CellContext]:
    """Optional: Leiden clusters as contexts. Requires scanpy."""
    try:
        import scanpy as sc
        import anndata as ad
    except ImportError:
        return default_contexts_from_dataset(
            dataset,
            expression,
            gene_symbols,
            tf_list,
            default_cell_type=default_cell_type,
        )

    adata = ad.AnnData(X=expression)
    adata.var_names = gene_symbols
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=min(30, expression.shape[1] - 1, expression.shape[0] - 1))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
    contexts: list[CellContext] = []
    leiden = adata.obs["leiden"]
    categories = list(getattr(leiden, "cat", leiden).categories) if hasattr(leiden, "cat") else sorted(leiden.unique())
    for cl in categories:
        idx = np.where(np.asarray(leiden) == cl)[0].tolist()
        sub = expression[idx, :]
        var = sub.var(axis=0)
        top_idx = np.argsort(-var)[:50]
        module = [gene_symbols[i] for i in top_idx]
        tfs = [t for t in tf_list if t in gene_symbols]
        if not tfs:
            tfs = [gene_symbols[i] for i in top_idx[:3]]
        contexts.append(
            CellContext(
                context_id=f"{dataset.species}_leiden_{cl}",
                cell_type=f"cluster_{cl}",
                module_genes=module,
                candidate_tfs=tfs,
                cell_indices=idx,
                metadata={"cluster_algorithm": "leiden"},
            )
        )
    return contexts
