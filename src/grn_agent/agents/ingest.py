from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from grn_agent.schemas import Dataset, GeneMeta, SampleMeta


def _to_upper_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def ingest_from_synthetic(
    dataset_id: str,
    species: str,
    n_cells: int,
    n_genes: int,
    gene_symbols: list[str],
    seed: int = 0,
    modalities: list[str] | None = None,
) -> tuple[Dataset, np.ndarray]:
    """Dry-run: random expression (cells x genes)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_cells, n_genes)).astype(np.float64)
    gene_symbols_u = [_to_upper_symbol(g) for g in gene_symbols]
    genes = [GeneMeta(gene_id=g, symbol=g) for g in gene_symbols_u]
    samples = [SampleMeta(sample_id=f"cell_{i}") for i in range(n_cells)]
    ds = Dataset(
        dataset_id=dataset_id,
        species=species,
        modalities=modalities or ["scrna"],
        genes=genes,
        samples=samples,
        metadata={"source": "synthetic"},
        expression_matrix_key="inline",
    )
    return ds, x


def ingest_from_npy(dataset_id: str, species: str, path: str, gene_symbols: list[str], modalities: list[str] | None = None) -> tuple[Dataset, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    n_cells, n_genes = arr.shape
    if len(gene_symbols) != n_genes:
        raise ValueError("gene_symbols length must match matrix columns")
    gene_symbols_u = [_to_upper_symbol(g) for g in gene_symbols]
    genes = [GeneMeta(gene_id=g, symbol=g) for g in gene_symbols_u]
    samples = [SampleMeta(sample_id=f"cell_{i}") for i in range(n_cells)]
    ds = Dataset(
        dataset_id=dataset_id,
        species=species,
        modalities=modalities or ["scrna"],
        genes=genes,
        samples=samples,
        metadata={"expression_path": str(Path(path).resolve())},
        expression_matrix_key="inline",
    )
    return ds, arr


def ingest_from_beeline_csv(
    dataset_id: str,
    species: str,
    path: str,
    modalities: list[str] | None = None,
) -> tuple[Dataset, np.ndarray, list[str]]:
    """Load a BEELINE-style CSV (genes × cells) and return (Dataset, cells×genes array, gene_symbols)."""
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    # Rows are genes, columns are cells → transpose to (cells × genes)
    x = df.values.T.astype(np.float64)
    gene_symbols = [_to_upper_symbol(g) for g in df.index.tolist()]
    n_cells, n_genes = x.shape
    genes = [GeneMeta(gene_id=g, symbol=g) for g in gene_symbols]
    samples = [SampleMeta(sample_id=str(c)) for c in df.columns.tolist()]
    ds = Dataset(
        dataset_id=dataset_id,
        species=species,
        modalities=modalities or ["scrna"],
        genes=genes,
        samples=samples,
        metadata={"expression_path": str(Path(path).resolve()), "source": "beeline_csv"},
        expression_matrix_key="inline",
    )
    return ds, x, gene_symbols


def try_ingest_h5ad(path: str, dataset_id: str, species: str) -> tuple[Dataset, np.ndarray, list[str]]:
    """Load AnnData if scanpy/anndata available; raise ImportError otherwise."""
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install grn-agent with optional-dependencies scanpy") from exc
    adata = ad.read_h5ad(path)
    if adata.X is None:
        raise ValueError("AnnData has no X")
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    genes = [_to_upper_symbol(g) for g in adata.var_names.astype(str).tolist()]
    ds = Dataset(
        dataset_id=dataset_id,
        species=species,
        modalities=["scrna"],
        genes=[GeneMeta(gene_id=g, symbol=g) for g in genes],
        samples=[SampleMeta(sample_id=str(i)) for i in range(x.shape[0])],
        metadata={"h5ad_path": str(Path(path).resolve())},
        expression_matrix_key="inline",
    )
    return ds, x, genes
