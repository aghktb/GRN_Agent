"""
RNA-seq data processing: normalization, gene symbol harmonization, TF identification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_expression(
    counts: np.ndarray,
    method: str = "log1p_cpm",
) -> np.ndarray:
    """
    Normalize expression matrix.
    
    Args:
        counts: (n_cells, n_genes) raw or TPM
        method: "log1p_cpm", "log1p", "none"
    
    Returns:
        Normalized (n_cells, n_genes) array.
    """
    if method == "none":
        return counts
    if method == "log1p":
        return np.log1p(counts)
    if method == "log1p_cpm":
        cpm = counts / counts.sum(axis=1, keepdims=True) * 1e6
        return np.log1p(cpm)
    raise ValueError(f"Unknown normalization method: {method}")


def load_beeline_expression(expr_csv: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Load BEELINE ExpressionData.csv: genes × cells.
    
    Returns:
        expression: (n_cells, n_genes) transposed
        gene_symbols: list of gene names
    """
    df = pd.read_csv(expr_csv, index_col=0)
    # Canonicalize to uppercase across the pipeline.
    gene_symbols = [str(g).strip().upper() for g in df.index]
    expression = df.T.to_numpy(dtype=np.float32)
    return expression, gene_symbols


def load_beeline_gold_network(ref_csv: str | Path) -> tuple[set[str], set[str]]:
    """
    Load BEELINE refNetwork.csv.
    
    Returns:
        (all_genes, all_tfs)
    """
    df = pd.read_csv(ref_csv)
    # Canonicalize to uppercase across the pipeline.
    tfs = set(df.iloc[:, 0].astype(str).str.strip().str.upper())
    targets = set(df.iloc[:, 1].astype(str).str.strip().str.upper())
    all_genes = tfs | targets
    return all_genes, tfs


def identify_expressed_tfs(
    expression: np.ndarray,
    gene_symbols: list[str],
    tf_list: list[str],
    min_mean_expr: float = 0.1,
) -> set[str]:
    """
    Filter TFs by minimum mean expression.
    
    Args:
        expression: (n_cells, n_genes)
        gene_symbols: gene names
        tf_list: candidate TF names
        min_mean_expr: threshold
    
    Returns:
        Set of expressed TF names.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    out: set[str] = set()
    for tf in tf_list:
        tf_u = str(tf).strip().upper()
        if tf_u not in gene_to_idx:
            continue
        idx = gene_to_idx[tf_u]
        if expression[:, idx].mean() >= min_mean_expr:
            out.add(tf_u)
    return out
