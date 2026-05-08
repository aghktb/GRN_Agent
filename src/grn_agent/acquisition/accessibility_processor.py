"""
DNase/ATAC-seq processing: peak filtering, promoter feature extraction.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def load_peaks_bed(bed_path: str | Path) -> pd.DataFrame:
    """
    Load BED file (chr, start, end, ...).
    
    Returns:
        DataFrame with columns: chr, start, end, (optional: name, score, ...)
    """
    df = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    cols = ["chr", "start", "end"]
    if df.shape[1] >= 4:
        cols.append("name")
    if df.shape[1] >= 5:
        cols.append("score")
    df.columns = cols + [f"col{i}" for i in range(len(cols), df.shape[1])]
    return df[cols]


def compute_promoter_accessibility(
    peaks: pd.DataFrame,
    gene_coords: pd.DataFrame,
    window: int = 2000,
) -> dict[str, float]:
    """
    For each gene, check if any peak overlaps promoter (TSS ± window).
    
    Args:
        peaks: DataFrame with chr, start, end
        gene_coords: DataFrame with gene_symbol, chr, tss (or start/end)
        window: bp around TSS
    
    Returns:
        {gene_symbol: accessibility_score} where overlapping peak support is
        compressed with ``log1p`` to keep model inputs on a stable scale.
    """
    out: dict[str, float] = {}
    for _, g in gene_coords.iterrows():
        gene = str(g["gene_symbol"])
        chrom = str(g["chr"])
        tss = int(g.get("tss", g.get("start", 0)))
        prom_start = max(0, tss - window)
        prom_end = tss + window
        
        overlaps = peaks[
            (peaks["chr"] == chrom)
            & (peaks["end"] >= prom_start)
            & (peaks["start"] <= prom_end)
        ]
        if len(overlaps) > 0:
            if "score" in overlaps.columns:
                score_sum = float(overlaps["score"].sum())
                # Some ENCODE peak BEDs use score=0 placeholders.
                # Treat overlap presence as accessibility when scores are non-informative.
                raw_support = score_sum if score_sum > 0 else float(len(overlaps))
                out[gene] = float(math.log1p(raw_support))
            else:
                out[gene] = 1.0
        else:
            out[gene] = 0.0
    return out


def compute_coverage_fraction(
    accessibility_dict: dict[str, float],
    target_genes: set[str],
) -> float:
    """
    Fraction of target genes with nonzero accessibility.
    
    Args:
        accessibility_dict: {gene: score}
        target_genes: set of gene symbols
    
    Returns:
        Coverage fraction (0.0 to 1.0).
    """
    if not target_genes:
        return 0.0
    covered = sum(1 for g in target_genes if accessibility_dict.get(g, 0.0) > 0)
    return covered / len(target_genes)


def compute_promoter_peak_fraction(
    peaks: pd.DataFrame,
    gene_coords: pd.DataFrame,
    window: int = 2000,
) -> float:
    """
    Fraction of peaks that overlap at least one promoter window.
    """
    if peaks.empty or gene_coords.empty:
        return 0.0
    promoter_overlaps = 0
    genes_by_chr: dict[str, pd.DataFrame] = {
        str(chrom): sub.reset_index(drop=True)
        for chrom, sub in gene_coords.groupby("chr", sort=False)
    }
    for _, peak in peaks.iterrows():
        chrom = str(peak["chr"])
        genes = genes_by_chr.get(chrom)
        if genes is None or genes.empty:
            continue
        start = int(peak["start"])
        end = int(peak["end"])
        tss = genes["tss"]
        if ((tss >= start - window) & (tss <= end + window)).any():
            promoter_overlaps += 1
    return promoter_overlaps / max(1, len(peaks))
