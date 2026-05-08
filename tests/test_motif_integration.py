"""
Unit tests for the motif integration pipeline.

Tests are scoped to logic that does NOT require external tools
(bedtools, fimo) or network access.

Covered:
  - JASPAR PFM parser (JASPARClient._parse_pfm)
  - JASPAR MEME file filter (filter_meme_for_tfs)
  - JASPAR motif→TF map parser (parse_meme_tf_map)
  - FIMO TSV parser (_parse_fimo)
  - Peak BED loader (_load_peaks)
  - Peak-to-gene mapping (build_peak_to_gene_map)
  - GTF parser (_parse_gtf)
  - FIMO hit aggregation (_aggregate)
  - End-to-end _empty_df shape
"""

from __future__ import annotations

import subprocess
import textwrap
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grn_agent.acquisition.jaspar_client import (
    JASPARClient,
    Motif,
    filter_meme_for_tfs,
    parse_meme_tf_map,
)
from grn_agent.acquisition.gene_coords import _parse_gtf, build_peak_to_gene_map
from grn_agent.acquisition.motif_scanner import (
    _aggregate,
    _empty_df,
    _load_peaks,
    _normalize_peak_chromosomes_to_fasta,
    _parse_fimo,
    _run_fimo,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_MEME = textwrap.dedent("""\
    MEME version 4

    ALPHABET= ACGT

    strands: + -

    Background letter frequencies
    A 0.25 C 0.25 G 0.25 T 0.25

    MOTIF MA0139.1 CTCF
    letter-probability matrix: alength= 4 w= 4
    0.97 0.01 0.01 0.01
    0.01 0.97 0.01 0.01
    0.01 0.01 0.97 0.01
    0.01 0.01 0.01 0.97

    MOTIF MA0143.4 SOX2
    letter-probability matrix: alength= 4 w= 4
    0.50 0.30 0.10 0.10
    0.10 0.10 0.70 0.10
    0.10 0.10 0.10 0.70
    0.20 0.20 0.20 0.40

    MOTIF MA0048.2 NFIX
    letter-probability matrix: alength= 4 w= 4
    0.10 0.10 0.70 0.10
    0.50 0.10 0.10 0.30
    0.10 0.60 0.10 0.20
    0.10 0.10 0.10 0.70

    MOTIF MA0148.3 SOX2::NANOG
    letter-probability matrix: alength= 4 w= 4
    0.25 0.25 0.25 0.25
    0.25 0.25 0.25 0.25
    0.25 0.25 0.25 0.25
    0.25 0.25 0.25 0.25

""")

_MOCK_GTF = textwrap.dedent("""\
    chr1\t.\tgene\t1000\t2000\t.\t+\t.\tgene_name "GeneA"; gene_id "ENSG001";
    chr1\t.\tgene\t5000\t6000\t.\t-\t.\tgene_name "GeneB"; gene_id "ENSG002";
    chr2\t.\tgene\t100\t500\t.\t+\t.\tgene_name "GeneC"; gene_id "ENSG003";
""")

_MOCK_FIMO_TSV = textwrap.dedent("""\
    motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence
    MA0139.1\tCTCF\tpeak1\t10\t14\t+\t20.5\t1.2e-5\t0.01\tACGT
    MA0143.4\tSOX2\tpeak1\t5\t9\t+\t15.3\t3.1e-5\t0.02\tACGT
    MA0139.1\tCTCF\tpeak3\t2\t6\t-\t18.0\t2.5e-5\t0.01\tACGT
""")


# ─────────────────────────────────────────────────────────────────────────────
# JASPAR PFM parser
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_pfm_basic():
    pfm_raw = {"A": [10.0, 1.0], "C": [1.0, 10.0], "G": [1.0, 1.0], "T": [1.0, 1.0]}
    motif = JASPARClient._parse_pfm("MA0001", "TF1", pfm_raw)
    assert motif is not None
    assert motif.tf_name == "TF1"
    assert motif.length == 2
    np.testing.assert_allclose(motif.pfm.sum(axis=1), np.ones(2), atol=1e-6)


def test_parse_pfm_bad_keys_returns_none():
    result = JASPARClient._parse_pfm("BAD", "TF", {"X": [1], "Y": [2]})
    assert result is None


def test_motif_max_score_positive():
    pfm_raw = {
        "A": [9.0, 0.1, 0.1, 0.1],
        "C": [0.1, 9.0, 0.1, 0.1],
        "G": [0.1, 0.1, 9.0, 0.1],
        "T": [0.1, 0.1, 0.1, 9.0],
    }
    motif = JASPARClient._parse_pfm("MA0002", "TF2", pfm_raw)
    assert motif is not None
    assert motif.max_score > 0


def test_motif_ic_nonnegative():
    pfm_raw = {"A": [9.0, 1.0], "C": [1.0, 9.0], "G": [1.0, 1.0], "T": [1.0, 1.0]}
    motif = JASPARClient._parse_pfm("MA0003", "TF3", pfm_raw)
    assert motif is not None
    assert np.all(motif.ic >= -1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# MEME file filter
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_meme_keeps_matching_tfs(tmp_path):
    src = tmp_path / "jaspar.meme"
    src.write_text(_MOCK_MEME, encoding="utf-8")
    out = tmp_path / "filtered.meme"

    filter_meme_for_tfs(src, ["CTCF", "SOX2"], out)

    tf_map = parse_meme_tf_map(out)
    assert "MA0139.1" in tf_map   # CTCF
    assert "MA0143.4" in tf_map   # SOX2
    assert "MA0148.3" in tf_map   # SOX2::NANOG (matched via SOX2 token)
    assert "MA0048.2" not in tf_map  # NFIX not requested


def test_filter_meme_case_insensitive(tmp_path):
    src = tmp_path / "jaspar.meme"
    src.write_text(_MOCK_MEME, encoding="utf-8")
    out = tmp_path / "filtered.meme"

    filter_meme_for_tfs(src, ["ctcf"], out)  # lowercase
    tf_map = parse_meme_tf_map(out)
    assert "MA0139.1" in tf_map


def test_filter_meme_matches_composite_dimer_name(tmp_path):
    """JASPAR uses 'TF1::TF2' for complexes; either symbol should match."""
    src = tmp_path / "jaspar.meme"
    src.write_text(_MOCK_MEME, encoding="utf-8")
    out = tmp_path / "filtered.meme"

    filter_meme_for_tfs(src, ["NANOG"], out)
    tf_map = parse_meme_tf_map(out)
    assert "MA0148.3" in tf_map


def test_filter_meme_no_matches_produces_header_only(tmp_path):
    src = tmp_path / "jaspar.meme"
    src.write_text(_MOCK_MEME, encoding="utf-8")
    out = tmp_path / "filtered.meme"

    filter_meme_for_tfs(src, ["NONEXISTENT_TF"], out)
    tf_map = parse_meme_tf_map(out)
    assert len(tf_map) == 0


def test_parse_meme_tf_map(tmp_path):
    meme = tmp_path / "test.meme"
    meme.write_text(_MOCK_MEME, encoding="utf-8")
    tf_map = parse_meme_tf_map(meme)
    assert tf_map == {
        "MA0139.1": "CTCF",
        "MA0143.4": "SOX2",
        "MA0048.2": "NFIX",
        "MA0148.3": "SOX2::NANOG",
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIMO TSV parser
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_fimo_basic(tmp_path):
    fimo_out = tmp_path / "fimo_out"
    fimo_out.mkdir()
    (fimo_out / "fimo.tsv").write_text(_MOCK_FIMO_TSV, encoding="utf-8")

    df = _parse_fimo(fimo_out / "fimo.tsv")
    assert len(df) == 3
    assert set(df["motif_id"]) == {"MA0139.1", "MA0143.4"}


def test_parse_fimo_missing_file_returns_empty(tmp_path):
    df = _parse_fimo(tmp_path / "nonexistent" / "fimo.tsv")
    assert df.empty


def test_parse_fimo_score_numeric(tmp_path):
    fimo_out = tmp_path / "fimo_out"
    fimo_out.mkdir()
    (fimo_out / "fimo.tsv").write_text(_MOCK_FIMO_TSV, encoding="utf-8")
    df = _parse_fimo(fimo_out / "fimo.tsv")
    assert df["score"].dtype in (float, "float64")


# ─────────────────────────────────────────────────────────────────────────────
# BED loader
# ─────────────────────────────────────────────────────────────────────────────

def test_load_peaks_with_name(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t1000\t2000\tpeak1\nchr2\t3000\t4000\tpeak2\n", encoding="utf-8")
    peaks = _load_peaks(bed)
    assert len(peaks) == 2
    assert list(peaks.columns[:4]) == ["chr", "start", "end", "name"]
    assert peaks.iloc[0]["name"] == "peak1"


def test_load_peaks_auto_name(tmp_path):
    bed = tmp_path / "noname.bed"
    bed.write_text("chr1\t1000\t2000\nchr1\t3000\t4000\n", encoding="utf-8")
    peaks = _load_peaks(bed)
    assert peaks.iloc[0]["name"] == "peak_0"
    assert peaks.iloc[1]["name"] == "peak_1"


def test_normalize_peak_chromosomes_to_fasta_rewrites_ucsc_to_ensembl(tmp_path):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">1 dna:chromosome\nACGT\n>MT dna:chromosome\nACGT\n", encoding="utf-8")
    peaks = pd.DataFrame(
        {
            "chr": ["chr1", "chrM", "chr2"],
            "start": [1, 5, 9],
            "end": [4, 8, 12],
            "name": ["peak1", "peak2", "peak3"],
        }
    )
    normalized = _normalize_peak_chromosomes_to_fasta(peaks, fasta)
    assert list(normalized["chr"]) == ["1", "MT", "chr2"]


def test_run_fimo_converts_timeout_to_runtimeerror(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["fimo"], timeout=3600)

    monkeypatch.setattr(subprocess, "run", _boom)

    with pytest.raises(RuntimeError, match="FIMO timed out after 3600"):
        _run_fimo(tmp_path / "a.meme", tmp_path / "b.fa", tmp_path / "out", 1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Peak-to-gene mapping
# ─────────────────────────────────────────────────────────────────────────────

def _make_peaks() -> pd.DataFrame:
    return pd.DataFrame(
        {"chr": ["chr1", "chr1", "chr2"], "start": [900, 5000, 200],
         "end": [1500, 5500, 600], "name": ["peak1", "peak2", "peak3"]}
    )


def _make_gene_coords() -> pd.DataFrame:
    return pd.DataFrame(
        {"gene_symbol": ["GeneA", "GeneB", "GeneC"],
         "chr": ["chr1", "chr1", "chr2"],
         "tss": [1100, 9000, 300],
         "strand": ["+", "+", "+"]}
    )


def test_peak_to_gene_overlap():
    mapping = build_peak_to_gene_map(_make_peaks(), _make_gene_coords(), window=2000)
    assert "GeneA" in mapping["peak1"]   # TSS 1100 within 900–1500 ±2000


def test_peak_to_gene_no_overlap():
    mapping = build_peak_to_gene_map(_make_peaks(), _make_gene_coords(), window=2000)
    # GeneB TSS 9000 is far from peak2 (5000–5500)
    assert "GeneB" not in mapping.get("peak2", [])


def test_peak_to_gene_cross_chr():
    mapping = build_peak_to_gene_map(_make_peaks(), _make_gene_coords(), window=2000)
    assert "GeneC" in mapping["peak3"]
    assert "GeneA" not in mapping["peak3"]


# ─────────────────────────────────────────────────────────────────────────────
# GTF parser
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_gtf_tss_plus_strand(tmp_path):
    gtf = tmp_path / "test.gtf"
    gtf.write_text(_MOCK_GTF, encoding="utf-8")
    df = _parse_gtf(gtf, gene_symbols=None)
    row = df[df["gene_symbol"] == "GeneA"].iloc[0]
    assert row["tss"] == 1000    # start on + strand
    assert row["strand"] == "+"


def test_parse_gtf_tss_minus_strand(tmp_path):
    gtf = tmp_path / "test.gtf"
    gtf.write_text(_MOCK_GTF, encoding="utf-8")
    df = _parse_gtf(gtf, gene_symbols=None)
    row = df[df["gene_symbol"] == "GeneB"].iloc[0]
    assert row["tss"] == 6000    # end on - strand
    assert row["strand"] == "-"


def test_parse_gtf_filter_by_symbol(tmp_path):
    gtf = tmp_path / "test.gtf"
    gtf.write_text(_MOCK_GTF, encoding="utf-8")
    df = _parse_gtf(gtf, gene_symbols=["GeneA"])
    assert len(df) == 1
    assert df.iloc[0]["gene_symbol"] == "GeneA"


# ─────────────────────────────────────────────────────────────────────────────
# FIMO hit aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _make_fimo_df() -> pd.DataFrame:
    return pd.DataFrame({
        "motif_id":      ["MA0139.1", "MA0139.1", "MA0143.4"],
        "sequence_name": ["peak1",    "peak3",    "peak1"],
        "score":         [20.5,       18.0,       15.3],
        "p_value":       [1.2e-5,     2.5e-5,     3.1e-5],
    })


def test_aggregate_basic():
    fimo_df = _make_fimo_df()
    motif_to_tf = {"MA0139.1": "CTCF", "MA0143.4": "SOX2"}
    peak_to_genes = {"peak1": ["GeneA", "GeneB"], "peak3": ["GeneB"]}

    df = _aggregate(fimo_df, motif_to_tf, peak_to_genes, pair_filter=None)

    assert "source_tf" in df.columns
    assert "motif_present" in df.columns
    assert df["motif_present"].all()   # all rows have a FIMO hit → present


def test_aggregate_max_score_pct_in_range():
    fimo_df = _make_fimo_df()
    motif_to_tf = {"MA0139.1": "CTCF", "MA0143.4": "SOX2"}
    peak_to_genes = {"peak1": ["GeneA"], "peak3": ["GeneA"]}

    df = _aggregate(fimo_df, motif_to_tf, peak_to_genes, pair_filter=None)
    assert (df["max_score_pct"] >= 0).all()
    assert (df["max_score_pct"] <= 1).all()


def test_aggregate_pair_filter():
    fimo_df = _make_fimo_df()
    motif_to_tf = {"MA0139.1": "CTCF", "MA0143.4": "SOX2"}
    peak_to_genes = {"peak1": ["GeneA", "GeneB"], "peak3": ["GeneB"]}
    pair_filter = {("CTCF", "GeneA")}

    df = _aggregate(fimo_df, motif_to_tf, peak_to_genes, pair_filter=pair_filter)
    assert len(df) == 1
    assert df.iloc[0]["source_tf"] == "CTCF"
    assert df.iloc[0]["target_gene"] == "GeneA"


def test_aggregate_peak_count():
    fimo_df = _make_fimo_df()
    motif_to_tf = {"MA0139.1": "CTCF"}
    # peak1 and peak3 both link to GeneB
    peak_to_genes = {"peak1": ["GeneB"], "peak3": ["GeneB"]}

    df = _aggregate(fimo_df, motif_to_tf, peak_to_genes, pair_filter=None)
    row = df[df["target_gene"] == "GeneB"].iloc[0]
    assert row["peak_count"] == 2   # hits in both peak1 and peak3


def test_aggregate_no_matching_motifs():
    fimo_df = _make_fimo_df()
    motif_to_tf: dict[str, str] = {}   # empty map
    peak_to_genes = {"peak1": ["GeneA"]}
    df = _aggregate(fimo_df, motif_to_tf, peak_to_genes, pair_filter=None)
    assert df.empty


def test_empty_df_columns():
    df = _empty_df()
    expected = {"source_tf", "target_gene", "motif_id",
                "motif_present", "max_score_pct", "peak_count"}
    assert expected == set(df.columns)
