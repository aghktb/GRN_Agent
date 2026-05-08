"""
Unit tests for multimodal data acquisition module.
"""

from __future__ import annotations

from argparse import Namespace
import math

import pandas as pd
import pytest

from grn_agent.acquisition.accessibility_processor import compute_promoter_accessibility
from grn_agent.acquisition.compatibility import validate_dataset_compatibility
from grn_agent.acquisition.geo_client import GEOClient, is_likely_accessibility_series
from grn_agent.acquisition.rna_processor import identify_expressed_tfs, normalize_expression
from grn_agent.acquisition.accessibility_processor import compute_coverage_fraction
from scripts.acquire_multimodal_data import (
    _extract_encode_accessibility_qc_metrics,
    _extract_best_geo_accessibility_meta_from_series,
    _extract_geo_sample_accessibility_meta,
    _infer_request_geo_accessions,
    _is_candidate_acceptable,
    _is_same_series_accessibility_candidate,
    _load_tf_symbols,
    _select_expressed_tfs,
    _score_accessibility_candidate,
)


def test_normalize_expression_log1p_cpm():
    import numpy as np
    counts = np.array([[100, 200], [300, 400]], dtype=np.float32)
    norm = normalize_expression(counts, method="log1p_cpm")
    assert norm.shape == counts.shape
    assert np.all(norm >= 0)


def test_normalize_expression_none():
    import numpy as np
    counts = np.array([[1, 2], [3, 4]], dtype=np.float32)
    norm = normalize_expression(counts, method="none")
    assert np.array_equal(norm, counts)


def test_identify_expressed_tfs():
    import numpy as np
    expr = np.array([[0.5, 0.01], [0.6, 0.02]], dtype=np.float32)
    genes = ["TF1", "TF2"]
    tfs = ["TF1", "TF2", "TF3"]
    expressed = identify_expressed_tfs(expr, genes, tfs, min_mean_expr=0.1)
    assert "TF1" in expressed
    assert "TF2" not in expressed
    assert "TF3" not in expressed


def test_load_tf_symbols_reads_tf_universe_without_gold(tmp_path):
    tf_file = tmp_path / "TFs.csv"
    tf_file.write_text("tf\nNanog\n Sox2 \nPou5f1,extra\n", encoding="utf-8")
    assert _load_tf_symbols(tf_file) == {"NANOG", "SOX2", "POU5F1"}


def test_select_expressed_tfs_uses_tf_file_not_gold():
    import numpy as np

    expr = np.array([[0.5, 0.9, 0.01], [0.6, 0.8, 0.02]], dtype=np.float32)
    genes = ["GOLDTF", "FILETF", "LOWTF"]
    expressed, source = _select_expressed_tfs(
        expr,
        genes,
        expr.mean(axis=0),
        {"FILETF", "LOWTF"},
        min_mean_expr=0.1,
    )
    assert source == "tf_file"
    assert expressed == {"FILETF"}
    assert "GOLDTF" not in expressed


def test_select_expressed_tfs_fallback_is_expression_only():
    import numpy as np

    expr = np.eye(20, dtype=np.float32)
    genes = [f"G{i}" for i in range(20)]
    mean_expr = np.arange(20, dtype=np.float32)
    expressed, source = _select_expressed_tfs(expr, genes, mean_expr, set(), min_mean_expr=0.1)
    assert source == "expression_top_decile"
    assert expressed == {"G18", "G19"}


def test_compute_coverage_fraction():
    accessibility = {"G1": 1.0, "G2": 0.0, "G3": 0.5}
    targets = {"G1", "G2", "G3"}
    cov = compute_coverage_fraction(accessibility, targets)
    assert cov == pytest.approx(2 / 3)


def test_compute_coverage_fraction_empty():
    accessibility = {}
    targets = set()
    cov = compute_coverage_fraction(accessibility, targets)
    assert cov == 0.0


def test_compute_promoter_accessibility_log_scales_peak_scores():
    peaks = pd.DataFrame(
        {
            "chr": ["chr1", "chr1"],
            "start": [90, 105],
            "end": [110, 125],
            "score": [1000.0, 1000.0],
        }
    )
    genes = pd.DataFrame({"gene_symbol": ["WLS"], "chr": ["chr1"], "tss": [100]})
    acc = compute_promoter_accessibility(peaks, genes, window=20)
    assert acc["WLS"] == pytest.approx(math.log1p(2000.0))


def test_extract_encode_accessibility_qc_metrics_reads_frip_and_tss():
    payloads = [
        {
            "quality_metric_of": ["ENCFF000AAA"],
            "tss_enrichment": 8.7,
            "nested": {"FRiP": 0.18},
        }
    ]
    metrics = _extract_encode_accessibility_qc_metrics(payloads)
    assert metrics["frip"] == pytest.approx(0.18)
    assert metrics["tss_enrichment"] == pytest.approx(8.7)


def test_extract_encode_accessibility_qc_metrics_accepts_spot_alias():
    payloads = [{"quality_metrics": [{"spot_score": 0.11}]}]
    metrics = _extract_encode_accessibility_qc_metrics(payloads)
    assert metrics["frip"] == pytest.approx(0.11)


def test_geo_search_filter_keeps_mixed_rna_atac_series():
    assert is_likely_accessibility_series(
        "SHARE-seq profiling of chromatin accessibility and gene expression",
        "single-cell RNA-seq and ATAC-seq were generated from matched samples",
        "Expression profiling by high throughput sequencing",
    )


def test_geo_seed_accessions_are_inferred_from_expr_path_and_context():
    args = Namespace(
        rna_accession="",
        expr="/tmp/downloads/GSM999002_expression.csv",
        cell_context="same donor as GSE999001 RNA profile",
        dataset_id="skin_context",
    )
    assert _infer_request_geo_accessions(args) == ["GSM999002", "GSE999001"]


def test_geo_family_soft_parser_extracts_sample_level_files():
    soft = """
^SERIES = GSE999001
!Series_title = SHARE-seq profiling of chromatin accessibility and gene expression
!Series_summary = This series contains paired RNA-seq and ATAC-seq samples.
^SAMPLE = GSM999001
!Sample_title = ATAC-seq from human adult skin
!Sample_organism_ch1 = Homo sapiens
!Sample_source_name_ch1 = human adult skin
!Sample_library_strategy = ATAC-seq
!Sample_characteristics_ch1 = tissue: skin
!Sample_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM999nnn/GSM999001/suppl/GSM999001_peaks.bed.gz
!Sample_series_id = GSE999001
^SAMPLE = GSM999002
!Sample_title = RNA-seq from human adult skin
!Sample_organism_ch1 = Homo sapiens
!Sample_source_name_ch1 = human adult skin
!Sample_library_strategy = RNA-Seq
!Sample_characteristics_ch1 = tissue: skin
!Sample_series_id = GSE999001
"""
    parsed = GEOClient._parse_family_soft(soft)
    assert parsed["samples"] == ["GSM999001", "GSM999002"]
    atac = parsed["sample_metadata"][0]
    assert atac["accession"] == "GSM999001"
    assert atac["library_strategy"] == "ATAC-seq"
    assert atac["supplementary_files"][0].startswith("https://")
    assert atac["supplementary_files"][0].endswith("GSM999001_peaks.bed.gz")


def test_geo_sample_page_parser_recovers_gsm_supplementary_beds():
    html = """
    <a href="ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM999nnn/GSM999099/suppl/GSM999099_fragments.bed.gz">(ftp)</a>
    <a href="/geo/download/?acc=GSM999099&format=file&file=GSM999099_peaks.bed.gz">(http)</a>
    GSM999099_celltype.txt.gz
    """
    links = GEOClient._extract_supplementary_links_from_geo_page("GSM999099", html)
    assert "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM999nnn/GSM999099/suppl/GSM999099_fragments.bed.gz" in links
    assert "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM999nnn/GSM999099/suppl/GSM999099_peaks.bed.gz" in links
    assert "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM999nnn/GSM999099/suppl/GSM999099_celltype.txt.gz" in links


def test_geo_sample_metadata_refetches_stale_accessibility_cache(monkeypatch, tmp_path):
    geo = GEOClient(cache_dir=tmp_path)
    cache_path = tmp_path / "GSM999099.json"
    cache_path.write_text(
        """
{
  "accession": "GSM999099",
  "title": "ATAC-seq from sample",
  "organism": "Homo sapiens",
  "library_strategy": "ATAC-seq",
  "series_accessions": ["GSE999099"],
  "supplementary_files": []
}
""".strip(),
        encoding="utf-8",
    )

    class Resp:
        status_code = 200
        text = """
^SAMPLE = GSM999099
!Sample_title = ATAC-seq from sample
!Sample_organism_ch1 = Homo sapiens
!Sample_library_strategy = ATAC-seq
!Sample_series_id = GSE999099
<a href="/geo/download/?acc=GSM999099&format=file&file=GSM999099_peaks.bed.gz">(http)</a>
"""

        def raise_for_status(self):
            return None

    monkeypatch.setattr(geo, "_get_with_retry", lambda *args, **kwargs: Resp())
    meta = geo.get_sample_metadata("GSM999099")
    assert meta["supplementary_files"] == [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM999nnn/GSM999099/suppl/GSM999099_peaks.bed.gz"
    ]


def test_geo_rna_gsm_resolves_same_series_atac_gsm(monkeypatch, tmp_path):
    geo = GEOClient(cache_dir=tmp_path)
    rna = {
        "accession": "GSM999002",
        "title": "RNA-seq from human adult skin",
        "organism": "Homo sapiens",
        "source_name": "human adult skin",
        "series_accessions": ["GSE999001"],
        "characteristics": {"tissue": ["skin"]},
        "characteristics_text": "tissue: skin",
        "supplementary_files": [],
    }
    atac = {
        "accession": "GSM999001",
        "title": "ATAC-seq from human adult skin",
        "organism": "Homo sapiens",
        "source_name": "human adult skin",
        "library_strategy": "ATAC-seq",
        "series_accessions": ["GSE999001"],
        "characteristics": {"tissue": ["skin"]},
        "characteristics_text": "tissue: skin",
        "supplementary_files": [
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM999nnn/GSM999001/suppl/GSM999001_peaks.bed.gz"
        ],
    }
    wrong_species_atac = {
        "accession": "GSM999000",
        "title": "ATAC-seq from mouse adult skin",
        "organism": "Mus musculus",
        "source_name": "mouse adult skin",
        "library_strategy": "ATAC-seq",
        "series_accessions": ["GSE999001"],
        "characteristics": {"tissue": ["skin"]},
        "characteristics_text": "tissue: skin",
        "supplementary_files": [
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM999nnn/GSM999000/suppl/GSM999000_peaks.bed.gz"
        ],
    }
    series = {
        "accession": "GSE999001",
        "title": "SHARE-seq profiling of chromatin accessibility and gene expression",
        "sample_metadata": [wrong_species_atac, atac, rna],
    }

    monkeypatch.setattr(geo, "get_sample_metadata", lambda acc: rna if acc == "GSM999002" else atac)
    monkeypatch.setattr(geo, "get_series_metadata", lambda acc: series)
    monkeypatch.setattr(geo, "_list_sample_supplementary_files", lambda acc: [])

    hits = geo.find_accessibility_samples_for_rna("GSM999002")
    assert hits[0]["accession"] == "GSM999001"
    assert hits[0]["parent_series_accession"] == "GSE999001"
    assert hits[0]["matched_rna_accession"] == "GSM999002"

    meta = _extract_geo_sample_accessibility_meta(hits[0], genome="hg38", requested_species="human")
    assert meta["accession"] == "GSM999001"
    assert meta["parent_series_accession"] == "GSE999001"
    assert meta["matched_rna_accession"] == "GSM999002"
    assert meta["has_peak_file"] is True

    score, detail = _score_accessibility_candidate(
        meta,
        requested_species="human",
        requested_cell_type="human adult skin",
        requested_cell_line=None,
        requested_lineage=None,
        requested_state=None,
        requested_cell_context=None,
        requested_perturbation=None,
        allow_perturbation=False,
    )
    assert score >= 0.60
    assert detail["s_study"] == pytest.approx(1.0)


def test_geo_series_sample_selection_filters_mixed_species(tmp_path):
    geo = GEOClient(cache_dir=tmp_path)
    series = {
        "accession": "GSE999010",
        "title": "mixed-species chromatin accessibility series",
        "sample_metadata": [
            {
                "accession": "GSM999010",
                "title": "ATAC-seq mouse skin",
                "organism": "Mus musculus",
                "source_name": "mouse skin",
                "library_strategy": "ATAC-seq",
                "supplementary_files": ["https://example.org/GSM999010_peaks.bed.gz"],
            },
            {
                "accession": "GSM999011",
                "title": "ATAC-seq human skin",
                "organism": "Homo sapiens",
                "source_name": "human skin",
                "library_strategy": "ATAC-seq",
                "supplementary_files": ["https://example.org/GSM999011_peaks.bed.gz"],
            },
        ],
    }
    meta = _extract_best_geo_accessibility_meta_from_series(
        geo,
        series,
        genome="hg38",
        requested_species="human",
        requested_cell_type="skin",
        requested_cell_line=None,
        requested_lineage=None,
        requested_state=None,
    )
    assert meta["accession"] == "GSM999011"
    assert meta["species"] == "human"


def test_validate_dataset_compatibility_accepts_public_geo_status():
    rna = {
        "source": "user_provided",
        "species": "human",
        "cell_type": "skin",
        "genome_build": "hg38",
        "n_replicates": 1,
        "gene_symbols": ["G1", "G2"],
        "expressed_tfs": ["TF1"],
        "status": "released",
    }
    acc = {
        "source": "GEO",
        "species": "human",
        "cell_type": "skin",
        "genome_build": "hg38",
        "assay": "ATAC-seq",
        "n_replicates": 1,
        "promoter_coverage_of_targets": 0.8,
        "has_peak_file": True,
        "has_signal_file": False,
        "status": "public",
        "qc_flags": {},
    }

    result = validate_dataset_compatibility(rna, acc, {"G1", "G2"}, {"TF1"}, strict=True)
    assert result["pass"] is True
    assert all("status not released" not in r for r in result["rejection_reasons"])


def test_same_series_candidate_marker_drives_priority():
    assert _is_same_series_accessibility_candidate(
        {"source": "GEO", "accession": "GSM999001", "matched_rna_accession": "GSM999002"}
    )
    assert _is_same_series_accessibility_candidate(
        {"source": "GEO", "accession": "GSM999001", "pairing_quality": "same_series"}
    )
    assert _is_same_series_accessibility_candidate(
        {"source": "GEO", "accession": "GSM999001", "pairing_quality": "request_geo_seed"}
    )
    assert not _is_same_series_accessibility_candidate(
        {"source": "ENCODE", "accession": "ENCSR000AAA", "pairing_quality": ""}
    )


def test_rna_geo_sample_with_files_is_not_acceptable_accessibility():
    meta = {
        "source": "GEO",
        "accession": "GSM999200",
        "assay": "RNA-seq",
        "species": "human",
        "cell_type": "skin",
        "files": ["https://example.org/GSM999200_matrix.txt.gz"],
        "has_peak_file": False,
        "has_signal_file": False,
        "status": "public",
    }
    score, detail = _score_accessibility_candidate(
        meta,
        requested_species="human",
        requested_cell_type="skin",
        requested_cell_line=None,
        requested_lineage=None,
        requested_state=None,
        requested_cell_context=None,
        requested_perturbation=None,
        allow_perturbation=False,
    )
    assert not _is_candidate_acceptable(
        meta,
        score,
        detail,
        requested_species="human",
        requested_cell_type="skin",
        requested_cell_line=None,
        requested_cell_context=None,
    )


def test_geo_atac_with_peak_can_pass_selection_despite_low_bio_score():
    meta = {
        "source": "GEO",
        "accession": "GSM999201",
        "assay": "ATAC-seq",
        "species": "mouse",
        "cell_type": "mouse brain",
        "description": "mouse brain ATAC-seq",
        "files": ["https://example.org/GSM999201_brain.peaks.bed.gz"],
        "has_peak_file": True,
        "has_signal_file": False,
        "status": "public",
        "n_replicates": 1,
    }
    score, detail = _score_accessibility_candidate(
        meta,
        requested_species="mouse",
        requested_cell_type="epithelial cell",
        requested_cell_line=None,
        requested_lineage=None,
        requested_state=None,
        requested_cell_context=None,
        requested_perturbation=None,
        allow_perturbation=False,
    )
    assert detail["tier"] == "REJECT"
    assert detail["s_bio"] == 0.0
    assert _is_candidate_acceptable(
        meta,
        score,
        detail,
        requested_species="mouse",
        requested_cell_type="epithelial cell",
        requested_cell_line=None,
        requested_cell_context=None,
    )


def test_validate_dataset_compatibility_pass():
    rna = {
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "genome_build": "mm10",
        "n_replicates": 3,
        "gene_symbols": ["G1", "G2", "G3", "G4", "G5"],
        "expressed_tfs": ["TF1", "TF2", "TF3"],
        "status": "released",
    }
    acc = {
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "genome_build": "mm10",
        "assay": "ATAC-seq",
        "n_replicates": 2,
        "promoter_coverage_of_targets": 0.8,
        "has_peak_file": True,
        "status": "released",
        "qc_flags": {},
    }
    gold_genes = {"G1", "G2", "G3", "G4", "G5"}
    gold_tfs = {"TF1", "TF2", "TF3", "TF4", "TF5"}
    
    result = validate_dataset_compatibility(rna, acc, gold_genes, gold_tfs, strict=True)
    assert result["pass"] is True
    assert result["biological_match"] == "pass"
    assert result["technical_match"] == "pass"
    assert result["feature_compatibility"] == "pass"
    assert result["accessibility_coverage"] == "pass"
    assert len(result["rejection_reasons"]) == 0
    assert result["accessibility_qc"]["decision"] == "conditional_accept"
    assert result["accessibility_qc"]["score"] >= 0.50


def test_validate_dataset_compatibility_species_mismatch():
    rna = {
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "genome_build": "mm10",
        "n_replicates": 3,
        "gene_symbols": ["G1", "G2"],
        "expressed_tfs": ["TF1"],
        "status": "released",
    }
    acc = {
        "species": "human",
        "cell_type": "embryonic stem cell",
        "genome_build": "hg38",
        "assay": "ATAC-seq",
        "n_replicates": 2,
        "promoter_coverage_of_targets": 0.8,
        "has_peak_file": True,
        "status": "released",
        "qc_flags": {},
    }
    gold_genes = {"G1", "G2"}
    gold_tfs = {"TF1"}
    
    result = validate_dataset_compatibility(rna, acc, gold_genes, gold_tfs, strict=True)
    assert result["pass"] is False
    assert result["biological_match"] == "fail"
    assert "Species mismatch" in result["rejection_reasons"][0]


def test_validate_dataset_compatibility_does_not_hard_reject_mid_coverage():
    rna = {
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "genome_build": "mm10",
        "n_replicates": 3,
        "gene_symbols": ["G1", "G2"],
        "expressed_tfs": ["TF1"],
        "status": "released",
    }
    acc = {
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "genome_build": "mm10",
        "assay": "ATAC-seq",
        "n_replicates": 2,
        "promoter_coverage_of_targets": 0.55,
        "has_peak_file": True,
        "status": "released",
        "qc_flags": {},
    }
    gold_genes = {"G1", "G2"}
    gold_tfs = {"TF1"}
    
    result = validate_dataset_compatibility(rna, acc, gold_genes, gold_tfs, strict=True)
    assert result["pass"] is True
    assert result["accessibility_coverage"] == "pass"
    assert result["accessibility_qc"]["components"]["Q_prom"] == pytest.approx(1.0)


def test_validate_dataset_compatibility_rejects_if_no_peak_or_signal_files():
    rna = {
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "genome_build": "mm10",
        "n_replicates": 3,
        "gene_symbols": ["G1", "G2"],
        "expressed_tfs": ["TF1"],
        "status": "released",
    }
    acc = {
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "genome_build": "mm10",
        "assay": "ATAC-seq",
        "n_replicates": 2,
        "promoter_coverage_of_targets": 0.8,
        "has_peak_file": False,
        "has_signal_file": False,
        "status": "released",
        "qc_flags": {},
    }
    gold_genes = {"G1", "G2"}
    gold_tfs = {"TF1"}

    result = validate_dataset_compatibility(rna, acc, gold_genes, gold_tfs, strict=True)
    assert result["pass"] is False
    assert "missing_peak_and_signal_files" in result["accessibility_qc"]["hard_rejects"]


def test_validate_dataset_compatibility_cell_type_mismatch_fails_even_non_strict():
    rna = {
        "species": "human",
        "cell_type": "embryonic stem cell",
        "genome_build": "hg38",
        "n_replicates": 3,
        "gene_symbols": ["G1", "G2"],
        "expressed_tfs": ["TF1"],
        "status": "released",
    }
    acc = {
        "species": "human",
        "cell_type": "esophagus mucosa",
        "genome_build": "hg38",
        "assay": "ATAC-seq",
        "n_replicates": 2,
        "promoter_coverage_of_targets": 0.8,
        "has_peak_file": True,
        "status": "released",
        "qc_flags": {},
    }
    gold_genes = {"G1", "G2"}
    gold_tfs = {"TF1"}

    result = validate_dataset_compatibility(rna, acc, gold_genes, gold_tfs, strict=False)
    assert result["pass"] is False
    assert result["biological_match"] == "fail"
    assert any("Cell type mismatch" in r for r in result["rejection_reasons"])
    assert "cell_type_mismatch" in result["accessibility_qc"]["hard_rejects"]
