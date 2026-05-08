# Multimodal Data Acquisition

Overview
--------

This module fetches, ranks, validates, and materializes multimodal inputs for
GRN training:

- expression matrix
- accessibility peaks and promoter accessibility
- motif hits
- QC metadata and manifest paths

The current implementation is centered on ENCODE and GEO acquisition plus a
scored accessibility QC model.

What the acquisition step does
------------------------------

1. Load RNA expression and identify expressed TFs.
2. Search or accept user-provided accessibility data.
3. Download or reuse a local peak BED when available.
4. Build promoter accessibility from peaks and TSS coordinates.
5. Optionally run motif scanning with `bedtools getfasta` + `fimo`.
6. Score dataset compatibility and accessibility QC.
7. Write a multimodal manifest consumed downstream.

Strict biological rule
----------------------

Cell-type mismatch in acquisition QC is a hard failure. A dataset such as:

- RNA: embryonic stem cell
- accessibility: esophagus mucosa

does not pass QC.

Accessibility QC model
----------------------

Accessibility QC is no longer a simple hard threshold on promoter coverage.
The current code computes a scored `accessibility_qc` block with weighted
components:

- `Q_signal`
- `Q_TSS`
- `Q_prom`
- `Q_peakdist`
- `Q_rep`
- `Q_match`
- `Q_usable`

Decision bands:

- `score >= 0.70`: `accept`
- `0.50 <= score < 0.70`: `conditional_accept`
- `< 0.50`: `reject`

Hard reject conditions still apply for:

- species mismatch
- unresolved genome
- no usable peak or signal files
- severe biological mismatch
- critically low signal and TSS together

Implication:

- promoter coverage around `0.55` is not automatically rejected
- QC is treated as a reliability prior, not just a binary filter

Current metric sources
----------------------

Already extracted:

- promoter coverage
- promoter-peak fraction
- peak/signal file usability
- replicate counts
- biological matching metadata
- ENCODE QC metrics when available:
  - `FRiP`
  - `TSS enrichment`

Still optional / missing in some datasets:

- replicate concordance from explicit peak overlap
- FRiP / TSS for GEO and user-provided local files unless supplied or derived

Manifest schema
---------------

The manifest includes:

- `rna`
- `accessibility`
- `motifs`
- `qc`
- `qc_report`
- `paths`

The accessibility section can now include fields such as:

- `promoter_coverage_of_rnaseq_genes`
- `promoter_peak_fraction`
- `frip`
- `tss_enrichment`
- `has_peak_file`
- `has_signal_file`

The QC payload includes:

- `pass`
- `biological_match`
- `technical_match`
- `feature_compatibility`
- `accessibility_coverage`
- `accessibility_qc`
- `rejection_reasons`
- `warnings`

Usage
-----

Minimal:

```bash
python scripts/acquire_multimodal_data.py \
  --expr data/expression.csv \
  --species mouse \
  --out-manifest data/manifest.json
```

Cell-type aware:

```bash
python scripts/acquire_multimodal_data.py \
  --expr data/expression.csv \
  --species human \
  --cell-type "embryonic stem cell" \
  --genome hg38 \
  --out-manifest data/manifest.json \
  --strict
```

User-provided peaks:

```bash
python scripts/acquire_multimodal_data.py \
  --expr data/expression.csv \
  --species mouse \
  --atac-file data/peaks.bed \
  --out-manifest data/manifest.json
```

Important flags
---------------

- `--strict`
- `--min-promoter-coverage`
- `--coverage-denominator`
- `--max-atac-candidates`
- `--skip-atac-search`
- `--skip-motif`
- `--fimo-pvalue`
- `--motif-window`
- `--genome`
- `--genome-fasta`
- `--gtf`

Notes on motif integration
--------------------------

- motif scanning uses promoter-overlapping peaks only
- chromosome names are normalized to the FASTA contig style before
  `bedtools getfasta`
- `fimo` and `bedtools` timeouts are downgraded into motif acquisition failure
  states so the broader acquisition workflow can continue without motif support

Downstream use
--------------

The TF-EAGER and integrated workflows consume the generated manifest directly.
The main downstream fields are:

- `paths.expression_matrix`
- `paths.promoter_accessibility`
- `paths.motif_hits`
- `qc` / `qc_report`

Relevant entrypoints
--------------------

- [scripts/acquire_multimodal_data.py](/home/aghktb/GRN_Agent/scripts/acquire_multimodal_data.py)
- [src/grn_agent/acquisition/compatibility.py](/home/aghktb/GRN_Agent/src/grn_agent/acquisition/compatibility.py)
- [src/grn_agent/acquisition/accessibility_processor.py](/home/aghktb/GRN_Agent/src/grn_agent/acquisition/accessibility_processor.py)
- [src/grn_agent/acquisition/motif_scanner.py](/home/aghktb/GRN_Agent/src/grn_agent/acquisition/motif_scanner.py)
- [src/grn_agent/acquisition/manifest_builder.py](/home/aghktb/GRN_Agent/src/grn_agent/acquisition/manifest_builder.py)
