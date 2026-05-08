# Multimodal Data Acquisition Pipeline (Current)

## Overview

This pipeline builds a validated multimodal dataset by integrating:
- RNA expression
- chromatin accessibility peaks (ATAC/DNase)
- TF motif evidence (JASPAR + FIMO)

Primary entry point: `scripts/acquire_multimodal_data.py`.

The workflow is designed to be:
- **strict** (quality-gated),
- **non-leaky** (no gold-label-derived feature filtering),
- **reproducible** (cached references, explicit manifest output).

---

## End-to-End Architecture

```text
Inputs (required)
  - expression CSV (--expr)
  - species (--species)
  - output manifest path (--out-manifest)

Inputs (optional)
  - biological context: --cell-type, --cell-line, --lineage, --state
  - accessibility hints: --atac-accession / --geo-accession / --atac-file
  - genome hints: --genome, --genome-fasta, --gtf
  - optional benchmark file: --gold-network (for evaluation context only)
  - QC knobs: --strict, --min-promoter-coverage, --coverage-denominator
  - motif knobs: --motif-window, --fimo-pvalue, --jaspar-release

    ↓
[1] RNA processing
  - load expression matrix
  - normalize (log1p_cpm)
  - identify expressed TFs
  - compute expressed-gene set for coverage denominator

    ↓
[2] Accessibility discovery
  - search ENCODE (ATAC, DNase)
  - search GEO series
  - score/rank candidates by species/cell-line/cell-type/assay/status/etc.
  - select best candidate

    ↓
[3] Local data resolution
  - download selected peak BED if needed
  - resolve genome FASTA + GTF (GenomeDB / user-provided)

    ↓
[4] Promoter accessibility feature build
  - map peaks to promoter windows (TSS ± motif-window)
  - write promoter_accessibility.bed
  - compute promoter coverage on RNA-derived denominator

    ↓
[5] Motif integration
  - fetch validated JASPAR MEME (default 2026)
  - filter motifs to expressed TFs
  - extract promoter-overlapping peak sequences (bedtools getfasta)
  - scan motifs (FIMO)
  - aggregate to TF-gene table

    ↓
[6] Compatibility QC
  - biological checks
  - technical checks
  - coverage threshold gate

    ↓
[7] Manifest writing
  - write multimodal_manifest.json with metadata, QC, and output paths
```

---

## CLI Parameters (Current)

### Required
- `--expr`: expression matrix CSV
- `--species`: species label
- `--out-manifest`: output manifest path

### Core optional inputs
- `--gold-network`: optional benchmark network CSV (not used to filter motif features)
- `--cell-type`, `--cell-line`, `--lineage`, `--state`
- `--allow-perturbation`
- `--genome` (auto-infers from species if omitted)
- `--dataset-id`

### Accessibility options
- `--atac-accession`
- `--geo-accession`
- `--atac-file`
- `--skip-atac-search`
- `--max-atac-candidates`

### Motif/genome options
- `--genome-fasta`
- `--gtf`
- `--skip-motif`
- `--no-auto-genome`
- `--motif-window` (default `2000`)
- `--fimo-pvalue` (default `1e-4`)
- `--jaspar-release` (default `2026`)

### QC policy knobs
- `--strict`
- `--min-promoter-coverage` (default currently `0.6`)
- `--coverage-denominator`:
  - `rnaseq_expressed` (default)
  - `rnaseq_all`
- `--coverage-min-mean-expr` (default `0.1`; used for `rnaseq_expressed`)

Notes:
- `--min-promoter-coverage` is now one input into the broader scored
  accessibility QC policy, not the entire acceptance rule by itself.
- Cell-type mismatch is treated as a hard acquisition QC failure.

---

## Data Processing Details

### 1) RNA processing

- Expression is loaded as `(cells × genes)`.
- Normalization method: `log1p_cpm`.
- Expressed TFs:
  - if TF list available, retain TFs with mean normalized expression `>= 0.1`.
  - fallback without TF list: top 10% by mean expression.
- Expressed genes for coverage denominator:
  - genes with mean normalized expression `>= --coverage-min-mean-expr`.

### 2) Accessibility candidate search and selection

- ENCODE and GEO are both queried unless user provides file/accession directly.
- Candidates are scored by:
  - species,
  - cell-type semantic fit,
  - cell-line fit/missingness,
  - assay type,
  - replicate count,
  - status,
  - perturbation policy.
- Highest score is selected; top candidates are logged.

### 3) Genome and annotation resolution

- If `--genome-fasta` is absent and auto mode is on:
  - genome FASTA and GTF are resolved via GenomeDB.
- GTF is used to derive TSS coordinates for promoter mapping.

### 4) Promoter accessibility

- Promoter window: `TSS ± --motif-window` (default 2000 bp).
- Peak overlap produces per-gene accessibility values.
- Coverage denominator is RNA-derived only:
  - `rnaseq_expressed`: expressed genes only,
  - `rnaseq_all`: all RNA genes.
- Coverage metric:
  - `promoter_coverage_of_rnaseq_genes` in manifest.
- Promoter-peak distribution is also recorded when peaks are available.

### 5) Motif integration

- JASPAR MEME retrieval (default release 2026) with validation/fallbacks.
- MEME filtered to expressed TF set.
- Peak names are normalized to stable unique IDs to avoid collisions.
- Sequence extraction uses:
  - `bedtools getfasta -nameOnly -s`
- Peak chromosome names are normalized to the genome FASTA contig style before
  sequence extraction.
- FIMO run uses:
  - `--thresh <pvalue>`
  - `--no-qvalue`
  - intentionally no `--parse-genomic-coord` (to preserve peak IDs).
- `bedtools getfasta` and `fimo` timeouts are downgraded into motif failure
  states so acquisition can continue without motif support.
- Aggregation outputs:
  - `source_tf`
  - `target_gene`
  - `motif_id`
  - `motif_present`
  - `max_score_pct`
  - `peak_count`
- Composite motif TF labels are handled in aggregation/filter logic.

---

## QC and Acceptance Logic (Current)

QC combines:
- Biological checks:
  - species
  - cell type semantic compatibility
  - cell line compatibility
  - lineage/state compatibility
  - perturbation policy
- Technical checks:
  - genome build consistency
  - replicate constraints
  - release status
  - QC flags
- Feature compatibility (non-label-gated)
- Coverage gate:
  - fail if promoter coverage < `--min-promoter-coverage`

Output includes:
- `pass` (bool)
- per-dimension pass/fail labels
- `rejection_reasons`
- `warnings`

---

## Non-Leakage Policy (Current)

The current pipeline removes and avoids gold-coupled manifest fields:
- removed: `gene_overlap_with_gold`
- removed: `promoter_coverage_of_gold_targets`
- removed: `tf_overlap_with_gold`

Current leakage-safe metrics:
- `promoter_coverage_of_rnaseq_genes`
- `tf_overlap_with_rnaseq`

Motif feature generation is not filtered by gold TF-target edge labels.

---

## Current Manifest Schema

```json
{
  "dataset_id": "mouse_embryonic_stem_cell",
  "species": "mouse",
  "cell_type": "embryonic stem cell",
  "genome_build": "mm10",
  "rna": {
    "source": "user_provided",
    "accession": "N/A",
    "files": [],
    "n_replicates": 1,
    "n_genes": 500
  },
  "accessibility": {
    "source": "ENCODE",
    "accession": "ENCSR000CMW",
    "assay": "DNase-seq",
    "files": [".../ENCFF178MGP.bed"],
    "n_replicates": 2,
    "promoter_coverage_of_rnaseq_genes": 0.614
  },
  "motifs": {
    "database": "JASPAR2026",
    "tf_motif_count": 16,
    "tf_overlap_with_rnaseq": 0.5926
  },
  "qc": {
    "pass": false,
    "biological_match": "pass",
    "technical_match": "pass",
    "feature_compatibility": "pass",
    "accessibility_coverage": "fail",
    "rejection_reasons": ["Promoter accessibility coverage < 70%: 61.40%"],
    "warnings": []
  },
  "paths": {
    "expression_matrix": "...",
    "gene_symbols": "...",
    "promoter_accessibility": "...",
    "motif_hits": "..."
  }
}
```

---

## Dependencies

Required runtime tools:
- `bedtools`
- `fimo` (MEME suite)

Python dependencies:
- `requests`
- `pandas`
- `numpy`

---

## Example Command (Current)

```bash
conda run --live-stream -n om-agent python3 scripts/acquire_multimodal_data.py \
  --expr Data/sc-RNA-seq/mESC/mESC_nonspecific_chipseq_tf500ExpressionData.csv \
  --species mouse \
  --out-manifest artifacts/verify_multimodal/multimodal_manifest.json \
  --gold-network Data/sc-RNA-seq/mESC/mESC_tf500_gold_edges.csv \
  --cell-type "embryonic stem cell" \
  --cell-line "ES-E14" \
  --genome mm10 \
  --strict \
  --jaspar-release 2026
```
