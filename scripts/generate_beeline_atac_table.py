#!/usr/bin/env python3
"""
Generate BEELINE ATAC availability table based on curated metadata.

Outputs CSV with: dataset_id, rna_accession, atac_accession, pairing_quality, species, cell_type, stage, notes
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BEELINE_ATAC_DATA = [
    {
        "dataset_id": "mESC",
        "rna_accession": "GSE98664 (BEELINE)",
        "atac_accession": "GSE148746 (external)",
        "pairing_quality": "medium",
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "stage": "naive/primed ESC",
        "notes": "GSE148746: multi-omic (scRNA+scATAC) mESC, same lineage but not same study as BEELINE RNA",
    },
    {
        "dataset_id": "mHSC",
        "rna_accession": "GSE81682 (BEELINE)",
        "atac_accession": "GSE148746 (external)",
        "pairing_quality": "high",
        "species": "mouse",
        "cell_type": "hematopoietic stem cell",
        "stage": "adult bone marrow",
        "notes": "GSE148746: includes mHSC with matched scRNA+scATAC, high pairing quality for integration",
    },
    {
        "dataset_id": "hESC-hHep",
        "rna_accession": "GSE106540 (BEELINE)",
        "atac_accession": "GSE156021 (external)",
        "pairing_quality": "medium",
        "species": "human",
        "cell_type": "hepatocyte (hESC-derived)",
        "stage": "differentiation trajectory",
        "notes": "GSE156021: hESC→endoderm→hepatocyte scRNA+scATAC; same lineage, external study",
    },
    {
        "dataset_id": "mESC (alternative)",
        "rna_accession": "GSE98664 (BEELINE)",
        "atac_accession": "ENCODE ENCSR000CGE (external)",
        "pairing_quality": "low",
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "stage": "naive ESC",
        "notes": "ENCODE bulk DNase-seq on mESC; bulk vs single-cell mismatch, lower integration quality",
    },
    {
        "dataset_id": "hESC-hHep (alternative)",
        "rna_accession": "GSE106540 (BEELINE)",
        "atac_accession": "ENCODE ENCSR000CJH (external)",
        "pairing_quality": "low",
        "species": "human",
        "cell_type": "hepatocyte",
        "stage": "adult liver",
        "notes": "ENCODE bulk DNase-seq on primary hepatocytes; not ESC-derived, bulk vs single-cell",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BEELINE ATAC availability table")
    parser.add_argument("--output", default="data/beeline_atac_availability.csv", help="Output CSV path")
    args = parser.parse_args()
    
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_id",
                "rna_accession",
                "atac_accession",
                "pairing_quality",
                "species",
                "cell_type",
                "stage",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(BEELINE_ATAC_DATA)
    
    print(f"Wrote {len(BEELINE_ATAC_DATA)} rows to {out}")


if __name__ == "__main__":
    main()
