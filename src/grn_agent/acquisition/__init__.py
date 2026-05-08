"""
Automated multimodal data acquisition, validation, and mapping for BEELINE + ENCODE/GEO.

Modalities supported
--------------------
- RNA-seq   : expression matrix (user-provided, REQUIRED)
- DNase/ATAC: automated search in ENCODE, or user-provided accession/file
- Motifs    : JASPAR PWMs → per-(TF, gene) binary + score features

Key entry points
----------------
- run_motif_integration(...)    full motif pipeline
- JASPARClient(...)             download JASPAR PWMs
- validate_dataset_compatibility(...) enforce acceptance criteria A-E
- build_multimodal_manifest(...) write validated manifest JSON
"""

from .encode_client import ENCODEClient
from .geo_client import GEOClient
from .compatibility import validate_dataset_compatibility
from .manifest_builder import build_multimodal_manifest
from .jaspar_client import (
    JASPARClient,
    Motif,
    download_jaspar_meme_file,
    filter_meme_for_tfs,
    parse_meme_tf_map,
)
from .motif_scanner import run_motif_integration
from .gene_coords import load_gene_coords, build_peak_to_gene_map
from .genome_db import GenomeDB, GenomeEntry, ensure_genome, get_genome_db

__all__ = [
    # ENCODE / GEO
    "ENCODEClient",
    "GEOClient",
    # Validation & manifest
    "validate_dataset_compatibility",
    "build_multimodal_manifest",
    # JASPAR (MEME file download + REST API)
    "JASPARClient",
    "Motif",
    "download_jaspar_meme_file",
    "filter_meme_for_tfs",
    "parse_meme_tf_map",
    # Motif integration — standard protocol (bedtools + FIMO)
    "run_motif_integration",
    # Gene coordinates (TSS lookup, BioMart/GTF)
    "load_gene_coords",
    "build_peak_to_gene_map",
    # Genome database (auto-download + index)
    "GenomeDB",
    "GenomeEntry",
    "ensure_genome",
    "get_genome_db",
]
