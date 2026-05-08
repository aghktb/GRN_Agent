"""
Bridge between the multimodal acquisition manifest and the feature extraction pipeline.

After running ``acquire_multimodal_data.py`` you have:
  multimodal_manifest.json  — paths + QC metadata
  <dataset_id>/motif_hits.tsv — (source_tf, target_gene, motif_present, max_score_pct, peak_count)
  <dataset_id>/promoter_accessibility.bed — optional per-gene ATAC signal

This module reads those files ONCE and exposes fast per-edge lookups used by
``agents/features.py`` to populate ``FeatureBundle.motif`` and ``FeatureBundle.atac``.

Usage in pipeline config (mESC_tf500_lora.yml):
  multimodal_manifest: Data/sc-RNA-seq/mESC/multimodal_manifest.json
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd

from grn_agent.schemas import ATACFeatures, MotifFeatures

log = logging.getLogger(__name__)

_LEGACY_ACCESSIBILITY_LOG1P_THRESHOLD = 50.0


class MultimodalFeatureLoader:
    """
    Loads motif hits + ATAC accessibility once from a multimodal manifest.
    Provides O(1) lookups per (TF, gene) edge.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self._manifest_path = Path(manifest_path).resolve()
        self._motif_index: dict[tuple[str, str], MotifFeatures] = {}
        self._atac_index: dict[str, ATACFeatures] = {}
        self._loaded = False
        self._manifest: dict = {}

    def load(self) -> None:
        """Load and index all modalities from the manifest. Call once before pipeline."""
        if self._loaded:
            return
        if not self._manifest_path.is_file():
            log.warning("Multimodal manifest not found: %s — skipping modality injection", self._manifest_path)
            self._loaded = True
            return

        with self._manifest_path.open(encoding="utf-8") as fp:
            self._manifest = json.load(fp)

        output_paths: dict[str, str] = self._manifest.get("output_paths", {}) or self._manifest.get("paths", {})

        # ── Motif hits ────────────────────────────────────────────────────────
        motif_path = output_paths.get("motif_hits")
        if motif_path and Path(motif_path).is_file():
            self._load_motif_hits(Path(motif_path))
        else:
            log.info("No motif_hits file in manifest — MotifFeatures will be None for all edges.")

        # ── ATAC / promoter accessibility ─────────────────────────────────────
        atac_path = output_paths.get("promoter_accessibility")
        if atac_path and Path(atac_path).is_file():
            self._load_atac_accessibility(Path(atac_path))
        else:
            log.info("No promoter_accessibility file in manifest — ATACFeatures will be None for all edges.")

        self._loaded = True
        log.info(
            "MultimodalFeatureLoader ready: %d motif pairs, %d ATAC gene profiles",
            len(self._motif_index),
            len(self._atac_index),
        )

    def _load_motif_hits(self, path: Path) -> None:
        """
        Parse motif_hits.tsv from run_motif_integration().

        Expected columns:
            source_tf, target_gene, motif_id,
            motif_present, max_score_pct, peak_count
        """
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as exc:
            log.warning("Failed to load motif_hits.tsv: %s", exc)
            return

        required = {"source_tf", "target_gene", "motif_present"}
        if not required.issubset(df.columns):
            log.warning("motif_hits.tsv missing columns %s", required - set(df.columns))
            return

        for _, row in df.iterrows():
            k = (str(row["source_tf"]).upper(), str(row["target_gene"]).upper())
            self._motif_index[k] = MotifFeatures(
                motif_present=bool(row["motif_present"]),
                motif_score=float(row.get("max_score_pct", 0.0)),
                n_supporting_regions=int(row.get("peak_count", 0)),
            )

        log.info("Loaded %d motif pair entries from %s", len(self._motif_index), path.name)

    def _load_atac_accessibility(self, path: Path) -> None:
        """
        Load promoter accessibility file.

        Expects a TSV/BED with at least:
            gene_symbol, accessibility_score  (optional: peak_to_gene_linked)
        OR a plain two-column TSV from compute_promoter_accessibility().
        """
        try:
            df = pd.read_csv(path, sep="\t", header=None)
        except Exception as exc:
            log.warning("Failed to load promoter_accessibility: %s", exc)
            return

        # Handle both single-value (gene, score) and full BED formats
        if df.shape[1] == 2:
            df.columns = ["gene_symbol", "accessibility_score"]
        elif df.shape[1] >= 4:
            df.columns = ["chr", "start", "end", "gene_symbol"] + [f"c{i}" for i in range(4, df.shape[1])]
            if "accessibility_score" not in df.columns and df.shape[1] >= 5:
                df["accessibility_score"] = df.iloc[:, 4]
            elif "accessibility_score" not in df.columns:
                df["accessibility_score"] = 1.0
        else:
            log.warning("Unexpected promoter_accessibility format: %d columns", df.shape[1])
            return

        raw_scores = pd.to_numeric(df["accessibility_score"], errors="coerce").fillna(0.0).astype(float)
        needs_legacy_log1p = bool((raw_scores > _LEGACY_ACCESSIBILITY_LOG1P_THRESHOLD).any())
        if needs_legacy_log1p:
            df["accessibility_score"] = raw_scores.map(lambda x: float(math.log1p(x)) if x > 0.0 else float(x))
            log.info(
                "Detected legacy raw promoter accessibility scale in %s; applied log1p normalization on load",
                path.name,
            )
        else:
            df["accessibility_score"] = raw_scores

        for _, row in df.iterrows():
            gene = str(row["gene_symbol"]).upper()
            score = float(row.get("accessibility_score", 0.0))
            self._atac_index[gene] = ATACFeatures(
                peak_accessibility=score,
                peak_to_gene_linked=score > 0.0,
                celltype_specificity=None,
            )

        log.info("Loaded ATAC accessibility for %d genes from %s", len(self._atac_index), path.name)

    # ── Public lookups ────────────────────────────────────────────────────────

    def get_motif_features(self, source_tf: str, target_gene: str) -> MotifFeatures | None:
        """Return motif features for a TF→gene pair, or None if not available."""
        if not self._loaded:
            self.load()
        k = (source_tf.upper(), target_gene.upper())
        return self._motif_index.get(k)

    def get_atac_features(self, target_gene: str) -> ATACFeatures | None:
        """Return promoter accessibility features for a gene, or None if not available."""
        if not self._loaded:
            self.load()
        return self._atac_index.get(target_gene.upper())

    def motif_targets_for_tf(self, source_tf: str) -> set[str]:
        """Return all target genes with motif support for a TF."""
        if not self._loaded:
            self.load()
        tf_u = str(source_tf).strip().upper()
        return {tg for (tf, tg), m in self._motif_index.items() if tf == tf_u and bool(m.motif_present)}

    def accessible_genes(self) -> set[str]:
        """Return genes with non-zero promoter accessibility support."""
        if not self._loaded:
            self.load()
        return {g for g, a in self._atac_index.items() if (a.peak_accessibility or 0.0) > 0.0}

    @property
    def has_motif(self) -> bool:
        return bool(self._motif_index)

    @property
    def has_atac(self) -> bool:
        return bool(self._atac_index)

    @property
    def manifest(self) -> dict:
        return self._manifest

    # ── QC summary ────────────────────────────────────────────────────────────

    def qc_summary(self) -> dict:
        """Return a summary dict of loaded modalities and their coverage."""
        return {
            "motif_pairs_loaded": len(self._motif_index),
            "atac_genes_loaded": len(self._atac_index),
            "motif_pairs_with_hit": sum(1 for m in self._motif_index.values() if m.motif_present),
            "manifest_path": str(self._manifest_path),
            "manifest_qc": self._manifest.get("qc_report", {}) or self._manifest.get("qc", {}),
        }


# ── Module-level singleton factory ────────────────────────────────────────────

_LOADERS: dict[str, MultimodalFeatureLoader] = {}


def get_loader(manifest_path: str | Path) -> MultimodalFeatureLoader:
    """
    Return (and cache) a MultimodalFeatureLoader for a given manifest path.
    Safe to call multiple times — loads once.
    """
    key = str(Path(manifest_path).resolve())
    if key not in _LOADERS:
        _LOADERS[key] = MultimodalFeatureLoader(key)
    loader = _LOADERS[key]
    if not loader._loaded:
        loader.load()
    return loader
