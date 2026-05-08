from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExpressionFeatures(BaseModel):
    """Algorithm 8: expression node — z_t, z_g, activity(t), mean expression."""

    tf_zscore: float | None = None
    target_zscore: float | None = None
    tf_activity_proxy: float | None = None
    tf_mean_expr: float | None = None
    target_mean_expr: float | None = None
    tf_dropout_rate: float | None = None
    target_dropout_rate: float | None = None


class NetworkFeatures(BaseModel):
    """Algorithm 8: network node — corr(t,g), shared neighbors, module membership."""

    pearson_r: float | None = None
    partial_corr: float | None = None
    in_same_module: bool | None = None
    k_hop_distance: int | None = None
    shared_neighbors: int | None = None
    shared_neighbor_names: list[str] = Field(default_factory=list)


class ATACFeatures(BaseModel):
    """Algorithm 8: binding — accessibility, peak-to-gene score."""

    peak_accessibility: float | None = None
    peak_to_gene_linked: bool | None = None
    celltype_specificity: float | None = None


class MotifFeatures(BaseModel):
    """Algorithm 8: binding — motif score, n regions."""

    motif_present: bool | None = None
    motif_score: float | None = None
    n_supporting_regions: int | None = None


class OrthologyFeatures(BaseModel):
    """Algorithm 8: orthology node — ortholog support/confidence."""

    ortholog_support: float | None = None
    ortholog_confidence: str | None = None
    supporting_species: list[str] = Field(default_factory=list)
    conserved_in_human: bool | None = None
    conserved_in_mouse: bool | None = None


class LiteratureFeatures(BaseModel):
    """Structured literature features (Track 2/3 only, SDD §8)."""

    lit_activation_prob: float | None = None
    lit_repression_prob: float | None = None
    num_supporting_pmids: int | None = None
    best_assay_weight: float | None = None
    latest_year_included: int | None = None


class FeatureBundle(BaseModel):
    context_id: str
    source_tf: str
    target_gene: str
    expression: ExpressionFeatures = Field(default_factory=ExpressionFeatures)
    network: NetworkFeatures = Field(default_factory=NetworkFeatures)
    atac: ATACFeatures | None = None
    motif: MotifFeatures | None = None
    orthology: OrthologyFeatures | None = None
    literature: LiteratureFeatures | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
