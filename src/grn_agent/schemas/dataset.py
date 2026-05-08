from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GeneMeta(BaseModel):
    gene_id: str
    symbol: str | None = None
    ensembl_id: str | None = None


class SampleMeta(BaseModel):
    sample_id: str
    cell_type: str | None = None
    batch: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """Standardized multi-omics dataset (SDD §3.2)."""

    dataset_id: str
    species: str
    modalities: list[str] = Field(default_factory=list)
    genes: list[GeneMeta] = Field(default_factory=list)
    samples: list[SampleMeta] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Expression matrix: cells x genes (path to .npy/.csv or in-memory ref key)
    expression_matrix_key: str | None = None
