from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CellContext(BaseModel):
    """Biological context for GRN construction (SDD §3.3)."""

    context_id: str
    cell_type: str | None = None
    module_genes: list[str] = Field(default_factory=list)
    candidate_tfs: list[str] = Field(default_factory=list)
    cell_indices: list[int] = Field(default_factory=list, description="Indices into expression matrix rows")
    metadata: dict[str, Any] = Field(default_factory=dict)
