from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoredEdge(BaseModel):
    """Binary edge output: P(regulatory edge present) in (0, 1)."""

    source_tf: str
    target_gene: str
    p_present: float = Field(ge=0.0, le=1.0, description="Probability the TF→gene edge is present in context")
    logit: float | None = Field(default=None, description="Pre-sigmoid logit, if available")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Decode ordering; set equal to p_present for EAGER"
    )
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    mechanism_reasoning: str = ""


class GraphMeta(BaseModel):
    species: str = "human"
    cell_type: str | None = None
    model_version: str = "grnagent_eager_v1"


class Network(BaseModel):
    context_id: str
    edges: list[ScoredEdge] = Field(default_factory=list)
    graph_metadata: GraphMeta = Field(default_factory=GraphMeta)
