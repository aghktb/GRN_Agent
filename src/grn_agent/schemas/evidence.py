from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import EvidenceNodeType, RelationType
from .context import CellContext


class CandidateEdge(BaseModel):
    source_tf: str
    target_gene: str
    context_id: str = ""


class EvidenceNode(BaseModel):
    node_id: str
    node_type: EvidenceNodeType
    label: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EvidenceRelation(BaseModel):
    src_id: str
    dst_id: str
    relation: RelationType


class EvidenceGraph(BaseModel):
    """Serialized evidence graph (SDD §3.7)."""

    context: CellContext
    edge: CandidateEdge
    nodes: list[EvidenceNode] = Field(default_factory=list)
    relations: list[EvidenceRelation] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict, description="Summary dict mirror for APIs")
