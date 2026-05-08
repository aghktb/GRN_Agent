"""Emit JSON Schema for all public Pydantic models (plan Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

from .context import CellContext
from .dataset import Dataset
from .evidence import CandidateEdge, EvidenceGraph
from .features import FeatureBundle
from .manifest import RunManifest
from .priors import PriorBundle
from .scoring import Network, ScoredEdge


def export_json_schemas(output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [
        ("dataset", Dataset),
        ("cell_context", CellContext),
        ("feature_bundle", FeatureBundle),
        ("prior_bundle", PriorBundle),
        ("candidate_edge", CandidateEdge),
        ("evidence_graph", EvidenceGraph),
        ("scored_edge", ScoredEdge),
        ("network", Network),
        ("run_manifest", RunManifest),
    ]
    for name, model in models:
        schema = model.model_json_schema()
        (out / f"{name}.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
