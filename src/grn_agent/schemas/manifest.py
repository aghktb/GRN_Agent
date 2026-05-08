from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .enums import EvalTrack


class RunManifest(BaseModel):
    """Run provenance (plan Phase 1)."""

    run_id: str
    dataset_id: str
    split_id: str = "default"
    eval_track: EvalTrack = EvalTrack.NO_LITERATURE
    model_version: str = "grnagent_v1"
    git_sha: str | None = None
    seed: int = 0
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stages_completed: list[str] = Field(default_factory=list)
    artifact_dir: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
