from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from grn_agent.schemas import EvalTrack


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def save_yaml_config(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def parse_eval_track(s: str) -> EvalTrack:
    key = (s or "").strip().lower().replace("-", "_")
    mapping = {
        "track1": EvalTrack.NO_LITERATURE,
        "track1_no_literature": EvalTrack.NO_LITERATURE,
        "no_literature": EvalTrack.NO_LITERATURE,
        "track2": EvalTrack.TIME_SLICED_LIT,
        "track2_time_sliced_literature": EvalTrack.TIME_SLICED_LIT,
        "track3": EvalTrack.ASSISTED,
        "track3_assisted": EvalTrack.ASSISTED,
    }
    if key in mapping:
        return mapping[key]
    try:
        return EvalTrack(s)
    except ValueError:
        return EvalTrack.NO_LITERATURE
