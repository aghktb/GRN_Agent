from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_json(path: str | Path, obj: BaseModel | dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json")
    else:
        data = obj
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path, model: type[T]) -> T:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return model.model_validate(raw)
