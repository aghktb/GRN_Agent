from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SplitSubset(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class SplitStrategy(str, Enum):
    leave_one_tf_out = "leave_one_tf_out"
    dataset_holdout = "dataset_holdout"
    cell_type_holdout = "cell_type_holdout"
    species_transfer = "species_transfer"


class SplitManifestRow(BaseModel):
    split_name: SplitStrategy
    fold_id: str
    subset: SplitSubset
    source_tf: str
    target_gene: str
    lab_id: str | None = None
    dataset_id: str | None = None
    cell_type: str | None = None
    species: str | None = None
    tf_frequency_bucket: str | None = None
    time_cutoff_year: int | None = Field(default=None, ge=1900, le=2100)


class SplitManifest(BaseModel):
    rows: list[SplitManifestRow] = Field(default_factory=list)

