from __future__ import annotations

from pathlib import Path

import pandas as pd

from grn_agent.schemas import SplitManifest, SplitManifestRow, SplitStrategy, SplitSubset


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _pick_col(df: pd.DataFrame, *aliases: str, required: bool = True) -> str | None:
    cmap = {_normalize_col(c): c for c in df.columns}
    for n in aliases:
        k = _normalize_col(n)
        if k in cmap:
            return cmap[k]
    if required:
        raise ValueError(f"Missing required column one of {aliases}; got {list(df.columns)}")
    return None


def load_split_manifest(path: str | Path) -> SplitManifest:
    p = Path(path)
    sep = "\t" if p.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(p, sep=sep)
    c_split = _pick_col(df, "split_name", "strategy")
    c_fold = _pick_col(df, "fold_id", "fold")
    c_subset = _pick_col(df, "subset")
    c_tf = _pick_col(df, "source_tf", "tf", "source")
    c_tg = _pick_col(df, "target_gene", "target", "gene")
    c_lab = _pick_col(df, "lab_id", required=False)
    c_ds = _pick_col(df, "dataset_id", required=False)
    c_ct = _pick_col(df, "cell_type", required=False)
    c_sp = _pick_col(df, "species", required=False)
    c_tfb = _pick_col(df, "tf_frequency_bucket", required=False)
    c_cutoff = _pick_col(df, "time_cutoff_year", required=False)

    rows: list[SplitManifestRow] = []
    for _, row in df.iterrows():
        cutoff = row[c_cutoff] if c_cutoff is not None else None
        if isinstance(cutoff, float) and pd.isna(cutoff):
            cutoff = None
        rows.append(
            SplitManifestRow(
                split_name=SplitStrategy(str(row[c_split]).strip()),
                fold_id=str(row[c_fold]).strip(),
                subset=SplitSubset(str(row[c_subset]).strip()),
                source_tf=str(row[c_tf]).strip().upper(),
                target_gene=str(row[c_tg]).strip().upper(),
                lab_id=(str(row[c_lab]).strip() if c_lab is not None and not pd.isna(row[c_lab]) else None),
                dataset_id=(str(row[c_ds]).strip() if c_ds is not None and not pd.isna(row[c_ds]) else None),
                cell_type=(str(row[c_ct]).strip() if c_ct is not None and not pd.isna(row[c_ct]) else None),
                species=(str(row[c_sp]).strip() if c_sp is not None and not pd.isna(row[c_sp]) else None),
                tf_frequency_bucket=(str(row[c_tfb]).strip() if c_tfb is not None and not pd.isna(row[c_tfb]) else None),
                time_cutoff_year=(int(cutoff) if cutoff is not None else None),
            )
        )
    manifest = SplitManifest(rows=rows)
    if not manifest.rows:
        raise ValueError("Split manifest is empty")
    return manifest

