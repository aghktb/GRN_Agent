"""
Load gold TF→target labels as binary \ {0, 1\ } (0 = absent, 1 = present).

Signed columns (activation / repression) map to 1; explicit negative / none maps to 0.
Presence-only files (source_tf, target_gene only) treat every listed edge as 1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _pick_col(colmap: dict[str, str], *names: str, required: bool = True) -> str | None:
    for n in names:
        k = _normalize_col(n)
        if k in colmap:
            return colmap[k]
    if required:
        raise ValueError(f"Missing column (one of {names}); got {list(colmap.values())}")
    return None


def _detect_numeric_label_mode(s: "pd.Series") -> str:
    """Return 'binary' (0/1 = absent/present) or '3class' (0/1/2 = act/rep/none) or 'string'."""
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) == 0:
        return "string"
    u = {int(round(float(x))) for x in vals.tolist()}
    if u.issubset({0, 1}):
        return "binary"
    if u.issubset({0, 1, 2}):
        return "3class"
    return "string"


def _row_to_label(raw: object, num_mode: str) -> int:
    """Map cell to binary present (1) / absent (0)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return 0
    if num_mode == "binary" and isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return 1 if int(raw) == 1 else 0
    if num_mode == "3class" and isinstance(raw, (int, float)) and not isinstance(raw, bool):
        i = int(raw)
        return 0 if i == 2 else 1
    s = str(raw).strip()
    s_low = s.lower()
    for neg in ("none", "absent", "negative", "no"):
        if s_low == neg:
            return 0
    for pos in (
        "activation",
        "repression",
        "present",
        "positive",
        "yes",
        "activates",
        "represses",
    ):
        if s_low == pos:
            return 1
    if s in ("0", "1"):
        return int(s)
    raise ValueError(f"Unknown label cell: {raw!r}")


def load_gold_edge_labels(path: str | Path) -> dict[tuple[str, str], int]:
    """
    Return map (source_tf, target_gene) -> 0/1.
    If no label column, every row is 1.
    """
    p = Path(path)
    sep = "\t" if p.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(p, sep=sep)
    colmap = {_normalize_col(c): c for c in df.columns}
    c_tf = _pick_col(colmap, "source_tf", "tf", "source", "regulator", "gene1")
    c_tgt = _pick_col(colmap, "target_gene", "target", "gene", "gene2")
    c_lab = _pick_col(colmap, "regulation_type", "label", "class", "sign", required=False)
    num_mode = _detect_numeric_label_mode(df[c_lab]) if c_lab is not None else "string"

    out: dict[tuple[str, str], int] = {}
    for _, row in df.iterrows():
        tf = str(row[c_tf]).strip().upper()
        g = str(row[c_tgt]).strip().upper()
        if c_lab is None:
            lab = 1
        else:
            lab = _row_to_label(row[c_lab], num_mode)
        out[(tf, g)] = int(lab)
    return out


def load_gold_edge_presence(path: str | Path) -> set[tuple[str, str]]:
    """Load positive TF→target edges (binary gold without explicit zeros)."""
    p = Path(path)
    sep = "\t" if p.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(p, sep=sep)
    colmap = {_normalize_col(c): c for c in df.columns}
    c_tf = _pick_col(colmap, "source_tf", "tf", "source", "regulator", "gene1")
    c_tgt = _pick_col(colmap, "target_gene", "target", "gene", "gene2")
    return {(str(row[c_tf]).strip().upper(), str(row[c_tgt]).strip().upper()) for _, row in df.iterrows()}


def inspect_gold_label_mode(path: str | Path) -> str:
    """Return 'signed' if a sign/label column exists, else 'binary' (all rows positive)."""
    p = Path(path)
    sep = "\t" if p.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(p, sep=sep, nrows=1)
    colmap = {_normalize_col(c): c for c in df.columns}
    c_lab = _pick_col(colmap, "regulation_type", "label", "class", "sign", required=False)
    return "signed" if c_lab is not None else "binary"
