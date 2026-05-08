from __future__ import annotations

import numpy as np

from grn_agent.schemas import SplitManifest, SplitStrategy, SplitSubset


def make_random_train_mask(n_cells: int, train_frac: float = 0.8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros(n_cells, dtype=bool)
    idx = rng.permutation(n_cells)[: int(n_cells * train_frac)]
    m[idx] = True
    return m


def leave_one_tf_out_mask(
    pairs: list[tuple[str, str]],
    held_tf: str,
) -> np.ndarray:
    """
    Boolean mask over **edges** (not cells): True if edge is allowed in train for this fold.
    """
    m = np.array([tf != held_tf for tf, _ in pairs], dtype=bool)
    return m


def fold_ids(manifest: SplitManifest, strategy: SplitStrategy) -> list[str]:
    ids = sorted({r.fold_id for r in manifest.rows if r.split_name == strategy})
    if not ids:
        raise ValueError(f"No folds for strategy={strategy}")
    return ids


def pairs_for_subset(
    manifest: SplitManifest,
    strategy: SplitStrategy,
    fold_id: str,
    subset: SplitSubset,
) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for r in manifest.rows:
        if r.split_name == strategy and r.fold_id == fold_id and r.subset == subset:
            out.add((r.source_tf, r.target_gene))
    return out


def filter_pairs_for_subset(
    pairs: list[tuple[str, str]],
    manifest: SplitManifest,
    strategy: SplitStrategy,
    fold_id: str,
    subset: SplitSubset,
) -> np.ndarray:
    allowed = pairs_for_subset(manifest, strategy, fold_id, subset)
    return np.array([(tf, tg) in allowed for tf, tg in pairs], dtype=bool)


def _unique_values(manifest: SplitManifest, strategy: SplitStrategy, fold_id: str, subset: SplitSubset, field: str) -> set[str]:
    out: set[str] = set()
    for r in manifest.rows:
        if r.split_name != strategy or r.fold_id != fold_id or r.subset != subset:
            continue
        v = getattr(r, field)
        if v:
            out.add(v)
    return out


def validate_fold_no_leakage(
    manifest: SplitManifest,
    strategy: SplitStrategy,
    fold_id: str,
) -> None:
    tr = pairs_for_subset(manifest, strategy, fold_id, SplitSubset.train)
    va = pairs_for_subset(manifest, strategy, fold_id, SplitSubset.val)
    te = pairs_for_subset(manifest, strategy, fold_id, SplitSubset.test)
    overlap_tv = tr & va
    overlap_tt = tr & te
    overlap_vt = va & te
    if overlap_tv:
        raise ValueError(f"Fold {fold_id} has train/val pair overlap: {len(overlap_tv)} edges")
    if overlap_tt:
        raise ValueError(f"Fold {fold_id} has train/test pair overlap: {len(overlap_tt)} edges")
    if overlap_vt:
        raise ValueError(f"Fold {fold_id} has val/test pair overlap: {len(overlap_vt)} edges")

    if strategy == SplitStrategy.leave_one_tf_out:
        tr_tfs = {tf for tf, _ in tr}
        va_tfs = {tf for tf, _ in va}
        te_tfs = {tf for tf, _ in te}
        shared_train_val = tr_tfs & va_tfs
        shared_train_test = tr_tfs & te_tfs
        shared_val_test = va_tfs & te_tfs
        if shared_train_val:
            raise ValueError(f"LOTO leakage in fold {fold_id}: train/val share TFs {sorted(shared_train_val)[:5]}")
        if shared_train_test:
            raise ValueError(f"LOTO leakage in fold {fold_id}: train/test share TFs {sorted(shared_train_test)[:5]}")
        if shared_val_test:
            raise ValueError(f"LOTO leakage in fold {fold_id}: val/test share TFs {sorted(shared_val_test)[:5]}")
    elif strategy == SplitStrategy.dataset_holdout:
        tr_labs = _unique_values(manifest, strategy, fold_id, SplitSubset.train, "lab_id")
        te_labs = _unique_values(manifest, strategy, fold_id, SplitSubset.test, "lab_id")
        tr_ds = _unique_values(manifest, strategy, fold_id, SplitSubset.train, "dataset_id")
        te_ds = _unique_values(manifest, strategy, fold_id, SplitSubset.test, "dataset_id")
        if tr_labs and te_labs and (tr_labs & te_labs):
            raise ValueError(f"Dataset holdout leakage in fold {fold_id}: overlapping lab_id")
        if tr_ds and te_ds and (tr_ds & te_ds):
            raise ValueError(f"Dataset holdout leakage in fold {fold_id}: overlapping dataset_id")
    elif strategy == SplitStrategy.cell_type_holdout:
        tr_ct = _unique_values(manifest, strategy, fold_id, SplitSubset.train, "cell_type")
        te_ct = _unique_values(manifest, strategy, fold_id, SplitSubset.test, "cell_type")
        if tr_ct and te_ct and (tr_ct & te_ct):
            raise ValueError(f"Cell-type holdout leakage in fold {fold_id}: overlapping cell_type")
    elif strategy == SplitStrategy.species_transfer:
        tr_sp = _unique_values(manifest, strategy, fold_id, SplitSubset.train, "species")
        te_sp = _unique_values(manifest, strategy, fold_id, SplitSubset.test, "species")
        if tr_sp and te_sp and (tr_sp & te_sp):
            raise ValueError(f"Species transfer leakage in fold {fold_id}: overlapping species")
