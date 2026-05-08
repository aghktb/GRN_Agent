"""Tensorization for TF-centered EAGER windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

TF_EAGER_WINDOW_SIZE = 100
NO_TARGET_POSITION = TF_EAGER_WINDOW_SIZE
TARGET_POSITION_VOCAB = TF_EAGER_WINDOW_SIZE + 1
VALUE_DIM = 32
# Worst case: TF + CTX + 100 * [GENE, EXPR, NETWORK, MOTIF, ACC, LINK, PRIOR, ORTHO, LIT].
MAX_TOKENS = 2 + TF_EAGER_WINDOW_SIZE * 9
EDGE_COMPACT_MAX_TOKENS = 2 + TF_EAGER_WINDOW_SIZE
TOKEN_LAYOUT_EVIDENCE = "evidence_tokens"
TOKEN_LAYOUT_EDGE_COMPACT = "edge_compact"
NUM_TOKEN_KINDS = 12


class TfEagerTokenKind:
    TF = 0
    CTX = 1
    GENE = 2
    EXPR = 3
    NETWORK = 4
    MOTIF = 5
    ACC = 6
    LINK = 7
    PRIOR = 8
    ORTHO = 9
    LIT = 10
    PAD = 11


_TOKEN_KIND_NAME_TO_ID = {
    "tf": TfEagerTokenKind.TF,
    "ctx": TfEagerTokenKind.CTX,
    "context": TfEagerTokenKind.CTX,
    "gene": TfEagerTokenKind.GENE,
    "expr": TfEagerTokenKind.EXPR,
    "expression": TfEagerTokenKind.EXPR,
    "network": TfEagerTokenKind.NETWORK,
    "motif": TfEagerTokenKind.MOTIF,
    "acc": TfEagerTokenKind.ACC,
    "accessibility": TfEagerTokenKind.ACC,
    "link": TfEagerTokenKind.LINK,
    "linkage": TfEagerTokenKind.LINK,
    "prior": TfEagerTokenKind.PRIOR,
    "priors": TfEagerTokenKind.PRIOR,
    "ortho": TfEagerTokenKind.ORTHO,
    "orthology": TfEagerTokenKind.ORTHO,
    "lit": TfEagerTokenKind.LIT,
    "literature": TfEagerTokenKind.LIT,
}


@dataclass
class TfEagerWindowBatch:
    token_kind: torch.Tensor
    x_value: torch.Tensor
    conf: torch.Tensor
    token_target_pos: torch.Tensor
    token_mask: torch.Tensor
    modality: torch.Tensor
    mech_mask: torch.Tensor
    func_mask: torch.Tensor
    context_idx: torch.Tensor
    tf_idx: torch.Tensor
    gene_idx: torch.Tensor
    gene_pos: torch.Tensor
    gene_mask: torch.Tensor
    labels: torch.Tensor
    sample_weight: torch.Tensor


def stack_window_batches(batches: list[TfEagerWindowBatch]) -> TfEagerWindowBatch:
    if not batches:
        raise ValueError("stack_window_batches requires at least one batch")
    return TfEagerWindowBatch(
        token_kind=torch.cat([b.token_kind for b in batches], dim=0),
        x_value=torch.cat([b.x_value for b in batches], dim=0),
        conf=torch.cat([b.conf for b in batches], dim=0),
        token_target_pos=torch.cat([b.token_target_pos for b in batches], dim=0),
        token_mask=torch.cat([b.token_mask for b in batches], dim=0),
        modality=torch.cat([b.modality for b in batches], dim=0),
        mech_mask=torch.cat([b.mech_mask for b in batches], dim=0),
        func_mask=torch.cat([b.func_mask for b in batches], dim=0),
        context_idx=torch.cat([b.context_idx for b in batches], dim=0),
        tf_idx=torch.cat([b.tf_idx for b in batches], dim=0),
        gene_idx=torch.cat([b.gene_idx for b in batches], dim=0),
        gene_pos=torch.cat([b.gene_pos for b in batches], dim=0),
        gene_mask=torch.cat([b.gene_mask for b in batches], dim=0),
        labels=torch.cat([b.labels for b in batches], dim=0),
        sample_weight=torch.cat([b.sample_weight for b in batches], dim=0),
    )


def _hash_bucket(s: str, mod: int) -> int:
    h = 2166136261
    for c in str(s).encode("utf-8", errors="ignore"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h % mod)


def _as_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _float_or_zero(v: object) -> float:
    return _as_float(v, 0.0)


def _bool_value(v: object) -> float:
    return 1.0 if v is True else 0.0


def _confidence(*values: object) -> float:
    return 1.0 if any(v is not None for v in values) else 0.0


def _vec(values: list[object]) -> np.ndarray:
    out = np.zeros(VALUE_DIM, dtype=np.float32)
    for i, v in enumerate(values[:VALUE_DIM]):
        out[i] = _float_or_zero(v)
    return out


def _append_token(
    kinds: list[int],
    xs: list[np.ndarray],
    confs: list[float],
    target_pos: list[int],
    token_kind: int,
    values: list[object],
    conf: float,
    *,
    pos: int = NO_TARGET_POSITION,
) -> None:
    if len(kinds) >= MAX_TOKENS:
        return
    kinds.append(token_kind)
    xs.append(_vec(values))
    confs.append(float(np.clip(conf, 0.0, 1.0)))
    target_pos.append(int(np.clip(pos, 0, NO_TARGET_POSITION)))


def _has_confidence(*values: object) -> bool:
    return any(v is not None for v in values)


def _normalize_drop_token_kinds(drop_token_kinds: list[str] | tuple[str, ...] | None) -> set[int]:
    out: set[int] = set()
    for raw in drop_token_kinds or ():
        key = str(raw or "").strip().lower()
        if not key:
            continue
        token_kind = _TOKEN_KIND_NAME_TO_ID.get(key)
        if token_kind is None:
            raise ValueError(f"unknown tf-eager drop_token_kinds entry: {raw!r}")
        out.add(token_kind)
    return out


def _compact_edge_values(
    ev: dict[str, Any],
    expr: dict[str, Any],
    net: dict[str, Any],
    motif: dict[str, Any],
    acc: dict[str, Any],
    link: dict[str, Any],
    prior: dict[str, Any],
    orth: dict[str, Any],
    lit: dict[str, Any],
) -> list[object]:
    return [
        expr.get("z_t", ev.get("z_t")),
        expr.get("z_g", ev.get("z_g")),
        expr.get("activity_t", ev.get("activity_t")),
        expr.get("mean_expr_t"),
        expr.get("mean_expr_g"),
        expr.get("dropout_t"),
        expr.get("dropout_g"),
        net.get("pearson_r", ev.get("correlation")),
        net.get("partial_corr"),
        net.get("in_same_module", ev.get("in_same_module")),
        net.get("k_hop_distance"),
        net.get("shared_neighbors", ev.get("shared_neighbors")),
        _bool_value(motif.get("motif_present", ev.get("motif_present"))),
        motif.get("motif_score"),
        motif.get("n_motif_regions"),
        acc.get("peak_accessibility", ev.get("accessibility")),
        acc.get("celltype_specificity"),
        _bool_value(link.get("peak_to_gene_linked")),
        link.get("linkage_score"),
        ev.get("ensemble_prior"),
        prior.get("p_grnboost"),
        prior.get("p_genie3"),
        prior.get("p_pidc"),
        prior.get("scenic_regulon_support"),
        prior.get("bootstrap_stability"),
        prior.get("ensemble_prior", ev.get("ensemble_prior")),
        orth.get("ortholog_support"),
        orth.get("ortholog_confidence"),
        orth.get("conserved_in_human"),
        orth.get("conserved_in_mouse"),
        lit.get("lit_activation_prob"),
        lit.get("lit_repression_prob"),
    ]


def window_record_to_batch(
    record: dict[str, Any],
    *,
    token_layout: str = TOKEN_LAYOUT_EVIDENCE,
    drop_token_kinds: list[str] | tuple[str, ...] | None = None,
    tf_vocab: int = 8192,
    gene_vocab: int = 8192,
    context_vocab: int = 1024,
) -> TfEagerWindowBatch:
    tf = str(record["source_tf"]).strip().upper()
    context = dict(record.get("context", {}))
    context_id = str(context.get("context_id", "context"))
    cell_type = str(context.get("cell_type") or "")
    genes = list(record.get("genes", []))[:TF_EAGER_WINDOW_SIZE]
    layout = str(token_layout or TOKEN_LAYOUT_EVIDENCE).strip().lower()
    if layout not in {TOKEN_LAYOUT_EVIDENCE, TOKEN_LAYOUT_EDGE_COMPACT}:
        raise ValueError(f"unknown tf-eager token_layout: {token_layout!r}")
    dropped = _normalize_drop_token_kinds(drop_token_kinds)
    max_tokens = EDGE_COMPACT_MAX_TOKENS if layout == TOKEN_LAYOUT_EDGE_COMPACT else MAX_TOKENS

    kinds: list[int] = []
    xs: list[np.ndarray] = []
    confs: list[float] = []
    token_target_pos: list[int] = []
    _append_token(kinds, xs, confs, token_target_pos, TfEagerTokenKind.TF, [0.0], 1.0)
    _append_token(
        kinds,
        xs,
        confs,
        token_target_pos,
        TfEagerTokenKind.CTX,
        [len(genes), context.get("n_cells"), context.get("n_module_genes")],
        1.0,
    )

    gene_hashes = np.zeros(TF_EAGER_WINDOW_SIZE, dtype=np.int64)
    gene_positions = np.full(TF_EAGER_WINDOW_SIZE, NO_TARGET_POSITION, dtype=np.int64)
    gene_mask = np.zeros(TF_EAGER_WINDOW_SIZE, dtype=np.float32)
    labels = np.zeros(TF_EAGER_WINDOW_SIZE, dtype=np.float32)
    weights = np.zeros(TF_EAGER_WINDOW_SIZE, dtype=np.float32)

    has_expr = False
    has_motif = False
    has_acc = False
    has_link = False

    mech = np.zeros(max_tokens, dtype=np.float32)
    func = np.zeros(max_tokens, dtype=np.float32)

    for i, g in enumerate(genes):
        target = str(g.get("target_gene", "")).strip().upper()
        gene_hashes[i] = _hash_bucket(target, max(1, int(gene_vocab)))
        gene_positions[i] = i
        gene_mask[i] = 1.0
        labels[i] = float(g.get("label", 0.0))
        weights[i] = float(g.get("sample_weight", 1.0))
        ev = dict(g.get("evidence", {}))
        expr = dict(g.get("expression", {}))
        net = dict(g.get("network", {}))
        motif = dict(g.get("motif", {}))
        acc = dict(g.get("accessibility", {}))
        link = dict(g.get("linkage", {}))
        prior = dict(g.get("prior", {}))
        orth = dict(g.get("orthology", {}))
        lit = dict(g.get("literature", {}))
        keep_motif = TfEagerTokenKind.MOTIF not in dropped
        keep_acc = TfEagerTokenKind.ACC not in dropped
        keep_link = TfEagerTokenKind.LINK not in dropped
        keep_prior = TfEagerTokenKind.PRIOR not in dropped
        keep_ortho = TfEagerTokenKind.ORTHO not in dropped
        keep_lit = TfEagerTokenKind.LIT not in dropped

        if layout == TOKEN_LAYOUT_EDGE_COMPACT:
            token_idx = len(kinds)
            edge_vals = _compact_edge_values(ev, expr, net, motif, acc, link, prior, orth, lit)
            expr_conf = _has_confidence(
                expr.get("z_t", ev.get("z_t")),
                expr.get("z_g", ev.get("z_g")),
                expr.get("activity_t", ev.get("activity_t")),
                expr.get("mean_expr_t"),
                expr.get("mean_expr_g"),
                expr.get("dropout_t"),
                expr.get("dropout_g"),
                net.get("pearson_r", ev.get("correlation")),
                net.get("partial_corr"),
                net.get("in_same_module", ev.get("in_same_module")),
                net.get("k_hop_distance"),
                net.get("shared_neighbors", ev.get("shared_neighbors")),
            )
            motif_conf = _has_confidence(
                motif.get("motif_present", ev.get("motif_present")),
                motif.get("motif_score"),
                motif.get("n_motif_regions"),
            ) if keep_motif else False
            acc_conf = _has_confidence(
                acc.get("peak_accessibility", ev.get("accessibility")),
                acc.get("celltype_specificity"),
            ) if keep_acc else False
            link_conf = _has_confidence(
                link.get("peak_to_gene_linked"),
                link.get("linkage_score"),
                ev.get("ensemble_prior"),
            ) if keep_link else False
            prior_conf = _has_confidence(
                prior.get("p_grnboost"),
                prior.get("p_genie3"),
                prior.get("p_pidc"),
                prior.get("scenic_regulon_support"),
                prior.get("bootstrap_stability"),
                prior.get("ensemble_prior", ev.get("ensemble_prior")),
            ) if keep_prior else False
            ortho_conf = _has_confidence(
                orth.get("ortholog_support"),
                orth.get("ortholog_confidence"),
                orth.get("conserved_in_human"),
                orth.get("conserved_in_mouse"),
            ) if keep_ortho else False
            lit_conf = _has_confidence(
                lit.get("lit_activation_prob"),
                lit.get("lit_repression_prob"),
                lit.get("num_supporting_pmids"),
                lit.get("best_assay_weight"),
                lit.get("latest_year_included"),
            ) if keep_lit else False

            if not keep_motif:
                edge_vals[12:15] = [0.0, 0.0, 0.0]
            if not keep_acc:
                edge_vals[15:17] = [0.0, 0.0]
            if not keep_link:
                edge_vals[17:19] = [0.0, 0.0]
            if not keep_prior:
                edge_vals[19:26] = [0.0] * 7
            if not keep_ortho:
                edge_vals[26:30] = [0.0] * 4
            if not keep_lit:
                edge_vals[30:32] = [0.0, 0.0]
            _append_token(
                kinds,
                xs,
                confs,
                token_target_pos,
                TfEagerTokenKind.GENE,
                edge_vals,
                1.0 if expr_conf or motif_conf or acc_conf or link_conf or prior_conf or ortho_conf or lit_conf else 0.0,
                pos=i,
            )
            if token_idx < len(mech) and (motif_conf or acc_conf or link_conf):
                mech[token_idx] = 1.0
            if token_idx < len(func) and expr_conf:
                func[token_idx] = 1.0
            has_expr = has_expr or expr_conf
            has_motif = has_motif or motif_conf
            has_acc = has_acc or acc_conf
            has_link = has_link or link_conf
            continue

        _append_token(kinds, xs, confs, token_target_pos, TfEagerTokenKind.GENE, [0.0], 1.0, pos=i)

        expr_vals = [
            expr.get("z_t", ev.get("z_t")),
            expr.get("z_g", ev.get("z_g")),
            expr.get("activity_t", ev.get("activity_t")),
            expr.get("mean_expr_t"),
            expr.get("mean_expr_g"),
            expr.get("dropout_t"),
            expr.get("dropout_g"),
        ]
        _append_token(kinds, xs, confs, token_target_pos, TfEagerTokenKind.EXPR, expr_vals, _confidence(*expr_vals), pos=i)
        has_expr = has_expr or _confidence(*expr_vals) > 0.0

        network_vals = [
            net.get("pearson_r", ev.get("correlation")),
            net.get("partial_corr"),
            net.get("in_same_module", ev.get("in_same_module")),
            net.get("k_hop_distance"),
            net.get("shared_neighbors", ev.get("shared_neighbors")),
        ]
        _append_token(
            kinds,
            xs,
            confs,
            token_target_pos,
            TfEagerTokenKind.NETWORK,
            network_vals,
            _confidence(*network_vals),
            pos=i,
        )
        has_expr = has_expr or _confidence(*network_vals) > 0.0

        motif_vals = [
            _bool_value(motif.get("motif_present", ev.get("motif_present"))),
            motif.get("motif_score"),
            motif.get("n_motif_regions"),
        ]
        if keep_motif:
            _append_token(
                kinds,
                xs,
                confs,
                token_target_pos,
                TfEagerTokenKind.MOTIF,
                motif_vals,
                _confidence(motif.get("motif_present", ev.get("motif_present")), motif.get("motif_score")),
                pos=i,
            )
            has_motif = has_motif or motif.get("motif_present", ev.get("motif_present")) is not None

        acc_vals = [
            acc.get("peak_accessibility", ev.get("accessibility")),
            acc.get("celltype_specificity"),
        ]
        if keep_acc:
            _append_token(kinds, xs, confs, token_target_pos, TfEagerTokenKind.ACC, acc_vals, _confidence(*acc_vals), pos=i)
            has_acc = has_acc or acc.get("peak_accessibility", ev.get("accessibility")) is not None

        link_vals = [
            _bool_value(link.get("peak_to_gene_linked")),
            link.get("linkage_score"),
            ev.get("ensemble_prior"),
        ]
        if keep_link:
            _append_token(
                kinds,
                xs,
                confs,
                token_target_pos,
                TfEagerTokenKind.LINK,
                link_vals,
                _confidence(link.get("peak_to_gene_linked"), link.get("linkage_score"), ev.get("ensemble_prior")),
                pos=i,
            )
            has_link = has_link or link.get("peak_to_gene_linked") is not None

        prior_vals = [
            prior.get("p_grnboost"),
            prior.get("p_genie3"),
            prior.get("p_pidc"),
            prior.get("scenic_regulon_support"),
            prior.get("bootstrap_stability"),
            prior.get("ensemble_prior", ev.get("ensemble_prior")),
        ]
        if keep_prior:
            _append_token(
                kinds,
                xs,
                confs,
                token_target_pos,
                TfEagerTokenKind.PRIOR,
                prior_vals,
                _confidence(*prior_vals),
                pos=i,
            )

        orth_vals = [
            orth.get("ortholog_support"),
            orth.get("ortholog_confidence"),
            orth.get("conserved_in_human"),
            orth.get("conserved_in_mouse"),
        ]
        if keep_ortho:
            _append_token(
                kinds,
                xs,
                confs,
                token_target_pos,
                TfEagerTokenKind.ORTHO,
                orth_vals,
                _confidence(*orth_vals),
                pos=i,
            )

        if lit and keep_lit:
            lit_vals = [
                lit.get("lit_activation_prob"),
                lit.get("lit_repression_prob"),
                lit.get("num_supporting_pmids"),
                lit.get("best_assay_weight"),
                lit.get("latest_year_included"),
            ]
            _append_token(
                kinds,
                xs,
                confs,
                token_target_pos,
                TfEagerTokenKind.LIT,
                lit_vals,
                _confidence(*lit_vals),
                pos=i,
            )

    n_real = len(kinds)
    while len(kinds) < max_tokens:
        kinds.append(TfEagerTokenKind.PAD)
        xs.append(np.zeros(VALUE_DIM, dtype=np.float32))
        confs.append(0.0)
        token_target_pos.append(NO_TARGET_POSITION)

    token_mask = np.zeros(max_tokens, dtype=np.float32)
    token_mask[:n_real] = 1.0
    if layout == TOKEN_LAYOUT_EVIDENCE:
        for j, k in enumerate(kinds[:n_real]):
            if k in (TfEagerTokenKind.MOTIF, TfEagerTokenKind.ACC, TfEagerTokenKind.LINK):
                mech[j] = 1.0
            if k in (TfEagerTokenKind.EXPR, TfEagerTokenKind.NETWORK):
                func[j] = 1.0

    return TfEagerWindowBatch(
        token_kind=torch.from_numpy(np.asarray([kinds], dtype=np.int64)),
        x_value=torch.from_numpy(np.stack(xs)[None, ...].astype(np.float32)),
        conf=torch.from_numpy(np.asarray([confs], dtype=np.float32)),
        token_target_pos=torch.from_numpy(np.asarray([token_target_pos], dtype=np.int64)),
        token_mask=torch.from_numpy(np.asarray([token_mask], dtype=np.float32)),
        modality=torch.from_numpy(np.asarray([[1.0 if has_expr else 0.0, 1.0 if has_acc else 0.0, 1.0 if has_motif else 0.0, 1.0 if has_link else 0.0]], dtype=np.float32)),
        mech_mask=torch.from_numpy(np.asarray([mech], dtype=np.float32)),
        func_mask=torch.from_numpy(np.asarray([func], dtype=np.float32)),
        context_idx=torch.tensor([_hash_bucket(context_id + cell_type, max(1, int(context_vocab)))], dtype=torch.long),
        tf_idx=torch.tensor([_hash_bucket(tf, max(1, int(tf_vocab)))], dtype=torch.long),
        gene_idx=torch.from_numpy(gene_hashes[None, :]),
        gene_pos=torch.from_numpy(gene_positions[None, :]),
        gene_mask=torch.from_numpy(gene_mask[None, :]),
        labels=torch.from_numpy(labels[None, :]),
        sample_weight=torch.from_numpy(weights[None, :]),
    )
