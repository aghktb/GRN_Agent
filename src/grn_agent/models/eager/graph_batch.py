"""Convert ``EvidenceGraph`` to padded tensors for :class:`EagerRegulator`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from grn_agent.schemas import EvalTrack, EvidenceGraph, EvidenceNodeType, RelationType

# Order must match RelationType usage in typed MPNN (index = enum order)
_RELATION_LIST: list[RelationType] = list(RelationType)

# Node kinds for typing & stage masks (not the same as EvidenceNodeType strings)
class EagerNodeKind:
    TF = 0
    GENE = 1
    CTX = 2
    EXPR = 3
    NETWORK = 4
    BINDING = 5
    PRIOR = 6
    ORTHO = 7
    LIT = 8
    PAD = 9


NUM_NODE_KINDS = 10
VALUE_DIM = 32
MAX_NODES = 32


def _kind_for_node_id(nid: str, ntype: EvidenceNodeType) -> int:
    if nid == "n_tf":
        return EagerNodeKind.TF
    if nid == "n_target":
        return EagerNodeKind.GENE
    if nid == "n_ctx":
        return EagerNodeKind.CTX
    if nid == "ev_expr":
        return EagerNodeKind.EXPR
    if nid == "ev_network":
        return EagerNodeKind.NETWORK
    if nid == "ev_binding":
        return EagerNodeKind.BINDING
    if nid == "ev_prior":
        return EagerNodeKind.PRIOR
    if nid == "ev_orthology":
        return EagerNodeKind.ORTHO
    if nid == "ev_lit":
        return EagerNodeKind.LIT
    return EagerNodeKind.PAD


def _floats_from_payload(payload: dict[str, Any], keys: list[str], out: np.ndarray, start: int) -> int:
    i = start
    for k in keys:
        v = payload.get(k)
        if v is None:
            if i < VALUE_DIM:
                out[i] = 0.0
            i += 1
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if i < VALUE_DIM:
            out[i] = fv
        i += 1
    return i


def _node_value_vector(kind: int, payload: dict[str, Any]) -> np.ndarray:
    out = np.zeros(VALUE_DIM, dtype=np.float32)
    if kind == EagerNodeKind.TF:
        _floats_from_payload(payload, ["role"], out, 0)
    elif kind == EagerNodeKind.GENE:
        _floats_from_payload(payload, ["role"], out, 0)
    elif kind == EagerNodeKind.CTX:
        _floats_from_payload(payload, ["n_cells", "n_module_genes"], out, 0)
    elif kind == EagerNodeKind.EXPR:
        _floats_from_payload(
            payload,
            ["z_t", "z_g", "activity_t", "mean_expr_t", "mean_expr_g", "dropout_t", "dropout_g"],
            out,
            0,
        )
    elif kind == EagerNodeKind.NETWORK:
        _floats_from_payload(
            payload,
            [
                "pearson_r",
                "partial_corr",
                "in_same_module",
                "k_hop_distance",
                "shared_neighbors",
            ],
            out,
            0,
        )
    elif kind == EagerNodeKind.BINDING:
        _floats_from_payload(
            payload,
            [
                "motif_present",
                "motif_score",
                "n_motif_regions",
                "peak_accessibility",
                "peak_to_gene_linked",
                "celltype_specificity",
            ],
            out,
            0,
        )
    elif kind == EagerNodeKind.PRIOR:
        keys = [k for k in payload if isinstance(payload.get(k), (int, float))]
        for j, k in enumerate(keys[:VALUE_DIM]):
            out[j] = float(payload[k])
    elif kind == EagerNodeKind.ORTHO:
        _floats_from_payload(
            payload,
            ["ortholog_support", "ortholog_confidence", "conserved_in_human", "conserved_in_mouse"],
            out,
            0,
        )
    elif kind == EagerNodeKind.LIT:
        _floats_from_payload(payload, ["lit_activation_prob", "lit_repression_prob"], out, 0)
    # clip huge values
    out = np.clip(out, -30.0, 30.0)
    return out


def _conf_scalar(kind: int, payload: dict[str, Any]) -> float:
    if kind == EagerNodeKind.PRIOR:
        return float(np.clip(float(payload.get("ensemble_prior", 0.0) or 0.0), 0.0, 1.0))
    if kind == EagerNodeKind.BINDING:
        ms = payload.get("motif_score")
        if ms is not None:
            return float(np.clip(abs(float(ms)), 0.0, 1.0))
        return 0.5 if payload.get("motif_present") else 0.0
    return 0.5


@dataclass
class EagerGraphBatch:
    """Single-graph batch (B=1) or collated batch — all lengths along dim 0 batch."""

    node_kind: torch.Tensor  # (B, N)
    x_value: torch.Tensor  # (B, N, F)
    conf: torch.Tensor  # (B, N)
    edge_index: torch.Tensor  # (2, E) single graph; collate may use list
    edge_type: torch.Tensor  # (E,)
    node_mask: torch.Tensor  # (B, N) 1 = real
    modality: torch.Tensor  # (B, 4) m_expr, m_acc, m_motif, m_link
    mech_mask: torch.Tensor  # (B, N) 1 on mechanistic (binding) tokens
    func_mask: torch.Tensor  # (B, N) 1 on expression + network tokens
    context_idx: torch.Tensor  # (B,) bucket id
    tf_idx: torch.Tensor  # (B,)
    gene_idx: torch.Tensor  # (B,)


def _hash_bucket(s: str, mod: int) -> int:
    h = 2166136261
    for c in s.encode("utf-8", errors="ignore"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h % mod)


def evidence_graph_to_batch(
    eg: EvidenceGraph,
    eval_track: EvalTrack,
    literature_in_graph: bool = False,
) -> EagerGraphBatch:
    """
    Build tensors for one ``EvidenceGraph``. Drops literature nodes for training (Track 1 / EAGER default).
    """
    nodes = list(eg.nodes)
    if eval_track == EvalTrack.NO_LITERATURE or not literature_in_graph:
        nodes = [n for n in nodes if n.node_type != EvidenceNodeType.literature]

    id_to_idx: dict[str, int] = {}
    kinds: list[int] = []
    xs: list[np.ndarray] = []
    confs: list[float] = []

    for n in nodes[:MAX_NODES]:
        kind = _kind_for_node_id(n.node_id, n.node_type)
        id_to_idx[n.node_id] = len(id_to_idx)
        kinds.append(kind)
        xs.append(_node_value_vector(kind, n.payload))
        confs.append(_conf_scalar(kind, n.payload))

    n_real = len(kinds)
    mech = np.zeros(MAX_NODES, dtype=np.float32)
    func = np.zeros(MAX_NODES, dtype=np.float32)
    for i in range(n_real):
        if kinds[i] == EagerNodeKind.BINDING:
            mech[i] = 1.0
        if kinds[i] in (EagerNodeKind.EXPR, EagerNodeKind.NETWORK):
            func[i] = 1.0
    while len(kinds) < MAX_NODES:
        kinds.append(EagerNodeKind.PAD)
        xs.append(np.zeros(VALUE_DIM, dtype=np.float32))
        confs.append(0.0)

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_types: list[int] = []
    for r in eg.relations:
        a, b = r.src_id, r.dst_id
        if a not in id_to_idx or b not in id_to_idx:
            continue
        if r.relation not in _RELATION_LIST:
            continue
        ri = _RELATION_LIST.index(r.relation)
        edge_src.append(id_to_idx[a])
        edge_dst.append(id_to_idx[b])
        edge_types.append(ri)
        # Undirected information flow: add reverse for message passing
        edge_src.append(id_to_idx[b])
        edge_dst.append(id_to_idx[a])
        edge_types.append(ri)

    if not edge_src:
        # self-loop on TF so MPNN has at least one message
        if "n_tf" in id_to_idx:
            i = id_to_idx["n_tf"]
            edge_src.append(i)
            edge_dst.append(i)
            edge_types.append(0)

    node_mask = np.zeros(MAX_NODES, dtype=np.float32)
    node_mask[:n_real] = 1.0

    ev = eg.evidence
    has_expr = "ev_expr" in id_to_idx
    has_bind = "ev_binding" in id_to_idx
    m_expr = 1.0 if has_expr else 0.0
    m_motif = 0.0
    m_acc = 0.0
    m_link = 0.0
    if has_bind:
        bnode = next((x for x in nodes if x.node_id == "ev_binding"), None)
        if bnode:
            p = bnode.payload
            if p.get("motif_present") or (p.get("motif_score") is not None and float(p.get("motif_score") or 0) > 0):
                m_motif = 1.0
            if p.get("peak_accessibility") is not None and float(p.get("peak_accessibility") or 0) > 0:
                m_acc = 1.0
            if p.get("peak_to_gene_linked"):
                m_link = 1.0
    modal = np.array([m_expr, m_acc, m_motif, m_link], dtype=np.float32)

    ctx = eg.context.context_id
    cidx = _hash_bucket(str(ctx) + str(eg.context.cell_type or ""), 1024)
    tidx = _hash_bucket(eg.edge.source_tf.upper(), 8192)
    gidx = _hash_bucket(eg.edge.target_gene.upper(), 8192)

    t_node_kind = torch.from_numpy(np.asarray([kinds], dtype=np.int64))
    t_x = torch.from_numpy(np.stack(xs)[None, ...].astype(np.float32))
    t_conf = torch.from_numpy(np.asarray([confs], dtype=np.float32))
    t_m = torch.from_numpy(np.asarray([node_mask], dtype=np.float32))
    t_modal = torch.from_numpy(np.asarray([modal], dtype=np.float32))
    ei = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    et = torch.tensor(edge_types, dtype=torch.long)
    return EagerGraphBatch(
        node_kind=t_node_kind,
        x_value=t_x,
        conf=t_conf,
        edge_index=ei,
        edge_type=et,
        node_mask=t_m,
        modality=t_modal,
        mech_mask=torch.from_numpy(np.asarray([mech], dtype=np.float32)),
        func_mask=torch.from_numpy(np.asarray([func], dtype=np.float32)),
        context_idx=torch.tensor([cidx], dtype=torch.long),
        tf_idx=torch.tensor([tidx], dtype=torch.long),
        gene_idx=torch.tensor([gidx], dtype=torch.long),
    )
