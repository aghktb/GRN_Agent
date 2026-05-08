"""
Algorithm 8: Evidence Graph Construction Per Edge.

For each candidate edge e=(t,g) in context c, constructs EG(e,c) with:
  Nodes: TF, target, context, expression, network, binding (optional), priors, orthology, literature (Track 2/3)
  Relations: logical links (in_context, supports_activation, supports_repression, etc.)

Gold labels are NEVER included here — the EvidenceGraph is pure input to the scoring model.
"""
from __future__ import annotations

from grn_agent.schemas import (
    CandidateEdge,
    CellContext,
    EvidenceGraph,
    EvidenceNode,
    EvidenceRelation,
    EvidenceNodeType,
    FeatureBundle,
    PriorBundle,
    RelationType,
    EvalTrack,
)


def _relation(src: str, dst: str, rel: RelationType) -> EvidenceRelation:
    return EvidenceRelation(src_id=src, dst_id=dst, relation=rel)


def build_evidence_graph(
    ctx: CellContext,
    edge: CandidateEdge,
    features: FeatureBundle,
    priors: PriorBundle,
    eval_track: EvalTrack,
    literature_payload: dict | None = None,
) -> EvidenceGraph:
    """
    Algorithm 8 — Assemble typed nodes and logical relations.

    Node layout:
      n_tf       — identifier node for TF
      n_target   — identifier node for target gene
      n_ctx      — context (cell type, module membership)
      ev_expr    — expression evidence: z_t, z_g, activity(t), mean_expr, dropout
      ev_network — network evidence: corr(t,g), shared_neighbors, in_same_module
      ev_binding — binding evidence: motif_score, accessibility, peak_to_gene (optional)
      ev_prior   — inference priors: ensemble_prior + per-method scores, stability
      ev_ortho   — orthology: ortholog_support, conserved_in_human/mouse
      ev_lit     — literature (Track 2/3 only)
    """
    nodes: list[EvidenceNode] = []
    rels: list[EvidenceRelation] = []

    # ── Identifier nodes ──────────────────────────────────────────────────────
    n_tf = EvidenceNode(
        node_id="n_tf",
        node_type=EvidenceNodeType.expression,
        label=edge.source_tf,
        payload={"role": "TF", "symbol": edge.source_tf},
    )
    n_tg = EvidenceNode(
        node_id="n_target",
        node_type=EvidenceNodeType.expression,
        label=edge.target_gene,
        payload={"role": "target", "symbol": edge.target_gene},
    )
    n_ctx = EvidenceNode(
        node_id="n_ctx",
        node_type=EvidenceNodeType.expression,
        label=ctx.context_id,
        payload={
            "cell_type": ctx.cell_type,
            "n_cells": len(ctx.cell_indices),
            "n_module_genes": len(ctx.module_genes),
            "species": ctx.metadata.get("species"),
        },
    )
    nodes.extend([n_tf, n_tg, n_ctx])
    rels.append(_relation("n_tf", "n_ctx", RelationType.in_context))
    rels.append(_relation("n_target", "n_ctx", RelationType.in_context))

    # ── Algorithm 8: Expression node — z_t, z_g, activity(t) ─────────────────
    expr = features.expression
    expr_payload = {
        "z_t": expr.tf_zscore,
        "z_g": expr.target_zscore,
        "activity_t": expr.tf_activity_proxy,
        "mean_expr_t": expr.tf_mean_expr,
        "mean_expr_g": expr.target_mean_expr,
        "dropout_t": expr.tf_dropout_rate,
        "dropout_g": expr.target_dropout_rate,
    }
    ev_expr = EvidenceNode(
        node_id="ev_expr",
        node_type=EvidenceNodeType.expression,
        label="expression_evidence",
        payload=expr_payload,
    )
    nodes.append(ev_expr)
    rels.append(_relation("ev_expr", "n_tf", RelationType.in_context))
    rels.append(_relation("ev_expr", "n_target", RelationType.in_context))
    # Directional hints from expression levels
    if expr.tf_zscore is not None and expr.tf_zscore > 0.5:
        rels.append(_relation("ev_expr", "n_tf", RelationType.supports_activation))

    # ── Algorithm 8: Network node — corr(t,g), shared neighbors ──────────────
    net = features.network
    net_payload = {
        "pearson_r": net.pearson_r,
        "partial_corr": net.partial_corr,
        "in_same_module": net.in_same_module,
        "k_hop_distance": net.k_hop_distance,
        "shared_neighbors": net.shared_neighbors,
        "shared_neighbor_names": net.shared_neighbor_names[:10],  # truncate for prompt compactness
    }
    ev_net = EvidenceNode(
        node_id="ev_network",
        node_type=EvidenceNodeType.correlation,
        label="network_evidence",
        payload=net_payload,
    )
    nodes.append(ev_net)
    corr = net.pearson_r
    if corr is not None and corr > 0.1:
        rels.append(_relation("ev_network", "n_tf", RelationType.supports_activation))
    elif corr is not None and corr < -0.1:
        rels.append(_relation("ev_network", "n_tf", RelationType.supports_repression))

    # ── Algorithm 8: Binding node — motif score, accessibility, peak-to-gene ─
    binding_payload: dict = {}
    has_binding_support = False
    if features.motif:
        binding_payload["motif_present"] = features.motif.motif_present
        binding_payload["motif_score"] = features.motif.motif_score
        binding_payload["n_motif_regions"] = features.motif.n_supporting_regions
        if features.motif.motif_present:
            has_binding_support = True
    if features.atac:
        binding_payload["peak_accessibility"] = features.atac.peak_accessibility
        binding_payload["peak_to_gene_linked"] = features.atac.peak_to_gene_linked
        binding_payload["celltype_specificity"] = features.atac.celltype_specificity
        if features.atac.peak_to_gene_linked:
            has_binding_support = True

    if binding_payload:
        ev_binding = EvidenceNode(
            node_id="ev_binding",
            node_type=EvidenceNodeType.motif,
            label="binding_evidence",
            payload=binding_payload,
        )
        nodes.append(ev_binding)
        if has_binding_support:
            rels.append(_relation("ev_binding", "n_tf", RelationType.supports_activation))

    # ── Algorithm 8: Priors node — p_prior(e), method-specific, stability ────
    prior_payload = priors.model_dump()
    ev_prior = EvidenceNode(
        node_id="ev_prior",
        node_type=EvidenceNodeType.inference_prior,
        label="inference_priors",
        payload=prior_payload,
    )
    nodes.append(ev_prior)
    if priors.ensemble_prior >= 0.5:
        rels.append(_relation("ev_prior", "n_tf", RelationType.supports_activation))
    else:
        rels.append(_relation("ev_prior", "n_tf", RelationType.contradicts_activation))

    # ── Algorithm 8: Orthology node — ortholog support/confidence ─────────────
    ortho_payload: dict = {}
    if features.orthology:
        orth = features.orthology
        ortho_payload = {
            "ortholog_support": orth.ortholog_support,
            "ortholog_confidence": orth.ortholog_confidence,
            "supporting_species": orth.supporting_species,
            "conserved_in_human": orth.conserved_in_human,
            "conserved_in_mouse": orth.conserved_in_mouse,
        }
    ev_ortho = EvidenceNode(
        node_id="ev_orthology",
        node_type=EvidenceNodeType.orthology,
        label="orthology_evidence",
        payload=ortho_payload,
    )
    nodes.append(ev_ortho)
    if ortho_payload.get("ortholog_support") and ortho_payload["ortholog_support"] > 0.5:
        rels.append(_relation("ev_orthology", "n_tf", RelationType.supports_activation))

    # ── Algorithm 8: Literature node (Track 2/3 only, mode=none for Track 1) ─
    if eval_track != EvalTrack.NO_LITERATURE and literature_payload:
        ev_lit = EvidenceNode(
            node_id="ev_lit",
            node_type=EvidenceNodeType.literature,
            label="literature_evidence",
            payload=literature_payload,
        )
        nodes.append(ev_lit)

    # ── Evidence summary dict (compact mirror for APIs / prompt) ─────────────
    summary = {
        "correlation": corr,
        "ensemble_prior": priors.ensemble_prior,
        "in_same_module": net.in_same_module,
        "shared_neighbors": net.shared_neighbors,
        "z_t": expr.tf_zscore,
        "z_g": expr.target_zscore,
        "activity_t": expr.tf_activity_proxy,
        "motif_present": features.motif.motif_present if features.motif else None,
        "accessibility": features.atac.peak_accessibility if features.atac else None,
    }
    return EvidenceGraph(context=ctx, edge=edge, nodes=nodes, relations=rels, evidence=summary)
