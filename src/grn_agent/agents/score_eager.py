"""EAGER neural scoring: EvidenceGraph -> binary p(present)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from grn_agent.models.eager.checkpoint import load_eager_checkpoint
from grn_agent.models.eager.graph_batch import evidence_graph_to_batch
from grn_agent.schemas import RelationType
from grn_agent.schemas import EvalTrack, EvidenceGraph, ScoredEdge

if TYPE_CHECKING:
    from grn_agent.models.eager.eager_model import EagerRegulator


def score_evidence_graph(
    eg: EvidenceGraph,
    eval_track: EvalTrack,
    *,
    checkpoint: str | Path | None = None,
    model: "EagerRegulator | None" = None,
    device: str | None = None,
) -> ScoredEdge:
    """
    Run EAGER forward. Pass ``model`` (for tests) or ``checkpoint`` path.
    """
    if model is None:
        if checkpoint is None or not Path(checkpoint).is_file():
            raise ValueError("EAGER scoring requires a checkpoint path to a saved EAGER model (.pt)")
        _dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = load_eager_checkpoint(checkpoint, map_location=_dev)
        model = model.to(_dev)
    else:
        _dev = next(model.parameters()).device

    model.eval()
    batch = evidence_graph_to_batch(eg, eval_track, literature_in_graph=False)
    batch = _batch_to_device(batch, _dev)
    attn_summary: dict[str, float] | None = None
    with torch.no_grad():
        out = model(batch, return_attention_summary=True)
    if isinstance(out, tuple):
        logit, attn_summary = out
    else:
        logit = out
    p = float(torch.sigmoid(logit).squeeze().item())
    logit_f = float(logit.squeeze().item())
    reasoning = _mechanistic_reasoning_text(eg, p, logit_f, attn_summary=attn_summary)
    return ScoredEdge(
        source_tf=eg.edge.source_tf,
        target_gene=eg.edge.target_gene,
        p_present=p,
        logit=logit_f,
        confidence_score=p,
        evidence_summary=eg.evidence.copy(),
        mechanism_reasoning=reasoning,
    )


def _batch_to_device(batch, dev: str | torch.device) -> object:
    from grn_agent.models.eager.graph_batch import EagerGraphBatch

    b = batch
    if not isinstance(b, EagerGraphBatch):
        return batch
    return EagerGraphBatch(
        node_kind=b.node_kind.to(dev),
        x_value=b.x_value.to(dev),
        conf=b.conf.to(dev),
        edge_index=b.edge_index.to(dev),
        edge_type=b.edge_type.to(dev),
        node_mask=b.node_mask.to(dev),
        modality=b.modality.to(dev),
        mech_mask=b.mech_mask.to(dev),
        func_mask=b.func_mask.to(dev),
        context_idx=b.context_idx.to(dev),
        tf_idx=b.tf_idx.to(dev),
        gene_idx=b.gene_idx.to(dev),
    )


def _mechanistic_reasoning_text(
    eg: EvidenceGraph,
    p_present: float,
    logit: float,
    *,
    attn_summary: dict[str, float] | None = None,
) -> str:
    e = eg.evidence or {}
    tf = eg.edge.source_tf
    tg = eg.edge.target_gene

    corr = _as_float(e.get("correlation"))
    prior = _as_float(e.get("ensemble_prior"))
    shared = _as_float(e.get("shared_neighbors"))
    z_t = _as_float(e.get("z_t"))
    activity_t = _as_float(e.get("activity_t"))
    motif = e.get("motif_present")
    acc = _as_float(e.get("accessibility"))
    in_module = bool(e.get("in_same_module", False))

    n_sup_act = sum(1 for r in eg.relations if r.relation == RelationType.supports_activation)
    n_sup_rep = sum(1 for r in eg.relations if r.relation == RelationType.supports_repression)
    n_contra = sum(1 for r in eg.relations if r.relation == RelationType.contradicts_activation)

    dir_txt = "undirected/weak directionality"
    if corr is not None:
        if corr > 0.1:
            dir_txt = "activation-leaning"
        elif corr < -0.1:
            dir_txt = "repression-leaning"

    strength = "moderate"
    if p_present >= 0.8:
        strength = "strong"
    elif p_present < 0.55:
        strength = "weak"

    expr_bits: list[str] = []
    if corr is not None:
        expr_bits.append(f"TF-target correlation is {corr:.3f} ({dir_txt})")
    if z_t is not None:
        expr_bits.append(f"TF z-score={z_t:.2f}")
    if activity_t is not None:
        expr_bits.append(f"TF activity proxy={activity_t:.2f}")

    mech_bits: list[str] = []
    if motif is True:
        mech_bits.append("motif evidence is present")
    elif motif is False:
        mech_bits.append("no motif hit is detected")
    if acc is not None:
        mech_bits.append(f"accessibility support={acc:.3f}")
    if in_module:
        mech_bits.append("TF and target co-occur in the same module")
    if shared is not None:
        mech_bits.append(f"shared-neighbor support={int(shared)}")
    if prior is not None:
        mech_bits.append(f"ensemble prior={prior:.3f}")

    rel_bits = (
        f"Graph relations contribute {n_sup_act} activation-supporting, "
        f"{n_sup_rep} repression-supporting, and {n_contra} contradicting signals."
    )

    sentence1 = (
        f"EAGER predicts a {strength} probability of regulation for {tf}->{tg} "
        f"(p_present={p_present:.3f}, logit={logit:.3f})."
    )
    sentence2 = "Expression evidence: " + ("; ".join(expr_bits) if expr_bits else "limited signal available.")
    sentence3 = "Mechanistic/context evidence: " + ("; ".join(mech_bits) if mech_bits else "limited support available.")
    sentence4 = rel_bits
    if attn_summary:
        s2m = float(attn_summary.get("stage2_attn_mech", 0.0))
        s2f = float(attn_summary.get("stage2_attn_func", 0.0))
        s3m = float(attn_summary.get("stage3_attn_mech", 0.0))
        s3f = float(attn_summary.get("stage3_attn_func", 0.0))
        dom2 = "functional" if s2f >= s2m else "mechanistic"
        dom3 = "functional" if s3f >= s3m else "mechanistic"
        sentence5 = (
            "Attention attribution: "
            f"stage-2 focus is {dom2} (mech={s2m:.2f}, func={s2f:.2f}); "
            f"stage-3 integration is {dom3} (mech={s3m:.2f}, func={s3f:.2f})."
        )
        return " ".join([sentence1, sentence2, sentence3, sentence4, sentence5])
    return " ".join([sentence1, sentence2, sentence3, sentence4])


def _as_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# Type alias for configs that only use eager
ScoringBackend = Literal["eager"]


# Backward-compatible name for pipeline imports
def score_with_eager(
    eg: EvidenceGraph,
    eval_track: EvalTrack,
    checkpoint: str | Path,
    device: str | None = None,
) -> ScoredEdge:
    return score_evidence_graph(eg, eval_track, checkpoint=checkpoint, device=device)
