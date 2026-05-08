from __future__ import annotations

from collections import defaultdict

from grn_agent.schemas import GraphMeta, Network, ScoredEdge


def decode_grn(
    context_id: str,
    edges: list[ScoredEdge],
    max_regulators_per_target: int = 5,
    min_confidence: float = 0.0,
) -> Network:
    """Graph decoding: top-K TF per target (SDD §7)."""
    by_target: dict[str, list[ ScoredEdge]] = defaultdict(list)
    for e in edges:
        if e.confidence_score < min_confidence:
            continue
        by_target[e.target_gene].append(e)
    kept: list[ScoredEdge] = []
    for tgt, lst in by_target.items():
        lst_sorted = sorted(lst, key=lambda x: -x.confidence_score)[:max_regulators_per_target]
        kept.extend(lst_sorted)
    # dedupe (tf, tgt)
    seen: set[tuple[str, str]] = set()
    final: list[ScoredEdge] = []
    for e in sorted(kept, key=lambda x: -x.confidence_score):
        k = (e.source_tf, e.target_gene)
        if k in seen:
            continue
        seen.add(k)
        final.append(e)
    if not final:
        meta = GraphMeta()
    else:
        meta = GraphMeta(species="unknown", cell_type=None, model_version="grnagent_eager_v1")
    return Network(context_id=context_id, edges=final, graph_metadata=meta)
