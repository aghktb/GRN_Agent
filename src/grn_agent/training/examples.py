from __future__ import annotations

import numpy as np

from grn_agent.schemas import EvidenceGraph


def label_binary_from_evidence_graph(eg: EvidenceGraph) -> int:
    """Weak supervision: 1 if correlation evidence supports a link, else 0 (demo only)."""
    c = eg.evidence.get("correlation")
    if c is None:
        return 0
    c = float(c)
    return 1 if abs(c) > 0.12 else 0


def build_graphs_with_gold(
    graphs: list[EvidenceGraph],
    gold: dict[tuple[str, str], int],
) -> tuple[list[EvidenceGraph], np.ndarray]:
    """Supervision from gold edges only; graphs not in ``gold`` are skipped."""
    xs: list[EvidenceGraph] = []
    ys: list[int] = []
    for eg in graphs:
        k = (str(eg.edge.source_tf).strip().upper(), str(eg.edge.target_gene).strip().upper())
        if k not in gold:
            continue
        xs.append(eg)
        ys.append(int(gold[k]))
    if not xs:
        raise ValueError(
            "No evidence graphs matched gold (source_tf, target_gene) keys — check symbols and candidate overlap."
        )
    return xs, np.array(ys, dtype=np.float32)
