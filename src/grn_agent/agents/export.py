from __future__ import annotations

from pathlib import Path

import pandas as pd

from grn_agent.io.artifact_store import save_json
from grn_agent.io.graphml_out import write_network_graphml
from grn_agent.schemas import Network, ScoredEdge


def _scored_edge_rows(edges: list[ScoredEdge]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for e in edges:
        rows.append(
            {
                "source_tf": e.source_tf,
                "target_gene": e.target_gene,
                "p_present": e.p_present,
                "logit": e.logit,
                "confidence_score": e.confidence_score,
                "mechanism_reasoning": e.mechanism_reasoning,
            }
        )
    return rows


def export_scored_edges_csv(edges: list[ScoredEdge], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_scored_edge_rows(edges)).to_csv(p, index=False)


def export_network_csv(net: Network, path: str | Path) -> None:
    export_scored_edges_csv(net.edges, path)


def export_network_bundle(net: Network, out_dir: str | Path) -> None:
    out = Path(out_dir)
    save_json(out / "network.json", net)
    export_network_csv(net, out / "network.csv")
    write_network_graphml(net, out / "network.graphml")
