"""Minimal GraphML export for Network (nodes TF/target + edges)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from grn_agent.schemas import Network


def write_network_graphml(net: Network, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element(
        "graphml",
        xmlns="http://graphml.graphdrawing.org/xmlns",
    )
    key_w = ET.SubElement(root, "key", id="weight", for_="edge", attr_name="weight", attr_type="double")
    key_t = ET.SubElement(root, "key", id="rtype", for_="edge", attr_name="regulation", attr_type="string")
    key_s = ET.SubElement(root, "key", id="conf", for_="edge", attr_name="confidence", attr_type="double")
    _ = key_w, key_t, key_s

    graph_el = ET.SubElement(root, "graph", id="G", edgedefault="directed")

    node_ids: set[str] = set()
    for e in net.edges:
        node_ids.add(e.source_tf)
        node_ids.add(e.target_gene)
    for nid in sorted(node_ids):
        ET.SubElement(graph_el, "node", id=nid)

    for i, e in enumerate(net.edges):
        src, tgt = e.source_tf, e.target_gene
        edge = ET.SubElement(
            graph_el,
            "edge",
            id=f"e{i}",
            source=src,
            target=tgt,
        )
        ET.SubElement(edge, "data", key="weight").text = str(e.p_present)
        ET.SubElement(edge, "data", key="rtype").text = "present" if e.p_present >= 0.5 else "absent"
        ET.SubElement(edge, "data", key="conf").text = str(e.confidence_score)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(p, encoding="unicode", xml_declaration=True)
