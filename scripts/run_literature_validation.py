#!/usr/bin/env python3
"""
Post-inference literature validation.

Reads a scored_edges CSV from TF-EAGER inference, runs the Literature
Validation Agent on each predicted edge, and writes an enriched CSV
with literature scores alongside model predictions.

Usage:
    python scripts/run_literature_validation.py \
        --input artifacts/.../test_scored_edges.csv \
        --cell-type mESC \
        --output artifacts/.../literature_validated.csv

    # Or limit to top predictions only:
    python scripts/run_literature_validation.py \
        --input artifacts/.../test_scored_edges.csv \
        --cell-type mHSC \
        --min-p 0.5 \
        --limit 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.agents import lit_config as config
from grn_agent.agents import pubmed_search, text_filter, llm_classifier, score_aggregator


def validate_edge(tf: str, target: str, cell_type: str | None = None) -> tuple[dict, list[dict]]:
    abstracts = pubmed_search.fetch_abstracts_for_pair(tf, target, cell_type=cell_type)
    print(f"         Stage 1: {len(abstracts)} abstracts fetched")

    filtered, filter_scores = text_filter.filter_abstracts(
        tf, target, abstracts, cell_type=cell_type
    )

    if len(filtered) == 0 and cell_type is not None:
        abstracts_broad = pubmed_search.fetch_abstracts_for_pair(tf, target, cell_type=None)
        filtered, filter_scores = text_filter.filter_abstracts(
            tf, target, abstracts_broad, cell_type=None
        )

    print(f"         Stage 2: {len(filtered)} abstracts kept after filtering")

    classifications = llm_classifier.classify_all(tf, target, filtered, cell_type=cell_type)
    n_raw = sum(1 for c in classifications if c.get("supports_interaction"))
    n_eff = sum(1 for c in classifications if c.get("effective_support"))
    print(f"         Stage 3: {n_eff} supporting (raw={n_raw}, dropped {n_raw - n_eff} for grounding/negation/direction)")

    scores = score_aggregator.aggregate(classifications)
    print(f"         Stage 4: lit_score = {scores['lit_score']}")
    return scores, classifications


def main() -> None:
    ap = argparse.ArgumentParser(description="Literature validation of TF-EAGER predictions")
    ap.add_argument("--input", required=True, help="Scored edges CSV from infer_tf_eager.py")
    ap.add_argument("--output", default=None, help="Output CSV path (default: input_literature_validated.csv)")
    ap.add_argument("--cell-type", default=None, dest="cell_type",
                    help=f"Cell type for PubMed query enrichment. Known: {sorted(config.CELL_TYPE_ALIASES)}")
    ap.add_argument("--min-p", type=float, default=0.0,
                    help="Only validate edges with p_present >= this threshold")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max edges to validate (takes top by p_present)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip pairs already in the output file")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Input not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_literature_validated.csv")

    # Load input edges
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter by min-p
    if args.min_p > 0:
        rows = [r for r in rows if float(r.get("p_present", 0)) >= args.min_p]

    # Sort by p_present descending, apply limit
    rows.sort(key=lambda r: float(r.get("p_present", 0)), reverse=True)
    if args.limit:
        rows = rows[:args.limit]

    # Resume support
    done = set()
    if args.resume and output_path.is_file():
        with open(output_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["source_tf"], r["target_gene"]))

    # Output fields = original fields + literature fields
    lit_fields = [
        "lit_score", "n_papers", "n_supporting", "supporting_ratio",
        "avg_conf", "reg_type", "relationships", "evidence_types",
        "pmids", "conflict_detected",
    ]
    sample_row = rows[0] if rows else {}
    output_fields = list(sample_row.keys()) + lit_fields

    write_header = not args.resume or not output_path.is_file()
    mode = "a" if args.resume and output_path.is_file() else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:     {input_path} ({len(rows)} edges after filtering)")
    print(f"Output:    {output_path}")
    print(f"Cell type: {args.cell_type or 'none'}")
    if done:
        print(f"Resuming:  {len(done)} pairs already done")

    with open(output_path, mode, newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        total = len(rows)
        for i, row in enumerate(rows, 1):
            tf = row["source_tf"].strip()
            target = row["target_gene"].strip()
            p_present = float(row.get("p_present", 0))

            if (tf, target) in done:
                print(f"[{i}/{total}] SKIP {tf} -> {target} (already done)")
                continue

            print(f"[{i}/{total}] {tf} -> {target} (p_present={p_present:.4f})")

            scores, classifications = validate_edge(tf, target, cell_type=args.cell_type)

            result_row = {**row, **scores}
            writer.writerow(result_row)
            out_f.flush()

            # Save raw classifications for audit
            json_dir = output_path.parent / "literature_classifications"
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"{tf}_{target}_classifications.json"
            with open(json_path, "w") as jf:
                json.dump(classifications, jf, indent=2)

    print(f"Done. Results: {output_path}")


if __name__ == "__main__":
    main()
