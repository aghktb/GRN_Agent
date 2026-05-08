# score_aggregator.py
# Stage 4: Combine all per-abstract LLM classifications into a single
#          Literature Support Score for a TF-Target pair.
#
# Score formula (quality-driven):
#
#   lit_score = avg_confidence(supporting_abstracts)
#             + 0.10   (if direct physical evidence: ChIP-seq, EMSA, etc.)
#             + 0.10   (if K > 1 supporting abstracts — reproducibility bonus)
#   capped at 1.0
#
# This rewards high-confidence LLM classifications backed by direct binding
# evidence and independent replication, rather than rewarding volume alone.

import math
from collections import Counter

from grn_agent.agents import lit_config as config

_KNOWN_EVIDENCE_TYPES = {
    "ChIP-seq", "ChIP-chip", "knockdown", "luciferase", "EMSA", "co-expression", "other"
}


def _is_supporting(c: dict) -> bool:
    """
    Treat an abstract as supporting only when the LLM said yes AND the evidence
    sentence was grounded in the abstract AND no negation/speculation/direction
    flags fired. Falls back to raw `supports_interaction` for old records that
    predate the new schema.
    """
    if "effective_support" in c:
        return bool(c["effective_support"])
    return bool(c.get("supports_interaction"))


def evidence_diversity_bonus(classifications: list[dict]) -> float:
    """
    Returns a bonus (0.0–1.0) for the number of distinct evidence types seen.

    Rationale: an interaction supported by ChIP-seq + luciferase + knockdown is
    much more credible than one supported by 20 knockdown papers alone.

    Scale:
        0 types  → 0.0
        1 type   → 0.25
        2 types  → 0.5
        3 types  → 0.75
        4+ types → 1.0
    """
    supporting = [c for c in classifications if _is_supporting(c)]
    evidence_types = {
        c["evidence_type"] for c in supporting
        if c.get("evidence_type") and c["evidence_type"] not in ("none", "co-expression")
    }
    return min(len(evidence_types) / 4, 1.0)


def aggregate(classifications: list[dict]) -> dict:
    """
    Compute the final Literature Support Score from a list of per-abstract results.

    Args:
        classifications: Output of llm_classifier.classify_all()

    Returns:
        Dict with:
          lit_score          – final weighted score (0.0–1.0)
          n_papers           – total abstracts evaluated
          n_supporting       – abstracts classified as supporting the interaction
          supporting_ratio   – n_supporting / n_papers
          evidence_diversity – diversity bonus value
          avg_confidence     – mean LLM confidence across supporting abstracts
          evidence_types     – comma-separated unique evidence types found
          relationships      – comma-separated unique relationship labels found
          pmids              – comma-separated PMIDs of supporting abstracts
    """
    n_papers = len(classifications)
    if n_papers == 0:
        return {
            "lit_score": 0.0,
            "n_papers": 0,
            "n_supporting": 0,
            "n_grounded": 0,
            "n_negated": 0,
            "n_speculative": 0,
            "n_wrong_direction": 0,
            "supporting_ratio": 0.0,
            "evidence_diversity": 0.0,
            "avg_conf": 0.0,
            "reg_type": "none",
            "relationships": "none",
            "evidence_types": "none",
            "pmids": "",
            "conflict_detected": False,
        }

    supporting = [c for c in classifications if _is_supporting(c)]
    n_supporting = len(supporting)
    supporting_ratio = n_supporting / n_papers
    n_grounded = sum(1 for c in classifications if c.get("evidence_grounded"))
    n_negated = sum(1 for c in classifications if c.get("is_negated"))
    n_speculative = sum(1 for c in classifications if c.get("is_speculative"))
    n_wrong_direction = sum(
        1 for c in classifications
        if c.get("supports_interaction") and not c.get("direction_correct", True)
    )

    avg_confidence = (
        sum(c.get("confidence", 0.0) for c in supporting) / n_supporting
        if n_supporting > 0 else 0.0
    )

    diversity = evidence_diversity_bonus(classifications)

    # 4d. Conflict detection (Agentic logic)
    # Flag as conflict if some papers support but others were dropped for negation
    # or if supporting papers disagree on the relationship (e.g. activates vs represses)
    supporting_rels = {c["relationship"].lower() for c in supporting if c.get("relationship")}
    has_conflicting_rels = ("activates" in supporting_rels and "represses" in supporting_rels)
    
    # Also conflict if significant number of negated papers exist alongside supporting papers
    has_negation_conflict = (n_supporting > 0 and n_negated > (n_supporting * 0.5))
    
    conflict_detected = has_conflicting_rels or has_negation_conflict

    # Synthesize Regulation Type (Direct vs Indirect)
    direct_evidence_keywords = {"chip-seq", "chip-chip", "emsa", "pull-down"}
    reg_type = "none"
    if n_supporting > 0:
        has_direct = any(
            c.get("evidence_type", "").lower() in direct_evidence_keywords
            for c in supporting
        )
        reg_type = "dir" if has_direct else "indir"

    # Consensus Relationship
    if n_supporting == 0:
        consensus_rel = "none"
    else:
        has_act = "activates" in supporting_rels
        has_sup = "represses" in supporting_rels
        
        if has_act and has_sup:
            consensus_rel = "both"
        elif has_act:
            consensus_rel = "activates"
        elif has_sup:
            consensus_rel = "suppresses"
        else:
            consensus_rel = "none"

    # Quality-Driven lit_score
    if n_supporting == 0:
        lit_score = 0.0
    else:
        base_score = avg_confidence
        quality_bonus = 0.10 if reg_type == "dir" else 0.0
        reproducibility_bonus = 0.10 if n_supporting > 1 else 0.0
        lit_score = base_score + quality_bonus + reproducibility_bonus
        
    lit_score = round(min(lit_score, 1.0), 4)

    evidence_types = ", ".join(sorted({
        c["evidence_type"] for c in supporting
        if c.get("evidence_type") and c["evidence_type"] != "none"
    }))
    pmids = ", ".join(c["pmid"] for c in supporting if c.get("pmid"))

    return {
        "lit_score": lit_score,
        "n_papers": n_papers,
        "n_supporting": n_supporting,
        "n_grounded": n_grounded,
        "n_negated": n_negated,
        "n_speculative": n_speculative,
        "n_wrong_direction": n_wrong_direction,
        "supporting_ratio": round(supporting_ratio, 4),
        "evidence_diversity": round(diversity, 4),
        "avg_conf": round(avg_confidence, 4),
        "reg_type": reg_type,
        "relationships": consensus_rel,
        "evidence_types": evidence_types,
        "pmids": pmids,
        "conflict_detected": conflict_detected,
    }
