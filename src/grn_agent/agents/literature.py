from __future__ import annotations

from grn_agent.schemas import EvalTrack, LiteratureFeatures


def literature_features_for_track(
    eval_track: EvalTrack,
    tf: str,
    target: str,
    time_cutoff_year: int | None = 2020,
) -> dict | None:
    """
    Track 1: None (no lit in graph).
    Track 2/3: structured placeholders — replace with PubMed + NER pipeline.
    """
    if eval_track == EvalTrack.NO_LITERATURE:
        return None
    lf = LiteratureFeatures(
        lit_activation_prob=0.55 if eval_track == EvalTrack.ASSISTED else 0.5,
        lit_repression_prob=0.15,
        num_supporting_pmids=0,
        best_assay_weight=0.0,
        latest_year_included=time_cutoff_year,
    )
    return lf.model_dump()
