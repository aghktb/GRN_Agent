from enum import Enum


class EvalTrack(str, Enum):
    """Evaluation mode (SDD §9)."""

    NO_LITERATURE = "track1_no_literature"
    TIME_SLICED_LIT = "track2_time_sliced_literature"
    ASSISTED = "track3_assisted"


class EvidenceNodeType(str, Enum):
    expression = "expression"
    correlation = "correlation"
    motif = "motif"
    atac = "atac"
    inference_prior = "inference_prior"
    orthology = "orthology"
    literature = "literature"


class RelationType(str, Enum):
    supports_activation = "supports_activation"
    supports_repression = "supports_repression"
    contradicts_activation = "contradicts_activation"
    contradicts_repression = "contradicts_repression"
    in_context = "in_context"
