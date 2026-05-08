"""
Pure-Python (numpy) PWM scorer — no MEME/FIMO required.

Implements sliding-window log-odds scoring over DNA sequences:
  score(seq, PWM) = Σ_i PWM[i, seq[i]]

Score threshold is set as a fraction of the PWM's max possible score
(equivalent to a loose p-value cut without needing a null distribution).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from grn_agent.acquisition.jaspar_client import Motif

logger = logging.getLogger(__name__)

# IUPAC → column indices (A=0, C=1, G=2, T=3); ambiguous → None (skip position)
_NUC_MAP: dict[str, int | None] = {
    "A": 0, "C": 1, "G": 2, "T": 3,
    "N": None, "R": None, "Y": None, "S": None,
    "W": None, "K": None, "M": None, "B": None,
    "D": None, "H": None, "V": None,
}


@dataclass
class MotifHit:
    motif_id: str
    tf_name: str
    seq_id: str
    start: int     # 0-based, on the forward strand
    end: int
    strand: str    # "+" or "-"
    score: float   # log-odds sum
    score_pct: float  # score / max_score  (0–1)


def encode_sequence(seq: str) -> np.ndarray:
    """
    Convert an uppercase DNA string to a numpy integer array (A=0,C=1,G=2,T=3).
    Ambiguous bases → -1 (will be treated as missing).
    """
    seq = seq.upper()
    arr = np.full(len(seq), -1, dtype=np.int8)
    for i, ch in enumerate(seq):
        idx = _NUC_MAP.get(ch)
        arr[i] = idx if idx is not None else -1
    return arr


def revcomp_encoded(arr: np.ndarray) -> np.ndarray:
    """Reverse-complement an encoded integer array."""
    # complement: A↔T (0↔3), C↔G (1↔2); -1 stays -1
    comp = np.where(arr >= 0, 3 - arr, -1)
    return comp[::-1].copy()


def score_sequence(
    seq_enc: np.ndarray,
    pwm: np.ndarray,
    score_threshold: float,
) -> list[tuple[int, float]]:
    """
    Slide the PWM over an encoded sequence and return (start, score) for hits
    exceeding score_threshold on the **forward** strand.

    Args:
        seq_enc:         Encoded sequence (length L), values in {-1,0,1,2,3}.
        pwm:             (motif_len, 4) log-odds matrix.
        score_threshold: Minimum score to report.

    Returns:
        List of (start_pos, score) for positions where score >= threshold.
    """
    motif_len = pwm.shape[0]
    seq_len = len(seq_enc)
    if seq_len < motif_len:
        return []

    hits: list[tuple[int, float]] = []
    # Build lookup: for each position in the window, precompute PWM row sums
    # using vectorised indexing wherever the base is unambiguous.
    for start in range(seq_len - motif_len + 1):
        window = seq_enc[start : start + motif_len]
        # Positions with known bases
        valid = window >= 0
        if valid.sum() < motif_len // 2:
            # Too many ambiguous bases — skip
            continue
        score = 0.0
        for i, enc in enumerate(window):
            if enc >= 0:
                score += pwm[i, enc]
            # Ambiguous: contribute 0 (neutral)
        if score >= score_threshold:
            hits.append((start, score))
    return hits


def scan_sequence_both_strands(
    seq: str,
    motif: "Motif",
    score_threshold_pct: float = 0.7,
) -> list[MotifHit]:
    """
    Scan a DNA sequence on both strands and return all motif hits.

    Args:
        seq:                  DNA string (any case).
        motif:                Motif object with .pwm and .max_score.
        score_threshold_pct:  Hit if score ≥ max_score × this fraction.

    Returns:
        List of MotifHit objects.
    """
    threshold = motif.max_score * score_threshold_pct
    seq_enc = encode_sequence(seq)
    seq_enc_rc = revcomp_encoded(seq_enc)
    seq_len = len(seq)
    hits: list[MotifHit] = []

    for start, score in score_sequence(seq_enc, motif.pwm, threshold):
        hits.append(
            MotifHit(
                motif_id=motif.matrix_id,
                tf_name=motif.tf_name,
                seq_id="",
                start=start,
                end=start + motif.length,
                strand="+",
                score=score,
                score_pct=score / motif.max_score if motif.max_score > 0 else 0.0,
            )
        )

    for start_rc, score in score_sequence(seq_enc_rc, motif.pwm, threshold):
        # Convert RC coordinates back to forward-strand coordinates
        start_fwd = seq_len - (start_rc + motif.length)
        hits.append(
            MotifHit(
                motif_id=motif.matrix_id,
                tf_name=motif.tf_name,
                seq_id="",
                start=start_fwd,
                end=start_fwd + motif.length,
                strand="-",
                score=score,
                score_pct=score / motif.max_score if motif.max_score > 0 else 0.0,
            )
        )
    return hits


def best_hit_score(hits: list[MotifHit]) -> float:
    """Return the best (max) score_pct from a list of hits, or 0.0."""
    if not hits:
        return 0.0
    return max(h.score_pct for h in hits)
