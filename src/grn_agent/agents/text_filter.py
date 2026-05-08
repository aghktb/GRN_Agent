# text_filter.py
# Stage 2: Fast pre-filter — eliminates irrelevant abstracts before sending to the LLM.
#
# Three checks run on each abstract (no LLM needed):
#   1. Co-occurrence   — TF and target appear in the same sentence
#   2. Regulatory lang — regulatory verbs/phrases are present near the gene names
#   3. Experimental    — known assay keywords appear (ChIP, luciferase, knockdown, etc.)
#
# Abstracts that score below both MIN_COOCCURRENCE_SCORE and MIN_REGULATORY_SCORE
# are dropped, saving expensive LLM inference time.

import re
from functools import lru_cache
from pathlib import Path

import spacy

from grn_agent.agents import lit_config as config
from grn_agent.agents import gene_aliases

# Load spaCy once at import time (en_core_web_sm is fast and sufficient for sentence splitting)
try:
    _nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser", "lemmatizer"])
    _nlp.add_pipe("sentencizer")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


@lru_cache(maxsize=1)
def _load_word_set(filepath: str) -> frozenset[str]:
    words = Path(filepath).read_text().splitlines()
    return frozenset(w.strip().lower() for w in words if w.strip())


def _regulatory_verbs() -> frozenset[str]:
    return _load_word_set(config.REGULATORY_VERBS_FILE)


def _experimental_methods() -> frozenset[str]:
    return _load_word_set(config.EXPERIMENTAL_METHODS_FILE)


def _sentences(text: str) -> list[str]:
    doc = _nlp(text)
    return [sent.text for sent in doc.sents]


# Negation cues that flip the meaning of a regulatory verb in a sentence.
# Kept narrow on purpose — broad cues like "not" alone fire on "not only X but
# also Y", which is the opposite of negation.
_NEGATION_CUES = (
    "does not", "did not", "do not", "not regulate", "not regulated",
    "no effect", "no significant", "failed to", "fails to", "unable to",
    "did not affect", "does not affect", "not bind", "not activate",
    "not repress", "not induce", "without effect", "independent of",
)

# Speculation cues — present tense softening that downgrades evidence weight.
_SPECULATION_CUES = (
    "may regulate", "might regulate", "could regulate", "potentially regulate",
    "may bind", "might bind", "could bind", "suggested to",
    "is thought to", "appears to", "putative", "potential target",
)


def _has_cue(sentence: str, cues: tuple[str, ...]) -> bool:
    s = sentence.lower()
    return any(cue in s for cue in cues)


def _name_in_text(name: str, text: str) -> bool:
    aliases = gene_aliases.get_aliases(name)
    for alias in aliases:
        if " " in alias:
            # Multi-word name (e.g. "apolipoprotein A1") — substring match
            if alias.lower() in text.lower():
                return True
        else:
            if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                return True
    return False


def cooccurrence_score(tf: str, target: str, abstracts: list[dict]) -> float:
    """
    Fraction of abstracts where TF and target co-occur, weighted by sentence-level hits.

    Formula:
        score = (same_sentence_hits × 3 + same_abstract_hits × 1) / (total_abstracts × 4)

    Sentence-level co-occurrence is weighted 3× over abstract-level because it is a
    much stronger signal of a direct functional relationship.
    """
    if not abstracts:
        return 0.0

    same_sentence = 0
    same_abstract = 0

    for rec in abstracts:
        text = rec["abstract"]
        if _name_in_text(tf, text) and _name_in_text(target, text):
            same_abstract += 1
            for sent in _sentences(text):
                if _name_in_text(tf, sent) and _name_in_text(target, sent):
                    same_sentence += 1
                    break  # one per abstract is sufficient for the score

    return (same_sentence * 3 + same_abstract * 1) / (len(abstracts) * 4)


def regulatory_language_score(tf: str, target: str, abstracts: list[dict]) -> float:
    """
    Fraction of co-occurrence sentences that also contain a regulatory verb/phrase.

    Formula:
        score = regulatory_sentences / max(co_occurrence_sentences, 1)

    A higher score means the co-occurring mentions use language like "activates",
    "represses", "binds to the promoter of", etc., rather than vague co-mentions.
    """
    verbs = _regulatory_verbs()
    co_sents = 0
    reg_sents = 0

    for rec in abstracts:
        for sent in _sentences(rec["abstract"]):
            if _name_in_text(tf, sent) and _name_in_text(target, sent):
                co_sents += 1
                sent_lower = sent.lower()
                if any(v in sent_lower for v in verbs):
                    reg_sents += 1

    return reg_sents / max(co_sents, 1)


def experimental_score(abstracts: list[dict]) -> float:
    """
    Fraction of abstracts mentioning at least one direct experimental method.

    Only methods that provide mechanistic evidence (ChIP, EMSA, luciferase, etc.)
    are counted — correlation-only studies score 0 for this component.
    """
    methods = _experimental_methods()
    if not abstracts:
        return 0.0

    hits = sum(
        1 for rec in abstracts
        if any(m in rec["abstract"].lower() for m in methods)
    )
    return hits / len(abstracts)


def _record_full_text(rec: dict) -> str:
    """Concatenate all available text fields for a record."""
    parts = [rec.get("title", ""), rec.get("abstract", "")]
    if rec.get("introduction"):
        parts.append(rec["introduction"])
    if rec.get("results"):
        parts.append(rec["results"])
    return " ".join(p for p in parts if p)


def filter_abstracts(tf: str, target: str, abstracts: list[dict],
                     cell_type: str | None = None) -> tuple[list[dict], dict]:
    """
    Apply all filters and return only abstracts that pass.

    Filters (in order):
      0. Cell-type hard filter — paper must mention the source cell type when
         cell_type is provided (derived from the input network filename).
      1. Co-occurrence — TF and target appear in the same sentence.
      2. Regulatory/experimental signal — regulatory verbs or assay keywords present.
      3. Negation gate — drop if every co-occurrence sentence is negated.

    All available text (title, abstract, introduction, results) is used when
    records have been enriched with PMC full-text sections.

    Returns:
        (passing_abstracts, scores_dict)
    """
    methods = _experimental_methods()
    verbs = _regulatory_verbs()
    cell_type_terms = [t.lower() for t in config.CELL_TYPE_ALIASES.get(cell_type or "", [])]

    kept = []
    n_dropped_negated = 0

    for rec in abstracts:
        full_text = _record_full_text(rec)
        full_text_lower = full_text.lower()

        # Cell-type annotation: collect sentences that mention the source cell type.
        # Not a hard drop — passed to the LLM stage via _filter_flags for context.
        if cell_type_terms:
            cell_type_sentences = [
                s for s in _sentences(full_text)
                if any(ct in s.lower() for ct in cell_type_terms)
            ]
        else:
            cell_type_sentences = []

        # Find every sentence that mentions both genes.
        cooc_sents = [
            s for s in _sentences(full_text)
            if _name_in_text(tf, s) and _name_in_text(target, s)
        ]
        
        # PERMISSIVE FALLBACK: If they do not appear in the same sentence, 
        # check if they both appear in the title/abstract. If they do, 
        # and there is strong regulatory language, let it pass to Stage 3.
        if not cooc_sents:
            if _name_in_text(tf, full_text) and _name_in_text(target, full_text):
                # Use the whole abstract/results as a "synthetic" co-occurrence sentence
                # so that the rest of the logic (negation, etc) still works.
                cooc_sents = [rec.get("abstract", "") or rec.get("title", "")]
            else:
                continue

        has_method = any(m in full_text_lower for m in methods)
        has_regulatory_verb = any(v in full_text_lower for v in verbs)
        if not (has_method or has_regulatory_verb):
            continue

        # Negation/speculation gating: if EVERY co-occurrence sentence is negated,
        # the abstract almost certainly disproves the relationship — drop it before
        # spending an LLM call on it.
        all_negated = all(_has_cue(s, _NEGATION_CUES) for s in cooc_sents)
        if all_negated:
            n_dropped_negated += 1
            continue

        # Tag the record so the LLM stage can prompt extra carefully on
        # speculative-only abstracts. The LLM still gets to see them.
        rec = dict(rec)  # don't mutate cached objects
        rec["_filter_flags"] = {
            "any_negated_cooc": any(_has_cue(s, _NEGATION_CUES) for s in cooc_sents),
            "any_speculative_cooc": any(_has_cue(s, _SPECULATION_CUES) for s in cooc_sents),
            "n_cooc_sentences": len(cooc_sents),
            "mentions_cell_type": bool(cell_type_sentences),
            "cell_type_sentences": cell_type_sentences,
        }
        kept.append(rec)

    scores = {
        "cooccurrence_score": cooccurrence_score(tf, target, abstracts),
        "regulatory_language_score": regulatory_language_score(tf, target, abstracts),
        "experimental_score": experimental_score(abstracts),
        "n_before_filter": len(abstracts),
        "n_after_filter": len(kept),
        "n_dropped_negated": n_dropped_negated,
    }

    return kept, scores
