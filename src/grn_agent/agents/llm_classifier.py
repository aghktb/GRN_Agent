# llm_classifier.py
# Stage 3: Classify each abstract using Gemma 4 26B running locally via Ollama.
#
# For each abstract that survived Stage 2, the LLM is asked:
#   "Does this abstract provide evidence that {TF} directly regulates {TARGET}?"
#
# The model returns structured JSON with four fields:
#   - supports_interaction: bool
#   - evidence_type: ChIP-seq | knockdown | luciferase | EMSA | co-expression | none
#   - relationship: activates | represses | binds | unclear | none
#   - confidence: float 0.0–1.0
#
# The 256K context window of Gemma 4 allows batching multiple abstracts in one
# call, but we classify one-at-a-time by default for auditability.

import json
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from grn_agent.agents import lit_config as config

# Cell-type-specific context strings injected into the LLM prompt.
# Each entry describes the biology of the source cell type so the model can
# weight evidence appropriately. A generic fallback is used when the cell type
# is unknown or not listed.
CELL_TYPE_CONTEXTS: dict[str, str] = {
    "hESC": """
CELL TYPE: human Embryonic Stem Cells (hESC)

These interactions are from a ChIP-seq gold standard network for human embryonic stem cells.
hESCs are pluripotent cells. The core pluripotency regulatory network (OCT4/POU5F1, SOX2, NANOG) is highly active.

Evidence priorities:
- MUST demonstrate regulation in hESC or human pluripotent stem cells.
- AGGRESSIVELY REJECT evidence if the interaction is specific to a completely different differentiated lineage (e.g., rejecting muscle-specific, neuron-specific, or liver-specific regulation if it is not active in the pluripotent state).
""",

    "mESC": """
CELL TYPE: mouse Embryonic Stem Cells (mESC)

These interactions come from a regulatory network derived from mouse embryonic stem cells.

Evidence priorities:
- MUST demonstrate regulation in mESC or mouse pluripotent stem cells.
- AGGRESSIVELY REJECT evidence if the interaction is specific to a differentiated lineage (e.g., rejecting muscle, liver, or immune specific regulation).
""",

    "mHSC": """
CELL TYPE: mouse Hematopoietic Stem Cells (mHSC)

These interactions come from a regulatory network derived from mouse hematopoietic stem cells (blood progenitors).

Evidence priorities:
- MUST demonstrate regulation in HSCs, bone marrow progenitors, or closely related blood cells.
- AGGRESSIVELY REJECT evidence if the interaction occurs in unrelated solid tissues (e.g., brain, liver, muscle).
""",

    "mDC": """
CELL TYPE: mouse Dendritic Cells (mDC)

These interactions come from a regulatory network derived from mouse dendritic cells (immune APCs).

Evidence priorities:
- MUST demonstrate regulation in dendritic cells, macrophages, or closely related myeloid immune cells.
- AGGRESSIVELY REJECT evidence from unrelated tissues (e.g., embryonic stem cells, neurons, muscle).
""",

    "HepG2": """
CELL TYPE: HepG2 (Human Liver Cancer Cell Line)

These interactions come from a regulatory network derived from HepG2 cells (hepatocyte-like).

Evidence priorities:
- MUST demonstrate regulation in HepG2, primary hepatocytes, or liver tissue.
- AGGRESSIVELY REJECT evidence from unrelated cell lineages (e.g., stem cells, neurons, muscle) unless the paper explicitly bridges the context to liver function.
""",
}

_GENERIC_CONTEXT = """
CELL TYPE: unspecified

These TF-target interactions represent direct transcriptional regulatory relationships.
Each edge should be supported by mechanistic experimental evidence of direct binding or functional regulation.

Valid supporting evidence includes:
- ChIP-seq, EMSA, or luciferase reporter assays demonstrating direct binding/activation.
- Perturbation experiments (knockdown, CRISPR) showing target gene changes.
"""


def get_cell_type_context(cell_type: str | None) -> str:
    """Return the context string for the given cell type, or the generic fallback."""
    if cell_type and cell_type in CELL_TYPE_CONTEXTS:
        return CELL_TYPE_CONTEXTS[cell_type].strip()
    return _GENERIC_CONTEXT.strip()

_PROMPT_TEMPLATE = """You are a molecular biology expert specializing in transcriptional regulation.

Read the text excerpt below and determine whether it provides direct experimental evidence that {tf} regulates {target}.

Criteria for "supports_interaction: true":
  - Direct binding evidence (ChIP-seq, ChIP-chip, EMSA, pull-down)
  - Promoter activation/repression assays (luciferase, reporter gene)
  - Functional perturbation showing target gene changes (knockdown, CRISPR, overexpression)

Do NOT count as support:
  - Pure co-expression or correlation data with no mechanistic evidence
  - Mentions in unrelated biological contexts
  - Negated claims ("{tf} does NOT regulate {target}", "no effect on {target}")
  - Speculative claims ("may regulate", "could potentially bind") with no shown evidence
  - Reverse direction ({target} regulates {tf} — the direction matters)

Required evidence_sentence:
  - Copy the EXACT sentence from the text excerpt that proves the interaction.
  - It must contain BOTH gene names (or a clear pronoun reference).
  - If you cannot find one, set supports_interaction=false and evidence_sentence="".

Background context:
{context}

Text excerpt:
{text}

Respond with ONLY valid JSON — no prose, no markdown:
{{
  "supports_interaction": true or false,
  "evidence_type": "ChIP-seq" | "ChIP-chip" | "knockdown" | "luciferase" | "EMSA" | "co-expression" | "other" | "none",
  "relationship": "activates" | "represses" | "binds" | "unclear" | "none",
  "direction_correct": true or false,
  "is_negated": true or false,
  "is_speculative": true or false,
  "evidence_sentence": "<exact sentence from the text excerpt, or empty string>",
  "confidence": <float 0.0 to 1.0>
}}"""

_NULL_RESULT = {
    "supports_interaction": False,
    "evidence_type": "none",
    "relationship": "none",
    "direction_correct": False,
    "is_negated": False,
    "is_speculative": False,
    "evidence_sentence": "",
    "confidence": 0.0,
}


def _call_ollama(prompt: str, retries: int = 2) -> str:
    """Send a prompt to the local Ollama server and return the raw response text."""
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=(30, 900),  # 30s connect, 15m read (large contexts on 26B take time)
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.Timeout:
            if attempt == retries:
                print(f"  [WARN] Ollama timed out after {retries+1} attempts — skipping abstract")
                return ""
            print(f"  [WARN] Ollama timeout, retry {attempt+1}/{retries}...")
    return ""


def _parse_response(raw: str) -> dict:
    """
    Extract the JSON block from the model's response.

    The model is instructed to return only JSON, but may occasionally wrap it
    in markdown fences. This handles both cases.
    """
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Greedy match — the schema now contains nested quotes in evidence_sentence,
    # so the first balanced {...} block is the safest extraction.
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return dict(_NULL_RESULT)

    try:
        result = json.loads(match.group())
        return {
            "supports_interaction": bool(result.get("supports_interaction", False)),
            "evidence_type": str(result.get("evidence_type", "none")),
            "relationship": str(result.get("relationship", "none")),
            "direction_correct": bool(result.get("direction_correct", True)),
            "is_negated": bool(result.get("is_negated", False)),
            "is_speculative": bool(result.get("is_speculative", False)),
            "evidence_sentence": str(result.get("evidence_sentence", "")).strip(),
            "confidence": float(result.get("confidence", 0.0)),
        }
    except (json.JSONDecodeError, ValueError):
        return dict(_NULL_RESULT)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy substring matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _build_text(record: dict) -> str:
    """
    Assemble the full text to send to the LLM.

    Uses title + abstract always, then appends Introduction and Results sections
    when the record was enriched with PMC full-text.
    """
    parts = []
    if record.get("title"):
        parts.append(record["title"])
    if record.get("abstract"):
        parts.append(record["abstract"])
    if record.get("results"):
        parts.append("RESULTS: " + record["results"])
    if record.get("introduction"):
        parts.append("INTRODUCTION: " + record["introduction"])
        
    full_text = "\n\n".join(parts)
    
    # SAFETY TRUNCATION: Local 26B models will freeze or timeout if fed 
    # excessively long PMC full-texts. We cap at ~15,000 characters 
    # (approx 3,000-4,000 tokens) to ensure stable inference times.
    MAX_CHARS = 4000
    if len(full_text) > MAX_CHARS:
        full_text = full_text[:MAX_CHARS] + "\n\n... [TEXT TRUNCATED FOR LENGTH]"
        
    return full_text


def _evidence_grounded(sentence: str, record: dict) -> bool:
    """
    Check whether the LLM's evidence_sentence actually appears in the record text.

    Checks title+abstract+introduction+results so evidence drawn from full-text
    sections is not incorrectly flagged as hallucinated.
    A loose substring match (lowercased, whitespace-collapsed) catches paraphrases
    where the model slightly modified punctuation but still copied the span.
    Returns False for fabricated/hallucinated sentences.
    """
    if not sentence:
        return False
    full_text = _build_text(record)
    norm_sent = _normalize(sentence)
    norm_full = _normalize(full_text)
    if norm_sent in norm_full:
        return True
    # Allow up to ~15% drift: check that 80% of the sentence's 4-grams hit the text.
    tokens = norm_sent.split()
    if len(tokens) < 4:
        return False
    ngrams = [" ".join(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    hits = sum(1 for ng in ngrams if ng in norm_full)
    return hits / len(ngrams) >= 0.8


def classify_abstract(tf: str, target: str, record: dict,
                      cell_type: str | None = None) -> dict:
    """
    Classify a single abstract record for TF-Target regulatory support.

    Uses all available text (title, abstract, introduction, results) when present.
    Adds two derived fields on top of the raw LLM output:
      - evidence_grounded: True if the evidence_sentence is traceable to the text.
      - effective_support: True only if supports_interaction AND grounded AND
        direction_correct AND not negated/speculative.
    """
    text = _build_text(record)
    context = get_cell_type_context(cell_type)
    prompt = _PROMPT_TEMPLATE.format(tf=tf, target=target, text=text, context=context)
    raw = _call_ollama(prompt)
    result = _parse_response(raw)
    result["pmid"] = record["pmid"]

    grounded = _evidence_grounded(result["evidence_sentence"], record)
    result["evidence_grounded"] = grounded

    flags = record.get("_filter_flags", {})
    result["mentions_cell_type"] = flags.get("mentions_cell_type", False)
    result["cell_type_sentences"] = flags.get("cell_type_sentences", [])

    # Cell-type gate: when classifier was called with a cell_type, the abstract
    # MUST mention that context for `effective_support` to fire. When cell_type
    # is None (non-specific / STRING network), gate passes through.
    cell_type_ok = (cell_type is None) or result["mentions_cell_type"]

    result["effective_support"] = (
        result["supports_interaction"]
        and grounded
        and result["direction_correct"]
        and not result["is_negated"]
        and not result["is_speculative"]
        and cell_type_ok
    )

    return result


def classify_all(tf: str, target: str, abstracts: list[dict],
                 cell_type: str | None = None) -> list[dict]:
    """
    Classify every abstract in the list and return all classification dicts.
    Parallelized via ThreadPoolExecutor to speed up Ollama inference.
    """
    if not abstracts:
        return []

    # Use 4 parallel workers by default. This saturates most GPUs without
    # exceeding VRAM limits for models like Gemma 7B/27B.
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(classify_abstract, tf, target, rec, cell_type=cell_type)
            for rec in abstracts
        ]
        return [f.result() for f in futures]
