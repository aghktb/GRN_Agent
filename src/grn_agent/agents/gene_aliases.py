# gene_aliases.py
# Resolve official gene symbols + synonyms via mygene.info (NCBI Gene mirror).
#
# The pipeline previously held a 6-entry hardcoded alias dict, which silently
# failed for any new gene. This module:
#   - looks up symbol + aliases for a gene name once
#   - caches results to disk so re-runs don't hit the network
#   - returns the same fallback (just the input name) if lookup fails
#
# Used by both pubmed_search (to widen the PubMed query) and text_filter
# (to recognize the gene in fetched abstracts).

import json
import os
import threading
from pathlib import Path

import requests

from grn_agent.agents import lit_config as config

_CACHE_PATH = Path(config.ABSTRACTS_CACHE_DIR).parent / "gene_alias_cache.json"
_MYGENE_URL = "https://mygene.info/v3/query"
_LOCK = threading.Lock()
_MEMORY_CACHE: dict[str, list[str]] | None = None


def _load_cache() -> dict[str, list[str]]:
    global _MEMORY_CACHE
    if _MEMORY_CACHE is not None:
        return _MEMORY_CACHE
    if _CACHE_PATH.exists():
        try:
            _MEMORY_CACHE = json.loads(_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            _MEMORY_CACHE = {}
    else:
        _MEMORY_CACHE = {}
    return _MEMORY_CACHE


def _save_cache() -> None:
    if _MEMORY_CACHE is None:
        return
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_MEMORY_CACHE, indent=2, sort_keys=True))


def _query_mygene(name: str, species: str = "human") -> list[str]:
    """
    Return [official_symbol, alias1, alias2, ...] for `name`. Empty list on miss.

    Prioritises exact symbol match over alias match so that e.g. querying "ALB"
    returns the albumin gene (symbol=ALB), not FBF1 (which lists "Alb" as an alias).
    """
    # Try exact symbol match first
    for query in [f"symbol:{name}", f"alias:{name}"]:
        params = {
            "q": query,
            "species": species,
            "fields": "symbol,alias,name",
            "size": 1,
        }
        try:
            resp = requests.get(_MYGENE_URL, params=params, timeout=15)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except (requests.RequestException, ValueError):
            continue
        if hits:
            break
    else:
        return []
    if not hits:
        return []
    top = hits[0]
    symbol = top.get("symbol", "")
    aliases = top.get("alias", []) or []
    if isinstance(aliases, str):
        aliases = [aliases]
    gene_name = top.get("name", "")  # full product name, e.g. "albumin"
    out = []
    seen = set()
    for token in [symbol, *aliases, name]:
        if token and token.lower() not in seen:
            out.append(token)
            seen.add(token.lower())
    # Add full gene/product name (e.g. "albumin", "apolipoprotein A1").
    # Split multi-word names and add each significant word (≥4 chars) as well
    # as the full name, so text matching catches "albumin promoter" etc.
    if gene_name and gene_name.lower() not in seen:
        out.append(gene_name)
        seen.add(gene_name.lower())
    return out


def get_aliases(name: str, species: str = "human") -> list[str]:
    """
    Return all known names for `name` (official symbol + synonyms + the input itself).

    Always includes the input name as a fallback so unknown genes still work.
    """
    key = f"{species}:{name.upper()}"
    with _LOCK:
        cache = _load_cache()
        if key in cache:
            return cache[key]
        aliases = _query_mygene(name, species=species)
        if not aliases:
            aliases = [name]
        cache[key] = aliases
        _save_cache()
        return aliases
