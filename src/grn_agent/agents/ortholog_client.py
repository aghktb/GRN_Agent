"""
Ortholog client — Algorithm 8 orthology evidence node.

Data source: mygene.io REST API (wraps NCBI Gene / Ensembl orthologs).
Cache: JSON on disk at ~/.cache/grn_agent/orthologs/ — one file per gene symbol.

For a pair (TF, target) in a given species:
  ortholog_support  = fraction of queried species where BOTH genes have an ortholog
  conserved_in_X    = True if an ortholog exists in species X
  supporting_species = list of species that have orthologs for this TF→target pair
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "grn_agent" / "orthologs"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# mygene.io taxid → human-readable name
_TAXID_NAMES: dict[int, str] = {
    9606: "human",
    10090: "mouse",
    10116: "rat",
    9913: "cow",
    7955: "zebrafish",
    7227: "fly",
    6239: "worm",
}

# Default companion species to query
_DEFAULT_COMPANION_TAXIDS: list[int] = [9606, 10090]

_SPECIES_TAXID: dict[str, int] = {
    "human": 9606,
    "homo sapiens": 9606,
    "mouse": 10090,
    "mus musculus": 10090,
    "rat": 10116,
    "rattus norvegicus": 10116,
}


def _cache_path(symbol: str, taxid: int) -> Path:
    safe = str(symbol).strip().upper().replace("/", "_").replace("\\", "_")
    return _CACHE_DIR / f"{safe}__{taxid}.json"


def _fetch_orthologs_mygene(symbol: str, source_taxid: int, target_taxids: list[int]) -> dict[int, str | None]:
    """
    Query mygene.io for orthologs of *symbol* (in source_taxid) in each target taxid.

    Returns {taxid: ortholog_symbol_or_None}.
    """
    try:
        import mygene  # type: ignore
    except ImportError:
        log.warning("mygene not installed — orthology evidence unavailable. pip install mygene")
        return {t: None for t in target_taxids}

    symbol = str(symbol).strip().upper()
    mg = mygene.MyGeneInfo()
    try:
        result = mg.query(
            symbol,
            scopes="symbol",
            species=source_taxid,
            fields="symbol,homologene,ensembl.gene",
            verbose=False,
        )
    except Exception as exc:
        log.debug("mygene query failed for %s: %s", symbol, exc)
        return {t: None for t in target_taxids}

    hits = result.get("hits", []) if isinstance(result, dict) else []
    if not hits:
        return {t: None for t in target_taxids}

    best = hits[0]
    homologene = best.get("homologene", {})
    genes_list: list[list] = homologene.get("genes", [])  # [[taxid, entrez_id], ...]

    taxid_to_entrez: dict[int, int] = {}
    for entry in genes_list:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            taxid_to_entrez[int(entry[0])] = int(entry[1])

    out: dict[int, str | None] = {}
    for tgt in target_taxids:
        if tgt == source_taxid:
            out[tgt] = symbol
            continue
        entrez = taxid_to_entrez.get(tgt)
        if entrez is None:
            out[tgt] = None
        else:
            try:
                sym_result = mg.getgene(entrez, fields="symbol", verbose=False)
                out[tgt] = sym_result.get("symbol") if sym_result else None
            except Exception:
                out[tgt] = None
        time.sleep(0.05)  # polite rate limiting

    return out


def prefetch_orthologs(
    symbols: list[str],
    source_species: str = "mouse",
    companion_taxids: list[int] | None = None,
) -> None:
    """
    Batch pre-fetch orthologs for all symbols and write to disk cache.
    Call this once before the per-edge feature extraction loop.
    mygene.io supports up to 1000 symbols per batch POST.
    """
    source_taxid = _SPECIES_TAXID.get(source_species.lower(), 10090)
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if companion_taxids is None:
        companion_taxids = [t for t in _DEFAULT_COMPANION_TAXIDS if t != source_taxid]

    # Filter to symbols not already fully cached
    to_fetch: list[str] = []
    for sym in symbols:
        if any(not _cache_path(sym, t).exists() for t in companion_taxids):
            to_fetch.append(sym)

    if not to_fetch:
        log.info("Ortholog cache warm for all %d symbols.", len(symbols))
        return

    try:
        import mygene  # type: ignore
    except ImportError:
        log.warning("mygene not installed — skip ortholog prefetch.")
        return

    mg = mygene.MyGeneInfo()
    log.info("Prefetching orthologs for %d symbols via mygene.io…", len(to_fetch))

    CHUNK = 500
    for i in range(0, len(to_fetch), CHUNK):
        chunk = to_fetch[i : i + CHUNK]
        try:
            results = mg.querymany(
                chunk,
                scopes="symbol",
                species=source_taxid,
                fields="symbol,homologene",
                verbose=False,
                as_dataframe=False,
            )
        except Exception as exc:
            log.warning("mygene batch query failed: %s", exc)
            continue

        for hit in results:
            if hit.get("notfound"):
                continue
            sym = str(hit.get("query", "")).strip().upper()
            homologene = hit.get("homologene", {})
            genes_list = homologene.get("genes", [])
            taxid_to_entrez: dict[int, int] = {}
            for entry in genes_list:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    taxid_to_entrez[int(entry[0])] = int(entry[1])

            for taxid in companion_taxids:
                cp = _cache_path(sym, taxid)
                if cp.exists():
                    continue
                if taxid == source_taxid:
                    cp.write_text(json.dumps({"symbol": sym}))
                elif taxid in taxid_to_entrez:
                    # We have the entrez ID but not the symbol yet — write entrez as placeholder;
                    # symbol resolved lazily on first use.
                    cp.write_text(json.dumps({"symbol": None, "entrez": taxid_to_entrez[taxid]}))
                else:
                    cp.write_text(json.dumps({"symbol": None}))

        time.sleep(0.2)

    log.info("Ortholog prefetch complete. Cache at %s", _CACHE_DIR)


def get_ortholog_info(
    tf_symbol: str,
    target_symbol: str,
    source_species: str = "mouse",
    companion_taxids: list[int] | None = None,
) -> dict[str, Any]:
    """
    Return orthology evidence for a TF→target pair.

    Parameters
    ----------
    tf_symbol, target_symbol : gene symbols in source_species
    source_species : "mouse" / "human" / "rat" / …
    companion_taxids : species to check conservation in; defaults to [human, mouse]

    Returns
    -------
    dict with keys matching OrthologyFeatures fields:
        ortholog_support, ortholog_confidence, supporting_species,
        conserved_in_human, conserved_in_mouse
    """
    tf_symbol = str(tf_symbol).strip().upper()
    target_symbol = str(target_symbol).strip().upper()
    source_taxid = _SPECIES_TAXID.get(source_species.lower(), 10090)
    if companion_taxids is None:
        companion_taxids = [t for t in _DEFAULT_COMPANION_TAXIDS if t != source_taxid]
        if source_taxid not in companion_taxids:
            companion_taxids = list(_DEFAULT_COMPANION_TAXIDS)

    def _get_for_gene(symbol: str) -> dict[int, str | None]:
        orthologs: dict[int, str | None] = {}
        missing: list[int] = []
        for taxid in companion_taxids:
            cp = _cache_path(symbol, taxid)
            if cp.exists():
                try:
                    orthologs[taxid] = json.loads(cp.read_text()).get("symbol")
                except Exception:
                    missing.append(taxid)
            else:
                missing.append(taxid)
        if missing:
            fetched = _fetch_orthologs_mygene(symbol, source_taxid, missing)
            for taxid, sym in fetched.items():
                _cache_path(symbol, taxid).write_text(json.dumps({"symbol": sym}))
                orthologs[taxid] = sym
        return orthologs

    tf_orthologs = _get_for_gene(tf_symbol)
    tg_orthologs = _get_for_gene(target_symbol)

    supporting: list[str] = []
    n_checked = len(companion_taxids)
    n_both = 0
    for taxid in companion_taxids:
        if tf_orthologs.get(taxid) is not None and tg_orthologs.get(taxid) is not None:
            n_both += 1
            species_name = _TAXID_NAMES.get(taxid, str(taxid))
            supporting.append(species_name)

    ortholog_support = n_both / n_checked if n_checked > 0 else 0.0
    confidence = "high" if ortholog_support >= 0.8 else ("medium" if ortholog_support >= 0.5 else "low")

    return {
        "ortholog_support": ortholog_support,
        "ortholog_confidence": confidence,
        "supporting_species": supporting,
        "conserved_in_human": tf_orthologs.get(9606) is not None and tg_orthologs.get(9606) is not None,
        "conserved_in_mouse": tf_orthologs.get(10090) is not None and tg_orthologs.get(10090) is not None,
    }
