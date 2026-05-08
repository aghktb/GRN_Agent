"""
ENCODE portal REST API client for automated DNase/ATAC-seq metadata + file retrieval.

API docs: https://www.encodeproject.org/help/rest-api/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"cell", "cells", "primary", "normal", "tissue", "of", "and", "the"}


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(str(text or "").lower()) if t}


def _significant_tokens(tokens: set[str]) -> set[str]:
    return {t for t in tokens if t not in _STOPWORDS}


def _abbrev_candidates(requested_tokens: set[str]) -> set[str]:
    sig = [t for t in sorted(requested_tokens) if t not in _STOPWORDS and len(t) >= 3]
    out: set[str] = set()
    if len(sig) >= 2:
        out.add("".join(t[0] for t in sig))
    # ENCODE often uses short stem-cell line codes in biosample summaries.
    if "stem" in requested_tokens and "cell" in requested_tokens:
        out.add("es")
    return {x for x in out if x}


def _biosample_term_matches(requested: str, experiment_term: str, biosample_summary: str = "") -> bool:
    """
    Match ENCODE biosample_ontology.term_name (+ optional free-text summary) to the
    user's --cell-type string.

    Uses token Jaccard on ``term_name`` (no raw substring between phrases): otherwise
    ``embryo`` would match ``embryonic stem cell`` and whole-embryo DNase (e.g.
    ENCSR723IXU) would be selected for mESC workflows.
    """
    requested_lc = str(requested or "").strip().lower()
    term_lc = str(experiment_term or "").strip().lower()
    summ_lc = str(biosample_summary or "").strip().lower()
    if not requested_lc:
        return True
    if not term_lc and not summ_lc:
        return False
    if requested_lc == term_lc:
        return True
    req_tokens = _tokenize(requested_lc)
    term_tokens = _tokenize(term_lc)
    blob_tokens = _tokenize(f"{term_lc} {summ_lc}")
    req_sig = _significant_tokens(req_tokens)
    term_sig = _significant_tokens(term_tokens)
    blob_sig = _significant_tokens(blob_tokens)
    if not req_sig:
        return requested_lc in term_lc
    if not term_sig:
        return False
    # Strict term-name Jaccard protects against embryo-vs-embryonic false positives.
    j_term = len(req_sig & term_sig) / max(1, len(req_sig | term_sig))
    if j_term >= 0.6:
        return True
    # Some ENCODE entries abbreviate cell lines in summary text.
    j_blob = len(req_sig & blob_sig) / max(1, len(req_sig | blob_sig))
    if j_blob >= 0.5 and len(req_sig & blob_sig) >= 2:
        return True
    compact_blob = f" {term_lc} {summ_lc} "
    for abbr in _abbrev_candidates(req_tokens):
        if re.search(rf"\\b{re.escape(abbr)}\\b", compact_blob):
            return True
    # Generic stem-cell code fallback; gated by stem/cell query intent.
    if {"stem", "cell"}.issubset(req_tokens) and re.search(r"\bes[- ]?[a-z0-9]{2,}\b", compact_blob):
        return True
    return False


class ENCODEClient:
    BASE_URL = "https://www.encodeproject.org"
    
    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".encode_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        # ENCODE sits behind AWS WAF: custom / library User-Agents often get HTTP 202
        # with an empty body (then JSON parse fails with "Expecting value line 1 column 1").
        # Browser-like headers match what worked historically for programmatic access.
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
                    "Gecko/20100101 Firefox/128.0"
                ),
                "Referer": f"{self.BASE_URL}/",
            }
        )
    
    def search_experiments(
        self,
        assay_title: str,
        organism: str,
        biosample_term_name: str | None = None,
        status: str = "released",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search for experiments matching criteria.
        
        Args:
            assay_title: e.g., "DNase-seq", "ATAC-seq"
            organism: e.g., "Homo sapiens", "Mus musculus"
            biosample_term_name: e.g., "embryonic stem cell", "hematopoietic stem cell"
            status: "released" (default), "in progress", etc.
            limit: max results
        
        Returns:
            List of experiment metadata dicts.
        """
        # NOTE: Including biosample_ontology.term_name in the query string causes HTTP 404
        # on encodeproject.org/search (verified 2026-04). Fetch broadly, then filter locally.
        # ENCODE search is paginated; scan deeper windows to avoid missing valid matches.
        fetch_limit = max(1, int(limit))
        params_base: dict[str, Any] = {
            "type": "Experiment",
            "assay_title": assay_title,
            "replicates.library.biosample.organism.scientific_name": organism,
            "status": status,
            "limit": fetch_limit,
            "format": "json",
        }
        def _fetch_once(base_params: dict[str, Any]) -> list[dict[str, Any]]:
            resp = self.session.get(f"{self.BASE_URL}/search/", params=base_params, timeout=60)
            resp.raise_for_status()
            if not (resp.text or "").strip():
                raise RuntimeError(
                    f"ENCODE /search/ returned HTTP {resp.status_code} with an empty body "
                    "(often AWS WAF). Retry later, use another network, or pass --atac-file."
                )
            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                _snippet = (resp.text or "")[:400].replace("\n", " ")
                logger.warning(
                    "ENCODE /search/ returned non-JSON (status=%s len=%s): %s",
                    resp.status_code,
                    len(resp.text or ""),
                    _snippet,
                )
                raise RuntimeError(
                    f"ENCODE search returned non-JSON (HTTP {resp.status_code}); "
                    "check network or portal status"
                ) from exc
            graph = data.get("@graph", [])
            return graph if isinstance(graph, list) else []

        graph = _fetch_once(params_base)
        # Reliability fallback: ENCODE indexing for the organism filter can be incomplete
        # for some experiments. Mix in assay-wide results to avoid blind spots.
        params_relaxed = dict(params_base)
        params_relaxed.pop("replicates.library.biosample.organism.scientific_name", None)
        relaxed = _fetch_once(params_relaxed)
        if relaxed:
            merged: dict[str, dict[str, Any]] = {}
            preferred = graph
            # Reserve capacity for relaxed-only entries even when filtered already fills limit.
            if len(preferred) >= fetch_limit:
                keep_filtered = max(1, int(fetch_limit * 0.7))
                preferred = preferred[:keep_filtered]
            for exp in preferred + relaxed:
                acc = str(exp.get("accession", "")).strip()
                if not acc or acc in merged:
                    continue
                merged[acc] = exp
                if len(merged) >= fetch_limit:
                    break
            graph = list(merged.values())
        if len(graph) > fetch_limit:
            graph = graph[:fetch_limit]

        if biosample_term_name:
            out: list[dict[str, Any]] = []
            for exp in graph:
                term = exp.get("biosample_ontology", {})
                term_name = term.get("term_name", "") if isinstance(term, dict) else ""
                summary = str(exp.get("biosample_summary", "") or "")
                if _biosample_term_matches(biosample_term_name, str(term_name), summary):
                    out.append(exp)
            return out[:limit]
        return graph[:limit]

    def search_experiments_free_text(
        self,
        *,
        query_text: str,
        assay_title: str,
        status: str = "released",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        ENCODE free-text fallback search (closer to portal UI behavior).
        """
        q = str(query_text or "").strip()
        if not q:
            return []

        params_base: dict[str, Any] = {
            "type": "Experiment",
            "assay_title": assay_title,
            "status": status,
            "limit": max(1, int(limit)),
            "format": "json",
        }
        # Try UI-like search params first; some ENCODE deployments reject these with 404.
        for key in ("searchTerm", "q"):
            try:
                params = dict(params_base)
                params[key] = q
                resp = self.session.get(f"{self.BASE_URL}/search/", params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                graph = data.get("@graph", [])
                if graph:
                    return graph[:limit]
            except Exception:
                pass

        # Final fallback: broad assay-wide pull + local text ranking.
        # This avoids endpoint-specific searchTerm quirks while preserving semantic recall.
        params = dict(params_base)
        params["limit"] = max(500, int(limit))
        resp = self.session.get(f"{self.BASE_URL}/search/", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        graph = data.get("@graph", [])
        if not isinstance(graph, list) or not graph:
            return []

        q_tokens = _significant_tokens(_tokenize(q))
        if not q_tokens:
            return graph[:limit]

        scored: list[tuple[float, dict[str, Any]]] = []
        for exp in graph:
            term = exp.get("biosample_ontology", {})
            term_name = str(term.get("term_name", "") if isinstance(term, dict) else "")
            summary = str(exp.get("biosample_summary", "") or "")
            aliases = term.get("aliases", []) if isinstance(term, dict) else []
            syns = term.get("synonyms", []) if isinstance(term, dict) else []
            blob = " ".join([term_name, summary, " ".join(str(a) for a in aliases), " ".join(str(s) for s in syns)])
            b_tokens = _significant_tokens(_tokenize(blob))
            if not b_tokens:
                continue
            overlap = len(q_tokens & b_tokens) / max(1, len(q_tokens))
            if overlap > 0.0:
                scored.append((overlap, exp))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:limit]]
    
    def get_experiment_metadata(self, accession: str) -> dict[str, Any]:
        """Fetch full experiment metadata by accession (e.g., ENCSR...)."""
        cache_path = self.cache_dir / f"{accession}.json"
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        
        resp = self.session.get(f"{self.BASE_URL}/experiments/{accession}/?format=json", timeout=60)
        resp.raise_for_status()
        data = resp.json()
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
    
    def get_file_metadata(self, file_accession: str) -> dict[str, Any]:
        """Fetch file metadata (e.g., ENCFF...)."""
        resp = self.session.get(f"{self.BASE_URL}/files/{file_accession}/?format=json", timeout=60)
        resp.raise_for_status()
        return resp.json()
    
    def download_file(self, file_accession: str, output_path: str | Path) -> None:
        """Download ENCODE file to local path (resolve canonical URL from file metadata)."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        meta = self.get_file_metadata(file_accession)
        candidates: list[str] = []

        # Best source: cloud URL when present
        cloud = meta.get("cloud_metadata", {}) if isinstance(meta.get("cloud_metadata"), dict) else {}
        cloud_url = str(cloud.get("url", "")).strip()
        if cloud_url:
            candidates.append(cloud_url)

        # ENCODE canonical relative href
        href = str(meta.get("href", "")).strip()
        if href:
            if href.startswith("http://") or href.startswith("https://"):
                candidates.append(href)
            else:
                candidates.append(f"{self.BASE_URL}{href}")

        # Fallback patterns by known file format/type
        fmt = str(meta.get("file_format", "")).strip().lower()
        fmt_type = str(meta.get("file_format_type", "")).strip().lower()
        ext = "bed.gz"
        if "narrowpeak" in fmt_type or "narrowpeak" in fmt:
            ext = "narrowPeak.gz"
        elif "broadpeak" in fmt_type or "broadpeak" in fmt:
            ext = "broadPeak.gz"
        candidates.append(f"{self.BASE_URL}/files/{file_accession}/@@download/{file_accession}.{ext}")
        # Legacy fallback
        candidates.append(f"{self.BASE_URL}/files/{file_accession}/@@download/{file_accession}.bed.gz")

        last_exc: Exception | None = None
        for url in list(dict.fromkeys(candidates)):
            try:
                resp = self.session.get(url, stream=True, timeout=120)
                resp.raise_for_status()
                with out.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                continue

        raise RuntimeError(
            f"Failed to download ENCODE file {file_accession}; tried {len(candidates)} URL variants"
        ) from last_exc
    
    def filter_files_by_output_type(
        self,
        experiment: dict[str, Any],
        output_type: str = "peaks",
        file_format: str = "bed",
        assembly: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract files matching criteria from experiment metadata.
        
        Args:
            experiment: full experiment dict from get_experiment_metadata
            output_type: e.g., "peaks", "alignments", "signal"
            file_format: e.g., "bed", "bigWig", "bam"
            assembly: e.g., "mm10", "hg38" (None = any)
        
        Returns:
            List of file metadata dicts.
        """
        files = experiment.get("files", [])
        out = []
        for f in files:
            if f.get("output_type") != output_type:
                continue
            if f.get("file_format") != file_format:
                continue
            if assembly and f.get("assembly") != assembly:
                continue
            if f.get("status") not in ("released", "in progress"):
                continue
            out.append(f)
        return out
    
    def get_qc_metrics(self, file_accession: str) -> dict[str, Any]:
        """Fetch QC metrics for a file (e.g., SPOT score for DNase)."""
        meta = self.get_file_metadata(file_accession)
        qc = meta.get("quality_metrics", [])
        if not qc:
            return {}
        return qc[0] if isinstance(qc, list) else qc
