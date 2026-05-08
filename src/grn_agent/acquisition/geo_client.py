"""
GEO metadata fetcher using NCBI E-utilities and GEO FTP.

Docs: https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html
"""

from __future__ import annotations

import gzip
import html
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import requests

from .compatibility import canonical_species_label

log = logging.getLogger(__name__)


def _soft_key_value(line: str) -> tuple[str, str] | None:
    if " = " not in line:
        return None
    key, value = line.split(" = ", 1)
    return key.strip(), value.strip()


def _series_bucket(accession: str) -> str:
    series_num = str(accession).upper().replace("GSE", "")
    return f"GSE{series_num[:-3]}nnn"


def _sample_bucket(accession: str) -> str:
    sample_num = str(accession).upper().replace("GSM", "")
    return f"GSM{sample_num[:-3]}nnn"


def _is_peak_like_url(raw: str) -> bool:
    low = str(raw or "").lower()
    if not low:
        return False
    if any(bad in low for bad in ("bigwig", ".bw", ".bam", ".bai", "matrix", "count", "fpkm")):
        return False
    return any(ext in low for ext in ("bed", "narrowpeak", "broadpeak", "peaks", "peak_", "_peaks"))


def _normalize_geo_download_url(raw: str) -> str:
    url = str(raw or "").strip()
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return "https://ftp.ncbi.nlm.nih.gov/" + url[len("ftp://ftp.ncbi.nlm.nih.gov/"):]
    if url.startswith("http://ftp.ncbi.nlm.nih.gov/"):
        return "https://ftp.ncbi.nlm.nih.gov/" + url[len("http://ftp.ncbi.nlm.nih.gov/"):]
    return url


def _is_accessibility_text(text: str) -> bool:
    blob = str(text or "").lower()
    return any(tok in blob for tok in ("atac", "dnase", "chromatin accessibility", "open chromatin"))


def _geo_context_query(text: str, *, max_terms: int = 5) -> str:
    stop = {
        "from",
        "with",
        "data",
        "sample",
        "samples",
        "expression",
        "rnaseq",
        "rna",
        "seq",
        "sequencing",
        "single",
        "cell",
        "cells",
        "accession",
    }
    terms: list[str] = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", str(text or "")):
        low = tok.lower()
        if low in stop or low.startswith(("gsm", "gse")):
            continue
        if low not in terms:
            terms.append(low)
        if len(terms) >= max_terms:
            break
    if not terms:
        return ""
    return "(" + " OR ".join(terms) + ")"


def is_likely_accessibility_series(title: str, summary: str, gdstype: str) -> bool:
    """
    Keep GEO series with accessibility evidence, including mixed RNA+ATAC studies.

    Older logic rejected any summary mentioning RNA-seq, which drops multimodal
    series where the series contains paired RNA and ATAC GSMs.
    """
    blob = f"{title or ''} {summary or ''} {gdstype or ''}".lower()
    if not _is_accessibility_text(blob):
        return False
    expression_only_type = "expression profiling" in str(gdstype or "").lower()
    # A series with explicit ATAC/DNase/chromatin-accessibility text is not
    # expression-only, even if it also mentions RNA-seq.
    return not (expression_only_type and not _is_accessibility_text(f"{title} {summary}"))


def is_likely_accessibility_sample(sample_meta: dict[str, Any]) -> bool:
    library_strategy = str(sample_meta.get("library_strategy", "") or "").strip().lower()
    if library_strategy in {"rna-seq", "rna seq", "scrna-seq", "scrna seq", "single-cell rna-seq"}:
        return False
    text_parts = [
        sample_meta.get("title", ""),
        sample_meta.get("source_name", ""),
        sample_meta.get("library_strategy", ""),
        sample_meta.get("library_source", ""),
        sample_meta.get("library_selection", ""),
        sample_meta.get("extract_protocol", ""),
        sample_meta.get("data_processing", ""),
        sample_meta.get("characteristics_text", ""),
        " ".join(str(x) for x in sample_meta.get("supplementary_files", []) or []),
    ]
    return _is_accessibility_text(" ".join(str(x) for x in text_parts))


class GEOClient:
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo"
    
    def __init__(self, email: str = "user@example.com", cache_dir: str | Path | None = None):
        self.email = email
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".geo_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._last_request_ts = 0.0
        # NCBI E-utilities recommendation: <= 3 req/s without API key.
        self._min_request_interval_s = 0.35

    def _get_with_retry(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: int = 30,
        stream: bool = False,
        max_retries: int = 5,
    ) -> requests.Response:
        """Retry transient GEO throttling/errors (429/5xx) with exponential backoff."""
        delay_s = 0.6
        for attempt in range(max_retries):
            now = time.time()
            delta = now - self._last_request_ts
            if delta < self._min_request_interval_s:
                time.sleep(self._min_request_interval_s - delta)
            resp = self.session.get(url, params=params, timeout=timeout, stream=stream)
            self._last_request_ts = time.time()
            if resp.status_code not in (429, 500, 502, 503, 504):
                resp.raise_for_status()
                return resp
            log.warning(
                "GEO request throttled/failed: status=%s url=%s attempt=%d/%d",
                resp.status_code,
                url,
                attempt + 1,
                max_retries,
            )
            if attempt == max_retries - 1:
                log.error("GEO request exhausted retries: status=%s url=%s", resp.status_code, url)
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after and retry_after.isdigit() else delay_s
            log.warning("GEO retrying after %.2fs for %s", wait_s, url)
            time.sleep(wait_s)
            delay_s = min(delay_s * 2.0, 8.0)
        raise RuntimeError("GEO request failed after retries")
    
    def get_series_metadata(self, accession: str, geo_id: str | None = None) -> dict[str, Any]:
        """
        Fetch GEO series metadata (GSE...).
        
        Returns dict with:
          - title, summary, organism, experiment_type
          - samples: list of GSM accessions
          - supplementary_files: list of file URLs
        """
        cache_path = self.cache_dir / f"{accession}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if "sample_metadata" in cached:
                return cached
        
        resolved_geo_id = str(geo_id or "").strip()
        if not resolved_geo_id:
            url = f"{self.EUTILS_BASE}/esearch.fcgi"
            params = {"db": "gds", "term": accession, "retmode": "json", "email": self.email}
            resp = self._get_with_retry(url, params=params, timeout=30)
            search_data = resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                raise ValueError(f"No GEO record found for {accession}")
            resolved_geo_id = id_list[0]

        summary_url = f"{self.EUTILS_BASE}/esummary.fcgi"
        summary_params = {"db": "gds", "id": resolved_geo_id, "retmode": "json", "email": self.email}
        resp2 = self._get_with_retry(summary_url, params=summary_params, timeout=30)
        summary = resp2.json().get("result", {}).get(resolved_geo_id, {})
        
        soft_meta = self._extract_series_from_soft(accession)
        out = {
            "accession": accession,
            "title": soft_meta.get("title") or summary.get("title", ""),
            "summary": soft_meta.get("summary") or summary.get("summary", ""),
            "organism": soft_meta.get("organism") or summary.get("taxon", ""),
            "experiment_type": summary.get("gdstype", ""),
            "samples": soft_meta.get("samples", []),
            "sample_metadata": soft_meta.get("sample_metadata", []),
            "supplementary_files": self._list_supplementary_files(accession),
        }
        cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    def search_series(
        self,
        *,
        species: str,
        cell_type: str | None = None,
        lineage: str | None = None,
        state: str | None = None,
        cell_context: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search GEO series (GSE) using broad accessibility keywords + context terms.
        Returns lightweight summary rows with accession/title/summary/taxon/gdstype.
        """
        terms = [
            f"({species})",
            "(ATAC OR DNase OR \"chromatin accessibility\")",
            "(\"high throughput sequencing\" OR sequencing)",
        ]
        if cell_type:
            terms.append(f"({cell_type})")
        if lineage:
            terms.append(f"({lineage})")
        if state:
            terms.append(f"({state})")
        context_query = _geo_context_query(cell_context or "")
        if context_query:
            terms.append(context_query)
        query = " AND ".join(terms)

        url = f"{self.EUTILS_BASE}/esearch.fcgi"
        params = {"db": "gds", "term": query, "retmax": max(1, limit), "retmode": "json", "email": self.email}
        resp = self._get_with_retry(url, params=params, timeout=30)
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        summary_url = f"{self.EUTILS_BASE}/esummary.fcgi"
        time.sleep(0.4)
        resp2 = self._get_with_retry(
            summary_url,
            params={"db": "gds", "id": ",".join(ids[:limit]), "retmode": "json", "email": self.email},
            timeout=30,
        )
        res = resp2.json().get("result", {})
        out: list[dict[str, Any]] = []
        for gid in ids[:limit]:
            it = res.get(gid, {})
            acc = str(it.get("accession", "")).strip()
            if not acc.startswith("GSE"):
                continue
            title = str(it.get("title", "") or "")
            summary = str(it.get("summary", "") or "")
            gdstype = str(it.get("gdstype", "") or "")
            if not is_likely_accessibility_series(title, summary, gdstype):
                continue
            out.append(
                {
                    "geo_id": str(gid),
                    "accession": acc,
                    "title": title,
                    "summary": summary,
                    "organism": it.get("taxon", ""),
                    "experiment_type": gdstype,
                }
            )
        return out

    def _family_soft_url(self, accession: str) -> str:
        acc = str(accession).upper().strip()
        return f"{self.GEO_FTP}/series/{_series_bucket(acc)}/{acc}/soft/{acc}_family.soft.gz"

    def _sample_supplementary_url(self, accession: str) -> str:
        acc = str(accession).upper().strip()
        return f"{self.GEO_FTP}/samples/{_sample_bucket(acc)}/{acc}/suppl/"

    def _get_family_soft_text(self, accession: str) -> str:
        resp = self._get_with_retry(self._family_soft_url(accession), timeout=60)
        if resp.status_code != 200:
            return ""
        return gzip.decompress(resp.content).decode("utf-8", errors="ignore")

    @staticmethod
    def _parse_sample_stanza(accession: str, rows: dict[str, list[str]]) -> dict[str, Any]:
        characteristics: dict[str, list[str]] = {}
        characteristics_text: list[str] = []
        for raw in rows.get("!Sample_characteristics_ch1", []):
            val = str(raw).strip()
            if not val:
                continue
            characteristics_text.append(val)
            if ":" in val:
                key, cval = val.split(":", 1)
                k = key.strip().lower().replace(" ", "_")
                v = cval.strip()
                if k and v:
                    characteristics.setdefault(k, []).append(v)

        def first(*keys: str) -> str:
            for key in keys:
                vals = rows.get(key, [])
                if vals:
                    return str(vals[0]).strip()
            return ""

        supp_files = [
            _normalize_geo_download_url(str(v).strip())
            for v in rows.get("!Sample_supplementary_file", [])
            if str(v).strip() and str(v).strip().upper() != "NONE"
        ]
        relations = [str(v).strip() for v in rows.get("!Sample_relation", []) if str(v).strip()]
        series_accessions = [
            str(v).strip()
            for v in rows.get("!Sample_series_id", [])
            if str(v).strip().startswith("GSE")
        ]

        return {
            "accession": accession,
            "title": first("!Sample_title"),
            "organism": first("!Sample_organism_ch1"),
            "source_name": first("!Sample_source_name_ch1"),
            "molecule": first("!Sample_molecule_ch1"),
            "library_strategy": first("!Sample_library_strategy"),
            "library_source": first("!Sample_library_source"),
            "library_selection": first("!Sample_library_selection"),
            "extract_protocol": first("!Sample_extract_protocol_ch1"),
            "data_processing": "\n".join(rows.get("!Sample_data_processing", [])),
            "platform_id": first("!Sample_platform_id"),
            "series_accessions": list(dict.fromkeys(series_accessions)),
            "relations": relations,
            "characteristics": {k: list(dict.fromkeys(v)) for k, v in characteristics.items()},
            "characteristics_text": "; ".join(characteristics_text),
            "supplementary_files": supp_files,
        }

    @classmethod
    def _parse_family_soft(cls, text: str) -> dict[str, Any]:
        series: dict[str, list[str]] = {}
        samples: list[dict[str, Any]] = []
        current_kind = ""
        current_acc = ""
        current_rows: dict[str, list[str]] = {}

        def flush_sample() -> None:
            nonlocal current_kind, current_acc, current_rows
            if current_kind == "SAMPLE" and current_acc:
                samples.append(cls._parse_sample_stanza(current_acc, current_rows))

        for line in str(text or "").splitlines():
            if line.startswith("^SERIES = "):
                flush_sample()
                current_kind = "SERIES"
                current_acc = line.split(" = ", 1)[1].strip()
                current_rows = {}
                continue
            if line.startswith("^SAMPLE = "):
                flush_sample()
                current_kind = "SAMPLE"
                current_acc = line.split(" = ", 1)[1].strip()
                current_rows = {}
                continue
            kv = _soft_key_value(line)
            if not kv:
                continue
            key, value = kv
            if current_kind == "SERIES":
                series.setdefault(key, []).append(value)
            elif current_kind == "SAMPLE":
                current_rows.setdefault(key, []).append(value)
        flush_sample()

        organisms = [
            str(s.get("organism", "")).strip()
            for s in samples
            if str(s.get("organism", "")).strip()
        ]
        return {
            "title": (series.get("!Series_title") or [""])[0],
            "summary": "\n".join(series.get("!Series_summary", [])),
            "organism": organisms[0] if organisms else "",
            "samples": [str(s.get("accession", "")).strip() for s in samples if str(s.get("accession", "")).strip()],
            "sample_metadata": samples,
        }

    def _extract_series_from_soft(self, accession: str) -> dict[str, Any]:
        try:
            return self._parse_family_soft(self._get_family_soft_text(accession))
        except Exception:
            return {"samples": [], "sample_metadata": []}
    
    def _extract_samples_from_soft(self, accession: str) -> list[str]:
        """Parse SOFT file for sample accessions."""
        try:
            text = self._get_family_soft_text(accession)
            samples = re.findall(r"\^SAMPLE = (GSM\d+)", text)
            return list(dict.fromkeys(samples))
        except Exception:
            return []
    
    def _list_supplementary_files(self, accession: str) -> list[str]:
        """List supplementary file URLs for a series."""
        acc = str(accession).upper().strip()
        suppl_url = f"{self.GEO_FTP}/series/{_series_bucket(acc)}/{acc}/suppl/"
        try:
            resp = self._get_with_retry(suppl_url, timeout=30)
            if resp.status_code != 200:
                return []
            return self._links_from_ftp_listing(suppl_url, resp.text)
        except Exception:
            return []

    @staticmethod
    def _links_from_ftp_listing(base_url: str, html: str) -> list[str]:
        links = re.findall(r'href="([^"]+)"', html)
        out: list[str] = []
        for link in links:
            if not link or link in ("../",) or link.startswith("?"):
                continue
            if "vulnerability-disclosure-policy" in link:
                continue
            if link.startswith("http://") or link.startswith("https://"):
                continue
            if "/" in link and not link.endswith("/"):
                continue
            if link.endswith("/"):
                continue
            out.append(_normalize_geo_download_url(urljoin(base_url, link)))
        return out

    def _list_sample_supplementary_files(self, accession: str) -> list[str]:
        """List sample-level supplementary file URLs for a GSM accession."""
        acc = str(accession).upper().strip()
        suppl_url = self._sample_supplementary_url(acc)
        out: list[str] = []
        try:
            resp = self._get_with_retry(suppl_url, timeout=30)
            if resp.status_code == 200:
                out.extend(self._links_from_ftp_listing(suppl_url, resp.text))
        except Exception:
            pass

        if out:
            return list(dict.fromkeys(out))

        # Some GEO sample directories are not discoverable via plain directory
        # listing from all networks, while the accession page still exposes the
        # processed-data table. Parse that page as a fallback.
        try:
            resp = self._get_with_retry(
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi",
                params={"acc": acc},
                timeout=60,
            )
            out.extend(self._extract_supplementary_links_from_geo_page(acc, resp.text))
        except Exception:
            pass
        return list(dict.fromkeys(out))

    @staticmethod
    def _extract_supplementary_links_from_geo_page(accession: str, text: str) -> list[str]:
        """Extract direct sample supplementary URLs from a GEO HTML/text page."""
        acc = str(accession or "").upper().strip()
        if not acc:
            return []
        out: list[str] = []
        body = html.unescape(str(text or ""))
        for raw_href in re.findall(r'href=["\']([^"\']+)["\']', body, flags=re.IGNORECASE):
            href = html.unescape(raw_href.strip())
            if not href:
                continue
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = urljoin("https://www.ncbi.nlm.nih.gov", href)

            parsed = urlparse(href)
            low_path = parsed.path.lower()
            if "ftp.ncbi.nlm.nih.gov" in parsed.netloc.lower() and f"/{acc.lower()}/suppl/" in low_path:
                out.append(_normalize_geo_download_url(href))
                continue

            if parsed.netloc.lower().endswith("ncbi.nlm.nih.gov") and low_path.endswith("/geo/download/"):
                qs = parse_qs(parsed.query)
                q_acc = str((qs.get("acc") or [""])[0]).upper()
                filename = str((qs.get("file") or [""])[0]).strip()
                if q_acc == acc and filename:
                    out.append(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{_sample_bucket(acc)}/{acc}/suppl/{filename}")

        # Last-resort recovery from visible supplementary table filenames.
        file_re = re.compile(rf"\b({re.escape(acc)}[A-Za-z0-9_.+-]+\.(?:bed|narrowPeak|broadPeak|bw|bigWig|txt)(?:\.gz)?)\b")
        for filename in file_re.findall(body):
            out.append(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{_sample_bucket(acc)}/{acc}/suppl/{filename}")
        return list(dict.fromkeys(_normalize_geo_download_url(u) for u in out if u))
    
    def download_supplementary_file(self, url: str, output_path: str | Path) -> None:
        """Download a supplementary file from GEO FTP."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        resp = self._get_with_retry(_normalize_geo_download_url(url), stream=True, timeout=120)
        with out.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def get_sample_metadata(self, gsm_accession: str) -> dict[str, Any]:
        """Fetch sample-level metadata."""
        gsm_accession = str(gsm_accession).upper().strip()
        cache_path = self.cache_dir / f"{gsm_accession}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if "series_accessions" in cached and "supplementary_files" in cached:
                # Older cache entries may have captured ATAC/DNase metadata
                # before GSM-page supplementary parsing existed. Re-fetch those
                # likely accessibility samples when their file list is empty.
                if cached.get("supplementary_files") or not is_likely_accessibility_sample(cached):
                    return cached

        out: dict[str, Any] = {"accession": gsm_accession}
        try:
            url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            resp = self._get_with_retry(
                url,
                params={"acc": gsm_accession, "targ": "self", "form": "text", "view": "full"},
                timeout=60,
            )
            parsed = self._parse_sample_soft_text(gsm_accession, resp.text)
            if parsed:
                out.update(parsed)
        except Exception:
            pass

        if not out.get("title") or not out.get("organism"):
            url = f"{self.EUTILS_BASE}/esearch.fcgi"
            params = {"db": "gds", "term": gsm_accession, "retmode": "json", "email": self.email}
            resp = self._get_with_retry(url, params=params, timeout=30)
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                out["error"] = "not_found"
            else:
                geo_id = id_list[0]
                summary_url = f"{self.EUTILS_BASE}/esummary.fcgi"
                time.sleep(0.4)
                resp2 = self._get_with_retry(
                    summary_url,
                    params={"db": "gds", "id": geo_id, "retmode": "json", "email": self.email},
                    timeout=30,
                )
                summary = resp2.json().get("result", {}).get(geo_id, {})
                if not out.get("title"):
                    out["title"] = summary.get("title", "")
                if not out.get("organism"):
                    out["organism"] = summary.get("taxon", "")

        supp_files = list(out.get("supplementary_files", []) or [])
        supp_files.extend(self._list_sample_supplementary_files(gsm_accession))
        out["supplementary_files"] = list(dict.fromkeys(str(u) for u in supp_files if str(u).strip()))
        cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    @classmethod
    def _parse_sample_soft_text(cls, accession: str, text: str) -> dict[str, Any]:
        current_acc = ""
        rows: dict[str, list[str]] = {}
        for line in str(text or "").splitlines():
            if line.startswith("^SAMPLE = "):
                current_acc = line.split(" = ", 1)[1].strip()
                rows = {}
                continue
            if current_acc and current_acc != accession:
                continue
            kv = _soft_key_value(line)
            if kv:
                key, value = kv
                rows.setdefault(key, []).append(value)
        if not current_acc:
            return {}
        return cls._parse_sample_stanza(accession, rows)

    @staticmethod
    def _match_tokens(sample_meta: dict[str, Any]) -> set[str]:
        fields: list[str] = [
            str(sample_meta.get("title", "") or ""),
            str(sample_meta.get("source_name", "") or ""),
            str(sample_meta.get("organism", "") or ""),
            str(sample_meta.get("characteristics_text", "") or ""),
        ]
        chars = sample_meta.get("characteristics", {})
        if isinstance(chars, dict):
            for vals in chars.values():
                if isinstance(vals, list):
                    fields.extend(str(v) for v in vals)
                else:
                    fields.append(str(vals))
        return {
            tok
            for tok in re.findall(r"[a-z0-9]+", " ".join(fields).lower())
            if len(tok) >= 3 and tok not in {"rna", "seq", "atac", "dnase", "sample"}
        }

    def find_accessibility_samples_for_rna(
        self,
        rna_accession: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Find ATAC/DNase GSMs in the same parent GSE(s) as an RNA GSM.

        This supports paired/multimodal GEO studies where the series is mixed
        RNA+ATAC and the usable peaks are attached to an ATAC sample, not the GSE.
        """
        rna_meta = self.get_sample_metadata(rna_accession)
        parent_series = [
            str(acc).strip()
            for acc in rna_meta.get("series_accessions", []) or []
            if str(acc).strip().startswith("GSE")
        ]
        out: list[dict[str, Any]] = []
        rna_tokens = self._match_tokens(rna_meta)
        rna_species = canonical_species_label(rna_meta.get("organism"))

        for gse in parent_series:
            series = self.get_series_metadata(gse)
            for sample in series.get("sample_metadata", []) or []:
                if not isinstance(sample, dict):
                    continue
                acc = str(sample.get("accession", "")).strip()
                if not acc or acc == str(rna_accession).upper().strip():
                    continue
                sample_files = list(sample.get("supplementary_files", []) or [])
                sample_files.extend(self._list_sample_supplementary_files(acc))
                sample = dict(sample)
                sample["supplementary_files"] = list(dict.fromkeys(str(u) for u in sample_files if str(u).strip()))
                if not is_likely_accessibility_sample(sample):
                    continue
                sample_species = canonical_species_label(sample.get("organism"))
                if rna_species and sample_species and sample_species != rna_species:
                    continue
                sample_tokens = self._match_tokens(sample)
                overlap = len(rna_tokens & sample_tokens) / max(1, len(rna_tokens | sample_tokens))
                has_peak = any(_is_peak_like_url(u) for u in sample["supplementary_files"])
                score = overlap + (1.0 if has_peak else 0.0)
                sample["parent_series_accession"] = gse
                sample["parent_series_title"] = series.get("title", "")
                sample["matched_rna_accession"] = str(rna_accession).upper().strip()
                sample["pairing_score"] = float(score)
                sample["pairing_quality"] = "same_series_sample"
                out.append(sample)

        out.sort(key=lambda s: (-float(s.get("pairing_score", 0.0)), str(s.get("accession", ""))))
        return out[: max(1, int(limit))]
