"""
JASPAR REST API client: download and parse TF motif PWMs.

Two complementary download modes
---------------------------------
1. Prebuilt MEME file (recommended for FIMO)
   The JASPAR project publishes ready-to-use MEME-format files for the
   CORE collection per species group.  Download once; subset by TF name
   for faster FIMO runs.

   Entry point: ``download_jaspar_meme_file()``

2. Per-TF JSON from the REST API (lightweight, no large download)
   Useful when you only need a handful of TFs and don't want the full
   ~700-motif MEME file.

   Entry point: ``JASPARClient.fetch_motifs_for_tfs()``

API docs: https://jaspar.elixir.no/api/v1/
MEME files: https://jaspar.elixir.no/downloads/
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prebuilt MEME file download (standard field practice)
# ---------------------------------------------------------------------------

# Reject truncated downloads / wrong ZIP members (seen: ~471 B file with 1 MOTIF).
_MIN_MEME_BYTES = 50_000
_MIN_MEME_MOTIF_LINES = 40


def _count_meme_motifs(meme_text: str) -> int:
    return sum(1 for line in meme_text.splitlines() if line.startswith("MOTIF "))


def _meme_text_looks_valid(meme_text: str) -> bool:
    if not meme_text or "MEME version" not in meme_text:
        return False
    if len(meme_text.encode("utf-8")) < _MIN_MEME_BYTES:
        return False
    if _count_meme_motifs(meme_text) < _MIN_MEME_MOTIF_LINES:
        return False
    return True


def _meme_urls_for_year(tax_group: str, year: int) -> list[str]:
    """Ordered fallbacks for a given JASPAR CORE release year."""
    y = year
    base = "https://jaspar.elixir.no/download/data"
    slug = f"JASPAR{y}_CORE_{tax_group}_non-redundant_pfms_meme"
    return [
        f"{base}/{y}/CORE/{slug}.zip",
        f"{base}/{y}/CORE/{slug}.txt",
        f"https://jaspar.elixir.no/download/CORE/{slug}.zip",
        f"https://jaspar.elixir.no/download/CORE/{slug}.txt",
    ]


def _meme_from_zip_bytes(raw: bytes) -> str:
    """Pick the largest plausible MEME file inside the ZIP (avoid readme/side files)."""
    with zipfile.ZipFile(BytesIO(raw)) as zf:
        names = [
            n
            for n in zf.namelist()
            if not n.endswith("/") and (n.endswith(".meme") or n.endswith(".txt"))
        ]
        if not names:
            raise RuntimeError("No .meme or .txt matrix file found inside ZIP")
        names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        last_err: Exception | None = None
        for name in names:
            try:
                text = zf.read(name).decode("utf-8")
            except UnicodeDecodeError as exc:
                last_err = exc
                continue
            if _meme_text_looks_valid(text):
                return text
        # Fall back to largest decoded text even if below thresholds (caller may retry other URLs)
        try:
            return zf.read(names[0]).decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Could not decode MEME from ZIP: {exc}") from last_err

# Tax-group alias mapping (species name → JASPAR tax_group key)
_SPECIES_TO_TAX: dict[str, str] = {
    "human": "vertebrates",
    "mouse": "vertebrates",
    "rat": "vertebrates",
    "zebrafish": "vertebrates",
    "drosophila": "insects",
    "arabidopsis": "plants",
}


def download_jaspar_meme_file(
    species_or_tax_group: str = "vertebrates",
    cache_dir: str | Path | None = None,
    *,
    jaspar_release: int | None = None,
) -> Path:
    """
    Download the prebuilt JASPAR CORE MEME file for a tax group.

    The file is cached locally; re-running returns the cached copy if it passes
    sanity checks (size + motif count). Corrupt or partial caches are removed
    and re-downloaded.

    Args:
        species_or_tax_group: "vertebrates" | "insects" | "plants" |
                              "mouse" | "human" | "rat" etc.
        cache_dir:            Where to store the downloaded file.
        jaspar_release:       Year, e.g. 2026. If None, tries 2026 then 2024.

    Returns:
        Path to the local .meme file.
    """
    tax_group = _SPECIES_TO_TAX.get(species_or_tax_group.lower(), species_or_tax_group.lower())
    if tax_group not in ("vertebrates", "insects", "plants"):
        raise ValueError(
            f"Unknown tax group '{tax_group}'. Choose from vertebrates, insects, plants"
        )

    release_years: tuple[int, ...]
    if jaspar_release is not None:
        release_years = (jaspar_release,)
    else:
        release_years = (2026, 2024)

    out_dir = Path(cache_dir) if cache_dir else Path(".cache") / "jaspar"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for year in release_years:
        meme_path = out_dir / f"JASPAR{year}_CORE_{tax_group}.meme"
        if meme_path.is_file():
            cached = meme_path.read_text(encoding="utf-8")
            if _meme_text_looks_valid(cached):
                logger.info("Using cached JASPAR MEME file: %s", meme_path)
                return meme_path
            logger.warning(
                "Cached JASPAR MEME looks invalid (%d bytes, %d MOTIF lines) — re-downloading: %s",
                meme_path.stat().st_size,
                _count_meme_motifs(cached),
                meme_path,
            )
            try:
                meme_path.unlink()
            except OSError:
                pass

        meme_text: str | None = None
        for url in _meme_urls_for_year(tax_group, year):
            try:
                logger.info("Downloading JASPAR %d MEME file from %s …", year, url)
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                raw = b"".join(resp.iter_content(chunk_size=65536))
                if url.endswith(".zip"):
                    meme_text = _meme_from_zip_bytes(raw)
                else:
                    meme_text = raw.decode("utf-8")
                if meme_text and _meme_text_looks_valid(meme_text):
                    break
                raise RuntimeError(
                    f"Download from {url} failed validation "
                    f"({len(meme_text or '')} chars, {_count_meme_motifs(meme_text or '')} motifs)"
                )
            except Exception as exc:
                last_exc = exc
                logger.warning("JASPAR MEME download failed via %s: %s", url, exc)
                meme_text = None

        if meme_text and _meme_text_looks_valid(meme_text):
            meme_path.write_text(meme_text, encoding="utf-8")
            logger.info(
                "JASPAR MEME file saved: %s (%d bytes, %d motifs)",
                meme_path,
                meme_path.stat().st_size,
                _count_meme_motifs(meme_text),
            )
            return meme_path

    raise RuntimeError(
        f"Unable to download a valid JASPAR MEME file for {tax_group} "
        f"(tried years {release_years})"
    ) from last_exc


def filter_meme_for_tfs(
    meme_path: str | Path,
    tf_names: list[str],
    output_path: str | Path,
) -> Path:
    """
    Extract only the motifs for the specified TF names from a MEME file.

    FIMO runs much faster when the motif database is small, so pre-filtering
    to the expressed TF set is standard practice.

    The MEME motif header format is::

        MOTIF MA0139.1 CTCF

    Matching is case-insensitive on the TF name (second token).

    Args:
        meme_path:    Full JASPAR MEME file (from ``download_jaspar_meme_file``).
        tf_names:     List of TF gene symbols to retain.
        output_path:  Where to write the filtered MEME file.

    Returns:
        Path to the filtered MEME file.
    """
    meme_path = Path(meme_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tf_upper = {t.upper() for t in tf_names}
    lines = meme_path.read_text(encoding="utf-8").splitlines(keepends=True)

    header_lines: list[str] = []
    motif_blocks: list[list[str]] = []
    current_block: list[str] = []
    in_motif = False

    for line in lines:
        if line.startswith("MOTIF "):
            if in_motif and current_block:
                motif_blocks.append(current_block)
            current_block = [line]
            in_motif = True
        elif in_motif:
            current_block.append(line)
        else:
            header_lines.append(line)

    if in_motif and current_block:
        motif_blocks.append(current_block)

    # Filter blocks whose TF name matches (supports dimers like "SOX2::NANOG" in JASPAR)
    def _motif_tf_tokens(tf_field: str) -> set[str]:
        s = tf_field.strip()
        if not s:
            return set()
        # Split composite labels; keep alphanumerics / common gene symbols
        for sep in ("::", "-", "+"):
            s = s.replace(sep, " ")
        return {p.upper() for p in s.split() if p}

    kept: list[list[str]] = []
    for block in motif_blocks:
        header = block[0]  # e.g. "MOTIF MA0139.1 CTCF\n" or "MOTIF MAxxx SOX2::NANOG\n"
        parts = header.strip().split()
        tf_in_file = parts[2] if len(parts) >= 3 else ""
        tokens = _motif_tf_tokens(tf_in_file)
        if (tf_in_file.upper() in tf_upper) or (tf_upper & tokens):
            kept.append(block)

    with output_path.open("w", encoding="utf-8") as out:
        out.writelines(header_lines)
        for block in kept:
            out.writelines(block)

    logger.info(
        "Filtered MEME: %d / %d motifs kept for %d TFs → %s",
        len(kept),
        len(motif_blocks),
        len(tf_names),
        output_path,
    )
    return output_path


def parse_meme_tf_map(meme_path: str | Path) -> dict[str, str]:
    """
    Parse a MEME file and return {matrix_id: tf_name}.

    Useful for resolving FIMO's ``motif_id`` column back to a gene symbol.
    """
    meme_path = Path(meme_path)
    result: dict[str, str] = {}
    for line in meme_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MOTIF "):
            parts = line.strip().split()
            if len(parts) >= 3:
                result[parts[1]] = parts[2]
            elif len(parts) == 2:
                result[parts[1]] = parts[1]
    return result

_BASE = "https://jaspar.elixir.no/api/v1"

# Nucleotide index: A=0, C=1, G=2, T=3
_NUC_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


class Motif:
    """
    A single TF motif with its count matrix and derived PWM.

    Attributes:
        matrix_id:  JASPAR ID (e.g., "MA0139.1")
        tf_name:    TF gene symbol (e.g., "CTCF")
        pwm:        (L, 4) log-odds matrix, columns = A C G T
        pfm:        (L, 4) probability matrix (pseudocount-smoothed)
        ic:         (L,) information content per position
    """

    def __init__(
        self,
        matrix_id: str,
        tf_name: str,
        pfm: np.ndarray,
        bg: np.ndarray | None = None,
    ) -> None:
        self.matrix_id = matrix_id
        self.tf_name = tf_name
        self.pfm = pfm  # (L, 4) row-normalized probabilities
        _bg = bg if bg is not None else np.full(4, 0.25)
        # PWM = log2(pfm / bg), clipped to avoid -inf
        with np.errstate(divide="ignore"):
            self.pwm = np.log2(np.clip(pfm / _bg[None, :], 1e-10, None))
        # IC per position = sum_j pfm[i,j] * log2(pfm[i,j] / bg[j])
        with np.errstate(divide="ignore"):
            self.ic: np.ndarray = np.sum(
                pfm * np.where(pfm > 0, np.log2(np.clip(pfm / _bg[None, :], 1e-10, None)), 0.0),
                axis=1,
            )

    @property
    def length(self) -> int:
        return self.pfm.shape[0]

    @property
    def max_score(self) -> float:
        return float(self.pwm.max(axis=1).sum())

    @property
    def min_score(self) -> float:
        return float(self.pwm.min(axis=1).sum())

    def __repr__(self) -> str:
        return f"Motif({self.matrix_id}, {self.tf_name}, L={self.length})"


class JASPARClient:
    """
    Downloads and caches TF motif PWMs from the JASPAR 2024 REST API.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        collection: str = "CORE",
        tax_group: str = "vertebrates",
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache") / "jaspar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.collection = collection
        self.tax_group = tax_group
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_motifs_for_tfs(
        self,
        tf_names: list[str],
        max_per_tf: int = 1,
    ) -> dict[str, list[Motif]]:
        """
        Fetch JASPAR motifs for a list of TF gene names.

        Args:
            tf_names:    List of TF gene symbols (e.g., ["CTCF", "SOX2"]).
            max_per_tf:  Max motifs per TF (highest IC picked first).

        Returns:
            {tf_name: [Motif, ...]}  (empty list if none found)
        """
        result: dict[str, list[Motif]] = {}
        for tf in tf_names:
            motifs = self._search_by_name(tf)
            if motifs:
                # Prefer highest total IC
                motifs.sort(key=lambda m: float(m.ic.sum()), reverse=True)
                result[tf] = motifs[:max_per_tf]
            else:
                result[tf] = []
        return result

    def fetch_all_vertebrate_motifs(self) -> list[Motif]:
        """
        Fetch all JASPAR CORE vertebrate motifs (bulk download, then parse).
        Uses cached bulk JSON to avoid hammering the API.
        """
        bulk_cache = self.cache_dir / f"jaspar_{self.collection}_{self.tax_group}_all.json"
        if bulk_cache.is_file():
            raw = json.loads(bulk_cache.read_text(encoding="utf-8"))
        else:
            logger.info("Downloading all JASPAR %s/%s motifs …", self.collection, self.tax_group)
            raw = self._paginate_search(
                collection=self.collection,
                tax_group=self.tax_group,
                page_size=100,
            )
            bulk_cache.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return [self._parse_matrix_entry(e) for e in raw if e.get("pfm")]

    def build_tf_to_motif_index(
        self, motifs: list[Motif]
    ) -> dict[str, list[Motif]]:
        """Group motif list by TF name (case-insensitive key → original case)."""
        idx: dict[str, list[Motif]] = {}
        for m in motifs:
            key = m.tf_name.upper()
            idx.setdefault(key, []).append(m)
        return idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_by_name(self, tf_name: str) -> list[Motif]:
        cache_path = self.cache_dir / f"search_{tf_name.upper()}.json"
        if cache_path.is_file():
            entries = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            params = {
                "name": tf_name,
                "collection": self.collection,
                "tax_group": self.tax_group,
                "page_size": 10,
                "format": "json",
            }
            try:
                resp = self._session.get(f"{_BASE}/matrix/", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                entries = data.get("results", [])
                cache_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.warning("JASPAR search failed for %s: %s", tf_name, exc)
                return []

        motifs: list[Motif] = []
        for entry in entries:
            # entries from search endpoint have condensed pfm; fetch full if needed
            pfm_raw = entry.get("pfm")
            if not pfm_raw:
                pfm_raw = self._fetch_pfm(entry.get("matrix_id", ""))
            if pfm_raw:
                m = self._parse_pfm(
                    matrix_id=entry.get("matrix_id", ""),
                    tf_name=entry.get("name", tf_name),
                    pfm_raw=pfm_raw,
                )
                if m is not None:
                    motifs.append(m)
        return motifs

    def _fetch_pfm(self, matrix_id: str) -> dict[str, list[float]] | None:
        if not matrix_id:
            return None
        cache_path = self.cache_dir / f"matrix_{matrix_id}.json"
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8")).get("pfm")
        try:
            resp = self._session.get(
                f"{_BASE}/matrix/{matrix_id}/", params={"format": "json"}, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data.get("pfm")
        except Exception as exc:
            logger.warning("Failed to fetch JASPAR matrix %s: %s", matrix_id, exc)
            return None

    def _paginate_search(
        self, collection: str, tax_group: str, page_size: int = 100
    ) -> list[dict[str, Any]]:
        url = f"{_BASE}/matrix/"
        params = {
            "collection": collection,
            "tax_group": tax_group,
            "page_size": page_size,
            "format": "json",
        }
        results: list[dict[str, Any]] = []
        page = 1
        while True:
            params["page"] = page
            try:
                resp = self._session.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("results", [])
                results.extend(batch)
                if not data.get("next"):
                    break
                page += 1
            except Exception as exc:
                logger.warning("JASPAR pagination failed at page %d: %s", page, exc)
                break
        return results

    def _parse_matrix_entry(self, entry: dict[str, Any]) -> Motif | None:
        pfm_raw = entry.get("pfm")
        if not pfm_raw:
            return None
        return self._parse_pfm(
            matrix_id=entry.get("matrix_id", ""),
            tf_name=entry.get("name", ""),
            pfm_raw=pfm_raw,
        )

    @staticmethod
    def _parse_pfm(
        matrix_id: str,
        tf_name: str,
        pfm_raw: dict[str, list[float]],
        pseudocount: float = 0.1,
    ) -> Motif | None:
        """
        Convert JASPAR PFM (count dict) → probability matrix → Motif.

        pfm_raw is {'A': [c0, c1, ...], 'C': [...], 'G': [...], 'T': [...]}
        """
        try:
            nucs = ["A", "C", "G", "T"]
            rows = [pfm_raw[n] for n in nucs]  # 4 × L counts
            mat = np.array(rows, dtype=np.float64).T  # L × 4 counts
            mat += pseudocount
            mat /= mat.sum(axis=1, keepdims=True)  # row-normalize → L × 4 probs
            return Motif(matrix_id=matrix_id, tf_name=tf_name, pfm=mat)
        except Exception as exc:
            logger.debug("Failed to parse PFM for %s/%s: %s", matrix_id, tf_name, exc)
            return None
