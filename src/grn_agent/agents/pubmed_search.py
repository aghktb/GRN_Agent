# pubmed_search.py
# Stage 1: Query PubMed via NCBI E-utilities and fetch abstracts.
#
# Two API calls per query:
#   ESearch  → returns a list of PMIDs matching the query
#   EFetch   → returns full records (title + abstract) for those PMIDs
#
# Results are cached to disk so re-runs skip already-fetched pairs.

import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import requests

from grn_agent.agents import lit_config as config
from grn_agent.agents import gene_aliases


def _base_params() -> dict:
    """Common params for every E-utils call. API key (if set) raises rate limit."""
    p = {"tool": "lit_mining_agent", "email": config.NCBI_EMAIL}
    if config.NCBI_API_KEY:
        p["api_key"] = config.NCBI_API_KEY
    return p


def _request_with_retry(url: str, params: dict, timeout: int) -> requests.Response:
    """
    GET with exponential backoff on 429 / 5xx / connection errors.

    NCBI E-utils returns 429 when you exceed 3 req/sec without an API key
    (10 req/sec with one). Retry pattern:
      attempt 1: wait base × 2^0 = 1s
      attempt 2: wait base × 2^1 = 2s
      ...
      attempt N: wait base × 2^(N-1)  (+ jitter)
    """
    last_exc: Exception | None = None
    for attempt in range(config.PUBMED_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                # Honor Retry-After if NCBI sent it; else exponential w/ jitter
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = int(retry_after)
                else:
                    sleep_s = config.PUBMED_BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"  [WARN] NCBI {resp.status_code}, backing off {sleep_s:.1f}s "
                      f"(attempt {attempt + 1}/{config.PUBMED_MAX_RETRIES})")
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            sleep_s = config.PUBMED_BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"  [WARN] NCBI {type(e).__name__}, backing off {sleep_s:.1f}s "
                  f"(attempt {attempt + 1}/{config.PUBMED_MAX_RETRIES})")
            time.sleep(sleep_s)
    raise RuntimeError(
        f"NCBI E-utils gave up after {config.PUBMED_MAX_RETRIES} retries: {url}"
    ) from last_exc


def _query_term(name: str) -> str:
    """Build a PubMed OR clause covering all known aliases for a gene name."""
    aliases = gene_aliases.get_aliases(name)
    parts = " OR ".join(f'"{a}"[tiab]' for a in aliases)
    return f"({parts})"


# ----------------------------- Composite ranking ----------------------------
# Surfaces foundational papers buried by PubMed's relevance ranking.
# See test_ranking.py for the validated test cases that informed weights.

def icite_lookup(pmids: list[str]) -> dict[str, dict]:
    """
    Batch citation metrics from NIH iCite. Returns {pmid: {citation_count, rcr, year, ...}}.

    Free, no auth, ~200 ms per chunk of 200 PMIDs. RCR (Relative Citation Ratio)
    is field-normalized — preferred over raw count to avoid age bias.
    """
    if not pmids:
        return {}
    out = {}
    for i in range(0, len(pmids), 200):
        chunk = pmids[i:i + 200]
        try:
            resp = requests.get(
                "https://icite.od.nih.gov/api/pubs",
                params={"pmids": ",".join(chunk)},
                timeout=30,
            )
            resp.raise_for_status()
            for d in resp.json().get("data", []):
                out[str(d["pmid"])] = d
        except (requests.RequestException, ValueError) as e:
            print(f"  [WARN] iCite lookup failed for {len(chunk)} PMIDs: {e}")
        time.sleep(0.2)
    return out


def check_has_pmc(pmids: list[str]) -> dict[str, bool]:
    """Use NCBI elink to check which PMIDs have a PMC full-text record. Chunked to avoid 414."""
    if not pmids:
        return {}
    has = {p: False for p in pmids}
    CHUNK = 100
    for i in range(0, len(pmids), CHUNK):
        chunk = pmids[i:i + CHUNK]
        params = {
            **_base_params(),
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": ",".join(chunk),
            "retmode": "json",
        }
        try:
            resp = _request_with_retry(
                f"{config.NCBI_BASE_URL}/elink.fcgi", params=params, timeout=30
            )
            # NCBI sometimes embeds raw control chars in JSON — strip before parsing
            raw = resp.text
            bad = [(idx, hex(ord(c)), repr(c)) for idx, c in enumerate(raw) if ord(c) < 0x20 and c not in "\t\n\r"]
            if bad:
                print(f"  [DEBUG] elink bad chars: {bad[:5]}")
            clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
            data = json.loads(clean)
        except (requests.RequestException, ValueError) as e:
            print(f"  [WARN] PMC elink chunk {i}-{i+len(chunk)} failed: {e}")
            continue
        for ls in data.get("linksets", []):
            ids_in = [str(i_) for i_ in ls.get("ids", [])]
            for db in ls.get("linksetdbs", []):
                if db.get("dbto") == "pmc" and db.get("links"):
                    for pmid in ids_in:
                        has[pmid] = True
        time.sleep(config.PUBMED_DELAY_SEC)
    return has


def _percentile_rank(values: list[float]) -> list[float]:
    """Tie-aware percentile rank in [0, 1]. Higher value -> higher rank."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    out = []
    for v in values:
        less = sum(1 for x in values if x < v)
        equal = sum(1 for x in values if x == v)
        rank = (less + (equal - 1) * 0.5) / (n - 1)
        out.append(rank)
    return out


def _title_keyword_score(title: str, tf_aliases: list[str], target_aliases: list[str],
                         ct_aliases: list[str], assay_terms: list[str]) -> float:
    """
    Categorical fraction of keyword groups present in title + title-pair bonus.

    Categories: TF, target, (optional) cell-type, (optional) assay/regulation.
    Bonus: +TITLE_PAIR_BONUS if BOTH TF and target appear in title (capped at 1.0).
    """
    title_low = title.lower()
    tf_in_title = any(a.lower() in title_low for a in tf_aliases)
    target_in_title = any(a.lower() in title_low for a in target_aliases)

    cats = [tf_in_title, target_in_title]
    if ct_aliases:
        # Word-boundary match for cell-type — short tokens like "ESC", "H1"
        # would otherwise hit "fluorescence", "H1N1", etc.
        ct_pattern = re.compile(
            r"\b(" + "|".join(re.escape(a.lower()) for a in ct_aliases) + r")\b"
        )
        cats.append(bool(ct_pattern.search(title_low)))
    if assay_terms:
        # Substring OK — "regulates" should match "upregulates"
        cats.append(any(t.lower() in title_low for t in assay_terms))
    base = sum(cats) / len(cats) if cats else 0.0

    pair_bonus = config.TITLE_PAIR_BONUS if (tf_in_title and target_in_title) else 0.0
    return min(base + pair_bonus, 1.0)


def _build_pool_metadata(pool_pmids: list[str], pubmed_rank_order: list[str]) -> dict:
    """
    Run iCite + has_pmc + percentile ranks once for the candidate pool.

    Citation impact uses RCR (field-normalized) when ≥25% of pool has it,
    otherwise falls back to raw citation count.
    """
    icite = icite_lookup(pool_pmids)
    has_pmc = check_has_pmc(pool_pmids)

    cites = [icite.get(p, {}).get("citation_count", 0) or 0 for p in pool_pmids]
    rcrs = [icite.get(p, {}).get("relative_citation_ratio") or 0.0 for p in pool_pmids]
    years = [icite.get(p, {}).get("year") or 2000 for p in pool_pmids]
    is_research = [bool(icite.get(p, {}).get("is_research_article", True)) for p in pool_pmids]

    nonzero_rcr = sum(1 for r in rcrs if r > 0)
    if nonzero_rcr >= len(pool_pmids) * 0.25:
        impact_values = [float(r) for r in rcrs]
        impact_metric = "rcr"
    else:
        impact_values = [float(c) for c in cites]
        impact_metric = "raw_citations"
    impact_pct = _percentile_rank(impact_values)

    yr_min, yr_max = min(years), max(years)
    yr_range = max(yr_max - yr_min, 1)
    rec_score = [(y - yr_min) / yr_range for y in years]

    rank_pos = {p: i for i, p in enumerate(pubmed_rank_order)}
    n = len(pubmed_rank_order)
    rel_rank = {p: 1.0 - (rank_pos.get(p, n) / max(n - 1, 1)) for p in pool_pmids}

    meta = {
        p: {
            "citation_percentile": impact_pct[i],
            "recency_score": rec_score[i],
            "relevance_rank_score": rel_rank[p],
            "has_pmc": has_pmc.get(p, False),
            "citation_count": cites[i],
            "rcr": rcrs[i],
            "year": years[i],
            "is_research_article": is_research[i],
        }
        for i, p in enumerate(pool_pmids)
    }
    meta["__impact_metric__"] = impact_metric  # for diagnostics
    return meta


def _composite_score_for(record: dict, meta: dict, weights: dict,
                         tf_aliases: list[str], target_aliases: list[str],
                         ct_aliases: list[str], assay_terms: list[str]) -> dict:
    """Composite [0,1] score + breakdown for one record."""
    pmid = record["pmid"]
    m = meta[pmid]
    cite_s = m["citation_percentile"]
    rel_s = 0.5 * m["relevance_rank_score"] + 0.5 * m["recency_score"]
    ft_s = 1.0 if m.get("has_pmc") else 0.0
    title_s = _title_keyword_score(
        record.get("title", ""), tf_aliases, target_aliases, ct_aliases, assay_terms
    )
    base = (
        weights["citations"] * cite_s
        + weights["relevance_recency"] * rel_s
        + weights["full_text"] * ft_s
        + weights["title_keywords"] * title_s
    )
    research_mult = 1.0 if m.get("is_research_article", True) else config.REVIEW_PENALTY
    return {
        "composite": base * research_mult,
        "research_mult": research_mult,
        "cite_score": cite_s, "rel_rec_score": rel_s,
        "ft_score": ft_s, "title_score": title_s,
        "raw_citations": m["citation_count"], "rcr": m["rcr"],
        "year": m["year"], "is_research_article": m["is_research_article"],
        "has_pmc": m["has_pmc"],
    }


def rank_and_select(pool_pmids: list[str], pubmed_rank_order: list[str],
                    records: list[dict], tf: str, target: str,
                    cell_type: str | None, top_n: int) -> tuple[list[dict], dict]:
    """
    Rank a candidate pool of records by composite score and return the top_n.

    Returns (top_records, ranking_diagnostic) where ranking_diagnostic is a
    pmid -> score-breakdown dict suitable for caching/audit.
    """
    if not records:
        return [], {}
    meta = _build_pool_metadata(pool_pmids, pubmed_rank_order)
    impact_metric = meta.pop("__impact_metric__", "rcr")
    rec_by_pmid = {r["pmid"]: r for r in records}

    tf_aliases = gene_aliases.get_aliases(tf)
    target_aliases = gene_aliases.get_aliases(target)
    ct_aliases = config.CELL_TYPE_ALIASES.get(cell_type or "", [])
    assay_terms = config.TITLE_ASSAY_TERMS

    scored: list[tuple[str, float, dict]] = []
    for pmid, rec in rec_by_pmid.items():
        if pmid not in meta:
            continue
        s = _composite_score_for(rec, meta, config.RANK_WEIGHTS,
                                 tf_aliases, target_aliases, ct_aliases, assay_terms)
        scored.append((pmid, s["composite"], s))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_pmids = [p for p, _, _ in scored[:top_n]]
    top_records = [rec_by_pmid[p] for p in top_pmids if p in rec_by_pmid]
    diagnostic = {
        "impact_metric": impact_metric,
        "weights": config.RANK_WEIGHTS,
        "scores": {p: s for p, _, s in scored},
        "top_order": top_pmids,
    }
    return top_records, diagnostic
# ----------------------------- end composite ranking ----------------------------


def build_queries(tf: str, target: str, cell_type: str | None = None) -> list[str]:
    """
    Generate a ranked list of PubMed queries for a TF-Target pair.

    Queries are ordered most→least specific. The paired tiers (both genes in
    title/abstract) run first. Single-gene tiers run last — they cast a wider
    net so papers where the partner gene only appears in the Introduction or
    Results sections (fetched from PMC) can still be caught by Stage 2's
    full-text co-occurrence filter.

    Cell type is added to single-gene tiers only — those queries are already
    broad, so the cell type narrows them without over-restricting the paired tiers.
    """
    tf_term = _query_term(tf)
    target_term = _query_term(target)
    exclude = 'NOT "network pharmacology" NOT "molecular docking"'

    reg_terms = (
        '"transcriptional regulation" OR "transcriptional target" OR '
        '"directly regulates" OR "promoter activity" OR "transcription factor" OR '
        '"target gene" OR "downstream target"'
    )
    assay_terms = (
        '"ChIP-seq" OR "ChIP-chip" OR "chromatin immunoprecipitation" OR '
        '"luciferase" OR "reporter assay" OR "EMSA" OR "promoter" OR '
        '"knockdown" OR "siRNA" OR "CRISPR"'
    )

    if cell_type and cell_type in config.CELL_TYPE_ALIASES:
        aliases = config.CELL_TYPE_ALIASES[cell_type]
        ct_clause = " AND (" + " OR ".join(f'"{a}"[tiab]' for a in aliases) + ")"
    else:
        ct_clause = ""

    # Organism gate — keeps plant/yeast homologs out of paired tiers when the
    # network's cell type implies a species. MYB/MYC have heavily-cited
    # Arabidopsis homologs that otherwise dominate the pool.
    if cell_type and cell_type in config.CELL_TYPE_ALIASES:
        prefix = cell_type[0].lower()
        if prefix == "m":
            organism_clause = ' AND ("Mus musculus"[MeSH] OR mouse[tiab] OR murine[tiab] OR mice[tiab])'
        elif prefix == "h" or cell_type == "HepG2":
            organism_clause = ' AND ("Homo sapiens"[MeSH] OR human[tiab] OR humans[tiab])'
        else:
            organism_clause = ""
    else:
        organism_clause = ""

    # --- Paired tiers: both genes must appear in title/abstract ---

    # Most specific: both genes + regulatory language
    paired_reg = (
        f"{tf_term} AND {target_term} AND ({reg_terms}){organism_clause} {exclude}"
    )
    # Both genes + experimental assay keywords
    paired_assay = (
        f"{tf_term} AND {target_term} AND ({assay_terms}){organism_clause} {exclude}"
    )
    # Target promoter/transcription mentioned alongside TF
    paired_promoter = (
        f"{tf_term} AND "
        f'("{target}" promoter[tiab] OR "{target}" transcription[tiab])'
        f"{organism_clause} {exclude}"
    )
    # Bare co-mention fallback
    paired_fallback = f"{tf_term} AND {target_term}{organism_clause} {exclude}"

    # --- Single-gene tiers: one gene + regulatory terms (+ optional cell type) ---
    # Stage 2 full-text co-occurrence filter checks that the partner gene appears
    # somewhere in the fetched full text (intro/results), so these are safe to cast wide.

    # TF alone: catches papers where the target is only named in results/intro
    tf_only = (
        f"{tf_term} AND ({reg_terms}){ct_clause} {exclude}"
    )
    # Target alone: catches papers where the TF is only named in results/intro
    target_only = (
        f"{target_term} AND ({reg_terms}){ct_clause} {exclude}"
    )

    return [paired_reg, paired_assay, paired_promoter, paired_fallback,
            tf_only, target_only]


def esearch(query: str, max_results: int = None) -> list[str]:
    """
    Run an ESearch query and return a list of PMIDs (strings).

    Args:
        query:       PubMed query string
        max_results: cap on returned PMIDs; defaults to config.MAX_ABSTRACTS_PER_PAIR
    """
    if max_results is None:
        max_results = config.MAX_ABSTRACTS_PER_PAIR

    params = {
        **_base_params(),
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }

    resp = _request_with_retry(
        f"{config.NCBI_BASE_URL}/esearch.fcgi", params=params, timeout=30
    )
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def efetch(pmids: list[str]) -> list[dict]:
    """
    Fetch title + abstract for a list of PMIDs.

    Returns a list of dicts:
        {"pmid": str, "title": str, "abstract": str, "pmc_id": str | None}

    PMIDs with no abstract text are silently omitted.
    pmc_id is populated when the article has an open-access PMC record, enabling
    a follow-up call to efetch_pmc_sections() for intro/results text.
    """
    if not pmids:
        return []

    # Chunk to avoid 414 URI Too Long. NCBI accepts up to ~200 PMIDs per GET
    # comfortably; 100 is conservative and well under URI limits.
    CHUNK = 100
    records = []
    for i in range(0, len(pmids), CHUNK):
        chunk = pmids[i:i + CHUNK]
        params = {
            **_base_params(),
            "db": "pubmed",
            "id": ",".join(chunk),
            "rettype": "abstract",
            "retmode": "xml",
        }
        try:
            resp = _request_with_retry(
                f"{config.NCBI_BASE_URL}/efetch.fcgi", params=params, timeout=60
            )
        except Exception as e:
            print(f"  [WARN] efetch chunk {i}-{i+len(chunk)} failed: {e}")
            continue
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            print(f"  [WARN] efetch XML parse failed for chunk {i}: {e}")
            continue
        records.extend(_parse_efetch_xml(root))
        time.sleep(config.PUBMED_DELAY_SEC)
    return records


def _parse_efetch_xml(root) -> list[dict]:
    """Extract {pmid, title, abstract, pmc_id} records from an efetch XML root."""
    records = []
    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")

        pmid = pmid_el.text if pmid_el is not None else "unknown"
        title = title_el.text or "" if title_el is not None else ""

        # Structured abstracts have multiple <AbstractText> sections
        # (Background, Methods, Results, Conclusions) — join them all.
        abstract_parts = [
            (el.text or "").strip()
            for el in article.findall(".//AbstractText")
        ]
        abstract = " ".join(p for p in abstract_parts if p)

        # Extract PMC ID when present — used to fetch full text later.
        pmc_id = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "pmc":
                raw = (article_id.text or "").strip()
                # Strip "PMC" prefix; efetch?db=pmc wants bare integers.
                pmc_id = raw.removeprefix("PMC") if raw else None
                break

        if abstract.strip():
            records.append({"pmid": pmid, "title": title, "abstract": abstract, "pmc_id": pmc_id})

    return records


def _pmc_fetch_xml(pmc_ids: list[str]) -> "ET.Element | None":
    """
    Try to fetch PMC full-text XML for a list of IDs.
    Returns the parsed root element, or None on 400/failure.
    """
    params = {
        **_base_params(),
        "db": "pmc",
        "id": ",".join(pmc_ids),
        "retmode": "xml",
    }
    try:
        resp = requests.get(
            f"{config.NCBI_BASE_URL}/efetch.fcgi", params=params, timeout=120
        )
        if resp.status_code == 400:
            return None
        resp.raise_for_status()
        return ET.fromstring(resp.text)
    except Exception:
        return None


def efetch_pmc_sections(pmc_ids: list[str]) -> dict[str, dict]:
    """
    Fetch Introduction and Results section text from PubMed Central.

    Returns {pmc_id: {"introduction": str, "results": str}}.
    Papers not in PMC or with parsing errors are silently omitted.

    Strategy: try all IDs in one batch first. If the batch returns 400
    (one bad ID poisons the whole request), fall back to individual fetches
    so valid IDs still get their full text.
    """
    if not pmc_ids:
        return {}

    root = _pmc_fetch_xml(pmc_ids)
    if root is None:
        # Batch failed — retry one at a time, skip bad IDs silently.
        roots = []
        for pid in pmc_ids:
            r = _pmc_fetch_xml([pid])
            if r is not None:
                roots.append(r)
            time.sleep(config.PUBMED_DELAY_SEC)
        if not roots:
            return {}
        # Merge all individual results under a synthetic root for unified parsing.
        root = ET.Element("root")
        for r in roots:
            for child in r:
                root.append(child)

    _TARGET_SECTIONS = {
        "introduction": {"introduction", "background", "background and introduction",
                         "background/introduction"},
        "results": {"results", "results and discussion"},
    }

    def _section_text(sec_el) -> str:
        return " ".join(s.strip() for s in sec_el.itertext() if s.strip())

    sections: dict[str, dict] = {}
    for article in root.iter("article"):
        pmc_id = None
        for aid in article.findall(".//article-id"):
            if aid.get("pub-id-type") in ("pmc", "pmcid"):
                pmc_id = (aid.text or "").strip().removeprefix("PMC")
                break
        if not pmc_id:
            continue

        intro_parts: list[str] = []
        results_parts: list[str] = []

        for sec in article.findall(".//sec"):
            title_el = sec.find("title")
            if title_el is None:
                continue
            sec_title = (title_el.text or "").strip().lower()
            sec_type = (sec.get("sec-type") or "").lower()

            if sec_title in _TARGET_SECTIONS["introduction"] or sec_type in _TARGET_SECTIONS["introduction"]:
                intro_parts.append(_section_text(sec))
            elif sec_title in _TARGET_SECTIONS["results"] or sec_type in _TARGET_SECTIONS["results"]:
                results_parts.append(_section_text(sec))

        if intro_parts or results_parts:
            sections[pmc_id] = {
                "introduction": " ".join(intro_parts),
                "results": " ".join(results_parts),
            }

    return sections


def _cache_path(tf: str, target: str, cell_type: str | None = None) -> str:
    suffix = f"_{cell_type}" if cell_type else ""
    fname = f"{tf.upper()}_{target.upper()}{suffix}.json"
    return os.path.join(config.ABSTRACTS_CACHE_DIR, fname)


def fetch_abstracts_for_pair(tf: str, target: str,
                             cell_type: str | None = None) -> list[dict]:
    """
    Orchestrates ESearch + EFetch (+ optional PMC full-text) for a TF-Target pair.

    Strategy:
      - Try each query in build_queries() in order (cell_type enriches queries).
      - Stop once MAX_ABSTRACTS_PER_PAIR unique PMIDs are collected.
      - Enrich records that have PMC IDs with Introduction/Results section text.
      - Cache results to disk; return cached results on subsequent calls.

    Returns:
        List of records: {pmid, title, abstract, pmc_id,
                          introduction?, results?}
    """
    cache = _cache_path(tf, target, cell_type)
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    if config.USE_COMPOSITE_RANKING:
        all_records, ranking_diag = _fetch_with_composite_ranking(tf, target, cell_type)
    else:
        all_records = _fetch_with_relevance_only(tf, target, cell_type)
        ranking_diag = None

    # Enrich with PMC full text (Introduction + Results) where available.
    pmc_ids = [r["pmc_id"] for r in all_records if r.get("pmc_id")]
    if pmc_ids:
        print(f"         Fetching PMC full text for {len(pmc_ids)} papers...")
        pmc_sections = efetch_pmc_sections(pmc_ids)
        for rec in all_records:
            if rec.get("pmc_id") and rec["pmc_id"] in pmc_sections:
                rec.update(pmc_sections[rec["pmc_id"]])
        time.sleep(config.PUBMED_DELAY_SEC)

    os.makedirs(config.ABSTRACTS_CACHE_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(all_records, f, indent=2)

    # Audit trail — what got picked and why
    if ranking_diag is not None:
        diag_path = os.path.join(
            config.ABSTRACTS_CACHE_DIR,
            f"{tf.upper()}_{target.upper()}{('_' + cell_type) if cell_type else ''}_ranking.json",
        )
        with open(diag_path, "w") as f:
            json.dump(ranking_diag, f, indent=2)

    return all_records


def _fetch_with_relevance_only(tf: str, target: str, cell_type: str | None) -> list[dict]:
    """Original behaviour: walk queries in order, take first MAX_ABSTRACTS_PER_PAIR."""
    seen_pmids: set[str] = set()
    all_records: list[dict] = []

    for query in build_queries(tf, target, cell_type):
        if len(seen_pmids) >= config.MAX_ABSTRACTS_PER_PAIR:
            break
        pmids = esearch(query, max_results=config.MAX_ABSTRACTS_PER_PAIR - len(seen_pmids))
        new_pmids = [p for p in pmids if p not in seen_pmids]
        if not new_pmids:
            time.sleep(config.PUBMED_DELAY_SEC)
            continue
        records = efetch(new_pmids)
        all_records.extend(records)
        seen_pmids.update(new_pmids)
        time.sleep(config.PUBMED_DELAY_SEC)

    return all_records


def _fetch_with_composite_ranking(tf: str, target: str,
                                  cell_type: str | None) -> tuple[list[dict], dict]:
    """
    Wide ESearch (POOL_SIZE candidates) -> rank by composite -> EFetch top-N.

    Two query rounds: cell-type-enriched + plain (Tweak 3). The plain round
    catches foundational papers using older cell-type vocabulary that the
    enriched queries filter out (e.g. 1995 "ES cells" papers missed by hESC tags).
    """
    pool_pmids: list[str] = []
    seen = set()
    rounds = [("with_ct", build_queries(tf, target, cell_type))]
    if cell_type:
        rounds.append(("no_ct", build_queries(tf, target, cell_type=None)))

    for round_label, queries in rounds:
        for q in queries:
            if len(pool_pmids) >= config.POOL_SIZE:
                break
            try:
                pmids = esearch(q, max_results=config.POOL_SIZE)
            except Exception as e:
                print(f"         ESearch failed ({round_label}): {e}")
                continue
            for p in pmids:
                if p not in seen:
                    seen.add(p)
                    pool_pmids.append(p)
            time.sleep(config.PUBMED_DELAY_SEC)
        if len(pool_pmids) >= config.POOL_SIZE:
            break

    pool_pmids = pool_pmids[:config.POOL_SIZE]
    if not pool_pmids:
        return [], {"pool_size": 0, "top_order": []}

    # Fetch full records for the whole pool — we need titles for keyword scoring,
    # and we'll need abstracts for the top-N anyway. Single batched EFetch.
    print(f"         Pool: {len(pool_pmids)} candidates -> ranking...")
    records = efetch(pool_pmids)
    if not records:
        return [], {"pool_size": len(pool_pmids), "top_order": []}

    top_records, diag = rank_and_select(
        pool_pmids=pool_pmids,
        pubmed_rank_order=pool_pmids,
        records=records,
        tf=tf, target=target, cell_type=cell_type,
        top_n=config.MAX_ABSTRACTS_PER_PAIR,
    )
    diag["pool_size"] = len(pool_pmids)
    diag["selected"] = len(top_records)
    print(f"         Selected top {len(top_records)} by composite "
          f"(impact metric: {diag.get('impact_metric')})")
    return top_records, diag
