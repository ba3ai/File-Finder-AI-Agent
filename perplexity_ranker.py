# perplexity_ranker.py
#
# Ranking rules (as requested):
# 1) Keyword matching:
#    - Prefer ONLY files that contain the exact keyword phrase (word-boundary aware) in
#      the file name or content. If any such files exist, return ONLY those.
#    - If there are NO exact matches, return files that contain ALL individual words
#      (in any order/position) in the file name or content.
#
# 2) Strict year filtering:
#    - If the query includes any year(s) (e.g., 2025 or 25-09-30 etc.), keep ONLY files
#      whose metadata or text contain any of those years.
#
# 3) Custom date sort key (for final ordering):
#    - If Modified < Created  -> SortKey = Modified
#    - Else (Created <= Modified) -> SortKey = Created
#    - Fallbacks:
#        * If only one exists, use it.
#        * Else use the newest date parsed from name/path/url/extracted_text.
#        * Else use datetime.min (so it sinks).
#    - Sort DESC (newest first). Ties break by score (desc) then filename A→Z.
#
# Notes:
# - Word boundary checks avoid matching 'pike' inside 'spike'.
# - Dates are detected in filename/path/url/extracted_text (if provided via OCR).
# - DEBUG flag prints helpful traces without changing behavior.

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# ========== configuration ==========
DEBUG = False  # set to True to see debug prints


def _dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# ========== helpers: text + dates ==========

# Word-boundary wrappers (avoid matching "pike" within "spike")
_WORD_BOUNDARY   = r"(?:^|[^0-9A-Za-z_])"
_WORD_BOUNDARY_R = r"(?:$|[^0-9A-Za-z_])"


def _wb_contains(term: str, text: str) -> bool:
    """Word-boundary 'contains' to avoid matching 'spike' for 'pike'."""
    if not term:
        return False
    pattern = re.compile(_WORD_BOUNDARY + re.escape(term) + _WORD_BOUNDARY_R, flags=re.IGNORECASE)
    return bool(pattern.search(text))


def _extract_keywords_and_dates(query: str) -> Tuple[List[str], List[str], str]:
    """
    Returns (keywords, date_terms, exact_phrase).

    Date terms capture:
      - YYYY and YYYY.MM(.DD)? / YYYY-MM(-DD)? / YYYY/MM(/DD)?
      - YY.MM.DD / YY-MM-DD / YY/MM/DD
      - bare YY (e.g., '25') if not already part of a YY.MM.DD pattern
    """
    q = (query or "").strip().lower()

    # YYYY or YYYY.XX(.XX)? / YYYY-XX(-XX)? / YYYY/XX(/XX)?
    # or YY.MM.DD / YY-MM-DD / YY/MM/DD
    date_terms = re.findall(
        r"\b(?:20\d{2}(?:[./-]\d{2}(?:[./-]\d{2})?)?|\d{2}[./-]\d{2}[./-]\d{2})\b",
        q,
    )

    # consider bare two-digit year like '25' (avoid double-adding when it's already in YY.MM.DD)
    bare_yy = re.findall(r"\b(\d{2})\b", q)
    for yy in bare_yy:
        if not any(m.startswith(yy) and any(sep in m for sep in "./-") for m in date_terms):
            date_terms.append(yy)

    words = re.findall(r"\b\w+\b", q)
    keywords = [w for w in words if w not in date_terms]
    exact_phrase = " ".join(keywords)  # used for the "exact-phrase" bucket
    return keywords, date_terms, exact_phrase


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        # Keep naive datetimes for consistent comparisons with datetime.min
        return datetime.fromisoformat(dt_str.replace("Z", ""))
    except Exception:
        return None


def _extract_dates_from_string(s: str) -> List[datetime]:
    """
    Find dates like 2025, 2025.05.31, 2025-05-31, 25.05.31, 25-05-31, 25/05/31 in a string
    and convert to datetime (assuming 20YY for two-digit years).
    """
    s = s or ""
    out: List[datetime] = []

    # YYYY or YYYY.MM.DD / YYYY-MM-DD / YYYY/MM/DD
    for m in re.finditer(r"\b(20\d{2})(?:[./-](\d{2})(?:[./-](\d{2}))?)?\b", s):
        year = int(m.group(1))
        if m.group(2) and m.group(3):
            month, day = int(m.group(2)), int(m.group(3))
            try:
                out.append(datetime(year, month, day))
            except Exception:
                pass
        else:
            try:
                out.append(datetime(year, 1, 1))
            except Exception:
                pass

    # YY.MM.DD / YY-MM-DD / YY/MM/DD  (assume 20YY)
    for m in re.finditer(r"\b(\d{2})[./-](\d{2})[./-](\d{2})\b", s):
        yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 2000 + yy
        try:
            out.append(datetime(year, mm, dd))
        except Exception:
            pass

    return out


def _file_created_date(file: Dict[str, Any]) -> Optional[datetime]:
    """Prefer createdDateTime (or fileSystemInfo.createdDateTime)."""
    return (
        _parse_iso(file.get("createdDateTime"))
        or _parse_iso((file.get("fileSystemInfo") or {}).get("createdDateTime"))
    )


def _file_modified_date(file: Dict[str, Any]) -> Optional[datetime]:
    """Fallback: lastModifiedDateTime (or fileSystemInfo.lastModifiedDateTime)."""
    return (
        _parse_iso(file.get("lastModifiedDateTime"))
        or _parse_iso((file.get("fileSystemInfo") or {}).get("lastModifiedDateTime"))
    )


def _all_text_dates(file: Dict[str, Any]) -> List[datetime]:
    """Parse ALL dates from name/path/url/extracted_text (not just the latest one)."""
    text_blobs = [
        file.get("name", ""),
        (file.get("parentReference") or {}).get("path", ""),
        file.get("webUrl", ""),
        file.get("extracted_text") or "",
    ]
    out: List[datetime] = []
    for blob in text_blobs:
        out.extend(_extract_dates_from_string(blob))
    return out


def _best_any_date_from_texts(file: Dict[str, Any]) -> Optional[datetime]:
    """Convenience: the most recent text-derived date."""
    dates = _all_text_dates(file)
    return max(dates) if dates else None


# ---------- NEW: custom Sort Key per your rule ----------
def _custom_sort_key(file: Dict[str, Any]) -> datetime:
    """
    Your rule:
      If Modified < Created  -> use Modified
      Else (Created <= Modified) -> use Created
      Fallbacks:
        - If only one exists, use the one that exists.
        - If neither exists, use the newest date parsed from name/path/url/extracted_text.
        - If still nothing, return datetime.min (so it sinks).
    """
    created = _file_created_date(file)
    modified = _file_modified_date(file)

    if created and modified:
        return modified if modified < created else created
    if created:
        return created
    if modified:
        return modified

    any_text_dt = _best_any_date_from_texts(file)
    return any_text_dt or datetime.min


def _years_from_date_terms(date_terms: List[str]) -> List[int]:
    """
    Extract years from date terms:
      - 2025 -> 2025
      - 2025-05[-31] -> 2025
      - 25.05.31 -> 2025
      - bare '25' -> 2025
    """
    years: List[int] = []
    for dt in date_terms:
        if not dt:
            continue
        m4 = re.fullmatch(r"20(\d{2})", dt)
        if m4:
            years.append(int("20" + m4.group(1)))
            continue
        m4b = re.match(r"(20\d{2})[./-]\d{2}(?:[./-]\d{2})?$", dt)
        if m4b:
            years.append(int(m4b.group(1)))
            continue
        m2full = re.fullmatch(r"(\d{2})[./-]\d{2}[./-]\d{2}$", dt)
        if m2full:
            years.append(2000 + int(m2full.group(1)))
            continue
        m2bare = re.fullmatch(r"\d{2}$", dt)
        if m2bare:
            years.append(2000 + int(dt))
    # keep order but de-dup
    seen = set()
    uniq: List[int] = []
    for y in years:
        if y not in seen:
            uniq.append(y)
            seen.add(y)
    return uniq


def _file_matches_years(file: Dict[str, Any], years: List[int]) -> bool:
    """
    STRICT year match: the file is kept only if ANY of these contain one of the query years:
      - createdDateTime / fileSystemInfo.createdDateTime
      - lastModifiedDateTime / fileSystemInfo.lastModifiedDateTime
      - ANY date found in filename, path, URL, or extracted_text
    """
    # metadata dates
    for d in (_file_created_date(file), _file_modified_date(file)):
        if d and d.year in years:
            return True

    # any date seen in the text blobs (consider ALL, not just the latest)
    for d in _all_text_dates(file):
        if d.year in years:
            return True

    return False


# ========== matching & scoring ==========

def _build_search_text(file: Dict[str, Any]) -> str:
    """Combine name + key metadata + extracted text into one lowercase blob."""
    fields = [
        file.get("name", ""),
        file.get("createdDateTime", ""),
        file.get("lastModifiedDateTime", ""),
        file.get("webUrl", ""),
        (file.get("file") or {}).get("mimeType", ""),
        (file.get("fileSystemInfo") or {}).get("createdDateTime", ""),
        (file.get("fileSystemInfo") or {}).get("lastModifiedDateTime", ""),
        (file.get("parentReference") or {}).get("path", ""),
        file.get("extracted_text") or "",
    ]
    print(f"🔍 Building search text for file {file.get('id', 'unknown')}: {fields}")
    _dprint(f"🔍 Search text for file {file.get('id', 'unknown')}: {fields}")
    return " ".join(fields).lower()


# ========== public API ==========

def rank_files_rule_based(query: str, files: List[Dict[str, Any]], original_query: Optional[str] = None):
    """
    STRICT YEAR FILTERING:
      - If the query mentions any year(s), ONLY files whose metadata/text contain those year(s) are kept.

    Keyword logic:
      - Exact phrase match first (name or content). If any exact matches, return ONLY them.
      - Else, return files where ALL individual words are present somewhere (name or content).

    Sorting (applied to the chosen candidate set):
      - Primary: custom Sort Key (DESC) where SortKey = Modified if Modified < Created, else Created.
      - Secondary: score (desc; small boost if sort-key year matches query-year).
      - Tertiary: filename A→Z.
    """
    if not query or not files:
        return files

    keywords, date_terms, exact_phrase = _extract_keywords_and_dates(query)
    query_years = _years_from_date_terms(date_terms)

    _dprint(
        f"Ranking files for query={query!r}\n"
        f"  keywords={keywords}, date_terms={date_terms}, exact_phrase={exact_phrase!r}, years={query_years}"
    )

    # 🔒 STRICT FILTER: if any query year is present, keep ONLY files that match those years
    working_files = files
    if query_years:
        working_files = [f for f in files if _file_matches_years(f, query_years)]
        if not working_files:
            _dprint("No files remain after strict year filtering.")
            return []

    # Gather candidates by keyword policy
    exact_matches: List[Dict[str, Any]] = []
    all_word_matches: List[Dict[str, Any]] = []

    for f in working_files:
        text = _build_search_text(f)

        # Exact phrase (word-boundary aware)
        is_exact = False
        if exact_phrase:
            pattern = re.compile(_WORD_BOUNDARY + re.escape(exact_phrase) + _WORD_BOUNDARY_R, flags=re.IGNORECASE)
            is_exact = bool(pattern.search(text))

        if is_exact:
            exact_matches.append(f)
            continue

        # Else, require ALL words to be present (also word-boundary aware)
        if keywords and all(_wb_contains(kw, text) for kw in keywords):
            all_word_matches.append(f)

    # If any exact matches found → only return those, else all-word matches
    candidate_files = exact_matches if exact_matches else all_word_matches

    if not candidate_files:
        _dprint("No candidates found after keyword filtering.")
        return []

    # Build rows: (sort_key_dt, score, name, file)
    rows: List[Tuple[datetime, int, str, Dict[str, Any]]] = []
    for f in candidate_files:
        sort_key_dt = _custom_sort_key(f)
        name = (f.get("name") or "").lower()
        score = 0
        # Optional: extra weight if Sort Key year matches a year in the query
        if query_years and sort_key_dt and sort_key_dt != datetime.min and sort_key_dt.year in query_years:
            score += 25
        rows.append((sort_key_dt, score, name, f))

    # Sorting without relying on timestamp() (avoid platform issues with datetime.min)
    # Stable sort: first by name A→Z, then by (sort_dt desc, score desc)
    rows.sort(key=lambda r: r[2])  # name asc
    rows.sort(key=lambda r: (r[0] or datetime.min, r[1]), reverse=True)  # sort_dt desc, score desc

    return [row[3] for row in rows]
