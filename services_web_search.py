# search_service.py
# Real-time web search for your assistant.
# Defaults to DuckDuckGo (free, no API key). Optional Brave/Bing support if keys are set.
from __future__ import annotations

import os
import json
import time
import sqlite3
import threading
from typing import List, Dict, Any, Optional

import requests

# --- Optional: full-text DuckDuckGo search (keyless) ---
try:
    from ddgs import DDGS  # pip install ddgs
except Exception:
    DDGS = None  # We'll raise a helpful error if someone selects this provider without the package.

# --- Optional: HTML cleaning for fetched pages ---
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None

# --- Optional: your existing LLM helper (Ollama/ChatOllama in this repo) ---
try:
    from openai_api import answer_general_query  # provides: answer_general_query(user_input, history=None, system_prompt=None, max_tokens=900)
except Exception:
    answer_general_query = None  # graceful fallback will be used if this isn't available


# =============================================================================
# Configuration / Constants
# =============================================================================
_DB_PATH = os.getenv("WEB_CACHE_DB", "web_cache.sqlite3")
_LOCK = threading.Lock()


# =============================================================================
# Exceptions
# =============================================================================
class WebSearchError(Exception):
    pass


class WebFetchError(Exception):
    pass


# =============================================================================
# Tiny SQLite cache for SERPs and fetched pages
# =============================================================================
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS web_cache(
            k  TEXT PRIMARY KEY,
            v  TEXT NOT NULL,
            ts INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS web_pages(
            url TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            ts  INTEGER NOT NULL
        )
        """
    )
    return conn


def _cache_get(key: str, max_age_sec: int = 12 * 3600) -> Optional[Any]:
    try:
        with _LOCK:
            c = _db().cursor()
            c.execute("SELECT v, ts FROM web_cache WHERE k=?", (key,))
            row = c.fetchone()
        if not row:
            return None
        v, ts = row
        if int(time.time()) - int(ts) > max_age_sec:
            return None
        return json.loads(v)
    except Exception:
        return None


def _cache_set(key: str, value: Any) -> None:
    try:
        with _LOCK:
            _db().execute(
                "INSERT OR REPLACE INTO web_cache(k,v,ts) VALUES (?,?,?)",
                (key, json.dumps(value)[:2_000_000], int(time.time())),
            )
            _db().commit()
    except Exception:
        pass


def _page_get(url: str, max_age_sec: int = 24 * 3600) -> Optional[str]:
    try:
        with _LOCK:
            c = _db().cursor()
            c.execute("SELECT text, ts FROM web_pages WHERE url=?", (url,))
            row = c.fetchone()
        if not row:
            return None
        text, ts = row
        if int(time.time()) - int(ts) > max_age_sec:
            return None
        return text
    except Exception:
        return None


def _page_set(url: str, text: str) -> None:
    try:
        with _LOCK:
            _db().execute(
                "INSERT OR REPLACE INTO web_pages(url,text,ts) VALUES (?,?,?)",
                (url, text[:2_000_000], int(time.time())),
            )
            _db().commit()
    except Exception:
        pass


# =============================================================================
# Providers
#   - duckduckgo (default, keyless) via duckduckgo-search
#   - brave (keyed)
#   - bing  (keyed; legacy public API retired, kept for compatibility if you still have it)
# =============================================================================
def _ddgs_text(
    q: str,
    *,
    count: int = 6,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    DuckDuckGo full-text search via duckduckgo-search (no API key).
    Returns a normalized list of {'title','url','snippet','source'} dicts.
    """
    if DDGS is None:
        raise WebSearchError(
            "duckduckgo-search package is not installed. Run: pip install duckduckgo-search"
        )

    out: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(
            q,
            max_results=count,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
        ):
            if not isinstance(r, dict):
                continue
            title = r.get("title") or ""
            url = r.get("href") or r.get("url") or ""
            body = r.get("body") or r.get("snippet") or ""
            if url:
                out.append(
                    {"title": title, "url": url, "snippet": body, "source": "duckduckgo"}
                )
            if len(out) >= count:
                break
    return out


def _brave_search(q: str, *, count: int = 6) -> List[Dict[str, Any]]:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise WebSearchError("Missing BRAVE_API_KEY")
    url = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": q, "count": count, "freshness": "month"}
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        raise WebSearchError(f"Brave error {r.status_code}: {r.text[:200]}")
    data = r.json()
    items = (data.get("web", {}) or {}).get("results", []) or []
    out = []
    for it in items:
        out.append(
            {
                "title": it.get("title"),
                "url": it.get("url"),
                "snippet": it.get("description"),
                "source": "brave",
            }
        )
    return out


def _bing_search(q: str, *, count: int = 6) -> List[Dict[str, Any]]:
    api_key = os.getenv("BING_API_KEY")
    if not api_key:
        raise WebSearchError("Missing BING_API_KEY")
    url = "https://api.bing.microsoft.com/v7.0/search"
    params = {"q": q, "count": count, "responseFilter": "Webpages", "freshness": "Month"}
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        raise WebSearchError(f"Bing error {r.status_code}: {r.text[:200]}")
    data = r.json()
    items = (data.get("webPages", {}) or {}).get("value", []) or []
    out = []
    for it in items:
        out.append(
            {
                "title": it.get("name"),
                "url": it.get("url"),
                "snippet": it.get("snippet"),
                "source": "bing",
            }
        )
    return out


def web_search(
    q: str,
    *,
    provider: Optional[str] = None,
    count: int = 6,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Unified search facade.
    - Defaults to DuckDuckGo (free, no key) unless SEARCH_PROVIDER env is set.
    - If a keyed provider is selected but keys are missing, it falls back to DuckDuckGo.
    """
    if not q or not q.strip():
        return []

    prov = (provider or os.getenv("SEARCH_PROVIDER") or "duckduckgo").lower()
    providers_order = [prov] + [p for p in ("duckduckgo", "brave", "bing") if p != prov]

    cache_key = f"serp::{prov}::{q.strip().lower()}::{count}"
    cached = _cache_get(cache_key, max_age_sec=12 * 3600)
    if cached is not None:
        return cached

    last_err: Optional[Exception] = None
    for choice in providers_order:
        try:
            if choice in ("duckduckgo", "ddgs"):
                out = _ddgs_text(
                    q, count=count, region=region, safesearch=safesearch, timelimit=timelimit
                )
            elif choice == "brave":
                out = _brave_search(q, count=count)
            elif choice == "bing":
                out = _bing_search(q, count=count)
            else:
                continue
            _cache_set(cache_key, out)
            return out
        except Exception as e:
            last_err = e
            continue

    raise WebSearchError(str(last_err) if last_err else "No provider available")


# =============================================================================
# Fetch & clean pages (for better summaries and citations)
# =============================================================================
def _require_bs4():
    if BeautifulSoup is None:
        raise WebFetchError("beautifulsoup4 is required. Run: pip install beautifulsoup4")


def _clean_html(html: str, max_chars: int = 8000) -> str:
    _require_bs4()
    try:
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.extract()
        text = " ".join((soup.get_text(" ") or "").split())
        return text[:max_chars]
    except Exception:
        return ""


def _fetch_page(url: str, *, max_kb: int = 256, use_cache: bool = True) -> str:
    if use_cache:
        cached = _page_get(url, max_age_sec=24 * 3600)
        if cached is not None:
            return cached
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (ECHO Assistant)"})
        r.raise_for_status()
        text = r.text
        if len(text) > max_kb * 1024:
            text = text[: max_kb * 1024]
        cleaned = _clean_html(text, max_chars=8000)
        if use_cache and cleaned:
            _page_set(url, cleaned)
        return cleaned
    except Exception:
        return ""


# =============================================================================
# High-level helpers
# =============================================================================
def browse(
    q: str,
    *,
    max_results: int = 5,
    include_pages: bool = True,
    provider: Optional[str] = None,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run search and (optionally) fetch + clean page content.
    Returns: {"query": str, "results": [{"index","title","url","snippet","body","source"}...]}
    """
    serp = web_search(
        q,
        provider=provider,
        count=max_results,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
    )
    results = []
    for i, item in enumerate(serp[:max_results], 1):
        body = _fetch_page(item["url"]) if include_pages else ""
        results.append(
            {
                "index": i,
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("snippet"),
                "body": body,
                "source": item.get("source"),
            }
        )
    return {"query": q, "results": results}


def build_cited_system_prompt(web: Dict[str, Any]) -> str:
    """
    Build a system prompt that inlines WEB CONTEXT and enforces bracketed [n] citations.
    """
    items = (web or {}).get("results", []) or []
    lines = []
    lines.append("You are ECHO, a precise assistant. You MUST:")
    lines.append("- Use WEB CONTEXT for time-sensitive facts.")
    lines.append("- Add bracketed citations like [1], [2] next to claims from the web.")
    lines.append("- Do not quote more than 25 words from any single source.")
    lines.append("- If info is uncertain or conflicting, note it briefly and cite sources.")
    lines.append("- If WEB CONTEXT is empty, say you could not find enough info.")
    lines.append("")
    lines.append("WEB CONTEXT:")
    for it in items[:5]:
        t = (it.get("title") or "").strip()
        u = (it.get("url") or "").strip()
        b = (it.get("body") or "")[:1000].strip()
        idx = it.get("index", 0)
        lines.append(f"[{idx}] {t} — {u}")
        if b:
            lines.append(b)
        lines.append("")
    return "\n".join(lines)


def web_answer(
    user_input: str,
    *,
    provider: Optional[str] = None,
    count: int = 5,
    include_pages: bool = True,
    max_tokens: int = 900,
    region: str = "us-en",
    safesafety: str = "moderate",  # kept for backward-compat; not used directly
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One call: search -> fetch -> build prompt -> call LLM -> return text + sources.
    Returns: {"text": str, "sources": list, "web": dict}
    """
    web = browse(
        user_input,
        max_results=count,
        include_pages=include_pages,
        provider=provider,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
    )
    sys_prompt = build_cited_system_prompt(web)

    if answer_general_query is None:
        # Minimal fallback if the LLM helper isn't available.
        lines = ["Here are some sources I found:\n"]
        for r in web.get("results", [])[:5]:
            lines.append(f"- {r.get('title') or r.get('url')} [{r.get('url')}]")
        text = "\n".join(lines)
    else:
        text = answer_general_query(
            user_input=user_input,
            history=None,
            system_prompt=sys_prompt,
            max_tokens=max_tokens,
        ) or ""

    return {
        "text": (text or "").strip(),
        "sources": web.get("results", [])[:5],
        "web": web,
    }
