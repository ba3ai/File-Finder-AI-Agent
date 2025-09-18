# --- semantic_search.py ---
# Prompt-only global ranking (LLaMA 3 via Ollama)
# 1) Ask the model for a FULL permutation of all candidates (indices only).
# 2) Retry with an ultra-light catalog if the first pass omits indices.
# 3) If still incomplete, fall back to a QUERY-AWARE newest→oldest order:
#    - files matching the query tokens first (by recency), then the rest (by recency).

from typing import List, Dict, Any, Optional
import os
import re
import json
from datetime import datetime
from langchain_ollama import ChatOllama

# ---------------- LLM client ----------------
_PROMPT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # make sure: ollama pull llama3
_PROMPT_CTX   = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
_prompt_llm   = ChatOllama(
    model=_PROMPT_MODEL,
    temperature=0.0,
    num_ctx=_PROMPT_CTX,
    num_predict=1024,
)

# ---------------- Catalog builders ----------------
def _build_catalog(
    files: List[Dict[str, Any]],
    max_chars: int = 240,
    ultra_light: bool = False
) -> str:
    """
    Numbered list for the LLM.
      ultra_light=False -> include CONTENT_SNIPPET (short)
      ultra_light=True  -> metadata only (for retries / large sets)
    """
    rows = []
    for i, f in enumerate(files, start=1):
        name = f.get("name", f"file_{i}")
        meta = {
            "name": name,
            "extension": (name.split(".")[-1].lower() if "." in name else ""),
            "size": f.get("size"),
            "created": f.get("createdDateTime") or (f.get("fileSystemInfo") or {}).get("createdDateTime"),
            "modified": f.get("lastModifiedDateTime") or (f.get("fileSystemInfo") or {}).get("lastModifiedDateTime"),
            "driveId": (f.get("parentReference") or {}).get("driveId"),
            "siteId": (f.get("parentReference") or {}).get("siteId"),
            "webUrl": f.get("webUrl"),
        }
        if ultra_light:
            rows.append(f"[{i}] METADATA: {json.dumps(meta, ensure_ascii=False)}")
        else:
            text = (f.get("extracted_text") or name)[:max_chars].replace("\n", " ")
            rows.append(
                f"[{i}] METADATA: {json.dumps(meta, ensure_ascii=False)}\n"
                f"CONTENT_SNIPPET: {text}"
            )
    return "\n\n".join(rows)

# ---------------- Output parsing ----------------
_BRACKET_LINE = re.compile(r"^\[(\d+)\]\s*$")

def _parse_indices(raw: str, n: int) -> List[int]:
    """Parse bracketed indices into a 0-based, de-duplicated list within range."""
    seen = set()
    out: List[int] = []
    for line in raw.splitlines():
        m = _BRACKET_LINE.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out

# ---------------- Prompts ----------------
_BASE_SYSTEM = (
    "You are an expert file search and ranking assistant.\n"
    "Decide which files best satisfy the user's request using ONLY the candidates provided.\n\n"
    "IMPORTANT INPUTS:\n"
    "- Candidates are numbered like: [k] METADATA: {...} then optional CONTENT_SNIPPET.\n"
    "- METADATA includes: name, extension, created, modified, webUrl, etc. Use BOTH the User Query and Original Query.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "- Return ONLY a list of bracketed indices, ONE PER LINE, in final ranked order. Example:\n"
    "[7]\n[2]\n[15]\n"
    "- Do NOT add explanations, JSON, bullets, or any extra text. Do NOT invent indices.\n\n"
    "DECISION & RANKING POLICY (apply in this order):\n"
    "A) FILE-TYPE FILTER (BEFORE ranking): if the user asks for a specific type/extension, ONLY consider files whose "
    "METADATA.extension matches. Mappings: (xlsx|xls) for excel/spreadsheet; pdf; (doc|docx) for word; (ppt|pptx) for "
    "powerpoint/slides; csv; (jpg|jpeg|png) for image/picture. If none match, return nothing.\n"
    "B) EXACT/NEAR-EXACT NAME PRIORITY: if the query names a specific file, put exact/near-exact name matches first "
    "(ignore case/punctuation/spaces/extension; prefer FINAL over DRAFT unless draft requested). When the request includes "
    "dates like '25.06.30', treat 'YY.MM.DD' as '20YY-MM-DD' and match/favor accordingly.\n"
    "C) RELEVANCE THEN RECENCY: rank by textual relevance (name + snippet). Use recency as a tie-breaker (prefer more "
    "recent METADATA.modified/created; if only the filename has a date, infer from it). Never place irrelevant above relevant.\n"
)

_FULL_ORDER_RULE = (
    "OUTPUT REQUIREMENT:\n"
    "- You MUST output a COMPLETE ORDER of ALL candidates, exactly once each. "
    "Include EVERY index from the provided list, with no omissions or duplicates.\n"
)

_FULL_ORDER_RULE_RETRY = (
    "CRITICAL: Your previous output omitted indices. You MUST now return a COMPLETE permutation of ALL candidates.\n"
    "Include EVERY index exactly once, one per line, no explanations.\n"
)

def _user_prompt(original_query: Optional[str], query: str, catalog: str) -> str:
    return (
        f"Original Query: {original_query}\n"
        f"User Query: {query}\n\n"
        f"Candidates:\n{catalog}\n\n"
        "Return the indices for ALL candidates, in final ranked order, one per line."
    )

# ---------------- Helpers for recency + query-aware fallback ----------------
_DATE_PATS = [
    re.compile(r"\b(20\d{2}|19\d{2})[.\-_/ ]?(0[1-9]|1[0-2])[.\-_/ ]?(0[1-9]|[12]\d|3[01])\b"),  # YYYY.MM.DD / YYYYMMDD
    re.compile(r"\b(\d{2})[.\-_/ ]?(0[1-9]|1[0-2])[.\-_/ ]?(0[1-9]|[12]\d|3[01])\b"),            # YY.MM.DD / YYMMDD
]

def _iso_ts(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None

def _ts_from_name(name: str) -> float:
    for pat in _DATE_PATS:
        m = pat.search(name or "")
        if m:
            y = int(m.group(1))
            if y < 100:  # YY -> 20YY
                y += 2000
            return datetime(y, int(m.group(2)), int(m.group(3))).timestamp()
    return 0.0

def _best_ts(f: Dict[str, Any]) -> float:
    # Prefer metadata modified/created; fall back to filename date.
    for path in [
        ("lastModifiedDateTime",),
        ("fileSystemInfo", "lastModifiedDateTime"),
        ("createdDateTime",),
        ("fileSystemInfo", "createdDateTime"),
    ]:
        try:
            v = f[path[0]] if len(path) == 1 else f[path[0]][path[1]]
            ts = _iso_ts(v) if isinstance(v, str) else None
            if ts:
                return ts
        except Exception:
            pass
    return _ts_from_name(f.get("name", ""))

_STOPWORDS = {
    "the","a","an","and","or","of","to","for","in","on","at","by","with",
    "file","files","most","recent","latest","newest","show","me","please",
    "report","document"
}

def _norm_tokens(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [t for t in s.split() if t and t not in _STOPWORDS]

def _matches_all_tokens(f: Dict[str, Any], tokens: list[str]) -> bool:
    if not tokens:
        return False
    hay = (f.get("name","") + " " + (f.get("extracted_text") or "")).lower()
    return all(t in hay for t in tokens)

# ---------------- Main API ----------------
def rank_files_with_llama_prompt(
    query: str,
    files: List[Dict[str, Any]],
    original_query: Optional[str] = None,
    max_chars: int = 240,
    **kwargs,  # swallow legacy args like prefer_recent to stay backward-compatible
) -> List[Dict[str, Any]]:
    """
    Prompt-only global ranking.
    We require a FULL permutation so pagination remains consistent across pages.
    Two attempts:
      1) Normal catalog (with snippets) + strict 'all indices' rule.
      2) If the model omits indices, retry with ultra-light catalog + stricter rule.
    If still incomplete, fall back to a query-aware recency order:
      - Items containing the query tokens first (newest→oldest), then the rest (newest→oldest).
    """
    if not files:
        return []

    N = len(files)

    # --- Attempt 1: regular catalog
    catalog = _build_catalog(files, max_chars=max_chars, ultra_light=False)
    sys = _BASE_SYSTEM + "\n" + _FULL_ORDER_RULE
    usr = _user_prompt(original_query, query, catalog)
    raw = _prompt_llm.predict(f"{sys}\n\n{usr}").strip()
    print(f"🤖 [1] Raw LLaMA response:\n{raw}\n")
    order = _parse_indices(raw, N)
    print(f"🤖 [1] Parsed indices: {order}")

    if len(order) == N:
        return [files[i] for i in order]

    # --- Attempt 2: ultra-light catalog + harder instruction
    lite_catalog = _build_catalog(files, max_chars=0, ultra_light=True)
    sys2 = _BASE_SYSTEM + "\n" + _FULL_ORDER_RULE_RETRY
    usr2 = _user_prompt(original_query, query, lite_catalog)
    raw2 = _prompt_llm.predict(f"{sys2}\n\n{usr2}").strip()
    order2 = _parse_indices(raw2, N)

    if len(order2) == N:
        return [files[i] for i in order2]

    # --- Query-aware fallback: keep relevant items first, then the rest (both newest→oldest)
    q_tokens = _norm_tokens(f"{query} {original_query}")
    if q_tokens:
        primary = [f for f in files if _matches_all_tokens(f, q_tokens)]
        if primary:
            primary_sorted = sorted(primary, key=_best_ts, reverse=True)
            rest = [f for f in files if f not in primary]
            rest_sorted = sorted(rest, key=_best_ts, reverse=True)
            return primary_sorted + rest_sorted

    # --- Nothing matched tokens → pure recency
    return sorted(files, key=_best_ts, reverse=True)
