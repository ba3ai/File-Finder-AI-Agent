# llm_ranker.py
# LLM-based re-ranking with:
#  - STRICT intent guidance for the model (core phrase/tokens + year logic incl. YY forms)
#  - Tolerant JSON extraction + normalization (rescales 0–10/0–100 to 0–1; clamps; trims)
#  - Phrase-first, token-second candidate selection
#  - Year filtering: 4-digit first; if empty, allow two-digit YY patterns in filenames; if still empty, fall back
#  - Custom sort: primary SortKey DESC (Modified if Modified < Created, else Created),
#                 secondary LLM score (+ tiny boost if SortKey year in query years),
#                 tertiary filename A→Z
#  - Rich debug report when enabled

from __future__ import annotations

import os
import re
import json
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# ---------------- Optional OpenAI client ----------------
_USE_OPENAI = False
_openai_client = None
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        _USE_OPENAI = True
    except Exception:
        _USE_OPENAI = False
        _openai_client = None

# ---------------- Ollama via LangChain ------------------
_llm = None
try:
    from langchain_ollama import ChatOllama
    _llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "mistral:7b"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8000")),
        num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
    )
except Exception:
    _llm = None

DEBUG = bool(int(os.getenv("LLM_RANKER_DEBUG", "0")))
LLM_RANKER_TRACE = bool(int(os.getenv("LLM_RANKER_TRACE", "0")))  # global on/off for debug report

def _dprint(*a):
    if DEBUG:
        print("[llm_ranker]", *a)

def _warn(msg: str):
    if DEBUG or LLM_RANKER_TRACE:
        print("[llm_ranker][WARN]", msg)

# ---------------- Utilities ----------------
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _parse_dt(s: str) -> float:
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

def _dt_year(ts: float) -> Optional[int]:
    try:
        if ts <= 0:
            return None
        return datetime.utcfromtimestamp(ts).year
    except Exception:
        return None

def _first(text: str, n: int) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ").replace("\r", " ")
    return t[:n]

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = _WS_RE.sub(" ", s).strip()
    return s

def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 2]

def _extract_all_years(text: str) -> List[int]:
    return [int(m.group(0)) for m in re.finditer(_YEAR_RE, text or "")]

def _combined_text(f: Dict[str, Any]) -> str:
    name = f.get("name") or f.get("title") or ""
    desc = f.get("description") or ""
    body = f.get("extracted_text") or ""
    meta_parts = [name, desc, body]
    for k in ("lastModifiedDateTime", "created_at", "createdTime", "createdDateTime"):
        v = f.get(k)
        if v:
            meta_parts.append(str(v))
    return _normalize(" ".join([p for p in meta_parts if p]))

def _created_modified_ts(f: Dict[str, Any]) -> Tuple[float, float]:
    created = f.get("createdDateTime") or f.get("created_at") or f.get("createdTime") or ""
    modified = f.get("lastModifiedDateTime") or f.get("modified_at") or f.get("modifiedTime") or ""
    ct = _parse_dt(created)
    mt = _parse_dt(modified)
    if ct == 0 and mt != 0:
        ct = mt
    if mt == 0 and ct != 0:
        mt = ct
    return ct, mt

def _sort_key_ts(f):
    ct, mt = _created_modified_ts(f)
    # prefer the most recent timestamp
    if mt and ct:
        return max(mt, ct)      # newest of the two
    return mt or ct or 0.0

# --- Stopwords & core-phrase helpers -----------------------------------------
_STOPWORDS = {
    "show", "find", "fetch", "give", "get", "open", "list", "please",
    "me", "the", "a", "an", "this", "that", "my",
    "file", "files", "document", "documents", "doc", "pdf", "from", "of", "for", "in", "on", "to"
}

def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS and not t.isdigit()]

def _focus_phrases(query: str) -> List[str]:
    """
    Build likely core phrases from content words (3-gram -> 2-gram -> whole set).
    Example: 'show me the 2025 pike valuation file' -> ['pike valuation']
    """
    toks = _content_tokens(query)
    phrases: List[str] = []
    for n in (3, 2):
        for i in range(len(toks) - n + 1):
            phrases.append(" ".join(toks[i:i+n]))
    if toks:
        phrases.append(" ".join(toks))
    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for p in phrases:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _yy_patterns(years: List[int]) -> List[re.Pattern]:
    """
    Two-digit-year patterns for filenames (e.g., 2025 -> '25', '25.03.31', '-25-', '_25_', '.25', '25-').
    """
    pats: List[re.Pattern] = []
    for y in years:
        yy = y % 100
        pats.extend([
            re.compile(rf"\b{yy}\b"),
            re.compile(rf"\b{yy}[.\-_]"),
            re.compile(rf"[.\-_]{yy}\b"),
        ])
    return pats

# ---------------- Default pre-trim -------------------------------------------
def _default_sort(query: str, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cheap pre-trim: filename contains whole query, then token matches, then recency."""
    q = query.lower()
    q_tokens = [t for t in _TOKEN_RE.findall(q) if len(t) >= 2]

    def name_score(f):
        n = (f.get("name") or "").lower()
        score = 0.0
        if q and q in n:
            score += 3.0
        for t in set(q_tokens):
            if t in n:
                score += 0.5
        return score

    return sorted(
        files,
        key=lambda f: (name_score(f), _parse_dt(f.get("lastModifiedDateTime") or "")),
        reverse=True,
    )

# ---------------- Catalog + LLM ----------------------------------------------
STRICT_FILE_RANKING_PROMPT = (
    "You are ranking files for a search assistant.\n"
    "You receive: (1) a natural-language query and (2) a numbered catalog of files (1-based).\n\n"
    "A. Preprocessing (MUST)\n"
    "- Lowercase everything for matching.\n"
    "- Ignore filler words: show, find, fetch, give, get, open, list, please, me, the, a, an, this, that, my, "
    "file, files, document, documents, doc, pdf, from, of, for, in, on, to\n"
    "- Extract: Core phrase (main content words in order), Core tokens, and 4-digit years in the query. "
    "Treat 20YY as also matching two-digit YY patterns in filenames: YY.MM.DD, YY-MM-DD, _YY_, ' YY ', YY-, YY_, .YY.\n\n"
    "B. Candidate Filtering (MUST, in this order)\n"
    "1) Year pass: If query has a year, keep files that contain the 4-digit year in name/snippet/metadata. "
    "If none, keep files whose filename matches the two-digit YY forms. If still none, keep all.\n"
    "2) Phrase/tokens pass: First keep files whose filename contains the exact core phrase. "
    "If none, keep files that contain all core tokens across name/snippet when ≤3 tokens; "
    "if >3 tokens, require at least two-thirds of the tokens. If still none, keep the set from step 1.\n\n"
    "C. Scoring (MUST; clamp to [0,1])\n"
    "- Phrase in filename: +0.70\n"
    "- All core tokens across name/snippet (if no phrase): +0.55\n"
    "- Year evidence: 4-digit year present +0.20; otherwise two-digit YY form in filename +0.10\n"
    "- Longer n-gram boost (e.g., 'pike valuation ryan final'): +0.05\n"
    "- Recency tie-assist: if modified/created year equals query year +0.03, else +0.02 if newer than most peers\n"
    "Penalties: single-token-only (no phrase) −0.40; no year evidence while others have it (when query has a year) −0.20; "
    "filler-only/unrelated −0.50. Clamp final score to [0,1]. If all <0.50, still return top 5 by score.\n\n"
    "D. Tie-breakers (2-decimal ties): phrase > token; 4-digit year > 2-digit form; newer date; shorter filename; lower index.\n"
)

OUTPUT_FORMAT_PROMPT = (
    "Return JSON only. Do not include explanations, markdown, or code fences. Output at most 10 items.\n\n"
    "The JSON must have this exact shape:\n"
    "{\n"
    '  "ranking": [\n'
    '    { "index": <INT>, "score": <DECIMAL with exactly two digits>, "reason": "<<=140 chars, single line>" }\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    '- "index": 1-based integer referring to the provided catalog.\n'
    '- "score": number between 0.00 and 1.00 with EXACTLY two decimals. If you output a different scale, it will be clamped.\n'
    '- "reason": max 140 characters; single line; briefly state why (no newlines).\n'
    "- No trailing commas. No additional fields. Output ONLY the JSON object above."
)

_SYSTEM = STRICT_FILE_RANKING_PROMPT + "\n\n" + OUTPUT_FORMAT_PROMPT

def _build_catalog(files: List[Dict[str, Any]], max_chars: int = 280) -> str:
    rows = []
    for i, f in enumerate(files, 1):
        name = f.get("name") or f.get("title") or f"file_{i}"
        snippet = f.get("extracted_text") or f.get("description") or ""
        if not snippet:
            snippet = name
        rows.append(f"{i}. {name}\n   { _first(snippet, max_chars) }")
    return "\n".join(rows)

def _ask_llm(prompt: str) -> str:
    if _USE_OPENAI and _openai_client is not None:
        try:
            resp = _openai_client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            _dprint("OpenAI error:", e)
    if _llm is not None:
        try:
            out = _llm.invoke(_SYSTEM + "\n\n" + prompt)
            return (getattr(out, "content", None) or str(out) or "").strip()
        except Exception as e:
            _dprint("Ollama error:", e)
    return ""

# ---------------- JSON extraction & normalization ----------------------------
def _extract_llm_json(txt: str) -> dict | None:
    if not txt:
        return None
    # 1) fenced ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 2) first {...} blob
    m = re.search(r"(\{.*\})", txt, re.DOTALL)
    if m:
        raw = m.group(1)
        last = raw.rfind("}")
        try:
            return json.loads(raw[: last + 1] if last != -1 else raw)
        except Exception:
            return None
    return None

def _normalize_ranking(payload: dict, catalog_len: int, max_items: int = 10) -> list[dict]:
    out = []
    if not isinstance(payload, dict):
        return out
    items = payload.get("ranking") or []
    seen = set()
    for it in items:
        try:
            idx = int(it.get("index"))
            if not (1 <= idx <= catalog_len) or idx in seen:
                continue
            seen.add(idx)

            sc = it.get("score", 0)
            try:
                sc = float(sc)
            except Exception:
                sc = 0.0
            if not math.isfinite(sc):
                sc = 0.0

            # Rescale if model used 0–10 or 0–100
            if sc > 1.0:
                if sc <= 10.0:
                    sc /= 10.0
                elif sc <= 100.0:
                    sc /= 100.0
            sc = max(0.0, min(1.0, sc))

            rsn = str(it.get("reason", "")).replace("\n", " ").strip()[:140]
            out.append({"index": idx, "score": round(sc, 2), "reason": rsn})
        except Exception:
            continue
    return out[:max_items]

# ---------------- Public API -------------------------------------------------
def rank_files_with_llm(
    query: str,
    files: List[Dict[str, Any]],
    limit: int = 12,
    candidate_pool: int = 48,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Re-rank candidate files using an LLM with strict year filtering & keyword logic.
    Returns (ranked_files, debug_report or None).
    """
    if not files:
        return [], ({"query": query, "note": "no candidates"} if (debug or LLM_RANKER_TRACE) else None)

    # 1) Pre-trim (cheap)
    prelim = _default_sort(query, files)[: max(candidate_pool, limit)]
    q_years = set(_extract_all_years(query))

    # 2) STRICT YEAR FILTERING (4-digit first)
    after_year = prelim
    if q_years:
        kept = []
        for f in prelim:
            text = _combined_text(f)
            f_years = set(_extract_all_years(text))
            if f_years & q_years:
                kept.append(f)
        after_year = kept

        # Fallback: allow two-digit filename patterns if strict filter removed all
        if not after_year:
            pats = _yy_patterns(sorted(q_years))
            tmp = []
            for f in prelim:
                name = (f.get("name") or f.get("title") or "").lower()
                if any(p.search(name) for p in pats):
                    tmp.append(f)
            after_year = tmp or prelim  # last resort: don't block everything

    # 3) KEYWORD LOGIC (phrase → tokens → fallback)
    q_norm = _normalize(query)
    phrases = _focus_phrases(q_norm)           # likely phrases in order of usefulness
    content_toks = _content_tokens(q_norm)     # tokens w/o stopwords or digits
    token_exclude = {str(y) for y in q_years}
    tokens = [t for t in content_toks if t not in token_exclude]

    # phrase: check in filename first (strongest signal)
    phrase_hits: List[Dict[str, Any]] = []
    for ph in phrases[:6]:  # check a few best phrases
        if not ph:
            continue
        ph_hits = []
        ph_l = ph.lower()
        for f in after_year:
            name = (f.get("name") or f.get("title") or "").lower()
            if ph_l in name:
                ph_hits.append(f)
        if ph_hits:
            phrase_hits = ph_hits
            break

    if phrase_hits:
        candidate_set = phrase_hits
        kw_mode = "phrase"
    else:
        if tokens:
            need_all = len(tokens) <= 3
            thresh = len(tokens) if need_all else max(2, int(round(len(tokens) * 0.66)))
            tmp = []
            for f in after_year:
                comb = _combined_text(f)
                hits = sum(1 for t in tokens if t in comb)
                if hits >= thresh:
                    tmp.append(f)
            candidate_set = tmp or after_year
            kw_mode = "tokens_all" if need_all else f"tokens_{thresh}_of_{len(tokens)}"
        else:
            candidate_set = after_year
            kw_mode = "fallback_any"

    if not candidate_set:
        # Shouldn't happen with the fallbacks, but guard anyway
        candidate_set = after_year
        kw_mode = "safety_fallback"

    # 4) Trim again for the LLM catalog
    cand = candidate_set[: max(candidate_pool, limit)]

    # 5) Ask LLM for scores
    catalog = _build_catalog(cand, max_chars=280)
    user_prompt = f"Query: {query}\n\nCatalog:\n{catalog}\n\nReturn JSON now."
    print(f"LLM prompt ({len(user_prompt)} chars):\n{user_prompt}\n---")
    llm_raw = _ask_llm(user_prompt)
    print(f"LLM response: {llm_raw[:1000]}...")  # Show first 1000 chars for brevity

    payload = _extract_llm_json(llm_raw) or {}
    ranking = _normalize_ranking(payload, catalog_len=len(cand))
    print(f"LLM parsed (normalized): {ranking}")

    # Build index -> score map
    idx_to_score: Dict[int, float] = {}
    for item in ranking:
        i = int(item["index"])
        sc = float(item["score"])
        idx_to_score[i - 1] = sc

    # 6) Fallback if LLM didn't return usable scores
    used_fallback = False
    if not idx_to_score:
        used_fallback = True
        _dprint("LLM returned no usable scores; using deterministic filename/token fallback.")
        main_ph = phrases[0] if phrases else ""
        tokset = set(tokens)

        def score_fn(f):
            n = (f.get("name") or "").lower()
            s = 0.0
            if main_ph and main_ph in n:
                s += 2.0
            s += sum(0.6 for t in tokset if t in n)
            return s

        for i, f in enumerate(cand):
            idx_to_score[i] = score_fn(f)

    # 7) Final sort tuple:
    #    primary: custom SortKey DESC
    #    secondary: score DESC (+ small boost if SortKey year is in q_years)
    #    tertiary: filename A→Z
    triples = []
    file_debug = []
    for i, f in enumerate(cand):
        sort_ts = _sort_key_ts(f) or 0.0
        sort_year = _dt_year(sort_ts)
        score = idx_to_score.get(i, 0.0)

        boosted = False
        if q_years and sort_year in q_years:
            score += 0.05  # tiny boost
            boosted = True

        name = (f.get("name") or f.get("title") or "").lower()
        triples.append((i, sort_ts, score, name))

        if debug or LLM_RANKER_TRACE:
            combined = _combined_text(f)
            token_hits = {t: (t in combined) for t in tokens}
            file_debug.append({
                "i": i + 1,
                "name": f.get("name") or f.get("title") or "",
                "sort_ts": sort_ts,
                "sort_year": sort_year,
                "score": round(score, 6),
                "year_hit": bool(set(_extract_all_years(combined)) & q_years) if q_years else None,
                "phrase_used": (kw_mode == "phrase"),
                "main_phrase_hit": (phrases[0] in name) if (phrases and kw_mode == "phrase") else None,
                "token_logic": kw_mode if kw_mode.startswith("tokens") else None,
                "token_hits": token_hits if tokens else None,
                "boosted_by_year": boosted,
            })

    ordered = sorted(triples, key=lambda t: (-t[2], -t[1], t[3]))
    ranked = [cand[i] for (i, _, __, ___) in ordered]

    # 8) Sanity warnings
    if q_years:
        for f in ranked:
            if not (set(_extract_all_years(_combined_text(f))) & q_years):
                _warn(f"Year rule: '{f.get('name')}' lacks {sorted(q_years)}")

    # 9) Build debug report if requested
    report = None
    if debug or LLM_RANKER_TRACE:
        def fmt_file(i_f):
            i, sort_ts, sc, n = i_f
            return {
                "index": i + 1,
                "name": cand[i].get("name"),
                "sort_ts": sort_ts,
                "sort_year": _dt_year(sort_ts),
                "score": sc,
            }

        report = {
            "query": query,
            "q_years": sorted(q_years),
            "kw_mode": kw_mode,
            "sizes": {
                "prelim": len(prelim),
                "after_year": len(after_year),
                "candidates": len(cand),
            },
            "catalog_preview": (catalog[:1200] if catalog else ""),
            "llm_raw": llm_raw,
            "llm_payload": payload,
            "llm_normalized_ranking": ranking,
            "used_fallback": used_fallback,
            "final_order": [fmt_file(t) for t in ordered],
            "per_file": file_debug,
        }

    return ranked[:limit], report

# Convenience wrapper that keeps your old call sites working (returns only files)
def rank_files_with_llm_simple(
    query: str,
    files: List[Dict[str, Any]],
    limit: int = 12,
    candidate_pool: int = 48,
) -> List[Dict[str, Any]]:
    ranked, _ = rank_files_with_llm(query, files, limit=limit, candidate_pool=candidate_pool, debug=True)
    return ranked

# Backwards-compatible alias some code may already import
def rank_files_llama(query: str, files: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    return rank_files_with_llm_simple(query, files, limit=limit)

# --- Optional: tiny CLI smoke test ------------------------------------------
if __name__ == "__main__":
    samples = [
        {"name": "Pike Valuation Ryan Final.pdf", "lastModifiedDateTime": "2025-03-31T12:00:00Z"},
        {"name": "Pike Draft Valuation Exhibits.pdf", "lastModifiedDateTime": "2025-02-15T12:00:00Z"},
        {"name": "Pike Val EP 23.06.30.pdf", "lastModifiedDateTime": "2023-06-30T12:00:00Z"},
        {"name": "Random Notes.txt", "lastModifiedDateTime": "2025-01-01T09:00:00Z", "extracted_text": "valuation memo"},
    ]
    ranked, rep = rank_files_with_llm("show me the 2025 pike valuation file", samples, limit=3, debug=True)
    print(json.dumps(rep, indent=2, default=str))
