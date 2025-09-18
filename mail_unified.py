# mail_unified.py
from __future__ import annotations
import json, math
from typing import List, Dict, Optional, Iterable

from gmail import search_gmail
from ms_outlook_mail import search_outlook_messages
from openai_api import answer_general_query

_SYSTEM_PICK = (
    "You are an email selector. You get a list of EMAILS as JSON. "
    "Return ONLY those that satisfy the user’s request. Never invent items not in the list.\n"
    "Output format (Markdown list):\n"
    "- [Subject](url) — from (ISO datetime)\n"
    "If no matches in this chunk, output exactly: NONE"
)

_SYSTEM_MERGE = (
    "You are formatting the final answer. You get multiple small Markdown lists "
    "containing only the emails that matched earlier. Merge, deduplicate (by url+subject), "
    "sort newest→oldest by date if present, and output one Markdown list. "
    "If empty, print exactly: No messages found."
)

def _chunk(items: List[Dict], size: int) -> Iterable[List[Dict]]:
    for i in range(0, len(items), size):
        yield items[i:i+size]

def _pick_from_chunk(user_query: str, chunk: List[Dict]) -> str:
    payload = json.dumps(chunk, ensure_ascii=False)
    prompt = f"User request: {user_query}\n\nEMAILS JSON:\n{payload}\n\nSelect & format as specified. If none: NONE"
    return answer_general_query(prompt, system_prompt=_SYSTEM_PICK)

def _merge_lists(lists: List[str]) -> str:
    text = "\n\n".join(lists)
    prompt = f"Merge these lists into one, per the rules:\n\n{text}"
    return answer_general_query(prompt, system_prompt=_SYSTEM_MERGE)

def email_search_unified_smart(
    user_query: str,
    sources: List[str],
    ms_token: Optional[str],
    gtoken: Optional[str],
    time_min: Optional[str],
    time_max: Optional[str],
    max_total: int = 300,
    chunk_size: int = 50,
) -> str:
    """
    Pulls a date-window of emails from the selected providers, then runs a
    map-reduce LLM selection so the LLM 'reads' everything without hitting
    context limits.
    """
    rows: List[Dict] = []

    # Outlook
    if ("outlook" in sources) and ms_token:
        # Do not over-filter by the whole sentence; date window only — let the LLM decide relevance.
        ms_rows = search_outlook_messages(ms_token, user_query, limit=max_total, time_min=time_min, time_max=time_max) or []
        for r in ms_rows:
            r["provider"] = "outlook"
        rows.extend(ms_rows)

    # Gmail
    if ("gmail" in sources) and gtoken:
        gm_rows = search_gmail(gtoken, user_query, limit=max_total, time_min=time_min, time_max=time_max) or []
        for r in gm_rows:
            r["provider"] = "gmail"
        rows.extend(gm_rows)

    if not rows:
        return "No messages found."

    # Map: let the LLM pick from each chunk
    picked_lists: List[str] = []
    for ch in _chunk(rows, chunk_size):
        s = _pick_from_chunk(user_query, ch).strip()
        if s and s.upper() != "NONE":
            picked_lists.append(s)

    if not picked_lists:
        return "No messages found."

    # Reduce: merge & format
    return _merge_lists(picked_lists)
