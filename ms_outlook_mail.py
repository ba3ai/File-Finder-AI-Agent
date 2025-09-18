"""
Outlook Mail integration (Microsoft Graph)
- Expects Graph token with scope: Mail.Read
"""
from __future__ import annotations

from typing import List, Dict, Optional
import os
import json as _json
import requests

from openai_api import answer_general_query

def _simple_nlp_window(user_query: str) -> tuple[Optional[str], Optional[str]]:
    """If caller didn't pass a window, interpret 'today'/'yesterday'."""
    try:
        from datetime import datetime, timedelta, timezone
        from zoneinfo import ZoneInfo
        low = (user_query or "").lower()
        if "today" in low or "yesterday" in low:
            APP_TZV = ZoneInfo(os.getenv("APP_TZ", "Asia/Dhaka"))
            now_local = datetime.now(APP_TZV)
            def _z(dt_local): return dt_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if "yesterday" in low:
                y = now_local - timedelta(days=1)
                s = y.replace(hour=0, minute=0, second=0, microsecond=0)
                e = y.replace(hour=23, minute=59, second=59, microsecond=999999)
            else:
                s = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
                e = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)
            return _z(s), _z(e)
    except Exception:
        pass
    return None, None

def search_outlook_messages(
    ms_token: str,
    query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> List[Dict]:
    """
    Graph Mail search.
    If a date window is given, use $filter on receivedDateTime (and optional subject contains).
    Otherwise try $search and fall back to subject contains.
    Returns rows: {id, subject, from, date, url, text}
    """
    try:
        top = min(int(limit or 10), 25)
        url = "https://graph.microsoft.com/v1.0/me/messages"

        # Use fallback window if none provided and user typed today/yesterday
        if time_min is None and time_max is None:
            tmin_f, tmax_f = _simple_nlp_window(query)
            time_min = time_min or tmin_f
            time_max = time_max or tmax_f

        if time_min or time_max:
            clauses = []
            if time_min: clauses.append(f"receivedDateTime ge {time_min}")
            if time_max: clauses.append(f"receivedDateTime le {time_max}")
            flt = " and ".join(clauses)
            if query:
                safe = (query or "").replace("'", "''")
                flt = f"{flt} and contains(subject,'{safe}')"
            r = requests.get(
                url,
                headers={"Authorization": f"Bearer {ms_token}"},
                params={
                    "$filter": flt,
                    "$orderby": "receivedDateTime desc",
                    "$select": "id,subject,from,receivedDateTime,webLink,bodyPreview",
                    "$top": top,
                },
                timeout=20,
            )
        else:
            headers = {"Authorization": f"Bearer {ms_token}", "ConsistencyLevel": "eventual"}
            r = requests.get(
                url,
                headers=headers,
                params={
                    "$search": f'"{query or ""}"',
                    "$select": "id,subject,from,receivedDateTime,webLink,bodyPreview",
                    "$top": top,
                },
                timeout=20,
            )
            if not r.ok:
                safe = (query or "").replace("'", "''")
                r = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {ms_token}"},
                    params={
                        "$filter": f"contains(subject,'{safe}')",
                        "$select": "id,subject,from,receivedDateTime,webLink,bodyPreview",
                        "$top": top,
                    },
                    timeout=20,
                )

        if not r.ok:
            return []

        items = r.json().get("value", []) or []
        out: List[Dict] = []
        for m in items[:top]:
            sender = ((m.get("from") or {}).get("emailAddress") or {}).get("address")
            out.append({
                "id": m.get("id"),
                "subject": m.get("subject"),
                "from": sender or "",
                "date": m.get("receivedDateTime"),
                "url": m.get("webLink"),
                "text": m.get("bodyPreview") or "",
            })
        return out
    except Exception:
        return []

_SYSTEM_SMART_EMAIL = (
    "You are an email assistant. You are given a user request and a list of emails (JSON). "
    "Return ONLY results that answer the request. Never invent emails not in the list. "
    "Format as a short Markdown list:\n- [Subject](url) — from (ISO date)\n"
    "If there are no matches, return: \"No messages found.\""
)

def outlook_mail_smart_search(
    ms_token: str,
    user_query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> str:
    rows = search_outlook_messages(ms_token, user_query, limit=limit, time_min=time_min, time_max=time_max)
    data_json = _json.dumps(rows, ensure_ascii=False)
    prompt = f"User request: {user_query}\n\nEmails JSON:\n{data_json}\n\nReturn only matching items as specified. If none, say 'No messages found.'"
    return answer_general_query(prompt, system_prompt=_SYSTEM_SMART_EMAIL)
