"""
Gmail integration (REST)
- Expects Google OAuth access token with scope: gmail.readonly
"""
from __future__ import annotations

from typing import List, Dict, Optional
import os
import base64
import datetime as _dt
import json as _json
import re as _re
import requests

from openai_api import answer_general_query  # uses your Ollama-backed LLM

# -------------------- helpers --------------------

def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def _ymd(iso: str) -> str:
    # Gmail q expects YYYY/MM/DD
    d = _dt.date.fromisoformat(iso[:10])
    return d.strftime("%Y/%m/%d")

def _decode_urlsafe(b64: str) -> str:
    try:
        return base64.urlsafe_b64decode(b64.encode()).decode(errors="ignore")
    except Exception:
        return ""

def _extract_gmail_text(msg_json) -> str:
    """Extract best-effort plain text from a Gmail message JSON."""
    def _plain_from_part(part):
        mime = (part.get("mimeType") or "").lower()
        body = (part.get("body") or {}).get("data")
        text = _decode_urlsafe(body) if body else ""
        if "html" in mime:
            text = _re.sub(r"<[^>]+>", " ", text)
        return text

    payload = msg_json.get("payload") or {}
    parts = payload.get("parts") or []
    texts = []
    if parts:
        stack = parts[:]
        while stack:
            p = stack.pop(0)
            if p.get("parts"):
                stack.extend(p["parts"])
            texts.append(_plain_from_part(p))
    else:
        texts.append(_plain_from_part(payload))
    text = " ".join([t for t in texts if t]).strip() or (msg_json.get("snippet") or "")
    return _re.sub(r"\s+", " ", text)[:2000]

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

# -------------------- core search --------------------

def search_gmail(
    google_access_token: str,
    query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> List[Dict]:
    """
    Search Gmail and return rows with basic fields + text preview.
    If time_min/time_max are provided (ISO8601 Z), add after:/before: filters.

    Returns rows: {id, subject, from, date, url, text}
    """
    base = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    try:
        # build query
        q = (query or "").strip()
        # Use fallback window if none provided and user typed today/yesterday
        if time_min is None and time_max is None:
            tmin_f, tmax_f = _simple_nlp_window(query)
            time_min = time_min or tmin_f
            time_max = time_max or tmax_f

        if time_min:
            q = (q + f" after:{_ymd(time_min)}").strip()
        if time_max:
            end_inc = _dt.date.fromisoformat(time_max[:10]) + _dt.timedelta(days=1)  # before is exclusive
            q = (q + f" before:{end_inc.strftime('%Y/%m/%d')}").strip()

        params = {"q": q, "maxResults": min(int(limit or 10), 50)}
        r = requests.get(base, headers=_h(google_access_token), params=params, timeout=20)
        if not r.ok:
            return []

        ids = [m.get("id") for m in (r.json().get("messages") or [])][:limit]
        out: List[Dict] = []
        for mid in ids:
            m = requests.get(f"{base}/{mid}", headers=_h(google_access_token),
                             params={"format": "full"}, timeout=20)
            if not m.ok:
                continue
            j = m.json()
            headers = {h.get("name"): h.get("value") for h in (j.get("payload", {}) or {}).get("headers", [])}
            out.append({
                "id": mid,
                "subject": headers.get("Subject") or "(no subject)",
                "from": headers.get("From") or "",
                "date": headers.get("Date") or j.get("internalDate"),
                "url": f"https://mail.google.com/mail/u/0/#all/{mid}",
                "text": _extract_gmail_text(j),
            })
        return out
    except Exception:
        return []

# -------------------- LLM smart passthrough --------------------

_SYSTEM_SMART_EMAIL = (
    "You are an email assistant. You are given a user request and a list of emails (JSON). "
    "Return ONLY results that answer the request. Never invent emails not in the list. "
    "Format as a short Markdown list:\n- [Subject](url) — from (ISO date)\n"
    "If there are no matches, return: \"No messages found.\""
)

def gmail_smart_search(
    google_access_token: str,
    user_query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> str:
    rows = search_gmail(google_access_token, user_query, limit=limit, time_min=time_min, time_max=time_max)
    data_json = _json.dumps(rows, ensure_ascii=False)
    prompt = f"User request: {user_query}\n\nEmails JSON:\n{data_json}\n\nReturn only matching items as specified. If none, say 'No messages found.'"
    return answer_general_query(prompt, system_prompt=_SYSTEM_SMART_EMAIL)
