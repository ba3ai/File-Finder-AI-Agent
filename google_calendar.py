"""
Google Calendar integration (REST)
- Expects Google OAuth access token with scope: calendar.readonly
"""
from __future__ import annotations

from typing import List, Dict, Optional
import json as _json
import requests
import datetime as _dt
import os

from openai_api import answer_general_query

def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

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

def search_google_calendar(
    google_access_token: str,
    query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> List[Dict]:
    """
    Search primary calendar for matching events within a window.
    If no window is provided, defaults to last 30d to next 365d (or 'today/yesterday' if typed).

    Returns rows: {subject, start, end, location, url, description}
    """
    try:
        now = _dt.datetime.utcnow()

        if time_min is None and time_max is None:
            tmin_f, tmax_f = _simple_nlp_window(query)
            time_min = time_min or tmin_f
            time_max = time_max or tmax_f

        tmn = time_min or (now - _dt.timedelta(days=30)).isoformat() + "Z"
        tmx = time_max or (now + _dt.timedelta(days=365)).isoformat() + "Z"

        params = {
            "q": (query or ""),
            "timeMin": tmn,
            "timeMax": tmx,
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": min(int(limit or 10), 50),
        }
        r = requests.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers=_h(google_access_token),
            params=params,
            timeout=20,
        )
        if not r.ok:
            return []

        items = r.json().get("items", []) or []
        out: List[Dict] = []
        for ev in items[: min(int(limit or 10), 50)]:
            start = (ev.get("start") or {}).get("dateTime") or (ev.get("start") or {}).get("date")
            end   = (ev.get("end")   or {}).get("dateTime")   or (ev.get("end")   or {}).get("date")
            out.append({
                "subject": ev.get("summary"),
                "start": start,
                "end": end,
                "location": ev.get("location") or "",
                "url": ev.get("htmlLink"),
                "description": ev.get("description") or "",
            })
        return out
    except Exception:
        return []

_SYSTEM_SMART_CAL = (
    "You are a calendar assistant. You are given a user request and a list of events (JSON). "
    "Return ONLY events that answer the request. Never invent events not in the list. "
    "Format as a short Markdown list:\n- [Title](url) — start → end @ location\n"
    "If there are no matches, return: \"No events found.\""
)

def google_calendar_smart_search(
    google_access_token: str,
    user_query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> str:
    rows = search_google_calendar(google_access_token, user_query, limit=limit, time_min=time_min, time_max=time_max)
    data_json = _json.dumps(rows, ensure_ascii=False)
    prompt = f"User request: {user_query}\n\nEvents JSON:\n{data_json}\n\nReturn only matching items as specified. If none, say 'No events found.'"
    return answer_general_query(prompt, system_prompt=_SYSTEM_SMART_CAL)
