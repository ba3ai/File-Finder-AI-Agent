"""
Outlook Calendar integration (Microsoft Graph)
- Expects a Graph token with scope: Calendars.Read
- Includes an in-file LLM caller (Ollama via LangChain) so you don't need openai_api.py
"""
from __future__ import annotations

from typing import List, Dict, Optional
import os
import json as _json
import requests
import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log = logging.getLogger("ms_outlook_calendar")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[ms_outlook_calendar] %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

def _d(*a):  # tiny debug helper
    log.info(" ".join(str(x) for x in a))

# -----------------------------------------------------------------------------
# Optional LLM (Ollama via LangChain) – local, no URL required
# -----------------------------------------------------------------------------
ChatOllama = None
try:
    from langchain_ollama import ChatOllama  # preferred
except Exception:
    try:
        # older installs
        from langchain_community.chat_models import ChatOllama  # type: ignore
    except Exception:
        ChatOllama = None

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    try:
        from langchain.schema import SystemMessage, HumanMessage
    except Exception:
        SystemMessage = HumanMessage = None  # type: ignore

MODEL_NAME  = os.getenv("OLLAMA_CAL_MODEL", os.getenv("OLLAMA_MODEL", "qwen2:7b"))
NUM_CTX     = int(os.getenv("OLLAMA_NUM_CTX", "20000"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "800"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

_LLM = None
if ChatOllama is not None and SystemMessage is not None and HumanMessage is not None:
    try:
        _LLM = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            num_ctx=NUM_CTX,
            num_predict=NUM_PREDICT,
        )
        _d(f"LLM ready: {MODEL_NAME} (ctx={NUM_CTX}, predict={NUM_PREDICT}, temp={TEMPERATURE})")
    except Exception as e:
        _d("LLM init failed:", e)
        _LLM = None

def _llm_select(system_prompt: str, user_prompt: str) -> str:
    """Call the local LLM; return plain text (or empty on failure)."""
    if _LLM is None or SystemMessage is None or HumanMessage is None:
        return ""
    try:
        out = _LLM.invoke([SystemMessage(content=system_prompt),
                           HumanMessage(content=user_prompt)])
        return (getattr(out, "content", None) or str(out) or "").strip()
    except Exception as e:
        _d("LLM error:", e)
        return ""

# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
def resolve_tz(user_tz: Optional[str]) -> str:
    if user_tz and str(user_tz).strip():
        return str(user_tz).strip()
    return os.getenv("APP_TZ") or os.getenv("TZ") or "UTC"

# Map common Windows time zone IDs (Graph often returns these) → IANA.
# Add more as needed for your users.
WINDOWS_TZ_TO_IANA = {
    "UTC": "UTC",
    "Bangladesh Standard Time": "Asia/Dhaka",
    "India Standard Time": "Asia/Kolkata",
    "Nepal Standard Time": "Asia/Kathmandu",
    "Myanmar Standard Time": "Asia/Yangon",
    "SE Asia Standard Time": "Asia/Bangkok",
    "China Standard Time": "Asia/Shanghai",
    "Tokyo Standard Time": "Asia/Tokyo",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "W. Europe Standard Time": "Europe/Berlin",
    "GTB Standard Time": "Europe/Bucharest",
    "GMT Standard Time": "Europe/London",
    "Greenwich Standard Time": "Etc/GMT",
    "E. Europe Standard Time": "Europe/Kyiv",
    "Russian Standard Time": "Europe/Moscow",
    "Morocco Standard Time": "Africa/Casablanca",
    "Egypt Standard Time": "Africa/Cairo",
    "South Africa Standard Time": "Africa/Johannesburg",
    "Atlantic Standard Time": "America/Halifax",
    "Eastern Standard Time": "America/New_York",
    "Central Standard Time": "America/Chicago",
    "Mountain Standard Time": "America/Denver",
    "Pacific Standard Time": "America/Los_Angeles",
    "Alaskan Standard Time": "America/Anchorage",
    "Hawaiian Standard Time": "Pacific/Honolulu",
}

def _tz_from_windows(w: Optional[str]) -> ZoneInfo:
    if not w:
        return ZoneInfo("UTC")
    iana = WINDOWS_TZ_TO_IANA.get(w, None)
    try:
        return ZoneInfo(iana or w)
    except Exception:
        return ZoneInfo("UTC")

def _parse_graph_dt(dt_str: Optional[str], tz_hint: Optional[str]) -> Optional[datetime]:
    """
    Graph can return:
      - 'YYYY-MM-DDTHH:MM:SS[.fffffff]Z' (UTC)
      - 'YYYY-MM-DDTHH:MM:SS[.fffffff]+/-HH:MM' (offset)
      - 'YYYY-MM-DDTHH:MM:SS[.fffffff]' (naive) + separate 'timeZone' field
    Return aware **UTC** datetime or None.
    """
    if not dt_str:
        return None
    s = dt_str.strip()
    try:
        # has trailing 'Z' (UTC) → parse and keep UTC
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).astimezone(ZoneInfo("UTC"))
        # has explicit offset
        if re.search(r"[+-]\d{2}:\d{2}$", s):
            return datetime.fromisoformat(s).astimezone(ZoneInfo("UTC"))
        # naive → use tz_hint if provided (Windows or IANA), then convert to UTC
        local_tz = _tz_from_windows(tz_hint)
        local_dt = datetime.fromisoformat(s).replace(tzinfo=local_tz)
        return local_dt.astimezone(ZoneInfo("UTC"))
    except Exception:
        return None

def _fmt_date_label_local(dt_utc: datetime, tz: str) -> str:
    return dt_utc.astimezone(ZoneInfo(tz)).strftime("%a, %b %d, %Y")

def _fmt_time_range_local(start_utc: Optional[datetime], end_utc: Optional[datetime], tz: str) -> str:
    if not start_utc:
        return ""
    s = start_utc.astimezone(ZoneInfo(tz))
    e = end_utc.astimezone(ZoneInfo(tz)) if end_utc else None
    # all-day (midnight to midnight or no end)
    if s.hour == 0 and s.minute == 0 and (e is None or (e.hour == 0 and e.minute == 0)):
        return "All-day"
    if e:
        return f"{s.strftime('%I:%M %p').lstrip('0')}–{e.strftime('%I:%M %p').lstrip('0')}"
    return s.strftime("%I:%M %p").lstrip("0")

def _day_phrase_for(dt_utc: Optional[datetime], tz: str) -> str:
    """Return 'today', 'tomorrow', or weekday (e.g., 'Tuesday'), else a short date."""
    if not dt_utc:
        return ""
    loc = dt_utc.astimezone(ZoneInfo(tz))
    today = datetime.now(ZoneInfo(tz)).date()
    d = loc.date()
    if d == today:
        return "today"
    if d == (today + timedelta(days=1)):
        return "tomorrow"
    if 0 <= (d - today).days <= 6:
        return loc.strftime("%A")
    return loc.strftime("%b %d")

# -----------------------------------------------------------------------------
# Join-link extraction
# -----------------------------------------------------------------------------
_JOIN_PATTERNS = [
    r"https?://teams\.microsoft\.com/[^\s)>\]]+",
    r"https?://meet\.google\.com/[^\s)>\]]+",
    r"https?://zoom\.us/j/[^\s)>\]]+",
    r"https?://[a-zA-Z0-9.-]*zoom\.us/[^\s)>\]]+",
    r"https?://webex\.com/[^\s)>\]]+",
    r"https?://[a-zA-Z0-9.-]*webex\.com/[^\s)>\]]+",
    r"https?://meet\.jit\.si/[^\s)>\]]+",
    r"https?://join\.skype\.com/[^\s)>\]]+",
]
_JOIN_RE = re.compile("|".join(_JOIN_PATTERNS), re.IGNORECASE)

def _extract_join_link(ev: Dict) -> str:
    """
    Prefer, in order:
    1) onlineMeeting.joinUrl (Graph)
    2) onlineMeetingUrl (legacy field)
    3) Any known meeting link in bodyPreview
    4) Fall back to webLink (event page)
    """
    # 1) onlineMeeting.joinUrl
    try:
        om = ev.get("onlineMeeting") or {}
        if isinstance(om, dict):
            ju = (om.get("joinUrl") or "").strip()
            if ju:
                return ju
    except Exception:
        pass

    # 2) onlineMeetingUrl
    ju = (ev.get("onlineMeetingUrl") or "").strip()
    if ju:
        return ju

    # 3) search bodyPreview
    text = (ev.get("bodyPreview") or "").strip()
    if text:
        m = _JOIN_RE.search(text)
        if m:
            return m.group(0)

    # 4) fallback
    wl = (ev.get("webLink") or "").strip()
    return wl

# -----------------------------------------------------------------------------
# Graph fetch
# -----------------------------------------------------------------------------
def search_outlook_events(
    ms_token: str,
    query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> List[Dict]:
    """
    Search Outlook calendar events.
    1) Try $search on /me/events (subject/body/attendees)
    2) Fallback to /me/calendarView with a broad time window

    Returns rows: {subject, start, end, location, url, description, organizer, attendees}
    """
    try:
        top = min(int(limit or 10), 25)
        headers = {
            "Authorization": f"Bearer {ms_token}",
            "ConsistencyLevel": "eventual",
        }

        # --- First try: $search ---
        url = "https://graph.microsoft.com/v1.0/me/events"
        select = (
            "subject,start,end,location,webLink,bodyPreview,isCancelled,organizer,attendees,"
            "isOnlineMeeting,onlineMeeting,onlineMeetingUrl,onlineMeetingProvider"
        )
        params = {
            "$search": f"\"{query or ''}\"",
            "$select": select,
            "$orderby": "start/dateTime",
            "$top": top,
        }
        r = requests.get(url, headers=headers, params=params, timeout=20)

        # --- Fallback: calendarView (robust) ---
        if not r.ok:
            now = datetime.utcnow()
            start = time_min or (now - timedelta(days=30)).isoformat(timespec="seconds") + "Z"
            end   = time_max or (now + timedelta(days=365)).isoformat(timespec="seconds") + "Z"
            url = "https://graph.microsoft.com/v1.0/me/calendarView"
            params = {
                "startDateTime": start,
                "endDateTime": end,
                "$orderby": "start/dateTime",
                "$select": select,
                "$top": top,
            }
            r = requests.get(url, headers=headers, params=params, timeout=20)

        if not r.ok:
            _d("Graph error:", r.status_code, r.text[:200])
            return []

        items = r.json().get("value", []) or []
        out: List[Dict] = []
        for ev in items[:top]:
            if ev.get("isCancelled"):
                continue
            join_url = _extract_join_link(ev)
            out.append({
                "subject": ev.get("subject") or "(no title)",
                "start": (ev.get("start") or {}).get("dateTime"),
                "start_tz": (ev.get("start") or {}).get("timeZone"),  # <-- keep tz hint
                "end": (ev.get("end") or {}).get("dateTime"),
                "end_tz": (ev.get("end") or {}).get("timeZone"),      # <-- keep tz hint
                "location": (ev.get("location") or {}).get("displayName") or "",
                "url": join_url,  # <— join link preferred
                "weblink": ev.get("webLink") or "",  # keep Outlook page in case needed
                "description": (ev.get("bodyPreview") or "").strip(),
                "organizer": ((ev.get("organizer") or {}).get("emailAddress") or {}).get("name", ""),
                "attendees": [((a.get("emailAddress") or {}).get("name") or "") for a in (ev.get("attendees") or [])],
            })
        return out
    except Exception as e:
        _d("search_outlook_events error:", e)
        return []

# -----------------------------------------------------------------------------
# LLM instructions
# -----------------------------------------------------------------------------
_SYSTEM_CAL_SUMMARY = (
    "You are a calendar summarizer.\n"
    "INPUTS\n"
    "- USER_REQUEST: free-form text about which meetings to summarize.\n"
    "- USER_TZ: IANA time zone (e.g., \"Asia/Dhaka\").\n"
    "- EVENTS_JSON: array of events with fields: subject, start, end, location, url, description, organizer, attendees,\n"
    "  plus helper fields date_label_local, time_range_local, day_hint.\n"
    "\n"
    "TASK\n"
    "Return short, human-friendly summaries ONLY for meetings from EVENTS_JSON that satisfy USER_REQUEST.\n"
    "Do not invent meetings; do not add content not present in EVENTS_JSON.\n"
    "\n"
    "FORMAT (Markdown only)\n"
    "For each matching meeting, output:\n"
    "**[Subject](url)** (if url is '#' or empty, render as **Subject** without link).\n"
    "- When: <date_label_local> • <time_range_local>\n"
    "- Where: <Location or Online>\n"
    "- People: <Organizer; up to 5 attendees>\n"
    "- Summary: <one compact sentence from description, if any>\n"
    "\n"
    "If there are zero matches, output exactly: No events found.\n"
)

_SYSTEM_CAL_ASSISTANT = (
    "You are a friendly personal calendar assistant.\n"
    "INPUTS\n"
    "- USER_REQUEST: what the user asked (e.g., 'Do I have anything today at 2?').\n"
    "- USER_TZ: IANA zone.\n"
    "- EVENTS_JSON: events to consider (never invent anything). Each event already includes:\n"
    "  subject, url, location, date_label_local, time_range_local, day_hint.\n"
    "\n"
    "GOAL\n"
    "Write a short, conversational reply that directly answers the request.\n"
    "- If zero matches: say the user is free (e.g., 'You’re free in that window.').\n"
    "- If one match: 'You have <Subject> <day_hint> at <time_range_local> (<location>)' and include the clickable link if url is present.\n"
    "- If multiple: one or two concise sentences: how many items and a quick list like '9:00 <A>, 12:10 <B>, 3:30 <C>'.\n"
    "\n"
    "STYLE\n"
    "- Warm, helpful, concise (<= 3 sentences). No bullets, no code blocks.\n"
    "- Use local times already provided. Never fabricate details not in EVENTS_JSON.\n"
)

# -----------------------------------------------------------------------------
# Helper: enrich rows for LLM / deterministic rendering
# -----------------------------------------------------------------------------
def _enrich_rows(rows: List[Dict], tz: str) -> List[Dict]:
    enriched = []
    for ev in rows:
        s_utc = _parse_graph_dt(ev.get("start"), ev.get("start_tz"))
        e_utc = _parse_graph_dt(ev.get("end"),   ev.get("end_tz"))
        date_label = _fmt_date_label_local(s_utc, tz) if s_utc else ""
        tr = _fmt_time_range_local(s_utc, e_utc, tz)
        day_hint = _day_phrase_for(s_utc, tz)
        loc = (ev.get("location") or "").strip()
        # If empty location but we have a join link, treat as Online
        if not loc:
            loc = "Online" if (ev.get("url") or "").startswith("http") else ""
        enriched.append({
            **ev,
            "date_label_local": date_label,
            "time_range_local": tr,
            "day_hint": day_hint,
            "location": loc,
            "url": ev.get("url") or "#",
        })
    return enriched

# -----------------------------------------------------------------------------
# Fallback renderer (when LLM is unavailable)
# -----------------------------------------------------------------------------
def _fallback_render_calendar(rows: List[Dict], user_tz: str) -> str:
    if not rows:
        return "No events found."
    lines: List[str] = []
    for ev in _enrich_rows(rows, user_tz):
        subj = ev.get("subject") or "(no title)"
        url  = ev.get("url") or "#"
        meta = " • ".join([x for x in [ev["date_label_local"], ev["time_range_local"], ev.get("location","")] if x])
        if url and url != "#":
            lines.append(f"- **[{subj}]({url})** — {meta}")
        else:
            lines.append(f"- **{subj}** — {meta}")
    return "\n".join(lines) if lines else "No events found."

def _fallback_assistant(rows: List[Dict], tz: str) -> str:
    """Conversational deterministic summary when no LLM is available."""
    rows = _enrich_rows(rows, tz)
    if not rows:
        return "You’re free in that window."
    if len(rows) == 1:
        r = rows[0]
        subj = r["subject"]
        time_str = r["time_range_local"]
        day = r["day_hint"] or r["date_label_local"]
        loc = f" ({r['location']})" if r.get("location") else ""
        link = r["url"] if r["url"] and r["url"] != "#" else ""
        if link:
            return f"You have **[{subj}]({link})** {day} at {time_str}{loc}."
        return f"You have **{subj}** {day} at {time_str}{loc}."
    # multiple
    parts = []
    for r in rows[:5]:
        parts.append(f"{r['time_range_local']} — {r['subject']}")
    more = "" if len(rows) <= 5 else f" and {len(rows)-5} more"
    day = rows[0]["day_hint"] or rows[0]["date_label_local"]
    return f"You have {len(rows)} events {day}: " + "; ".join(parts) + more + "."

# -----------------------------------------------------------------------------
# Public function: structured summary (bulleted) + in-file LLM
# -----------------------------------------------------------------------------
def outlook_calendar_smart_search(
    ms_token: str,
    user_query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    user_tz: Optional[str] = None,
) -> str:
    """
    Fetches events from Outlook and asks the local LLM (if available)
    to select/format only the items that satisfy `user_query`.
    Falls back to a deterministic Markdown list if no LLM is available.
    """
    tz = resolve_tz(user_tz)
    rows = search_outlook_events(
        ms_token,
        user_query,
        limit=limit,
        time_min=time_min,
        time_max=time_max,
    )

    if not rows:
        return "No events found."

    if _LLM is None:
        # No LLM installed; deterministic pretty output
        return _fallback_render_calendar(rows, tz)

    erows = _enrich_rows(rows, tz)
    data_json = _json.dumps(erows, ensure_ascii=False)
    prompt = (
        f"USER_REQUEST: {user_query}\n"
        f"USER_TZ: {tz}\n"
        f"EVENTS_JSON:\n{data_json}\n\n"
        "Return only matching items as specified. If none, say 'No events found.'"
    )
    out = _llm_select(_SYSTEM_CAL_SUMMARY, prompt).strip()
    # Guard: if model returns empty, use fallback
    return out or _fallback_render_calendar(rows, tz)

# -----------------------------------------------------------------------------
# Public function: conversational assistant reply
# -----------------------------------------------------------------------------
def outlook_calendar_assistant_reply(
    ms_token: str,
    user_query: str,
    limit: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    user_tz: Optional[str] = None,
) -> str:
    """
    Friendly, short assistant-style answer (no bullets).
    Example: 'Hey! You have Morning Call today at 9:00 AM.'  (with a join link if present)
    """
    tz = resolve_tz(user_tz)
    rows = search_outlook_events(
        ms_token,
        user_query,
        limit=limit,
        time_min=time_min,
        time_max=time_max,
    )

    if not rows:
        return "You’re free in that window."

    # Deterministic conversational fallback
    if _LLM is None:
        return _fallback_assistant(rows, tz)

    erows = _enrich_rows(rows, tz)
    data_json = _json.dumps(erows, ensure_ascii=False)
    prompt = (
        f"USER_REQUEST: {user_query}\n"
        f"USER_TZ: {tz}\n"
        f"EVENTS_JSON:\n{data_json}\n\n"
        "Write the conversational reply now."
    )
    out = _llm_select(_SYSTEM_CAL_ASSISTANT, prompt).strip()
    return out or _fallback_assistant(rows, tz)
