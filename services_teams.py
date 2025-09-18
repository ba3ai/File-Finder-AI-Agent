# services_teams.py
# ------------------------------------------------------------
# Microsoft Teams helper utilities: search, filter & summarize
# - Graph v1.0 Search (chatMessage) with the correct body
# - Fallback: enumerate recent chats & messages
# - Time window (today/yesterday/this week) + ISO overrides
# - Smart, assistant-style answers:
#     • who the conversation is with
#     • when (localized)
#     • deep link to open in Teams (when available)
#     • short TL;DR
# - Lightweight intent-style filters: links-only / files-only / “from Alice”
# ------------------------------------------------------------
import os
import re
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # handled below

log = logging.getLogger("services_teams")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("TEAMS:%(levelname)s:%(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

GRAPH_V1  = "https://graph.microsoft.com/v1.0"
GRAPH_BETA = "https://graph.microsoft.com/beta"

# ─────────────────────────────────────────────────────────────
# Small name caches
_TEAM_CACHE: Dict[str, str] = {}
_CHANNEL_CACHE: Dict[Tuple[str, str], str] = {}
_CHAT_TITLE_CACHE: Dict[str, str] = {}
_CHAT_MEMBERS_CACHE: Dict[str, List[str]] = {}

# ─────────────────────────────────────────────────────────────
# HTTP helpers
def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def _get(url: str, token: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(url, headers=_auth_headers(token), params=params or {}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else {}

def _post(url: str, token: str, body: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, headers=_auth_headers(token), data=json.dumps(body), timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {url} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else {}

# ─────────────────────────────────────────────────────────────
# Time helpers
def _to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def _within_window(created_iso: str, tmin_iso: Optional[str], tmax_iso: Optional[str]) -> bool:
    if not (tmin_iso or tmax_iso):
        return True
    dt = _to_dt(created_iso)
    if not dt:
        return True
    if tmin_iso:
        mn = _to_dt(tmin_iso)
        if mn and dt < mn:
            return False
    if tmax_iso:
        mx = _to_dt(tmax_iso)
        if mx and dt > mx:
            return False
    return True

def _fmt_local_time(iso: Optional[str], tz_name: Optional[str] = None) -> str:
    if not iso:
        return ""
    tz_name = tz_name or os.getenv("APP_TZ") or "UTC"
    dt_utc = _to_dt(iso)
    if not dt_utc:
        return iso
    try:
        if ZoneInfo:
            dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
        else:  # pragma: no cover
            dt_local = dt_utc
        return dt_local.strftime("%Y-%m-%d %I:%M %p")
    except Exception:
        return dt_utc.strftime("%Y-%m-%d %I:%M %p")

# ─────────────────────────────────────────────────────────────
# LLM (local Ollama through LangChain community). Safe fallback included.
def get_local_llm():
    try:
        from langchain_community.chat_models import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "qwen2:7b")  # e.g., "llama3:8b"
        return ChatOllama(model=model, temperature=0.2)
    except Exception:
        class Dummy:
            def invoke(self, msgs):
                return type("Obj", (), {"content": msgs[-1]["content"][:900]})
        return Dummy()

def _summarize_with_llm(snippets: List[str], user_prompt: str) -> str:
    if not snippets:
        return ""
    llm = get_local_llm()
    joined = "\n".join(f"- {s}" for s in snippets[:40])
    sys = (
        "You are a concise executive assistant for Microsoft Teams.\n"
        "Summarize the messages into clear, actionable bullets.\n"
        "Prefer newest info. Mention owners, dates, and concrete next steps when obvious."
    )
    usr = (
        f"User request: {user_prompt}\n\n"
        f"Message snippets (newest first):\n{joined}\n\n"
        "Return a short TL;DR with 3–6 bullets."
    )
    try:
        out = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}])
        return getattr(out, "content", "") or ""
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────────
# Text helpers & filters
_HTML_RX = re.compile(r"<[^>]+>")
URL_RX  = re.compile(r"https?://\S+", re.I)

def _strip_html(s: str) -> str:
    return _HTML_RX.sub(" ", s or "")

def _clip(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ").strip()
    return (s[: n - 1] + "…") if len(s) > n else s

def _seems_link_text(text: str) -> bool:
    return bool(URL_RX.search(text or ""))

def _seems_file_text(text: str) -> bool:
    return bool(re.search(r"\.(pdf|docx?|xlsx?|pptx?|png|jpg|jpeg|gif|zip)\b", (text or ""), re.I))

# ─────────────────────────────────────────────────────────────
# Directory lookups + caching (for readable context lines)
def _get_team_name(token: str, team_id: Optional[str]) -> Optional[str]:
    if not team_id:
        return None
    if team_id in _TEAM_CACHE:
        return _TEAM_CACHE[team_id]
    try:
        data = _get(f"{GRAPH_V1}/teams/{team_id}", token)
        name = (data or {}).get("displayName")
        if name:
            _TEAM_CACHE[team_id] = name
        return name
    except Exception:
        return None

def _get_channel_name(token: str, team_id: Optional[str], channel_id: Optional[str]) -> Optional[str]:
    if not (team_id and channel_id):
        return None
    key = (team_id, channel_id)
    if key in _CHANNEL_CACHE:
        return _CHANNEL_CACHE[key]
    try:
        data = _get(f"{GRAPH_V1}/teams/{team_id}/channels/{channel_id}", token)
        name = (data or {}).get("displayName")
        if name:
            _CHANNEL_CACHE[key] = name
        return name
    except Exception:
        return None

def _get_chat_title(token: str, chat_id: str) -> Optional[str]:
    if not chat_id:
        return None
    if chat_id in _CHAT_TITLE_CACHE:
        return _CHAT_TITLE_CACHE[chat_id]
    try:
        data = _get(f"{GRAPH_V1}/chats/{chat_id}", token)
        topic = (data or {}).get("topic")
        if not topic:
            names = _get_chat_members(token, chat_id)
            topic = ", ".join(names[:4]) + ("…" if len(names) > 4 else "")
        if topic:
            _CHAT_TITLE_CACHE[chat_id] = topic
        return topic
    except Exception:
        return None

def _get_chat_members(token: str, chat_id: str) -> List[str]:
    if not chat_id:
        return []
    if chat_id in _CHAT_MEMBERS_CACHE:
        return _CHAT_MEMBERS_CACHE[chat_id]
    try:
        res = _get(f"{GRAPH_V1}/chats/{chat_id}/members", token, params={"$top": 50})
        names = []
        for m in (res or {}).get("value", []) or []:
            dn = m.get("displayName") or m.get("email") or ""
            if dn:
                names.append(dn)
        _CHAT_MEMBERS_CACHE[chat_id] = names
        return names
    except Exception:
        return []

def _context_label_for_item(token: str, item: Dict[str, Any]) -> str:
    ctype = item.get("conversationType")
    if ctype == "channel":
        team_id = item.get("teamId")
        chan_id = item.get("channelId")
        team_name = _get_team_name(token, team_id) or "Team"
        chan_name = _get_channel_name(token, team_id, chan_id) or "Channel"
        return f"in #{chan_name} — {team_name}"
    if ctype == "chat":
        title = item.get("chatTitle")
        if title:
            return f"with {title}"
        members = item.get("participants") or []
        if members:
            label = ", ".join(members[:4]) + ("…" if len(members) > 4 else "")
            return f"with {label}"
        return "in a chat"
    return ""

# ─────────────────────────────────────────────────────────────
# Normalizers
def _normalize_hit(hit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        res = hit.get("resource") or {}
        created = res.get("createdDateTime")
        body = (res.get("body") or {}).get("content") or ""
        author = ((res.get("from") or {}).get("user") or {}).get("displayName") or ""
        web_url = res.get("webUrl") or res.get("link")  # Graph returns webUrl in many cases
        channel_identity = (res.get("channelIdentity") or {})
        team_id = channel_identity.get("teamId")
        channel_id = channel_identity.get("channelId")

        return {
            "id": res.get("id"),
            "createdDateTime": created,
            "from": author,
            "text": _strip_html(body).strip(),
            "conversationType": "channel" if channel_id else "chat",
            "teamId": team_id,
            "channelId": channel_id,
            "chatId": None,          # not provided by search
            "chatTitle": None,
            "participants": [],
            "webUrl": web_url,
            "source": "teams",
        }
    except Exception:
        return None

def _normalize_chat_msg(m: Dict[str, Any], chat_id: str, chat_title: Optional[str], members: List[str]) -> Dict[str, Any]:
    body = (m.get("body") or {}).get("content") or ""
    author = ((m.get("from") or {}).get("user") or {}).get("displayName") or ""
    web_url = m.get("webUrl") or m.get("link")
    return {
        "id": m.get("id"),
        "createdDateTime": m.get("createdDateTime"),
        "from": author,
        "text": _strip_html(body).strip(),
        "conversationType": "chat",
        "teamId": None,
        "channelId": None,
        "chatId": chat_id,
        "chatTitle": chat_title,
        "participants": members,
        "webUrl": web_url,
        "source": "teams",
    }

# ─────────────────────────────────────────────────────────────
# Graph Search (v1.0) — correct body (no query_string/stored_fields)
def teams_search_messages(token: str, query: str, size: int = 25) -> Dict[str, Any]:
    body = {
        "requests": [{
            "entityTypes": ["chatMessage"],
            "query": {"queryString": query or "*"},
            "from": 0,
            "size": int(size),
        }]
    }
    return _post(f"{GRAPH_V1}/search/query", token, body)

# Fallback enumeration
def _list_recent_chats(token: str, top: int = 10) -> List[Dict[str, Any]]:
    data = _get(f"{GRAPH_V1}/me/chats", token, params={"$top": top})
    return data.get("value", []) or []

def _list_chat_messages(token: str, chat_id: str, top: int = 50) -> List[Dict[str, Any]]:
    data = _get(f"{GRAPH_BETA}/chats/{chat_id}/messages", token, params={"$top": top})
    return data.get("value", []) or []

# ─────────────────────────────────────────────────────────────
# Query analyzer (very light NLP)
def _analyze_query(q: str) -> Dict[str, Any]:
    ql = (q or "").lower()

    # Time windows
    now = datetime.now(timezone.utc)
    today_start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    week_start  = today_start - timedelta(days=today_start.weekday())  # Monday start

    time_min = time_max = None
    if "today" in ql or "latest" in ql:
        time_min = today_start.isoformat().replace("+00:00", "Z")
        time_max = None
    elif "yesterday" in ql:
        y0 = (today_start - timedelta(days=1))
        y1 = today_start - timedelta(microseconds=1)
        time_min = y0.isoformat().replace("+00:00", "Z")
        time_max = y1.isoformat().replace("+00:00", "Z")
    elif "this week" in ql:
        time_min = week_start.isoformat().replace("+00:00", "Z")
        time_max = None

    # Filters
    only_links = ("link" in ql or "links" in ql)
    only_files = ("file" in ql or "files" in ql or "document" in ql or "attachment" in ql)

    # “from Alice/Bob” naive capture
    m = re.search(r"(from|by)\s+([A-Za-z0-9._ -]{2,})", ql)
    who = []
    if m:
        cand = m.group(2).strip().strip(".!,?")
        if cand:
            who = [cand]

    return {
        "time_min": time_min,
        "time_max": time_max,
        "only_links": only_links,
        "only_files": only_files,
        "from_people": who,
    }

def _apply_filters(items: List[Dict[str, Any]], flt: Dict[str, Any]) -> List[Dict[str, Any]]:
    who = [w.lower() for w in (flt.get("from_people") or [])]
    only_links = bool(flt.get("only_links"))
    only_files = bool(flt.get("only_files"))

    out = []
    for it in items:
        txt = it.get("text") or ""
        author = (it.get("from") or "").lower()

        if who and not any(w in author for w in who):
            continue
        if only_links and not _seems_link_text(txt):
            continue
        if only_files and not (_seems_file_text(txt) or _seems_link_text(txt)):
            continue
        out.append(it)
    return out

# ─────────────────────────────────────────────────────────────
# Search → normalize → fallback
def teams_search_advanced(
    token: str,
    query: str,
    size: int = 25,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> Dict[str, Any]:
    flt = _analyze_query(query)
    time_min = time_min or flt.get("time_min")
    time_max = time_max or flt.get("time_max")

    # 1) Graph Search
    try:
        raw = teams_search_messages(token, query, size=size)
        containers = (raw.get("value") or [])[0].get("hitsContainers", []) if raw.get("value") else []
        hits = []
        for c in containers:
            hits.extend(c.get("hits", []) or [])
        items: List[Dict[str, Any]] = []
        for h in hits:
            n = _normalize_hit(h)
            if not n or not n.get("text"):
                continue
            if not _within_window(n.get("createdDateTime"), time_min, time_max):
                continue
            items.append(n)
        if items:
            items = _apply_filters(items, flt)
            items.sort(key=lambda x: (x.get("createdDateTime") or ""), reverse=True)
            return {"items": items[:size], "mode": "search"}
        log.info("Search returned 0 items; will try fallback.")
    except Exception as e:
        log.warning("Teams Search API failed; will try fallback. %s", e)

    # 2) Fallback: enumerate recent chats & messages
    try:
        chats = _list_recent_chats(token, top=10)
        collected: List[Dict[str, Any]] = []
        for ch in chats:
            cid = ch.get("id")
            if not cid:
                continue
            title = _get_chat_title(token, cid)
            members = _get_chat_members(token, cid)
            msgs = _list_chat_messages(token, cid, top=50)
            for m in msgs:
                n = _normalize_chat_msg(m, cid, title, members)
                if not n["text"]:
                    continue
                if not _within_window(n.get("createdDateTime"), time_min, time_max):
                    continue
                collected.append(n)
        collected = _apply_filters(collected, flt)
        collected.sort(key=lambda x: (x.get("createdDateTime") or ""), reverse=True)
        return {"items": collected[:size], "mode": "fallback"}
    except Exception as e:
        log.exception("Fallback enumeration failed: %s", e)
        return {"items": [], "mode": "error"}

# ─────────────────────────────────────────────────────────────
# Assistant-style formatting (group by conversation)
def _group_key(it: Dict[str, Any]) -> Tuple:
    if it.get("conversationType") == "channel":
        return ("channel", it.get("teamId"), it.get("channelId"))
    return ("chat", it.get("chatId") or "search-chat", it.get("chatTitle") or "")

def _format_group_header(token: str, items: List[Dict[str, Any]]) -> str:
    one = items[0]
    if one.get("conversationType") == "channel":
        team = _get_team_name(token, one.get("teamId")) or "Team"
        chan = _get_channel_name(token, one.get("teamId"), one.get("channelId")) or "Channel"
        return f"**{team} › #{chan}**"
    # chat
    label = _context_label_for_item(token, one).replace("with ", "")
    return f"**Chat with {label}**"

def _format_group_lines(token: str, items: List[Dict[str, Any]], tz: Optional[str]) -> List[str]:
    tz_name = tz or os.getenv("APP_TZ") or "UTC"
    lines = []
    for it in items:
        when = _fmt_local_time(it.get("createdDateTime"), tz_name)
        author = it.get("from") or "Unknown"
        snippet = _clip(it.get("text") or "", 160)
        link = it.get("webUrl")
        if link:
            lines.append(f"- {when} — {author}: “{snippet}”  [Open]({link})")
        else:
            lines.append(f"- {when} — {author}: “{snippet}”")
    return lines

def _format_items_for_user(token: str, items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    # group by conversation, keep most recent 3 per conversation
    grouped: Dict[Tuple, List[Dict[str, Any]]] = {}
    for it in items:
        grouped.setdefault(_group_key(it), []).append(it)
    for k in grouped:
        grouped[k].sort(key=lambda x: (x.get("createdDateTime") or ""), reverse=True)
        grouped[k] = grouped[k][:3]

    # render
    blocks: List[str] = []
    for _, group_items in sorted(grouped.items(), key=lambda kv: (kv[1][0].get("createdDateTime") or ""), reverse=True):
        header = _format_group_header(token, group_items)
        lines  = _format_group_lines(token, group_items, tz=os.getenv("APP_TZ"))
        blocks.append(header + "\n" + "\n".join(lines))
    return "\n\n".join(blocks)

# ─────────────────────────────────────────────────────────────
# Public entry points
def teams_answer(
    token: str,
    user_query: str,
    size: int = 25,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> Dict[str, Any]:
    log.info("Teams answer: q=%r size=%s tmin=%s tmax=%s", user_query, size, time_min, time_max)
    res = teams_search_advanced(token, user_query, size=size, time_min=time_min, time_max=time_max)
    items = res.get("items", []) or []
    if not items:
        return {"text": "I couldn’t find any Teams messages matching that request.", "items": [], "mode": res.get("mode")}

    # Nicely grouped list + TL;DR
    grouped_view = _format_items_for_user(token, items[: min(20, size)])
    summary = _summarize_with_llm([it.get("text") or "" for it in items], user_query)

    out = "Here’s what I found:\n\n" + grouped_view
    if summary:
        out += "\n\n**TL;DR:**\n" + summary.strip()

    return {"text": out, "items": items, "mode": res.get("mode")}

# Kept for backward compatibility
def teams_router(
    token: str,
    q: str,
    size: int = 25,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> Dict[str, Any]:
    return teams_answer(token, q, size=size, time_min=time_min, time_max=time_max)
