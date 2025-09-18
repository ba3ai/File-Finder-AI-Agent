# services_slack.py
# ------------------------------------------------------------
# Slack helper utilities: search, filter & summarize
# - OAuth token is stored in Prefs (bot and/or user token)
# - Workspace-wide search (if user token has search:read)
# - Fallback: enumerate channels/ims and scan messages
# - Smart, assistant-style answers:
#     • who the conversation is with / channel name
#     • when (localized)
#     • deep link + web permalink
#     • short TL;DR
# ------------------------------------------------------------
from __future__ import annotations

import os
import re
import json
import time
import math
import logging as log
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import requests

# Pref keys (mirror connections_api.py)
SLACK_BOT_TOKEN   = "slack_bot_token"
SLACK_USER_TOKEN  = "slack_user_token"
SLACK_TEAM_ID     = "slack_team_id"
SLACK_TEAM_NAME   = "slack_team_name"

# ---------- helpers ----------
def _hdr(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded"}

def _get_tz() -> str:
    return os.getenv("APP_TZ", "Asia/Dhaka")

def _to_local_str(ts_float: float, tz_name: str) -> str:
    try:
        dt_utc = datetime.fromtimestamp(ts_float, tz=timezone.utc)
        if ZoneInfo:
            dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
        else:
            dt_local = dt_utc
        return dt_local.strftime("%Y-%m-%d %I:%M %p")
    except Exception:
        return datetime.utcfromtimestamp(ts_float).strftime("%Y-%m-%d %I:%M %p")

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    # strip Slack formatting for links: <http://...|label>
    s = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\\2 (\\1)", s)
    # at-mentions like <@U12345> are left as-is; frontend can decorate if desired
    return s

def _normalize_item(team_id: str, channel: Dict[str, Any], msg: Dict[str, Any], users_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ch_id   = channel.get("id")
    ch_name = channel.get("name") or channel.get("user") or "Direct message"
    ch_type = channel.get("is_im") and "im" or channel.get("is_mpim") and "mpim" or channel.get("is_private") and "private_channel" or "channel"
    ts      = float(msg.get("ts", "0"))
    text    = _clean_text(msg.get("text", ""))
    user_id = msg.get("user") or msg.get("bot_id") or ""
    user    = users_map.get(user_id, {})
    message_id = f"{ch_id}:{msg.get('ts')}"

    deep_link = f"slack://channel?team={team_id}&id={ch_id}"
    permalink = None
    try:
        # chat.getPermalink for canonical URL
        # Note: requires channels:read or groups:read for private channels
        r = requests.get("https://slack.com/api/chat.getPermalink",
                         headers=_hdr(os.getenv('SLACK_FALLBACK_TOKEN') or ""),
                         params={"channel": ch_id, "message_ts": msg.get("ts")}, timeout=15)
        if r.ok and r.json().get("ok"):
            permalink = r.json().get("permalink")
    except Exception:
        pass

    return {
        "id": message_id,
        "conversationId": ch_id,
        "conversationType": ch_type,
        "conversationName": ch_name,
        "createdDateTime": datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "text": text,
        "from": {
            "id": user_id,
            "name": user.get("real_name") or user.get("name") or "Unknown",
        },
        "permalink": permalink,
        "deepLink": deep_link,
    }

def _paged_get(url: str, token: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    cursor = None
    while True:
        q = dict(params)
        if cursor:
            q["cursor"] = cursor
        r = requests.get(url, headers=_hdr(token), params=q, timeout=30)
        if not r.ok or not r.json().get("ok"):
            break
        data = r.json()
        part = data.get("channels") or data.get("messages", {}).get("matches") or data.get("members") or data.get("files") or data.get("items") or []
        if isinstance(part, dict):
            part = list(part.values())
        items.extend(part)
        cursor = data.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        # simple rate-limit safety
        time.sleep(0.4)
    return items

def _list_channels(token: str, types: str = "public_channel,private_channel,im,mpim") -> List[Dict[str, Any]]:
    return _paged_get("https://slack.com/api/conversations.list", token, {"types": types, "limit": 200})

def _users_map(token: str) -> Dict[str, Dict[str, Any]]:
    users = _paged_get("https://slack.com/api/users.list", token, {"limit": 200})
    return {u.get("id"): u for u in users}

def _history(token: str, channel_id: str, oldest: Optional[str] = None, latest: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    params = {"channel": channel_id, "limit": limit}
    if oldest: params["oldest"] = oldest
    if latest: params["latest"]  = latest
    items: List[Dict[str, Any]] = []
    cursor = None
    while True:
        q = dict(params)
        if cursor:
            q["cursor"] = cursor
        r = requests.get("https://slack.com/api/conversations.history", headers=_hdr(token), params=q, timeout=30)
        if not r.ok or not r.json().get("ok"):
            break
        data = r.json()
        msgs = data.get("messages", []) or []
        items.extend(msgs)
        cursor = data.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.3)
    return items

def _guess_time_window(q: str) -> Tuple[Optional[str], Optional[str]]:
    qs = (q or "").lower()
    now = time.time()
    if "today" in qs:
        # Slack wants epoch seconds as strings
        start = math.floor(time.time() - (time.time() % 86400))
        return str(start), None
    if "yesterday" in qs:
        start = math.floor(time.time() - (time.time() % 86400)) - 86400
        end   = start + 86400
        return str(start), str(end)
    return None, None

# ---------- public API ----------
def slack_answer(
    token: str,
    user_query: str,
    team_id: Optional[str] = None,
    size: int = 25,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Attempt a smart answer: prefer workspace search (user token with search:read),
    else enumerate channels the token has access to and filter by keywords.
    """
    tz = _get_tz()
    tmin, tmax = time_min, time_max
    if not tmin and not tmax:
        tmin, tmax = _guess_time_window(user_query)

    # First try search.messages if the token allows it
    try:
        sr = requests.get(
            "https://slack.com/api/search.messages",
            headers=_hdr(token),
            params={"query": user_query, "count": max(20, size)},
            timeout=20,
        )
        if sr.ok and sr.json().get("ok"):
            matches = sr.json().get("messages", {}).get("matches", []) or []
            # also need users map for names
            users = _users_map(token)
            # find channel objects for names/types
            channels_idx = {c.get("id"): c for c in _list_channels(token)}
            items = []
            for m in matches[:size]:
                ch = channels_idx.get(m.get("channel", {}).get("id"), {"id": m.get("channel", {}).get("id"), "name": m.get("channel", {}).get("name"), "is_private": m.get("channel", {}).get("is_private")})
                items.append(_normalize_item(team_id or "", ch, m, users))
            text = _render_items(items, tz)
            return {"text": text or "No matching Slack messages found.", "items": items, "mode": "search.messages"}
    except Exception as e:
        log.exception("Slack search.messages failed: %s", e)

    # Fallback: enumerate channels + history and do keyword filtering
    channels = _list_channels(token)
    users    = _users_map(token)
    kw = [w for w in re.split(r"\\W+", user_query.lower()) if len(w) >= 2]
    items: List[Dict[str, Any]] = []

    for ch in channels:
        ch_id = ch.get("id")
        msgs = _history(token, ch_id, oldest=tmin, latest=tmax, limit=200)
        for m in msgs:
            text = (m.get("text") or "").lower()
            if any(k in text for k in kw):
                items.append(_normalize_item(team_id or "", ch, m, users))
        if len(items) >= size:
            break

    items = sorted(items, key=lambda it: it.get("createdDateTime", ""), reverse=True)[:size]
    text  = _render_items(items, tz)
    return {"text": text or "No matching Slack messages found.", "items": items, "mode": "scan"}

# ---------- formatting ----------
def _group_key(it: Dict[str, Any]) -> Tuple[str, str]:
    return (it.get("conversationId") or "", it.get("conversationName") or "")

def _render_items(items: List[Dict[str, Any]], tz: str) -> str:
    if not items:
        return ""
    # group: keep up to 3 per conversation
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for it in items:
        grouped.setdefault(_group_key(it), []).append(it)
    for k in grouped:
        grouped[k].sort(key=lambda x: (x.get("createdDateTime") or ""), reverse=True)
        grouped[k] = grouped[k][:3]

    blocks: List[str] = []
    for (_, name), group_items in sorted(grouped.items(), key=lambda kv: (kv[1][0].get("createdDateTime") or ""), reverse=True):
        header = f"**{name or 'Direct message'}**"
        lines = []
        for it in group_items:
            # when + snippet + link
            iso = it.get("createdDateTime") or ""
            try:
                ts = datetime.fromisoformat(iso.replace("Z","")).replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                ts = time.time()
            when = _to_local_str(ts, tz)
            snip = (it.get("text") or "").strip()
            snip = snip[:180] + ("…" if len(snip) > 180 else "")
            link = it.get("permalink") or it.get("deepLink")
            lines.append(f"- {when} — {snip}  \n  {link}")
        blocks.append(header + "\\n" + "\\n".join(lines))
    return "\\n\\n".join(blocks)
