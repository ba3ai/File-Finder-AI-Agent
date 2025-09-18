# mail_all.py
# -----------------------------------------------------------------------------
# Smarter-than-ChatGPT email search:
# - strict, timezone-correct windows
# - provider-first filtering
# - parallel LLM map/reduce
# - digest renderer + ASSISTANT voice renderer (personal tone)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import sys
import json
import time
import logging
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, date, timezone as _tz
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────────────────────────────────────
# Logging / Debug
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_DEBUG = os.getenv("OPENAI_API_DEBUG", "0") not in ("0", "false", "False", "")
EMAIL_DEBUG_LEVEL = int(os.getenv("EMAIL_DEBUG_LEVEL", "1"))  # 1..3
EMAIL_DEBUG_ECHO  = os.getenv("EMAIL_DEBUG_ECHO", "0") not in ("0", "false", "False", "")

_logger = logging.getLogger("mail_all")
if not _logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[mail_all] %(message)s"))
    _logger.addHandler(h)
_logger.setLevel(logging.INFO if OPENAI_API_DEBUG else logging.WARNING)
_logger.propagate = False

def _d(level: int, *a):
    if level <= EMAIL_DEBUG_LEVEL:
        _logger.info(" ".join(str(x) for x in a))

def _djson(level: int, label: str, data):
    if level <= EMAIL_DEBUG_LEVEL:
        try:
            s = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            s = str(data)
        _logger.info(f"{label}: {s}")

# ──────────────────────────────────────────────────────────────────────────────
# Config caps / threads
# ──────────────────────────────────────────────────────────────────────────────
MAX_PER_PROVIDER_DEFAULT = int(os.getenv("EMAIL_READ_MAX_PER_PROVIDER", "400"))
CHUNK_DEFAULT            = int(os.getenv("EMAIL_READ_CHUNK", "50"))
PROVIDER_THREADS         = int(os.getenv("EMAIL_THREADS_PROVIDER", "2"))
GMAIL_MSG_THREADS        = int(os.getenv("GMAIL_THREADS_MESSAGES", "8"))
LLM_MAP_THREADS          = int(os.getenv("LLM_THREADS_MAP", "4"))

# Personal assistant toggle
EMAIL_PERSONAL_VOICE = os.getenv("EMAIL_PERSONAL_VOICE", "0").lower() not in ("0", "false", "no")

# ──────────────────────────────────────────────────────────────────────────────
# LLM (LangChain + Ollama) — local, no URL required
# ──────────────────────────────────────────────────────────────────────────────
ChatOllama = None
try:
    from langchain_ollama import ChatOllama  # preferred
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama  # fallback
    except Exception:
        ChatOllama = None

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    try:
        from langchain.schema import SystemMessage, HumanMessage
    except Exception:
        SystemMessage = HumanMessage = None

MODEL_NAME  = os.getenv("OLLAMA_MODEL", "qwen2:7b")
NUM_CTX     = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "800"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

_LLM = None
if ChatOllama is not None:
    try:
        _LLM = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            num_ctx=NUM_CTX,
            num_predict=NUM_PREDICT,
        )
        _d(1, f"LLM ready: model={MODEL_NAME}, ctx={NUM_CTX}, predict={NUM_PREDICT}, temp={TEMPERATURE}")
    except Exception as e:
        _d(1, "LLM init failed:", e)
        _LLM = None

def _llm_select(system_prompt: str, user_prompt: str) -> str:
    if _LLM is None or SystemMessage is None or HumanMessage is None:
        _d(1, "LLM unavailable; returning empty selection")
        return ""
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    t0 = time.perf_counter()
    try:
        out = _LLM.invoke(msgs)
        txt = (getattr(out, "content", None) or str(out) or "").strip()
        _d(2, f"LLM invoke ok in {time.perf_counter()-t0:.2f}s; out_len={len(txt)}")
        return txt
    except Exception as e:
        _d(1, "LLM invoke error:", e)
        return ""

# ──────────────────────────────────────────────────────────────────────────────
# Timezone & windows
# ──────────────────────────────────────────────────────────────────────────────
def resolve_tz(user_tz: Optional[str] = None) -> str:
    if user_tz and str(user_tz).strip():
        return user_tz.strip()
    env_tz = os.getenv("APP_TZ") or os.getenv("TZ")
    if env_tz:
        return env_tz
    try:
        import tzlocal
        nm = tzlocal.get_localzone_name()
        if nm:
            return nm
    except Exception:
        pass
    return "UTC"

def _as_utc_iso(dt_local: datetime) -> str:
    return dt_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

def window_for_day(day_local: date, user_tz: str) -> Tuple[str, str]:
    tz = ZoneInfo(user_tz)
    start_local = datetime(day_local.year, day_local.month, day_local.day, 0, 0, 0, tzinfo=tz)
    end_local   = start_local + timedelta(days=1)
    return _as_utc_iso(start_local), _as_utc_iso(end_local)

def window_today(user_tz: str) -> Tuple[str, str]:
    now_local = datetime.now(ZoneInfo(user_tz))
    return window_for_day(now_local.date(), user_tz)

def gmail_after_before_from_iso(time_min_iso: str, time_max_iso: str, user_tz: str) -> Tuple[str, str]:
    tz = ZoneInfo(user_tz)
    tmin_utc = datetime.strptime(time_min_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
    tmax_utc = datetime.strptime(time_max_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
    start_local = tmin_utc.astimezone(tz)
    end_local   = tmax_utc.astimezone(tz)
    after  = start_local.strftime("%Y/%m/%d")
    before = end_local.strftime("%Y/%m/%d")  # exclusive
    return after, before

def normalize_window_strict(tmin_iso: Optional[str], tmax_iso: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    def _parse_any(z: str) -> datetime:
        if "." in z:
            return datetime.strptime(z, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=ZoneInfo("UTC"))
        return datetime.strptime(z, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
    out_min = None
    out_max = None
    try:
        if tmin_iso:
            dt = _parse_any(tmin_iso).replace(microsecond=0)
            out_min = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if tmax_iso:
            dt = _parse_any(tmax_iso)
            dt = (dt + timedelta(seconds=1)).replace(microsecond=0)  # strict exclusive end
            out_max = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return tmin_iso, tmax_iso
    return out_min, out_max

def parse_relative_window(query: str, user_tz: str) -> Tuple[Optional[str], Optional[str]]:
    q = (query or "").lower().strip()
    tz = ZoneInfo(user_tz)
    now = datetime.now(tz)

    m = re.search(r"(\d{4}-\d{2}-\d{2})\D+(to|-|through|until)\D+(\d{4}-\d{2}-\d{2})", q)
    if m:
        d1 = date.fromisoformat(m.group(1)); d2 = date.fromisoformat(m.group(3))
        start_iso, _ = window_for_day(min(d1, d2), user_tz)
        _, end_iso   = window_for_day(max(d1, d2), user_tz)
        _d(1, f"Window rule: explicit range → {start_iso} .. {end_iso}")
        return start_iso, end_iso

    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
    if m:
        d = date.fromisoformat(m.group(1))
        s, e = window_for_day(d, user_tz)
        _d(1, f"Window rule: explicit date → {s} .. {e}")
        return s, e

    if "today" in q:
        s, e = window_today(user_tz); _d(1, f"Window rule: today → {s} .. {e}"); return s, e
    if "yesterday" in q:
        y = (now - timedelta(days=1)).date(); s, e = window_for_day(y, user_tz)
        _d(1, f"Window rule: yesterday → {s} .. {e}"); return s, e
    if any(x in q for x in ("last 7 days", "past 7 days", "last seven days")):
        start = (now - timedelta(days=7)).date(); end = now.date()
        s, _ = window_for_day(start, user_tz); _, e = window_for_day(end, user_tz)
        _d(1, f"Window rule: last7 → {s} .. {e}"); return s, e
    if "this week" in q:
        start = (now - timedelta(days=now.weekday())).date(); end = start + timedelta(days=7)
        s, _ = window_for_day(start, user_tz); _, e = window_for_day(end, user_tz)
        _d(1, f"Window rule: this week → {s} .. {e}"); return s, e
    if "last week" in q:
        start = (now - timedelta(days=now.weekday()+7)).date(); end = start + timedelta(days=7)
        s, _ = window_for_day(start, user_tz); _, e = window_for_day(end, user_tz)
        _d(1, f"Window rule: last week → {s} .. {e}"); return s, e
    if "this month" in q:
        start = date(now.year, now.month, 1)
        nm = date(now.year+1, 1, 1) if now.month == 12 else date(now.year, now.month+1, 1)
        s, _ = window_for_day(start, user_tz); _, e = window_for_day(nm, user_tz)
        _d(1, f"Window rule: this month → {s} .. {e}"); return s, e

    _d(1, "Window rule: none (will use default later)")
    return None, None

# ──────────────────────────────────────────────────────────────────────────────
# Query helpers (tokens, flags)
# ──────────────────────────────────────────────────────────────────────────────
STOP = {
    "the","a","an","of","to","for","in","on","at","and","or","with","me","my",
    "today","yesterday","this","week","month","email","emails","inbox","show","list",
    "give","from","subject","need","want","find","please","hi","hello","thanks","thank",
    "could","would","can","share","provide","send","get","please,"
}

def _tokens(q: str) -> list[str]:
    pat = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|[A-Za-z0-9']+"
    return [t for t in re.findall(pat, (q or "").lower())]

def _sender_terms(q: str) -> list[str]:
    out = []
    for tok in _tokens(q):
        if tok in STOP:
            continue
        if "@" in tok or len(tok) >= 3:
            out.append(tok)
    uniq = list(dict.fromkeys(out))[:4]
    _d(2, "Sender terms:", uniq)
    return uniq

def _subject_phrase(q: str) -> str:
    words = [w for w in _tokens(q) if "@" not in w and w not in STOP and len(w) >= 3]
    phrase = " ".join(words)[:60]
    _d(2, "Subject phrase:", phrase)
    return phrase

def _flags(q: str) -> dict:
    ql = (q or "").lower()
    f = {
        "unread": any(w in ql for w in ("unread", "is:unread")),
        "attach": any(w in ql for w in ("attachment", "attachments", "has:attachment")),
    }
    _d(2, "Flags:", f)
    return f

# ──────────────────────────────────────────────────────────────────────────────
# Outlook search hints — sender emails → $filter, names join $search
# ──────────────────────────────────────────────────────────────────────────────
def _outlook_build_filters_and_search(q: str) -> tuple[Optional[str], Optional[str]]:
    sender_emails = [t for t in _sender_terms(q) if "@" in t]
    sender_names  = [t for t in _sender_terms(q) if "@" not in t]
    subject = _subject_phrase(q)
    flags   = _flags(q)

    fil_parts = []
    if flags["unread"]:
        fil_parts.append("isRead eq false")
    if flags["attach"]:
        fil_parts.append("hasAttachments eq true")
    for em in sender_emails:
        fil_parts.append(f"from/emailAddress/address eq '{em}'")

    search_terms = " ".join([p for p in [subject] + sender_names if p]).strip()
    extra_filter  = " and ".join(fil_parts) if fil_parts else None
    search_phrase = search_terms or None

    if search_phrase and (len(search_phrase) < 3 or search_phrase in {"please","thanks","hello"}):
        search_phrase = None

    _d(1, "Outlook hints → filter:", extra_filter, "| search:", search_phrase)
    return extra_filter, search_phrase

# ──────────────────────────────────────────────────────────────────────────────
# Gmail query
# ──────────────────────────────────────────────────────────────────────────────
def _gmail_build_query(q: str) -> str:
    senders = _sender_terms(q)
    subject = _subject_phrase(q)
    flags   = _flags(q)

    parts = ["in:anywhere"]
    for s in senders:
        parts.append(f"from:{s}")
    if subject:
        parts.append(f"subject:{subject}")
        parts.append(f"\"{subject}\"")
    if flags["unread"]:
        parts.append("is:unread")
    if flags["attach"]:
        parts.append("has:attachment")
    qstr = " ".join(parts).strip()[:250]
    _d(1, "Gmail q=", qstr)
    return qstr

# ──────────────────────────────────────────────────────────────────────────────
# Outlook fetch (Graph) — strict end (lt), nextLink paging
# ──────────────────────────────────────────────────────────────────────────────
def fetch_outlook_all(
    ms_token: str,
    max_total: int = MAX_PER_PROVIDER_DEFAULT,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    extra_filter: Optional[str] = None,
    search_phrase: Optional[str] = None,
) -> List[Dict]:
    base_url = "https://graph.microsoft.com/v1.0/me/messages"

    def _params(with_search: bool) -> dict:
        p = {
            "$orderby": "receivedDateTime desc",
            "$select": "id,subject,from,receivedDateTime,webLink,bodyPreview,internetMessageId,isRead,hasAttachments",
            "$top": 50,
        }
        fil = []
        if time_min: fil.append(f"receivedDateTime ge {time_min}")
        if time_max: fil.append(f"receivedDateTime lt {time_max}")  # STRICT end
        if extra_filter: fil.append(extra_filter)
        if fil: p["$filter"] = " and ".join(fil)
        if with_search and search_phrase:
            p["$search"] = f"\"{search_phrase}\""
        return p

    headers = {
        "Authorization": f"Bearer {ms_token}",
        "ConsistencyLevel": "eventual",
    }

    rows: List[Dict] = []
    tried_without_search = False

    url = base_url
    params = _params(with_search=True)
    next_page_url: Optional[str] = None
    page = 0

    while True:
        page += 1
        request_url = next_page_url or url
        request_params = None if next_page_url else params

        t0 = time.perf_counter()
        r = requests.get(request_url, headers=headers, params=request_params, timeout=30)
        dt = time.perf_counter() - t0

        if not r.ok:
            _d(1, f"Outlook page{page} HTTP {r.status_code} in {dt:.2f}s")
            if not tried_without_search and search_phrase:
                tried_without_search = True
                url = base_url
                params = _params(with_search=False)
                next_page_url = None
                _d(1, "Graph rejected $search + $filter; retrying without $search")
                continue
            _d(2, "Outlook error body:", r.text[:300])
            return rows

        data = r.json()
        got = len(data.get("value", []))
        _d(1, f"Outlook page{page}: got={got} in {dt:.2f}s")
        for it in data.get("value", []):
            fobj = (it.get("from") or {}).get("emailAddress") or {}
            sender_name = (fobj.get("name") or "").strip()
            sender_addr = (fobj.get("address") or "").strip()
            sender = f"{sender_name} <{sender_addr}>" if (sender_name or sender_addr) else ""
            rows.append({
                "id": it.get("id"),
                "provider": "outlook",
                "subject": it.get("subject") or "(no subject)",
                "from": sender,
                "date": it.get("receivedDateTime"),
                "url": it.get("webLink"),
                "preview": it.get("bodyPreview") or "",
                "mid": it.get("internetMessageId") or "",
            })
            if len(rows) >= max_total:
                _d(1, f"Outlook cap reached ({len(rows)})")
                return rows

        next_page_url = data.get("@odata.nextLink")
        if not next_page_url:
            break

    _d(1, f"Outlook total rows={len(rows)}")
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Gmail fetch — metadata only + after/before window
# ──────────────────────────────────────────────────────────────────────────────
def _gmail_header(headers, name: str, default: str = "") -> str:
    for h in headers or []:
        if (h.get("name") or "").lower() == name.lower():
            return h.get("value") or default
    return default

def _gmail_get_message(token: str, msg_id: str) -> Dict:
    u = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}"
    try:
        t0 = time.perf_counter()
        r = requests.get(
            u,
            headers={"Authorization": f"Bearer {token}"},
            params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
            timeout=30,
        )
        _d(3, f"Gmail get {msg_id} in {time.perf_counter()-t0:.2f}s (ok={r.ok})")
        if not r.ok:
            return {}
        j = r.json()
        headers = (j.get("payload") or {}).get("headers") or []
        subject = _gmail_header(headers, "Subject", "(no subject)")
        frm     = _gmail_header(headers, "From", "")
        dte     = _gmail_header(headers, "Date", "")
        web     = f"https://mail.google.com/mail/u/0/#all/{msg_id}"
        return {
            "id": msg_id,
            "provider": "gmail",
            "subject": subject,
            "from": frm,
            "date": dte,
            "url": web,
            "preview": j.get("snippet") or "",
        }
    except Exception:
        return {}

def fetch_gmail_all(
    google_access_token: str,
    max_total: int = MAX_PER_PROVIDER_DEFAULT,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    user_tz: str = "UTC",
    base_query: Optional[str] = None,
) -> List[Dict]:
    base = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    headers = {"Authorization": f"Bearer {google_access_token}"}
    q_parts: List[str] = []

    if base_query:
        q_parts.append(base_query)

    if time_min and time_max:
        after, before = gmail_after_before_from_iso(time_min, time_max, user_tz)
        q_parts += [f"after:{after}", f"before:{before}"]
    elif time_min:
        after, _ = gmail_after_before_from_iso(time_min, _as_utc_iso(datetime.now(ZoneInfo(user_tz))), user_tz)
        q_parts += [f"after:{after}"]
    elif time_max:
        _, before = gmail_after_before_from_iso(
            _as_utc_iso(datetime.now(ZoneInfo(user_tz)) - timedelta(days=365*5)), time_max, user_tz
        )
        q_parts += [f"before:{before}"]

    params = {"q": " ".join(q_parts).strip(), "maxResults": 500, "includeSpamTrash": False}
    _djson(1, "Gmail params", params)

    rows: List[Dict] = []
    page = 0
    r = requests.get(base, headers=headers, params=params, timeout=30)
    while r.ok:
        page += 1
        j = r.json()
        ids = [m["id"] for m in j.get("messages", [])]
        _d(1, f"Gmail page{page}: ids={len(ids)}")

        with ThreadPoolExecutor(max_workers=max(1, GMAIL_MSG_THREADS)) as ex:
            futs = [ex.submit(_gmail_get_message, google_access_token, mid) for mid in ids]
            for fu in as_completed(futs):
                one = fu.result()
                if one:
                    rows.append(one)
                if len(rows) >= max_total:
                    _d(1, f"Gmail cap reached ({len(rows)})")
                    return rows

        tok = j.get("nextPageToken")
        if not tok:
            break
        r = requests.get(base, headers=headers, params={**params, "pageToken": tok}, timeout=30)
    if not r.ok:
        _d(1, "Gmail error:", r.status_code, r.text[:200])

    _d(1, f"Gmail total rows={len(rows)}")
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Unified fetch (providers in parallel)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_all_emails(
    sources: List[str],
    ms_token: Optional[str],
    g_token: Optional[str],
    max_total_each: int = MAX_PER_PROVIDER_DEFAULT,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    user_tz: str = "UTC",
    keywords: Optional[str] = None,
) -> List[Dict]:

    extra_filter, outlook_search = _outlook_build_filters_and_search(keywords or "")
    gmail_q = _gmail_build_query(keywords or "")

    # Critical: avoid $search when a window is provided (Graph can ignore filter)
    if time_min or time_max:
        outlook_search = None

    rows: List[Dict] = []

    def _job_outlook():
        if "outlook" in sources and ms_token:
            _d(1, "Outlook window:", time_min, time_max)
            return fetch_outlook_all(
                ms_token,
                max_total=max_total_each,
                time_min=time_min,
                time_max=time_max,
                extra_filter=extra_filter,
                search_phrase=outlook_search,
            )
        return []

    def _job_gmail():
        if "gmail" in sources and g_token:
            _d(1, "Gmail window:", time_min, time_max)
            return fetch_gmail_all(
                g_token,
                max_total=max_total_each,
                time_min=time_min,
                time_max=time_max,
                user_tz=user_tz,
                base_query=gmail_q,
            )
        return []

    with ThreadPoolExecutor(max_workers=max(1, PROVIDER_THREADS)) as ex:
        jobs = []
        if "outlook" in sources and ms_token:
            jobs.append(ex.submit(_job_outlook))
        if "gmail" in sources and g_token:
            jobs.append(ex.submit(_job_gmail))

        for fu in as_completed(jobs):
            try:
                rows.extend(fu.result() or [])
            except Exception as e:
                _d(1, "Provider job error:", e)

    _d(1, f"Unified fetch rows={len(rows)}")
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# LLM map/reduce
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_PICK = (
    "You are an expert email router. Your task: given a USER REQUEST, an optional WINDOW "
    "(time_min, time_max, user_tz), and a JSON array EMAILS, select ONLY the messages that "
    "best satisfy the user.\n\n"
    "STRICT RULES\n"
    "1) Never invent emails; only use items present in EMAILS.\n"
    "2) If a time window is provided, KEEP items whose timestamp is within [time_min, time_max) in UTC.\n"
    "3) Prefer exact sender matches, explicit date windows, and strong subject/preview keyword matches.\n"
    "4) Deduplicate within this chunk by (Subject + URL); keep the newest.\n"
    "5) Sort newest → oldest by the email date.\n"
    "6) OUTPUT STRICTLY a Markdown bullet list, one item per line, exactly:\n"
    "   - [Subject](url) — From (Date)\n"
    "   (If URL is missing, use (#) as the link target.)\n"
    "7) If no items in this chunk match, output exactly: NONE\n"
    "8) Do not add any other text before or after the list."
)

_SYSTEM_MERGE = (
    "You merge several short Markdown lists of emails into ONE list.\n"
    "Each input line follows: - [Subject](url) — From (Date)\n\n"
    "RULES\n"
    "1) Keep only items that appear in the input lists (no invention).\n"
    "2) Deduplicate by (Subject + URL); keep the newest item.\n"
    "3) Sort newest → oldest using the date at the end of each line.\n"
    "4) OUTPUT MUST BE ONLY a Markdown bullet list, one '- ' item per line, same format:\n"
    "   - [Subject](url) — From (Date)\n"
    "5) If AND ONLY IF there are zero items, output exactly: No messages found.\n"
    "6) Never append any extra text, headings, or explanations."
)

# Neutral digest (previous behavior)
_SYSTEM_DIGEST = (
    "You render a clean email digest in Markdown from a single merged list where each line is:\n"
    "- [Subject](url) — From (Date)\n"
    "You are also given:\n"
    "- header_date_local: the date string to show in the header\n"
    "- user_tz: IANA time zone name\n"
    "- OPTIONAL rows JSON that may contain 'preview' and 'local_time'\n\n"
    "RENDERING RULES\n"
    "1) If there are zero items, output exactly: No messages found.\n"
    "2) Otherwise, produce a digest like:\n"
    "   Here’s your **email list for** (header_date_local):\n"
    "   1.  **Subject**\n"
    "       From: From\n"
    "       Time: <local_time if present; else echo Date as-is>\n"
    "       Snippet: <preview trimmed to ≤240 chars; single line>   (omit if not available)\n"
    "       [[open]](url)                                           (omit if url is (#) or empty)\n"
    "   <blank line between items>\n"
    "3) If exactly one item is listed, append: That’s the only message delivered to your inbox.\n"
    "4) No extra commentary before or after the digest."
)

# New: Personal assistant style renderer
_SYSTEM_ASSISTANT = (
    "You are the user's friendly personal email assistant. Use a natural, upbeat tone.\n"
    "INPUTS:\n"
    "- USER_REQUEST (what they asked)\n"
    "- header_date_local (label like 'Wed, Sep 03, 2025')\n"
    "- MERGED_LIST (lines like '- [Subject](url) — From (Date)')\n"
    "- EMAILS_JSON (subset objects with subject, from, url, preview, local_time)\n"
    "- USER_TZ (IANA zone)\n\n"
    "GOAL:\n"
    "Write a short conversational answer describing the results for that date/window.\n"
    "Start with a helpful sentence (e.g., 'You’ve got two messages today…').\n"
    "Mention key times in LOCAL time if provided (local_time). If a single message, be specific.\n"
    "Include quick inline links as '[[open]](url)' for each item mentioned. Do not dump long metadata.\n"
    "Never invent emails; only refer to items present in MERGED_LIST / EMAILS_JSON.\n"
    "Keep it under ~6 short sentences. No headings or code blocks.\n"
    "If there are zero items, reply politely with 'I didn’t find anything for that period.'"
)

def _rows_used_in_merged(merged_md: str, rows: List[Dict]) -> List[Dict]:
    keys = set()
    for line in (merged_md or "").splitlines():
        m = re.match(r"\s*-\s*\[(?P<subj>.+?)\]\((?P<url>.*?)\)\s+—\s+.*\((?P<date>.+?)\)\s*$", line)
        if m:
            keys.add((m.group("subj").strip(), m.group("url").strip()))
    out = []
    for r in rows:
        k = ((r.get("subject") or "").strip(), (r.get("url") or "").strip())
        if k in keys:
            out.append(r)
    return out

def _parse_iso_utc(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(_tz.utc)

def _fmt_header_date_from_window(tmin_iso: str, user_tz: str) -> str:
    dt_local = _parse_iso_utc(tmin_iso).astimezone(ZoneInfo(user_tz))
    return dt_local.strftime("%a, %b %d, %Y")

def _fmt_local_time(utc_iso: str, user_tz: str) -> str:
    try:
        dt_local = _parse_iso_utc(utc_iso).astimezone(ZoneInfo(user_tz))
        return dt_local.strftime("%I:%M %p")
    except Exception:
        return ""

def _llm_digest(merged_md: str, user_query: str, time_min: Optional[str], time_max: Optional[str],
                user_tz: str, rows: List[Dict]) -> str:
    if not merged_md or merged_md.strip().lower() == "no messages found.":
        return "No messages found."
    used_rows = _rows_used_in_merged(merged_md, rows)
    compact = []
    for r in used_rows:
        local_time = ""
        dt = r.get("date") or ""
        if dt:
            local_time = _fmt_local_time(dt, user_tz)
        compact.append({
            "subject": r.get("subject"),
            "from": r.get("from"),
            "date": r.get("date"),
            "url": r.get("url"),
            "preview": (r.get("preview") or "")[:500],
            "local_time": local_time,
        })
    header_date_local = _fmt_header_date_from_window(time_min or _as_utc_iso(datetime.now(_tz.utc)), user_tz)
    payload = json.dumps(compact, ensure_ascii=False)
    prompt = (
        f"header_date_local: {header_date_local}\n"
        f"user_tz: {user_tz}\n"
        f"MERGED LIST (each line '- [Subject](url) — From (Date)'):\n{merged_md}\n\n"
        f"OPTIONAL EMAILS JSON (subset for snippets/local time):\n{payload}\n\n"
        "Render the final digest exactly per the rules."
    )
    out = _llm_select(_SYSTEM_DIGEST, prompt).strip()
    return out or merged_md

def _llm_assistant(merged_md: str, user_query: str, time_min: Optional[str],
                   user_tz: str, rows: List[Dict]) -> str:
    if not merged_md or merged_md.strip().lower() == "no messages found.":
        return "I didn’t find anything for that period."
    used_rows = _rows_used_in_merged(merged_md, rows)
    compact = []
    for r in used_rows:
        local_time = _fmt_local_time(r.get("date") or "", user_tz) if r.get("date") else ""
        compact.append({
            "subject": r.get("subject"),
            "from": r.get("from"),
            "url": r.get("url"),
            "preview": (r.get("preview") or "")[:300],
            "local_time": local_time,
        })
    header_date_local = _fmt_header_date_from_window(time_min or _as_utc_iso(datetime.now(_tz.utc)), user_tz)
    payload = json.dumps(compact, ensure_ascii=False)
    prompt = (
        f"USER_REQUEST: {user_query}\n"
        f"header_date_local: {header_date_local}\n"
        f"USER_TZ: {user_tz}\n"
        f"MERGED_LIST:\n{merged_md}\n\n"
        f"EMAILS_JSON:\n{payload}\n\n"
        "Write the friendly assistant reply now."
    )
    out = _llm_select(_SYSTEM_ASSISTANT, prompt).strip()
    return out or merged_md

def _chunks(items: List[Dict], n: int):
    for i in range(0, len(items), n):
        yield items[i:i+n]

def _llm_pick(user_query: str, chunk: List[Dict]) -> str:
    payload = json.dumps(chunk, ensure_ascii=False)
    prompt = (
        f"USER REQUEST: {user_query}\n\nEMAILS JSON:\n{payload}\n\n"
        "Select & format per rules. If none: NONE"
    )
    _d(2, f"LLM pick on chunk size={len(chunk)}")
    res = _llm_select(_SYSTEM_PICK, prompt) or "NONE"
    _d(3, "LLM pick output:", res[:300])
    return res

def _sanitize_merged_markdown(md: str) -> str:
    if not md:
        return md
    lines = [l.rstrip() for l in md.splitlines()]
    has_bullets = any(l.lstrip().startswith("- ") for l in lines)
    if not has_bullets:
        return md.strip()
    cleaned = []
    for l in lines:
        low = l.strip().lower()
        if low in {"no messages found.", "no messages found", "none"}:
            continue
        cleaned.append(l)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned).strip()

def _llm_merge(lists: List[str]) -> str:
    _d(2, f"LLM merge on {len(lists)} partial lists")
    text = "\n\n".join(lists)
    out = _llm_select(_SYSTEM_MERGE, f"Merge these lists into one per the rules:\n\n{text}").strip()
    out = _sanitize_merged_markdown(out)
    _d(3, "LLM merge output (sanitized):", out[:300])
    return out or "No messages found."

def _fallback_render(rows: List[Dict], limit: int = 20) -> str:
    def _fmt(r):
        subj = r.get("subject") or "(no subject)"
        url  = r.get("url") or ""
        frm  = r.get("from") or ""
        dt   = r.get("date") or ""
        return f"- [{subj}]({url}) — {frm} ({dt})"
    return "\n".join(_fmt(r) for r in rows[:limit]) or "No messages found."

# ──────────────────────────────────────────────────────────────────────────────
# Public API — core, parallel LLM map
# ──────────────────────────────────────────────────────────────────────────────
def email_search_all_mailbox(
    user_query: str,
    sources: List[str],
    ms_token: Optional[str],
    g_token: Optional[str],
    max_total_each: int = MAX_PER_PROVIDER_DEFAULT,
    chunk_size: int = CHUNK_DEFAULT,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    user_tz: Optional[str] = None,
    keywords: Optional[str] = None,
) -> str:
    """
    Fetch emails from providers, LLM-filter them per the user's request,
    then render either a neutral digest or a personal assistant reply.
    """
    tz = resolve_tz(user_tz)
    dbg = {"stage": "email_search_all_mailbox", "tz": tz, "sources": sources}
    _d(1, f"[mail] q='{user_query}' sources={sources} tmin={time_min} tmax={time_max} tz={tz}")

    if not time_min and not time_max:
        now = datetime.now(ZoneInfo(tz))
        start = (now - timedelta(days=30)).date(); end = now.date()
        time_min, _ = window_for_day(start, tz); _, time_max = window_for_day(end, tz)
        dbg["window_default"] = "last30"
    dbg["window"]   = {"time_min": time_min, "time_max": time_max}
    dbg["keywords"] = keywords or user_query

    t0 = time.perf_counter()
    rows = fetch_all_emails(
        sources=sources,
        ms_token=ms_token,
        g_token=g_token,
        max_total_each=max_total_each,
        time_min=time_min,
        time_max=time_max,
        user_tz=tz,
        keywords=keywords or user_query,
    )
    dbg["fetch_elapsed_s"] = round(time.perf_counter() - t0, 2)
    dbg["rows_total"]      = len(rows)
    _d(1, f"[mail] fetched rows={len(rows)} in {dbg['fetch_elapsed_s']}s")

    if not rows:
        result = "No messages found."
        if EMAIL_DEBUG_ECHO:
            result += f"\n\n<details><summary>Debug</summary>\n\n```json\n{json.dumps(dbg, indent=2)}\n```\n</details>"
        return result

    if _LLM is None:
        rows_sorted = sorted(rows, key=lambda r: (r.get("date") or ""), reverse=True)
        result = _fallback_render(rows_sorted)
        if EMAIL_DEBUG_ECHO:
            result += f"\n\n<details><summary>Debug</summary>\n\n```json\n{json.dumps(dbg, indent=2)}\n```\n</details>"
        return result

    picks: List[str] = []
    t1 = time.perf_counter()
    chunks = list(_chunks(rows, max(1, int(chunk_size))))
    with ThreadPoolExecutor(max_workers=max(1, LLM_MAP_THREADS)) as ex:
        futs = [ex.submit(_llm_pick, user_query, ch) for ch in chunks]
        for fu in as_completed(futs):
            try:
                s = (fu.result() or "").strip()
                if s and s.upper() != "NONE":
                    picks.append(s)
            except Exception as e:
                _d(1, "[mail] LLM map error:", e)
    dbg["llm_map_elapsed_s"]  = round(time.perf_counter() - t1, 2)
    dbg["llm_map_kept_lists"] = len(picks)

    result_merged = (_llm_merge(picks) if picks else "No messages found.")
    dbg["result_empty_after_merge"] = (result_merged.strip().lower() == "no messages found.")

    # PERSONAL voice when enabled, otherwise neutral digest
    use_assistant_voice = EMAIL_PERSONAL_VOICE
    if use_assistant_voice and result_merged.strip().lower() != "no messages found.":
        try:
            result = _llm_assistant(
                merged_md=result_merged,
                user_query=user_query,
                time_min=time_min,
                user_tz=tz,
                rows=rows,
            )
        except Exception as e:
            _d(1, "[mail] assistant render failed, falling back to digest:", e)
            result = _llm_digest(result_merged, user_query, time_min, None, tz, rows)
    else:
        result = _llm_digest(result_merged, user_query, time_min, None, tz, rows)

    if EMAIL_DEBUG_ECHO:
        result += f"\n\n<details><summary>Debug</summary>\n\n```json\n{json.dumps(dbg, indent=2)}\n```\n</details>"
    return result

# ──────────────────────────────────────────────────────────────────────────────
# ChatGPT-style entry — uses LLM/parsed windows and progressive widen
# ──────────────────────────────────────────────────────────────────────────────
def email_search_chatgpt_style(
    user_query: str,
    sources: List[str],
    ms_token: Optional[str],
    g_token: Optional[str],
    user_tz: Optional[str] = None,
    default_if_no_window: str = "today",
    max_total_each: int = MAX_PER_PROVIDER_DEFAULT,
    chunk_size: int = CHUNK_DEFAULT,
    llm_time_min_iso: Optional[str] = None,
    llm_time_max_iso: Optional[str] = None,
    llm_date_hint: Optional[str] = None,
) -> str:
    tz = resolve_tz(user_tz)

    if llm_date_hint and re.fullmatch(r"\s*\d{4}-\d{1,2}-\d{1,2}\s*", llm_date_hint):
        y, m, d = [int(x) for x in re.findall(r"\d+", llm_date_hint)]
        tmin, tmax = window_for_day(date(y, m, d), tz)
        _d(1, f"LLM date-only → local-day window: {tmin} .. {tmax} (tz={tz})")
    else:
        if llm_time_min_iso or llm_time_max_iso:
            tmin, tmax = normalize_window_strict(llm_time_min_iso, llm_time_max_iso)
            _d(1, f"LLM window(norm): {tmin} .. {tmax}")
        else:
            tmin, tmax = parse_relative_window(user_query, tz)
            if not tmin and not tmax:
                if default_if_no_window.lower() == "today":
                    tmin, tmax = window_today(tz)
                    _d(1, "No window → default=today (ge/lt)")
                else:
                    now = datetime.now(ZoneInfo(tz))
                    start = (now - timedelta(days=7)).date(); end = now.date()
                    tmin, _ = window_for_day(start, tz); _, tmax = window_for_day(end, tz)
                    _d(1, "No window → default=last7")

    result = email_search_all_mailbox(
        user_query=user_query,
        sources=sources,
        ms_token=ms_token,
        g_token=g_token,
        max_total_each=max_total_each,
        chunk_size=chunk_size,
        time_min=tmin,
        time_max=tmax,
        user_tz=tz,
        keywords=user_query,
    )
    if result.strip().lower() != "no messages found.":
        return result

    _d(1, "No results; widening to last30")
    now = datetime.now(ZoneInfo(tz))
    start30 = (now - timedelta(days=30)).date()
    tmin30, _ = window_for_day(start30, tz); _, tmax30 = window_for_day(now.date(), tz)
    result = email_search_all_mailbox(
        user_query=user_query,
        sources=sources,
        ms_token=ms_token,
        g_token=g_token,
        max_total_each=max_total_each,
        chunk_size=chunk_size,
        time_min=tmin30,
        time_max=tmax30,
        user_tz=tz,
        keywords=user_query,
    )
    if result.strip().lower() != "no messages found.":
        return result

    _d(1, "Still empty; widening to last90")
    start90 = (now - timedelta(days=90)).date()
    tmin90, _ = window_for_day(start90, tz); _, tmax90 = window_for_day(now.date(), tz)
    return email_search_all_mailbox(
        user_query=user_query,
        sources=sources,
        ms_token=ms_token,
        g_token=g_token,
        max_total_each=max_total_each,
        chunk_size=chunk_size,
        time_min=tmin90,
        time_max=tmax90,
        user_tz=tz,
        keywords=user_query,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Convenience: “today”
# ──────────────────────────────────────────────────────────────────────────────
def get_todays_emails_markdown(
    user_query: str,
    sources: List[str],
    ms_token: Optional[str],
    g_token: Optional[str],
    user_tz: Optional[str] = None,
    max_total_each: int = MAX_PER_PROVIDER_DEFAULT,
    chunk_size: int = CHUNK_DEFAULT,
) -> str:
    tz = resolve_tz(user_tz)
    tmin, tmax = window_today(tz)
    _d(1, f"Today window: {tmin} .. {tmax}")
    return email_search_all_mailbox(
        user_query=user_query,
        sources=sources,
        ms_token=ms_token,
        g_token=g_token,
        max_total_each=max_total_each,
        chunk_size=chunk_size,
        time_min=tmin,
        time_max=tmax,
        user_tz=tz,
        keywords=user_query,
    )
