# calendar_service.py — LLM-first Calendar Event Creator (Outlook / Google)
# Follows flow:
# 1) Intent → 2) Slot Filling (Title → Date/Time → Participants → Location → Description)
# 3) Smart LLM Assistance (auto title, draft/rephrase/summarize description)
# 4) Confirmation → 5) Execution (+ modifications)

from __future__ import annotations
import os, re, json, logging, requests
from datetime import datetime, timedelta
from typing import Optional, Callable, List, Tuple, Dict, Any
from flask import session, current_app
try:
    from flask_login import current_user
except Exception:
    current_user = None

# Python 3.9+ preferred
try:
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# ───────────────────────────────────────────────────────────────────────────────
# Intent trigger (heuristic + optional LLM)
# ───────────────────────────────────────────────────────────────────────────────

_HEURISTIC_CREATE = (
    "create event","create meeting","new event","schedule","set up a meeting",
    "book a meeting","add to my calendar","make a calendar event",
    "put on my calendar","calendar event","schedule a call","set a reminder",
)

def _heuristic_calendar_create(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _HEURISTIC_CREATE)

def is_calendar_create_intent(text: str) -> bool:
    if _heuristic_calendar_create(text):
        return True
    return bool(_llm_classify_intent(text))

# ───────────────────────────────────────────────────────────────────────────────
# Provider detection (same registration pattern as email_service)
# ───────────────────────────────────────────────────────────────────────────────

_custom_outlook_token_loader: Optional[Callable[[], Optional[str]]] = None
_custom_gmail_creds_loader: Optional[Callable[[], Optional["Credentials"]]] = None  # type: ignore[name-defined]

def register_outlook_token_loader(func: Callable[[], Optional[str]]) -> None:
    global _custom_outlook_token_loader
    _custom_outlook_token_loader = func

def register_gmail_creds_loader(func: Callable[[], Optional["Credentials"]]) -> None:  # type: ignore[name-defined]
    global _custom_gmail_creds_loader
    _custom_gmail_creds_loader = func

def _get_outlook_access_token() -> Optional[str]:
    if _custom_outlook_token_loader:
        try:
            tok = _custom_outlook_token_loader()
            if tok: return tok
        except Exception:
            logging.exception("Custom Outlook token loader failed")
    # Optional silent MSAL fallback (if you already ship it)
    try:
        from msal_auth import load_token_cache, build_msal_app
        cache = load_token_cache(); app = build_msal_app(cache=cache)
        scopes = ["Calendars.ReadWrite"]
        for acc in app.get_accounts():
            res = app.acquire_token_silent(scopes=scopes, account=acc)
            if res and res.get("access_token"): return res["access_token"]
        res = app.acquire_token_silent(scopes=["https://graph.microsoft.com/.default"], account=None)
        if res and res.get("access_token"): return res["access_token"]
    except Exception:
        logging.debug("MSAL token acquisition failed", exc_info=True)
    return None

def _get_gmail_credentials():
    if _custom_gmail_creds_loader:
        try:
            creds = _custom_gmail_creds_loader()
            if creds: return creds
        except Exception:
            logging.exception("Custom Gmail creds loader failed")
    # Try session → instance (same idea you use for email)
    try:
        data = session.get("gmail_credentials") or session.get("google_credentials")
        if data:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            creds = Credentials.from_authorized_user_info(
                data, scopes=["https://www.googleapis.com/auth/calendar.events"]
            )
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            return creds
    except Exception:
        logging.debug("Session Google creds load failed", exc_info=True)
    try:
        from pathlib import Path
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        inst = Path(getattr(current_app, "instance_path", ".")) / "google_calendar_token.json"
        if inst.exists():
            info = json.loads(inst.read_text())
            creds = Credentials.from_authorized_user_info(
                info, scopes=["https://www.googleapis.com/auth/calendar.events"]
            )
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            return creds
    except Exception:
        logging.debug("Instance Google calendar creds load failed", exc_info=True)
    return None

def _detect_calendar_provider() -> Tuple[Optional[str], Optional[object]]:
    tok = _get_outlook_access_token()
    if tok: return ("outlook", tok)
    creds = _get_gmail_credentials()
    if creds: return ("google", creds)
    return (None, None)

# ───────────────────────────────────────────────────────────────────────────────
# General helpers
# ───────────────────────────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)")

def _extract_emails(text: str) -> List[str]:
    raw = re.findall(EMAIL_RE, text or "")
    seen, out = set(), []
    for a in raw:
        a2 = a.strip().strip(",;")
        if a2 and a2.lower() not in seen:
            seen.add(a2.lower()); out.append(a2)
    return out

def _user_tz() -> str:
    return current_app.config.get("APP_TZ") or current_app.config.get("TZ") or "Asia/Dhaka"

def _aware_now(tz: str, like: Optional[datetime] = None) -> datetime:
    """Return a 'now' compatible with dt comparisons (aware if dt is aware)."""
    try:
        if like and like.tzinfo:
            return datetime.now(like.tzinfo)
        if ZoneInfo:
            return datetime.now(ZoneInfo(tz))  # type: ignore[arg-type]
    except Exception:
        pass
    return datetime.now()

def _align_tz(dt: datetime, tz: str) -> datetime:
    """Ensure dt has tzinfo if possible."""
    if dt and dt.tzinfo is None:
        try:
            if ZoneInfo:
                return dt.replace(tzinfo=ZoneInfo(tz))  # type: ignore[arg-type]
        except Exception:
            return dt
    return dt

# ───────────────────────────────────────────────────────────────────────────────
# LLM plumbing — Ollama first (qwen2:7b), OpenAI fallback via your answer wrapper
# ───────────────────────────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        import ollama  # type: ignore
        return True
    except Exception:
        return False

def _ollama_chat(messages: List[Dict[str,str]], json_expect: bool=False, max_tokens: int=600) -> str:
    import ollama  # type: ignore
    model = os.getenv("OLLAMA_MODEL", "qwen2:7b")
    try:
        resp = ollama.chat(model=model, messages=messages, options={"num_predict": max_tokens})
        txt = (resp.get("message",{}) or {}).get("content","") or ""
        if json_expect:
            t = txt.strip()
            if t.startswith("```"):
                import re as _re
                t = _re.sub(r"^```(?:json)?\s*|\s*```$","",t,flags=_re.S)
            return t
        return txt
    except Exception:
        logging.exception("Ollama chat call failed")
        return ""

def _openai_answer(user_prompt: str, system_prompt: str, max_tokens: int=600) -> str:
    try:
        from openai_api import answer_general_query
    except Exception:
        return ""
    try:
        return (answer_general_query(user_prompt, system_prompt=system_prompt, max_tokens=max_tokens) or "").strip()
    except TypeError:
        return (answer_general_query(f"{system_prompt}\n\n{user_prompt}") or "").strip()
    except Exception:
        logging.exception("OpenAI wrapper call failed")
        return ""

def _llm_json(system_prompt: str, user_prompt: str, attempts: int=2, max_tokens: int=600) -> Dict[str,Any]:
    for _ in range(attempts):
        if _ollama_available():
            txt = _ollama_chat(
                [{"role":"system","content":system_prompt+" Always reply with minified JSON only."},
                 {"role":"user","content":user_prompt}],
                json_expect=True, max_tokens=max_tokens
            )
        else:
            txt = _openai_answer(user_prompt, system_prompt=system_prompt+" Only return JSON.", max_tokens=max_tokens)
        s,e = txt.find("{"), txt.rfind("}")
        if s!=-1 and e!=-1:
            txt = txt[s:e+1]
        try:
            return json.loads(txt) if txt else {}
        except Exception:
            continue
    return {}

def _llm_text(system_prompt: str, user_prompt: str, max_tokens: int=600) -> str:
    if _ollama_available():
        return _ollama_chat(
            [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            json_expect=False, max_tokens=max_tokens
        ).strip()
    return _openai_answer(user_prompt, system_prompt=system_prompt, max_tokens=max_tokens).strip()

# ───────────────────────────────────────────────────────────────────────────────
# LLM tasks
# ───────────────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM = (
    "You are an intent classifier for a calendar assistant. "
    "Return JSON like {\"calendar_create\": true/false}. "
    "True when the user asks to schedule, create, add, book, or set a meeting/event."
)
def _llm_classify_intent(text: str) -> bool:
    if not text.strip(): return False
    data = _llm_json(_INTENT_SYSTEM, f"Text: {text}")
    return bool(isinstance(data, dict) and data.get("calendar_create") is True)

_EXTRACT_SYSTEM = (
    "Extract calendar event fields from natural language. Return JSON only with keys: "
    "title, date (YYYY-MM-DD or empty), time (HH:MM 24h or empty), end_time (HH:MM or empty), "
    "duration_minutes (int or null), location (string or empty), attendees (array of strings), "
    "description (string or empty), is_all_day (true/false). If unknown, leave empty/null."
)
def _llm_extract_fields(text: str) -> Dict[str,Any]:
    out = _llm_json(_EXTRACT_SYSTEM, f"Text: {text}")
    emails: List[str] = []
    for a in (out.get("attendees") or []):
        emails += _extract_emails(a if isinstance(a,str) else str(a))
    out["attendees"] = list(dict.fromkeys(emails))
    return out

def _llm_generate_title(context: Dict[str,Any]) -> str:
    sys = ("You generate concise, professional calendar titles (max 7 words). "
           "No quotes. Avoid redundancy. Return plain text only.")
    return _llm_text(sys, json.dumps(context)) or "Meeting"

def _llm_draft_or_rephrase(description_hint: str, context: Dict[str,Any]) -> str:
    sys = ("You are a helpful assistant who drafts or rephrases short event descriptions. "
           "Tone: clear, polite, professional. 2–4 short sentences. Return plain text.")
    prompt = f"Hint: {description_hint}\nContext JSON: {json.dumps(context)}"
    return _llm_text(sys, prompt)

# ───────────────────────────────────────────────────────────────────────────────
# Date/time helpers
# ───────────────────────────────────────────────────────────────────────────────

def _safe_dateparse(s: str, tz: str) -> Optional[datetime]:
    if not s: return None
    try:
        import dateparser
        # Prefer future dates and anchor parsing to a timezone-aware "now"
        if ZoneInfo:
            now = datetime.now(ZoneInfo(tz))  # type: ignore[arg-type]
        else:
            now = datetime.now()
        return dateparser.parse(
            s,
            settings={
                "TIMEZONE": tz,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": now,
            },
        )
    except Exception:
        return None

def _combine_dt(date_str: str, time_str: str, tz: str) -> Optional[datetime]:
    if date_str and time_str:
        return _safe_dateparse(f"{date_str} {time_str}", tz) or _safe_dateparse(f"{date_str}T{time_str}", tz)
    return None

def _ensure_future_if_no_year(original_text: str, dt: Optional[datetime], tz: str) -> Optional[datetime]:
    """If user didn't specify a year, push dt forward yearly until it's in the future."""
    if not dt:
        return dt
    # Respect explicit years in the user text.
    if re.search(r"\b\d{4}\b", original_text):
        return dt
    dt = _align_tz(dt, tz)
    now = _aware_now(tz, like=dt)
    while dt < now:
        try:
            dt = dt.replace(year=dt.year + 1)
        except ValueError:
            # Handle Feb 29 → Mar 1
            dt = dt.replace(month=3, day=1, year=dt.year + 1)
    return dt

def _merge_time_like(flow: dict, text: str, tz: str) -> None:
    data = flow["data"]
    # time range e.g. "3:10pm-4:00pm"
    rng = re.search(r'(\d{1,2}(:\d{2})?\s*(am|pm)?)\s*[-–]\s*(\d{1,2}(:\d{2})?\s*(am|pm)?)', text, re.I)
    if rng:
        st_txt = f"{text[:rng.start()]} {rng.group(1)}"; en_txt = f"{text[:rng.start()]} {rng.group(4)}"
        st = _safe_dateparse(st_txt, tz); en = _safe_dateparse(en_txt, tz)
        if st:
            st = _ensure_future_if_no_year(text, st, tz)
            data["start"] = st
        if en:
            en = _ensure_future_if_no_year(text, en, tz)
            if data.get("start") and en <= data["start"]:
                en = data["start"] + timedelta(minutes=30)
            data["end"] = en
        return
    # duration
    dur_m = re.search(r"\bfor\s+(\d{1,3})\s*(min|mins|minutes)\b", text, re.I)
    dt = _safe_dateparse(text, tz)
    if dt and not data.get("start"):
        dt = _ensure_future_if_no_year(text, dt, tz)
        data["start"] = dt
        if dur_m:
            data["end"] = dt + timedelta(minutes=int(dur_m.group(1)))
        return
    if dt and data.get("start") and not data.get("end"):
        dt = _ensure_future_if_no_year(text, dt, tz)
        if dt > data["start"]:
            data["end"] = dt; return
    if dur_m and data.get("start") and not data.get("end"):
        data["end"] = data["start"] + timedelta(minutes=int(dur_m.group(1)))

# ───────────────────────────────────────────────────────────────────────────────
# Providers: create event
# ───────────────────────────────────────────────────────────────────────────────

def _create_outlook_event(access_token: str, payload: Dict[str,Any]) -> bool:
    url = "https://graph.microsoft.com/v1.0/me/events"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code in (200,201): return True
    logging.warning("Graph create event failed: %s %s", r.status_code, r.text[:400])
    try: r.raise_for_status()
    except Exception: pass
    return False

def _create_google_event(creds, event: Dict[str,Any]) -> bool:
    from googleapiclient.discovery import build
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    service.events().insert(calendarId="primary", body=event).execute()
    return True

# ───────────────────────────────────────────────────────────────────────────────
# Flow state
# ───────────────────────────────────────────────────────────────────────────────

def _init_flow():
    session["cal_flow"] = {
        "stage": "slot_start",
        "provider": None,
        "data": {
            "title": None,
            "start": None,
            "end": None,
            "all_day": False,
            "location": None,
            "attendees": [],
            "description": None,
            "seed_text": ""   # remember first user utterance for better generation
        }
    }

def reset_calendar_flow():
    session.pop("cal_flow", None)

# ───────────────────────────────────────────────────────────────────────────────
# Slot helpers
# ───────────────────────────────────────────────────────────────────────────────

SLOTS_ORDER = ["title","datetime","participants","location","description"]

def _first_missing_slot(d: Dict[str,Any]) -> Optional[str]:
    if not d.get("title"): return "title"
    if not (d.get("start") and d.get("end")): return "datetime"
    if not d.get("attendees"): return "participants"
    # location/description are optional; we still ask once
    if d.get("_asked_location") != True: return "location"
    if d.get("_asked_description") != True: return "description"
    return None

def _ask_for_slot(slot: str) -> str:
    if slot == "title":
        return ("What should I name this event? "
                "You can also say **generate** and I’ll name it for you.")
    if slot == "datetime":
        return ("When should I schedule it? For example:\n"
                "• `tomorrow 3:00–4:00pm`\n"
                "• `Sep 21 at 10:30 for 45 minutes`")
    if slot == "participants":
        return ("Who should I invite? Please share emails (comma-separated is fine).")
    if slot == "location":
        return ("Do you want to add a location or link (Zoom/Meet/Teams)? "
                "Reply with the place or link, or say **skip**.")
    if slot == "description":
        return ("Any notes or details to include? You can paste text, or say **draft** "
                "to let me write a brief note for you. Say **skip** to leave it empty.")
    return "Tell me more about the event."

def _slot_finish_mark(flow: dict, slot: str) -> None:
    if slot == "location": flow["data"]["_asked_location"] = True
    if slot == "description": flow["data"]["_asked_description"] = True

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

def continue_calendar_create_flow(user_id: Optional[int], user_text: str) -> dict:
    text = (user_text or "").strip()
    flow = session.get("cal_flow")

    # Start
    if not flow:
        _init_flow()
        flow = session["cal_flow"]
        provider, _prov = _detect_calendar_provider()
        flow["provider"] = provider
        session.modified = True
        if not provider:
            reset_calendar_flow()
            return {
                "reply": "I can create events on Outlook or Google Calendar, but I’m not connected yet. "
                         "Please connect one and we’ll set this up. 📅",
                "meta": {"mode":"calendar_create","stage":"needs_connection"},
                "done": True
            }
        # seed
        flow["data"]["seed_text"] = text
        # quick extract from first utterance
        _initial_llm_fill(flow, text)
        slot = _first_missing_slot(flow["data"]) or "review"
        if slot == "review":
            flow["stage"] = "review"
            return _review_reply(flow)
        flow["stage"] = f"collect_{slot}"
        return {"reply": _ask_for_slot(slot),
                "meta":{"mode":"calendar_create","stage":flow["stage"],"slot":slot},"done":False}

    # cancel
    if text.lower() in {"cancel","stop","abort","quit"}:
        reset_calendar_flow()
        return {"reply":"All good — I’ll stop here. Say “create an event” when you’re ready again.",
                "meta":{"mode":"calendar_create","stage":"cancelled"},"done":True}

    provider = flow["provider"]
    if not provider:
        new_p, _pobj = _detect_calendar_provider()
        if not new_p:
            reset_calendar_flow()
            return {"reply":"Looks like I’m not connected to a calendar. Connect Outlook or Google and try again.",
                    "meta":{"mode":"calendar_create","stage":"error"},"done":True}
        flow["provider"] = new_p

    stage = flow["stage"]
    tz = _user_tz()

    # ── Slot: Title
    if stage == "collect_title":
        if text.lower() in {"generate","auto","g","ai"}:
            # generate title from context
            ctx = _ctx_for_llm(flow)
            flow["data"]["title"] = _llm_generate_title(ctx) or "Meeting"
        else:
            flow["data"]["title"] = text if text else (_llm_generate_title(_ctx_for_llm(flow)) or "Meeting")

        next_slot = _first_missing_slot(flow["data"]) or "review"
        flow["stage"] = f"collect_{next_slot}" if next_slot!="review" else "review"
        if flow["stage"] == "review": return _review_reply(flow)
        return {"reply": _ask_for_slot(next_slot),
                "meta":{"mode":"calendar_create","stage":flow["stage"],"slot":next_slot},"done":False}

    # ── Slot: DateTime
    if stage == "collect_datetime":
        _merge_time_like(flow, text, tz)
        if not (flow["data"]["start"] and flow["data"]["end"]):
            # try LLM extraction too
            ext = _llm_extract_fields(text)
            d, t, et, dur = (ext.get("date") or "").strip(), (ext.get("time") or "").strip(), (ext.get("end_time") or "").strip(), ext.get("duration_minutes")
            st = _combine_dt(d, t, tz) or _safe_dateparse(f"{d} {t}", tz)
            if st:
                st = _ensure_future_if_no_year(text, st, tz)
                flow["data"]["start"] = st
            if et and d:
                en = _combine_dt(d, et, tz) or _safe_dateparse(f"{d} {et}", tz)
                if en:
                    en = _ensure_future_if_no_year(text, en, tz)
                    flow["data"]["end"] = en
            elif dur and flow["data"]["start"]:
                flow["data"]["end"] = flow["data"]["start"] + timedelta(minutes=int(dur))
        if not (flow["data"]["start"] and flow["data"]["end"]):
            return {"reply":"Got it. I still need the date/time (e.g., `Wed 3pm–4pm` or `Sep 21 10:30 for 30 minutes`).",
                    "meta":{"mode":"calendar_create","stage":"collect_datetime"},"done":False}

        next_slot = _first_missing_slot(flow["data"]) or "review"
        flow["stage"] = f"collect_{next_slot}" if next_slot!="review" else "review"
        if flow["stage"] == "review": return _review_reply(flow)
        return {"reply": _ask_for_slot(next_slot),
                "meta":{"mode":"calendar_create","stage":flow["stage"],"slot":next_slot},"done":False}

    # ── Slot: Participants
    if stage == "collect_participants":
        emails = _extract_emails(text)
        if emails:
            flow["data"]["attendees"] = list(dict.fromkeys(flow["data"]["attendees"] + emails))
        else:
            return {"reply":"Please provide at least one email (comma-separated is fine).",
                    "meta":{"mode":"calendar_create","stage":"collect_participants"},"done":False}
        next_slot = _first_missing_slot(flow["data"]) or "review"
        flow["stage"] = f"collect_{next_slot}" if next_slot!="review" else "review"
        if flow["stage"] == "review": return _review_reply(flow)
        return {"reply": _ask_for_slot(next_slot),
                "meta":{"mode":"calendar_create","stage":flow["stage"],"slot":next_slot},"done":False}

    # ── Slot: Location (optional)
    if stage == "collect_location":
        if text.lower() not in {"skip","none","no"}:
            flow["data"]["location"] = text
        _slot_finish_mark(flow, "location")
        next_slot = _first_missing_slot(flow["data"]) or "review"
        flow["stage"] = f"collect_{next_slot}" if next_slot!="review" else "review"
        if flow["stage"] == "review": return _review_reply(flow)
        return {"reply": _ask_for_slot(next_slot),
                "meta":{"mode":"calendar_create","stage":flow["stage"],"slot":next_slot},"done":False}

    # ── Slot: Description (optional with LLM assistance)
    if stage == "collect_description":
        if text.lower() in {"skip","none","no"}:
            flow["data"]["description"] = None
        elif text.lower() in {"draft","rephrase","summarize","write","generate"}:
            flow["data"]["description"] = _llm_draft_or_rephrase("Write a brief meeting note.", _ctx_for_llm(flow))
        else:
            flow["data"]["description"] = text
        _slot_finish_mark(flow, "description")
        flow["stage"] = "review"
        return _review_reply(flow)

    # ── Review & confirmation
    if stage == "review":
        low = text.lower().strip()
        if low in {"yes","y","create","confirm"}:
            ok = _create_event(flow)
            reset_calendar_flow()
            if ok:
                return {"reply":"✅ All set — your event has been added to the calendar!",
                        "meta":{"mode":"calendar_create","stage":"done"},"done":True}
            return {"reply":"❌ I couldn’t create the event. Please reconnect your calendar and try again.",
                    "meta":{"mode":"calendar_create","stage":"done"},"done":True}
        if low in {"no","n","cancel"}:
            reset_calendar_flow()
            return {"reply":"Cancelled. I didn’t add anything to your calendar.",
                    "meta":{"mode":"calendar_create","stage":"cancelled"},"done":True}

        # “modify” flow — free-form change; use LLM to merge changes
        if low.startswith("title:"):
            flow["data"]["title"] = text.split(":",1)[1].strip() or flow["data"]["title"]
            return _review_reply(flow)
        if low.startswith("location:"):
            flow["data"]["location"] = text.split(":",1)[1].strip()
            return _review_reply(flow)
        if low.startswith(("desc:","description:")):
            flow["data"]["description"] = text.split(":",1)[1].strip()
            return _review_reply(flow)
        if _extract_emails(text):
            flow["data"]["attendees"] = list(dict.fromkeys(flow["data"]["attendees"] + _extract_emails(text)))
            return _review_reply(flow)
        # otherwise let LLM parse changes (timing/title/etc.)
        _merge_modification_from_llm(flow, text, tz)
        return _review_reply(flow)

    # fallback safety
    reset_calendar_flow()
    return {"reply":"I reset the calendar assistant — say “create an event” to start again.",
            "meta":{"mode":"calendar_create","stage":"reset"},"done":True}

# ───────────────────────────────────────────────────────────────────────────────
# Internals
# ───────────────────────────────────────────────────────────────────────────────

def _initial_llm_fill(flow: dict, text: str) -> None:
    tz = _user_tz()
    ext = _llm_extract_fields(text)
    title = (ext.get("title") or "").strip() or None
    date_s, time_s, end_time_s = (ext.get("date") or "").strip(), (ext.get("time") or "").strip(), (ext.get("end_time") or "").strip()
    dur = ext.get("duration_minutes")
    attendees = ext.get("attendees") or []
    loc = (ext.get("location") or "").strip() or None
    desc = (ext.get("description") or "").strip() or None

    start = _combine_dt(date_s, time_s, tz) or _safe_dateparse(f"{date_s} {time_s}", tz)
    end = None
    if end_time_s and date_s:
        end = _combine_dt(date_s, end_time_s, tz) or _safe_dateparse(f"{date_s} {end_time_s}", tz)
    elif dur and start:
        end = start + timedelta(minutes=int(dur))

    if start:
        start = _ensure_future_if_no_year(text, start, tz)
    if end:
        end = _ensure_future_if_no_year(text, end, tz)

    data = flow["data"]
    data["title"] = title
    data["start"] = start
    data["end"] = end
    data["attendees"] = list(dict.fromkeys(attendees + _extract_emails(text)))
    data["location"] = loc
    data["description"] = desc

    # Also try free-form time range in one go
    _merge_time_like(flow, text, tz)

def _merge_modification_from_llm(flow: dict, text: str, tz: str) -> None:
    if not text: return
    ext = _llm_extract_fields(text)
    if ext.get("title"): flow["data"]["title"] = ext["title"].strip()
    if ext.get("location"): flow["data"]["location"] = ext["location"].strip()
    if ext.get("description"): flow["data"]["description"] = ext["description"].strip()
    adds = _extract_emails(text) + (ext.get("attendees") or [])
    if adds:
        flow["data"]["attendees"] = list(dict.fromkeys(flow["data"]["attendees"] + _extract_emails(", ".join(adds))))
    d, t, et, dur = (ext.get("date") or "").strip(), (ext.get("time") or "").strip(), (ext.get("end_time") or "").strip(), ext.get("duration_minutes")
    st = _combine_dt(d,t,tz) or _safe_dateparse(f"{d} {t}", tz)
    if st:
        st = _ensure_future_if_no_year(text, st, tz)
        flow["data"]["start"] = st
    if et and d:
        en = _combine_dt(d, et, tz) or _safe_dateparse(f"{d} {et}", tz)
        if en:
            en = _ensure_future_if_no_year(text, en, tz)
            flow["data"]["end"] = en
    elif dur and flow["data"]["start"]:
        flow["data"]["end"] = flow["data"]["start"] + timedelta(minutes=int(dur))
    if not (flow["data"]["start"] and flow["data"]["end"]):
        _merge_time_like(flow, text, tz)

def _ctx_for_llm(flow: dict) -> Dict[str,Any]:
    d = flow["data"].copy()
    # Convert datetimes to readable strings
    if isinstance(d.get("start"), datetime): d["start"] = d["start"].isoformat()
    if isinstance(d.get("end"), datetime): d["end"] = d["end"].isoformat()
    return d

def _fmt_dt(dt: Optional[datetime]) -> str:
    if not dt: return "—"
    try:
        return dt.astimezone().strftime("%a, %d %b %Y %H:%M")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M")

def _review_reply(flow: dict) -> dict:
    prov = flow["provider"]
    d = flow["data"]

    # If end missing or not after start, make it a 30-minute slot
    if d.get("start") and d.get("end") and d["end"] <= d["start"]:
        d["end"] = d["start"] + timedelta(minutes=30)

    title = d.get("title") or "(no title)"
    start = _fmt_dt(d.get("start"))
    end   = _fmt_dt(d.get("end"))
    loc   = d.get("location") or "—"
    att   = ", ".join(d.get("attendees") or []) or "—"
    desc  = d.get("description") or "—"
    msg = (
        f"**Please confirm**\n\n"
        f"**Title:** {title}\n"
        f"**When:** {start} → {end}\n"
        f"**Participants:** {att}\n"
        f"**Location/Link:** {loc}\n"
        f"**Notes:** {desc}\n\n"
        "Shall I add this to your calendar? **yes** to create, **no** to cancel.\n"
        "To tweak, you can say things like: `title: ...`, `location: ...`, add emails, "
        "`tomorrow 4–4:30`, or `for 45 minutes`."
    )
    return {"reply": msg, "meta":{"mode":"calendar_create","stage":"review","provider":prov},"done":False}

def _create_event(flow: dict) -> bool:
    provider = flow["provider"]
    tz = _user_tz()
    title = flow["data"].get("title") or "New event"
    desc  = flow["data"].get("description") or ""
    loc   = flow["data"].get("location") or ""
    start = flow["data"]["start"]; end = flow["data"]["end"]
    attendees = flow["data"].get("attendees") or []
    if provider == "outlook":
        access_token = _get_outlook_access_token()
        if not access_token: return False
        payload = {
            "subject": title,
            "body": {"contentType":"HTML","content": desc.replace("\n","<br>")},
            "start": {"dateTime": start.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": tz},
            "end":   {"dateTime": end.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": tz},
            "location": {"displayName": loc} if loc else None,
            "attendees": [{"emailAddress":{"address": a},"type":"required"} for a in attendees],
            "allowNewTimeProposals": True,
        }
        payload = {k:v for k,v in payload.items() if v is not None}
        return _create_outlook_event(access_token, payload)
    # google
    creds = _get_gmail_credentials()
    if not creds: return False
    event = {
        "summary": title,
        "description": desc,
        "start": {"dateTime": start.isoformat(), "timeZone": tz},
        "end":   {"dateTime": end.isoformat(),   "timeZone": tz},
    }
    if loc: event["location"] = loc
    if attendees: event["attendees"] = [{"email": a} for a in attendees]
    return _create_google_event(creds, event)
