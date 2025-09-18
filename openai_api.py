# openai_api.py — smarter intent + provider & entity hints (Slack-friendly)
from __future__ import annotations
import json, os, re, string
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

# ------------- Ollama (LangChain) -------------
try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None  # type: ignore

MODEL_NAME       = os.getenv("OLLAMA_MODEL", "qwen2:7b")
NUM_CTX          = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
NUM_PREDICT      = int(os.getenv("OLLAMA_NUM_PREDICT", "800"))
LLM_TEMPERATURE  = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
DEBUG            = bool(int(os.getenv("OPENAI_API_DEBUG", "0")))
APP_TZ_NAME      = os.getenv("APP_TZ", "Asia/Dhaka")

_llm = None
if ChatOllama is not None:
    _llm = ChatOllama(
        model=MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
    )

def _dprint(*args, **kwargs):
    if DEBUG: print("[openai_api]", *args, **kwargs)

# ------------- text helpers -------------
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_GUARD = str.maketrans({c: " " for c in string.punctuation if c not in {'"', "'", "-", "_", ".", "@", "#"}})

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","what","which","who","whom","whose","where",
    "why","how","to","for","of","in","on","at","by","with","from","as","is","are","was","were","be","been","being",
    "this","that","these","those","i","you","he","she","it","we","they","me","him","her","them","my","your","our",
    "their","mine","yours","ours","theirs","do","does","did","doing","done","can","could","should","would","will",
    "shall","may","might","must","not","no","yes","please","kindly","show","give","find","open","need","want","see",
    "about","regarding","related","latest","recent","new","ready","user","request","result","results",
}

def _normalize_space(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s or "").strip()

def _json_from_text(txt: str) -> Optional[Dict[str, Any]]:
    if not txt: return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.DOTALL|re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r"(\{.*\})", txt, flags=re.DOTALL)
    if m:
        raw = m.group(1); last = raw.rfind("}")
        if last != -1: raw = raw[:last+1]
        try: return json.loads(raw)
        except Exception: return None
    return None

def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k); out.append(x)
    return out

def _split_phrases_and_tokens(text: str) -> List[str]:
    phrases = [p.strip() for p in re.findall(r'"([^"]+)"', text or "")]
    remainder = re.sub(r'"[^"]+"', " ", text or "")
    remainder = remainder.translate(_PUNCT_GUARD)
    tokens = [t.strip() for t in remainder.split()]
    return [t for t in (phrases + tokens) if t]

def _rule_keywords(text: str, max_terms: int = 8) -> str:
    if not text: return ""
    all_terms = _split_phrases_and_tokens(text)
    cleaned: List[str] = []
    for t in all_terms:
        low = t.lower().strip()
        if low.startswith(("@","#")):
            cleaned.append(low); continue
        low = low.strip(string.punctuation + " ")
        if not low: continue
        if low.isdigit() and len(low) > 4:  # drop big numbers (ids)
            continue
        if low in STOPWORDS: continue
        if len(low) == 1 and not low.isdigit(): continue
        cleaned.append(low)
    cleaned = _dedup_preserve_order(cleaned)
    return " ".join(cleaned[:max_terms])

# ------------- time helpers -------------
def _tz() -> timezone:
    try:
        if ZoneInfo is not None: return ZoneInfo(APP_TZ_NAME)  # type: ignore
    except Exception:
        pass
    return timezone.utc

def _sod(dt: datetime) -> datetime: return dt.replace(hour=0, minute=0, second=0, microsecond=0)
def _eod(dt: datetime) -> datetime: return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
def _Z(dt_local: datetime) -> str:  return dt_local.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

_DATE = r"(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})"

def _parse_window_local(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristic time window parser:
      - YYYY-MM-DD or between YYYY-MM-DD and YYYY-MM-DD
      - today, yesterday, this week, next week, last week
      - this month, last month
      - past/last N days|weeks|months (N<=180)
    Returns (time_min_isoZ, time_max_isoZ) or (None, None)
    """
    q = (text or "").lower()
    now = datetime.now(_tz())

    # absolute range "between A and B"
    m = re.search(rf"\bbetween\s+{_DATE}\s+(?:and|to|-)\s+{_DATE}\b", q)
    if m:
        y1,m1,d1,y2,m2,d2 = map(int, m.groups())
        s = _sod(datetime(y1,m1,d1,tzinfo=_tz()))
        e = _eod(datetime(y2,m2,d2,tzinfo=_tz()))
        return _Z(s), _Z(e)

    # single explicit date
    m = re.search(rf"\b{_DATE}\b", q)
    if m:
        y,mo,d = map(int, m.groups())
        s = _sod(datetime(y,mo,d,tzinfo=_tz()))
        e = _eod(datetime(y,mo,d,tzinfo=_tz()))
        return _Z(s), _Z(e)

    # relative short-hands
    if "today" in q:      return _Z(_sod(now)), _Z(_eod(now))
    if "yesterday" in q:  y = now - timedelta(days=1); return _Z(_sod(y)), _Z(_eod(y))
    if "this week" in q:  s = _sod(now - timedelta(days=now.weekday())); e = _eod(s + timedelta(days=6)); return _Z(s), _Z(e)
    if "next week" in q:  s = _sod(now - timedelta(days=now.weekday()) + timedelta(weeks=1)); e = _eod(s + timedelta(days=6)); return _Z(s), _Z(e)
    if "last week" in q:  s = _sod(now - timedelta(days=now.weekday(), weeks=1)); e = _eod(s + timedelta(days=6)); return _Z(s), _Z(e)

    if "this month" in q:
        s = _sod(now.replace(day=1))
        nm = (s.replace(year=s.year+1, month=1) if s.month==12 else s.replace(month=s.month+1))
        e = _eod(nm - timedelta(days=1))
        return _Z(s), _Z(e)
    if "last month" in q:
        first_this = _sod(now.replace(day=1))
        first_prev = (first_this.replace(year=first_this.year-1, month=12) if first_this.month==1
                      else first_this.replace(month=first_this.month-1))
        end_prev = _eod(first_this - timedelta(days=1))
        return _Z(first_prev), _Z(end_prev)

    # "past/last N days|weeks|months"
    m = re.search(r"\b(past|last)\s+(\d{1,3})\s*(day|days|week|weeks|month|months)\b", q)
    if m:
        n = min(int(m.group(2)), 180)
        unit = m.group(3)
        if unit.startswith("day"):
            s = _sod(now - timedelta(days=n))
        elif unit.startswith("week"):
            s = _sod(now - timedelta(weeks=n))
        else:
            s = _sod(now - timedelta(days=30*n))
        return _Z(s), _Z(_eod(now))

    return None, None

# ------------- LLM call -------------
def _llm_respond(prompt: str) -> str:
    if _llm is None: return ""
    try:
        out = _llm.invoke(prompt)
        return (getattr(out, "content", None) or str(out) or "").strip()
    except Exception as e:
        _dprint("LLM error:", e)
        try:
            return (_llm.predict(prompt) or "").strip()
        except Exception as e2:
            _dprint("LLM fallback error:", e2)
            return ""

# ------------- INTENT SYSTEM (JSON) -------------
_INTENT_SYSTEM_JSON = (
    "You are an intent router. Return STRICT JSON only.\n"
    "Schema:{"
    "\"intent\":\"send_email|email_search|calendar_create|calendar_search|file_search|file_content|web_search|message_search|hr_admin|general\","
    "\"confidence\":0.0-1.0,"
    "\"keywords\":[\"...\"],"
    "\"extracted_keywords\":[\"...\"],"
    "\"data\":\"string or empty\","
    "\"time_min_iso\":\"YYYY-MM-DDTHH:MM:SSZ|null\","
    "\"time_max_iso\":\"YYYY-MM-DDTHH:MM:SSZ|null\"}\n"
    "Routing notes:\n"
    "- send_email is for composing/sending an email (collect recipient, subject, body, then confirm).\n"
    "- calendar_create is for scheduling a new event/meeting (title, time, attendees, etc.).\n"
    "- message_search is for Slack/Teams chats/channels/DMs (#channel, @user, workspace).\n"
    "- email_search is for Gmail/Outlook email inbox.\n"
    "- file_search is to find/list files; file_content is to read/summarize a specific file.\n"
    "- web_search is for live/online facts (news, weather, prices, who/what queries).\n"
    "- hr_admin for HR/policy/leave/payroll questions.\n"
    "Time rules:\n"
    "- Do NOT convert relative time (today/this week/last month) to absolute; leave as null.\n"
    "- Only set time_* when the user gave explicit absolute dates.\n"
    "Keywords: 1–8 short tokens (preserve quoted phrases); extracted_keywords should equal keywords.\n"
    "Confidence: >=0.7 when clearly matched; otherwise lower.\n"
)

# ------------- Scoring heuristics -------------
_EMAIL_WORDS  = {"email","e-mail","inbox","outlook","gmail","subject:","from:","to:"}
_MSG_WORDS    = {"slack","workspace","teams","microsoft teams","channel","channels","dm","dms","direct message","thread","threads","#","@"}
_FILE_WORDS   = {"file","files","document","documents","pdf","ppt","pptx","xls","xlsx","doc","docx","sheet","slides","drive","sharepoint","dropbox","box"}
_FILE_READ    = {"read","summarize","summary","extract","parse","outline","highlights","key points","table of contents"}
_CAL_WORDS    = {"calendar","meeting","meetings","event","events","schedule","appointment","invite"}
_WEB_WORDS    = {"web","website","online","news","price","prices","weather","who is","what is","exchange rate","score","latest","trending"}
_HR_WORDS     = {"leave","vacation","holiday","payroll","attendance","policy","overtime","benefits","salary","hr"}

# Email compose lexicon
_SEND_VERBS   = {"send","compose","write","draft","email"}
_SEND_FIELDS  = {"subject","cc","bcc"}
_SEND_TO_RX   = re.compile(r"\b(email|mail)\s+(to|@)\b", re.I)
_TO_EMAIL_RX  = re.compile(r"\bto\s+([A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9\-.]+)\b")

# Calendar CREATE lexicon
_CAL_CREATE_VERBS = {"create","schedule","set up","setup","book","add","arrange","organize","organise","plan"}
_CAL_CREATE_NOUNS = {"meeting","event","appointment","call","sync","catchup","catch-up","demo","review","standup","stand-up","session"}

def _score_intents(text: str) -> Tuple[str, float, Dict[str, Any]]:
    low = (text or "").lower()

    # provider / entity hints
    channels = re.findall(r"#([a-z0-9_\-]+)", low)
    people   = re.findall(r"@([a-z0-9_\.\-]+)", low)
    provider_hint = None
    if "slack" in low or channels or ("workspace" in low and "teams" not in low):
        provider_hint = "slack"
    elif "teams" in low or "microsoft teams" in low:
        provider_hint = "teams"
    elif "gmail" in low:
        provider_hint = "gmail"
    elif "outlook" in low:
        provider_hint = "outlook"

    def has_any(words): return any(w in low for w in words)
    def weight(*pairs):
        return sum(w for cond, w in pairs if cond)

    # send_email (compose/send flow)
    has_send_verb   = has_any(_SEND_VERBS)
    has_email_word  = ("email" in low) or ("e-mail" in low) or (" mail" in low)
    has_email_to_rx = bool(_SEND_TO_RX.search(low))
    mentions_fields = has_any(_SEND_FIELDS) or ("subject:" in low)
    has_explicit_to = bool(_TO_EMAIL_RX.search(low))
    s_send = weight(
        (has_send_verb and has_email_word, 0.7),
        (has_email_to_rx or has_explicit_to, 0.3),
        (mentions_fields, 0.2),
    )

    # message_search (Slack/Teams)
    s_msg = weight(
        (has_any(_MSG_WORDS),                0.6),
        (bool(channels),                     0.2),
        (bool(people),                       0.1),
        ("chat" in low or "message" in low,  0.1),
    )

    # email_search (inbox queries)
    s_email = weight(
        (has_any(_EMAIL_WORDS),              0.6),
        ("inbox" in low,                     0.2),
        ("email" in low and not provider_hint, 0.1),
    )

    # file_search vs file_content
    s_file = weight(
        (has_any(_FILE_WORDS),               0.5),
        ("find" in low or "search" in low,   0.2),
        ("in drive" in low or "in sharepoint" in low, 0.1),
    )
    s_file_content = weight(
        (has_any(_FILE_WORDS),               0.3),
        (has_any(_FILE_READ),                0.5),
        ("summarise" in low or "summarize" in low, 0.2),
    )

    # calendar_create (explicit scheduling)
    has_create_verb = has_any(_CAL_CREATE_VERBS)
    has_meet_noun   = has_any(_CAL_CREATE_NOUNS) or ("calendar" in low)
    has_time_hint   = bool(re.search(r"\b\d{1,2}:\d{2}\b", low) or re.search(r"\b(am|pm)\b", low)) \
                      or any(w in low for w in ("today","tomorrow","next","this"))
    s_cal_create = weight(
        (has_create_verb and has_meet_noun, 0.7),
        (has_time_hint,                      0.2),
        ("invite" in low,                    0.1),
    )

    # calendar_search (querying existing calendar)
    s_cal_search = weight(
        (has_any(_CAL_WORDS),                0.7),
        ("today" in low or "tomorrow" in low,0.1),
    )

    # web_search
    s_web = weight(
        (has_any(_WEB_WORDS),                0.6),
        ("http://" in low or "https://" in low, 0.2),
        ("google" in low and "drive" not in low, 0.1),
    )

    # hr_admin
    s_hr = weight((has_any(_HR_WORDS), 0.7))

    scored = [
        ("send_email",       s_send),
        ("message_search",   s_msg),
        ("email_search",     s_email),
        ("file_content",     s_file_content),
        ("file_search",      s_file),
        ("calendar_create",  s_cal_create),
        ("calendar_search",  s_cal_search),
        ("web_search",       s_web),
        ("hr_admin",         s_hr),
        ("general",          0.0),
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    intent, score = scored[0]

    # resolve tie between file_search vs file_content
    if intent == "file_search" and s_file_content >= s_file - 0.05:
        intent, score = "file_content", s_file_content

    extras = {
        "provider_hint": provider_hint,
        "channels": [f"#{c}" for c in channels][:5],
        "people":   [f"@{p}" for p in people][:5],
    }
    return intent, float(score), extras

# ------------- Public API -------------
def detect_intent_and_extract(user_input: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "intent": "...",
        "data": "<keywords string>",        # back-compat for your callers
        "confidence": float,
        "time_min_iso": str|None,
        "time_max_iso": str|None,
        # NEW (safe to ignore by existing code):
        "provider_hint": "slack|teams|gmail|outlook|drive|sharepoint|web|None",
        "channels": ["#finance", ...],
        "people": ["@alice", ...],
      }
    """
    ui = (user_input or "").strip()
    if not ui:
        return {"intent":"general","data":"","confidence":0.0,"time_min_iso":None,"time_max_iso":None}

    # 1) Ask LLM for a first pass (strict JSON)
    sys = _INTENT_SYSTEM_JSON
    raw = answer_general_query(f"User: {ui}\nReturn STRICT JSON per the schema.", system_prompt=sys)
    parsed = _json_from_text(raw) or {}

    # 2) Read LLM fields
    intent = str(parsed.get("intent","") or "").strip().lower()
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    kw_list = parsed.get("keywords") or []
    if not isinstance(kw_list, list): kw_list = []
    kw = " ".join([str(k).strip() for k in kw_list if isinstance(k,str)])

    # 3) Our deterministic scorer (wins if LLM is low/uncertain)
    h_intent, h_score, extras = _score_intents(ui)
    if intent not in {"send_email","email_search","calendar_create","calendar_search","file_search","file_content","web_search","message_search","hr_admin","general"}:
        intent, confidence = h_intent, max(0.55, h_score)
    else:
        if h_score >= (confidence + 0.15):
            intent, confidence = h_intent, h_score

    # 4) Keywords
    data = _normalize_space(kw) or _rule_keywords(ui)

    # 5) Time window (relative parser takes precedence)
    tmin, tmax = _parse_window_local(ui)
    if not (tmin or tmax):
        ltmin = parsed.get("time_min_iso") or None
        ltmax = parsed.get("time_max_iso") or None
        def _sane(isoz: Optional[str]) -> bool:
            if not isoz: return False
            try:
                dt = datetime.strptime(isoz, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except Exception: return False
            now = datetime.now(timezone.utc)
            return abs(dt.year - now.year) <= 1
        if _sane(ltmin): tmin = ltmin
        if _sane(ltmax): tmax = ltmax

    out = {
        "intent": intent,
        "data": data,
        "confidence": float(round(confidence, 3)),
        "time_min_iso": tmin,
        "time_max_iso": tmax,
    }
    out.update(extras)
    return out

def extract_keywords(user_input: str, max_terms: int = 8) -> str:
    res = detect_intent_and_extract(user_input)
    data = res.get("data") or _rule_keywords(user_input, max_terms=max_terms)
    return " ".join(str(data).split()[:max_terms])

# ------------- Chat-flow rules (unchanged API) -------------
_RULE_WORDS_RE = re.compile(r"\b(?:answer|reply|respond|write|speak)\b.*?\b(?:in|using|with)\b.*?\b(\d+)\b\s*words?\b", re.I)
_RULE_ALWAYS_RE = re.compile(r"\b(?:always|every time|from now on|going forward|henceforth)\b.*?\b(\d+)\b\s*words?\b", re.I)
_RULE_RESET_RE = re.compile(r"\b(?:reset|forget|clear)\b.*?\b(?:rules|instructions|flow)\b", re.I)
_RULE_LANG_RE  = re.compile(r"\b(?:in|use|reply in)\b\s+(english|bangla|bengali|hindi|spanish|german|french|arabic|urdu|chinese)\b", re.I)

_LANG_MAP = {
    "bangla":"Bengali","bengali":"Bengali","english":"English","hindi":"Hindi",
    "spanish":"Spanish","german":"German","french":"French","arabic":"Arabic",
    "urdu":"Urdu","chinese":"Chinese",
}

def detect_chat_flow_rules(utterance: str) -> dict:
    u = (utterance or "").strip(); low = u.lower()
    if _RULE_RESET_RE.search(low): return {"reset": True}
    rules: dict = {}
    m = _RULE_ALWAYS_RE.search(low) or _RULE_WORDS_RE.search(low)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 50: rules["max_words"] = n
        except Exception: pass
    ml = _RULE_LANG_RE.search(low)
    if ml:
        lang = ml.group(1).strip().lower()
        rules["language"] = _LANG_MAP.get(lang, lang.title())
    if "words" in low and not rules.get("max_words"):
        lone = re.search(r"\b(\d+)\s*words?\b", low)
        if lone:
            try:
                n = int(lone.group(1))
                if 1 <= n <= 50: rules["max_words"] = n
            except Exception: pass
    return rules

def apply_chat_rules_to_text(text: str, rules: Optional[dict]) -> str:
    if not text or not rules: return text or ""
    out = text.strip()
    n = rules.get("max_words")
    if isinstance(n, int) and n > 0:
        words = _WHITESPACE_RE.split(out)
        if len(words) > n: out = " ".join(words[:n])
    return out

def _system_prompt_from_rules(base_prompt: str, rules: Optional[dict]) -> str:
    if not rules: return base_prompt
    extras = []
    if rules.get("language"):
        extras.append(f"Always reply in {rules['language']}. Do not switch languages.")
    if rules.get("max_words"):
        extras.append(f"Keep the entire reply to at most {int(rules['max_words'])} words. Avoid greetings or filler.")
    return base_prompt + (" " + " ".join(extras) if extras else "")

def answer_general_query(
    user_input: str,
    history: List[dict] | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 768,
    chat_rules: dict | None = None,
) -> str:
    if _llm is None: return ""
    base_sys = system_prompt or (
        "You are ECHO, a helpful assistant. Answer the user's question directly. "
        "If unsure, say you are unsure. Do not invent file links or claim you searched storage."
    )
    sys_msg = _system_prompt_from_rules(base_sys, chat_rules)
    convo = [f"system: {sys_msg}"]
    if history:
        for m in history[-6:]:
            role = m.get("role","user"); content = m.get("content","")
            convo.append(f"{role}: {content}")
    convo.append(f"user: {user_input}")
    prompt = "\n".join(convo)
    try:
        out = _llm.invoke(prompt)
        text = getattr(out, "content", None) or str(out)
        return apply_chat_rules_to_text((text or "").strip(), chat_rules)
    except Exception:
        try:
            text = _llm.predict(prompt)
            return apply_chat_rules_to_text((text or "").strip(), chat_rules)
        except Exception:
            return ""

# Back-compat alias
def answere_general_query(*args, **kwargs):
    return answer_general_query(*args, **kwargs)

if __name__ == "__main__":
    samples = [
        "send an email to john@acme.com about the Q2 report",
        "schedule a 45-min meeting with sara@acme.com on Thursday at 3pm",
        "search slack #finance for pike valuation last week",
        "show my inbox today",
        "what meetings do I have next week",
        "summarize the Q2 board deck PDF",
        "find p&l 2024 xlsx in drive",
        "what is the latest USD rate today",
        "from now on answer in 3 words",
    ]
    rules = {}
    for t in samples:
        print(">", t)
        print(" rules:", detect_chat_flow_rules(t))
        print(" intent:", detect_intent_and_extract(t))
