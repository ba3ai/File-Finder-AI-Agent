# email_service.py — Guided, LLM-driven personal email assistant
# Flow:
#   awaiting_recipient → choose_mode → (awaiting_generation_instructions | awaiting_user_body)
#   → review ↔ awaiting_mod_instructions → send
#
# Friendly tone + personalization + iterative LLM edits.

from __future__ import annotations
import re, json, base64, logging, requests
from datetime import datetime
from typing import Optional, Tuple, Callable, List, Iterable
from flask import session, current_app
try:
    from flask_login import current_user
except Exception:
    current_user = None  # graceful if flask_login isn't available

# ───────────────────────────────────────────────────────────
# Optional DB logging (best-effort)
# ───────────────────────────────────────────────────────────
_db = None
_EmailLogModel = None
try:
    from extensions import db as _db  # type: ignore
except Exception:
    try:
        from db import db as _db  # type: ignore
    except Exception:
        _db = None
if _db is not None:
    try:
        from models import EmailLog as _EmailLogModel  # type: ignore
    except Exception:
        _EmailLogModel = None

def _persist_log(user_id, provider, to, subject, body, status, err=None):
    try:
        if _db is None:
            return
        if _EmailLogModel is None:
            class EmailLog(_db.Model):  # type: ignore
                __tablename__ = "email_log"
                id = _db.Column(_db.Integer, primary_key=True)
                user_id = _db.Column(_db.Integer, nullable=True)
                provider = _db.Column(_db.String(20))
                to = _db.Column(_db.String(2048))
                subject = _db.Column(_db.String(998))
                body_preview = _db.Column(_db.Text)
                status = _db.Column(_db.String(20))
                error = _db.Column(_db.Text, nullable=True)
                created_at = _db.Column(_db.DateTime, default=datetime.utcnow)
            _db.create_all()
            globals()["_EmailLogModel"] = EmailLog  # type: ignore

        preview = (body or "")[:5000]
        rec = _EmailLogModel(  # type: ignore
            user_id=user_id,
            provider=provider,
            to=to if isinstance(to, str) else ", ".join(to or []),
            subject=subject,
            body_preview=preview,
            status=status,
            error=(err or "")[:4000],
            created_at=datetime.utcnow(),
        )
        _db.session.add(rec)
        _db.session.commit()
    except Exception as e:
        logging.exception("Email logging failed: %s", e)

# ───────────────────────────────────────────────────────────
# Intent trigger (so router can hand control to this flow)
# ───────────────────────────────────────────────────────────
_INTENT_PHRASES = (
    "send email","send an email","write an email","compose email",
    "email to","email someone","send a mail","compose a mail","mail to"
)
def is_send_email_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _INTENT_PHRASES)

# ───────────────────────────────────────────────────────────
# Provider loaders
# ───────────────────────────────────────────────────────────
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
    try:
        from msal_auth import load_token_cache, build_msal_app
        cache = load_token_cache(); app = build_msal_app(cache=cache)
        scopes = ["Mail.Send"]
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
    try:
        data = session.get("gmail_credentials") or session.get("google_credentials")
        if data:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            creds = Credentials.from_authorized_user_info(
                data, scopes=["https://www.googleapis.com/auth/gmail.send"]
            )
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            return creds
    except Exception:
        logging.debug("Session Gmail creds load failed", exc_info=True)
    try:
        from pathlib import Path
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        inst = Path(getattr(current_app, "instance_path", ".")) / "gmail_token.json"
        if inst.exists():
            info = json.loads(inst.read_text())
            creds = Credentials.from_authorized_user_info(
                info, scopes=["https://www.googleapis.com/auth/gmail.send"]
            )
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            return creds
    except Exception:
        logging.debug("Instance Gmail creds load failed", exc_info=True)
    return None

def _detect_provider_for_user() -> Tuple[Optional[str], Optional[object]]:
    tok = _get_outlook_access_token()
    if tok: return ("outlook", tok)
    creds = _get_gmail_credentials()
    if creds: return ("gmail", creds)
    return (None, None)

# ───────────────────────────────────────────────────────────
# User display name helpers (for personalization)
# ───────────────────────────────────────────────────────────
def _titlecase_from_email_local(local: str) -> str:
    if not local:
        return "You"
    local = local.replace(".", " ").replace("_", " ").replace("-", " ")
    return " ".join(w.capitalize() for w in local.split() if w)

def _get_user_display_name() -> str:
    cu = current_user
    for attr in ("full_name","name","display_name","given_name","first_name"):
        try:
            val = getattr(cu, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            pass
    email = None
    try:
        email = getattr(cu, "email", None)
    except Exception:
        pass
    if not email:
        email = session.get("user_email") or session.get("email")
    local = (email or "").split("@")[0]
    return _titlecase_from_email_local(local) or "You"

# Replace placeholder tokens and tidy the body
_NAME_TOKEN_RX = re.compile(r"(?i)\[\s*(your|sender|my)\s+name\s*\]")
_SUBJECT_LINE_RX = re.compile(r"(?im)^\s*subject\s*:\s*.+?$")
_SIGNOFF_LINE_RX = re.compile(r"(?im)^(best regards|regards|sincerely|thanks|thank you|kind regards)\s*,\s*$")

def _apply_sender_name(body: str, sender_name: str) -> str:
    if not body:
        return body
    b = _NAME_TOKEN_RX.sub(sender_name, body)

    # Append name after a bare sign-off line ONLY if the next line is empty/missing
    if _SIGNOFF_LINE_RX.search(b):
        lines = b.splitlines()
        for i, ln in enumerate(lines):
            if _SIGNOFF_LINE_RX.match(ln):
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if not next_line:  # no name already present
                    lines[i] = ln + "\n" + sender_name
                b = "\n".join(lines)
                break
    return b

def _strip_subject_line_in_body(body: str) -> str:
    if not body:
        return body
    lines = [ln for ln in body.splitlines() if not _SUBJECT_LINE_RX.match(ln)]
    return "\n".join(lines).strip()

# ───────────────────────────────────────────────────────────
# Address parsing
# ───────────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)")

def _extract_emails(text: str) -> List[str]:
    raw = re.findall(EMAIL_RE, text or "")
    seen, out = set(), []
    for a in raw:
        a2 = a.strip().strip(",;")
        if a2 and a2.lower() not in seen:
            seen.add(a2.lower()); out.append(a2)
    return out

def _normalize_recipients(to: str | Iterable[str]) -> List[str]:
    if isinstance(to, str):
        return _extract_emails(to)
    flat: List[str] = []
    for item in list(to or []):
        flat += _extract_emails(item if isinstance(item, str) else str(item))
    return flat

def _parse_cc_bcc(text: str) -> tuple[list[str], list[str]]:
    low = text or ""
    m_cc = re.search(r"\bcc[:\s]\s*([^\n]+)", low, flags=re.I)
    m_bc = re.search(r"\bbcc[:\s]\s*([^\n]+)", low, flags=re.I)
    cc = _extract_emails(m_cc.group(1)) if m_cc else []
    bcc = _extract_emails(m_bc.group(1)) if m_bc else []
    return cc, bcc

# ───────────────────────────────────────────────────────────
# LLM helpers (generation + revision + subject)
# ───────────────────────────────────────────────────────────
def _llm_call(user_prompt: str, system_prompt: str, max_tokens: int = 400) -> str:
    try:
        from openai_api import answer_general_query
    except Exception:
        logging.debug("openai_api.answer_general_query not available")
        return ""
    try:
        return (answer_general_query(user_prompt, system_prompt=system_prompt, max_tokens=max_tokens) or "").strip()
    except TypeError:
        return (answer_general_query(f"{system_prompt}\n\n{user_prompt}") or "").strip()
    except Exception:
        logging.exception("LLM call failed")
        return ""

_BODY_SYSTEM = (
    "You write clear, friendly email bodies in plain text. "
    "Keep it concise, helpful, and natural. Avoid robotic phrasing. "
    "Do not add a signature unless the user implies one."
)
_SUBJECT_SYSTEM = (
    "You generate concise, professional email subjects. "
    "Return ONLY the subject line, no quotes, <= 10 words."
)
_EDIT_SYSTEM = (
    "You are an email editor. Rewrite the email to follow the user's requested changes. "
    "Keep it plain text, concise, helpful, and natural. Do not add a signature unless asked."
)

def _generate_body_from_instructions(instructions: str) -> str:
    prompt = (
        "Draft a short, friendly email body following these instructions.\n"
        "Use plain text, no markdown, no quotes.\n\n"
        f"Instructions: {instructions.strip()}"
    )
    return _llm_call(prompt, _BODY_SYSTEM, max_tokens=350)

def _revise_body(original: str, change_instructions: str) -> str:
    prompt = (
        "Revise the following email body according to the user's change request.\n"
        "Return only the updated body in plain text.\n\n"
        f"Original:\n{original.strip()}\n\n"
        f"Changes: {change_instructions.strip()}"
    )
    return _llm_call(prompt, _EDIT_SYSTEM, max_tokens=350)

def _generate_subject_from_body(body_text: str) -> str:
    prompt = (
        "Generate a professional email subject based on this body. "
        "Return only the subject line (no quotes):\n\n"
        f"{body_text.strip()}"
    )
    return _llm_call(prompt, _SUBJECT_SYSTEM, max_tokens=40)

# ───────────────────────────────────────────────────────────
# Senders
# ───────────────────────────────────────────────────────────
def _send_via_outlook(access_token: str,
                      to: List[str] | str,
                      subject: str,
                      body_html: str,
                      cc: Optional[List[str]] = None,
                      bcc: Optional[List[str]] = None) -> bool:
    url = "https://graph.microsoft.com/v1.0/me/sendMail"
    recipients = [{"emailAddress": {"address": a}} for a in _normalize_recipients(to)]
    if not recipients:
        raise ValueError("No valid recipient address.")
    msg = {
        "message": {
            "subject": subject or "",
            "body": {"contentType": "HTML", "content": body_html or ""},
            "toRecipients": recipients,
        },
        "saveToSentItems": True,
    }
    if cc:
        msg["message"]["ccRecipients"] = [{"emailAddress": {"address": a}} for a in _normalize_recipients(cc)]
    if bcc:
        msg["message"]["bccRecipients"] = [{"emailAddress": {"address": a}} for a in _normalize_recipients(bcc)]
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=msg, timeout=25)
    if r.status_code in (202, 200): return True
    logging.warning("Graph sendMail failed: %s %s", r.status_code, r.text[:500])
    try: r.raise_for_status()
    except Exception: pass
    return False

def _send_via_gmail(creds,
                    to: List[str] | str,
                    subject: str,
                    body_html: str,
                    cc: Optional[List[str]] = None,
                    bcc: Optional[List[str]] = None) -> bool:
    from email.mime.text import MIMEText
    from googleapiclient.discovery import build
    to_list = _normalize_recipients(to)
    if not to_list:
        raise ValueError("No valid recipient address.")
    msg = MIMEText(body_html or "", "html")
    msg["to"] = ", ".join(to_list)
    msg["subject"] = subject or ""
    if cc:
        msg["cc"] = ", ".join(_normalize_recipients(cc))
    if bcc:
        msg["bcc"] = ", ".join(_normalize_recipients(bcc))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    body = {"raw": raw}
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    service.users().messages().send(userId="me", body=body).execute()
    return True

# ───────────────────────────────────────────────────────────
# Flow state
# ───────────────────────────────────────────────────────────
def _init_flow():
    session['email_flow'] = {
        # stages: awaiting_recipient → choose_mode → awaiting_generation_instructions | awaiting_user_body
        # → review ↔ awaiting_mod_instructions → send
        "stage": "awaiting_recipient",
        "provider": None,
        "to": [], "cc": [], "bcc": [],
        "subject": None,
        "body": None,
    }

def _reset_flow():
    session.pop('email_flow', None)

def reset_email_flow():
    _reset_flow()

# ───────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────
def continue_email_flow(user_id: Optional[int], user_text: str) -> dict:
    flow = session.get('email_flow')
    text = (user_text or "").strip()

    # start
    if not flow:
        _init_flow()
        flow = session['email_flow']
        provider, obj = _detect_provider_for_user()
        flow["provider"] = provider
        session.modified = True
        if not provider:
            _reset_flow()
            return {
                "reply": "Heads up — I’m not connected to Outlook or Gmail yet. Please connect one in the dashboard and I’ll handle your email from there. ✨",
                "meta": {"mode":"email_flow","stage":"needs_connection"},
                "done": True, "sent": False
            }
        return {
            "reply": "Great — who should I send this to? You can list multiple addresses, and add `cc:` / `bcc:` if you like.",
            "meta": {"mode":"email_flow","stage":"awaiting_recipient","provider":flow["provider"]},
            "done": False, "sent": False
        }

    # cancel
    if text.lower() in {"cancel","stop","abort","quit"}:
        _reset_flow()
        return {"reply":"All good — I’ve cancelled the email. If you change your mind, just say “send an email.”",
                "meta":{"mode":"email_flow","stage":"cancelled"},"done":True,"sent":False}

    stage = flow["stage"]
    provider = flow["provider"]

    # creds
    provider_obj = _get_outlook_access_token() if provider == "outlook" else _get_gmail_credentials()
    if not provider_obj:
        new_p, new_o = _detect_provider_for_user()
        if new_p and new_o:
            flow["provider"] = provider = new_p
            provider_obj = new_o
            session.modified = True
        else:
            _reset_flow()
            return {"reply":"Hmm, I lost access to your mail account. Please reconnect and we’ll pick up right where we left off.",
                    "meta":{"mode":"email_flow","stage":"error","provider":provider},"done":True,"sent":False}

    # ───── Stage: awaiting_recipient ─────
    if stage == "awaiting_recipient":
        found_to = _extract_emails(text)
        cc, bcc = _parse_cc_bcc(text)
        if found_to: flow["to"] = list({a.lower(): a for a in found_to}.values())
        if cc: flow["cc"] = list({a.lower(): a for a in cc}.values())
        if bcc: flow["bcc"] = list({a.lower(): a for a in bcc}.values())
        if not flow["to"]:
            return {"reply":"Could you share at least one recipient email? You can also add `cc:` and `bcc:`.",
                    "meta":{"mode":"email_flow","stage":stage,"provider":provider},
                    "done":False,"sent":False}

        flow["stage"] = "choose_mode"; session.modified = True
        who = ", ".join(flow["to"])
        return {"reply": f"Nice — I’ll email **{who}**. Would you like me to **draft the message** or do you want to **paste your own**?\n\nReply **draft** or **provide**.",
                "meta":{"mode":"email_flow","stage":"choose_mode","provider":provider},
                "done":False,"sent":False}

    # ───── Stage: choose_mode ─────
    if stage == "choose_mode":
        lower = text.lower()
        if EMAIL_RE.search(text) or "cc:" in lower or "bcc:" in lower:
            more_to = _extract_emails(text)
            if more_to:
                s = {a.lower() for a in flow["to"]}
                for a in more_to:
                    if a.lower() not in s:
                        flow["to"].append(a); s.add(a.lower())
            cc, bcc = _parse_cc_bcc(text)
            if cc:
                s = {a.lower() for a in flow["cc"]}
                for a in cc:
                    if a.lower() not in s: flow["cc"].append(a); s.add(a.lower())
            if bcc:
                s = {a.lower() for a in flow["bcc"]}
                for a in bcc:
                    if a.lower() not in s: flow["bcc"].append(a); s.add(a.lower())

        if lower.startswith(("draft", "write", "compose", "generate", "create")) or lower == "draft":
            flow["stage"] = "awaiting_generation_instructions"; session.modified = True
            return {"reply":"Awesome — tell me what kind of note you want (purpose, key points, tone, any must-say lines). I’ll whip up a draft.",
                    "meta":{"mode":"email_flow","stage":"awaiting_generation_instructions","provider":provider},
                    "done":False,"sent":False}
        if lower.startswith(("provide","paste","custom","my own","i will provide","i will write")) or lower == "provide":
            flow["stage"] = "awaiting_user_body"; session.modified = True
            return {"reply":"Perfect — paste your message body here. I’ll handle the subject automatically.",
                    "meta":{"mode":"email_flow","stage":"awaiting_user_body","provider":provider},
                    "done":False,"sent":False}

        return {"reply":"Just say **draft** (I’ll write it) or **provide** (you’ll paste it).",
                "meta":{"mode":"email_flow","stage":"choose_mode","provider":provider},
                "done":False,"sent":False}

    # ───── Stage: awaiting_generation_instructions ─────
    if stage == "awaiting_generation_instructions":
        instr = text.strip()
        if not instr:
            return {"reply":"Tell me the gist — what’s this email about and what tone should I use?",
                    "meta":{"mode":"email_flow","stage":stage,"provider":provider},
                    "done":False,"sent":False}
        body = _generate_body_from_instructions(instr) or "Hello,\n\n[Your Name]\n"
        body = _strip_subject_line_in_body(body)
        body = _apply_sender_name(body, _get_user_display_name())
        flow["body"] = body
        flow["subject"] = _generate_subject_from_body(body) or ""
        flow["stage"] = "review"; session.modified = True

        shown_to = ", ".join(flow["to"] or [])
        shown_cc = ", ".join(flow["cc"] or []) or "—"
        shown_bcc = ", ".join(flow["bcc"] or []) or "—"
        preview = (f"**Here’s a first pass — take a look**\n\n"
                   f"**To:** {shown_to}\n**CC:** {shown_cc}\n**BCC:** {shown_bcc}\n"
                   f"**Subject:** {flow['subject']}\n**Body:**\n{flow['body']}\n\n"
                   "Like it as-is? Reply **yes** to send, **modify** to tweak, or **no** to cancel.")
        return {"reply": preview,
                "meta":{"mode":"email_flow","stage":"review","provider":provider},
                "done":False,"sent":False}

    # ───── Stage: awaiting_user_body ─────
    if stage == "awaiting_user_body":
        body = text.strip()
        if not body:
            return {"reply":"Go ahead and paste your message. I’ll format the rest.",
                    "meta":{"mode":"email_flow","stage":stage,"provider":provider},
                    "done":False,"sent":False}
        body = _strip_subject_line_in_body(body)
        body = _apply_sender_name(body, _get_user_display_name())
        flow["body"] = body
        flow["subject"] = _generate_subject_from_body(body) or ""
        flow["stage"] = "review"; session.modified = True

        shown_to = ", ".join(flow["to"] or [])
        shown_cc = ", ".join(flow["cc"] or []) or "—"
        shown_bcc = ", ".join(flow["bcc"] or []) or "—"
        preview = (f"**Quick preview**\n\n"
                   f"**To:** {shown_to}\n**CC:** {shown_cc}\n**BCC:** {shown_bcc}\n"
                   f"**Subject:** {flow['subject']}\n**Body:**\n{flow['body']}\n\n"
                   "Shall I send it? Reply **yes** to send, **modify** to change, or **no** to cancel.")
        return {"reply": preview,
                "meta":{"mode":"email_flow","stage":"review","provider":provider},
                "done":False,"sent":False}

    # ───── Stage: review (confirm/modify) ─────
    if stage == "review":
        low = text.lower().strip()

        # quick recipient/cc/bcc changes allowed here
        if EMAIL_RE.search(text) or "cc:" in low or "bcc:" in low:
            more_to = _extract_emails(text)
            if more_to:
                s = {a.lower() for a in flow["to"]}
                for a in more_to:
                    if a.lower() not in s: flow["to"].append(a); s.add(a.lower())
            cc, bcc = _parse_cc_bcc(text)
            if cc:
                s = {a.lower() for a in flow["cc"]}
                for a in cc:
                    if a.lower() not in s: flow["cc"].append(a); s.add(a.lower())
            if bcc:
                s = {a.lower() for a in flow["bcc"]}
                for a in bcc:
                    if a.lower() not in s: flow["bcc"].append(a); s.add(a.lower())
            session.modified = True
            low = "preview"

        if low in {"yes","y"}:
            try:
                to_list  = flow["to"] or []
                cc_list  = flow.get("cc") or []
                bcc_list = flow.get("bcc") or []
                subject  = flow.get("subject") or ""
                body     = flow.get("body") or ""
                html_body = body.replace("\n", "<br>")

                if provider == "outlook":
                    ok = _send_via_outlook(provider_obj, to=to_list, subject=subject, body_html=html_body,
                                           cc=cc_list, bcc=bcc_list)
                else:
                    ok = _send_via_gmail(provider_obj, to=to_list, subject=subject, body_html=html_body,
                                         cc=cc_list, bcc=bcc_list)

                if ok:
                    _persist_log(user_id, provider, to_list, subject, body, "sent", None)
                    _reset_flow()
                    return {"reply":"✅ All set — your email is on its way.",
                            "meta":{"mode":"email_flow","stage":"done","provider":provider},
                            "done": True, "sent": True}
                else:
                    _persist_log(user_id, provider, to_list, subject, body, "failed", "Unknown error")
                    _reset_flow()
                    return {"reply":"❌ Shoot — I couldn’t send that. Mind reconnecting your mail account and trying again?",
                            "meta":{"mode":"email_flow","stage":"done","provider":provider},
                            "done": True, "sent": False}
            except Exception as e:
                _persist_log(user_id, provider, flow.get("to"), flow.get("subject"),
                             flow.get("body"), "failed", str(e))
                _reset_flow()
                return {"reply": f"❌ I hit an error while sending: {e}",
                        "meta":{"mode":"email_flow","stage":"done","provider":provider},
                        "done": True, "sent": False}

        if low in {"no","n","cancel"}:
            _reset_flow()
            return {"reply":"No worries — I won’t send it. If you want to try again, just say the word.",
                    "meta":{"mode":"email_flow","stage":"cancelled","provider":provider},
                    "done": True, "sent": False}

        # Modify flow — now supports a dedicated next-turn stage
        if low.startswith(("modify","edit","revise","change")):
            # Inline instructions: "modify: make it friendlier"
            if ":" in text:
                instr = text.split(":",1)[1].strip()
                if instr:
                    new_body = _revise_body(flow["body"] or "", instr)
                    new_body = _strip_subject_line_in_body(new_body)
                    new_body = _apply_sender_name(new_body, _get_user_display_name())
                    flow["body"] = new_body or (flow["body"] or "")
                    flow["subject"] = _generate_subject_from_body(flow["body"] or "") or (flow["subject"] or "")
                    session.modified = True
                else:
                    flow["stage"] = "awaiting_mod_instructions"; session.modified = True
                    return {"reply":"Sure — what should I change or add?",
                            "meta":{"mode":"email_flow","stage":"awaiting_mod_instructions","provider":provider},
                            "done":False,"sent":False}
            else:
                flow["stage"] = "awaiting_mod_instructions"; session.modified = True
                return {"reply":"Got it — tell me how to tweak the draft (tone, details, additions).",
                        "meta":{"mode":"email_flow","stage":"awaiting_mod_instructions","provider":provider},
                        "done":False,"sent":False}

        elif low.startswith("subject:"):
            flow["subject"] = text.split(":",1)[1].strip(); session.modified = True
        elif low.startswith("body:"):
            body = text.split(":",1)[1].strip()
            body = _strip_subject_line_in_body(body)
            body = _apply_sender_name(body, _get_user_display_name())
            flow["body"] = body; session.modified = True
        elif low == "preview":
            pass
        else:
            return {"reply":"Say **yes** to send, **no** to cancel, **modify** to edit (or `modify: …` with details), or update directly with `subject: …` / `body: …`. You can also add recipients and `cc:`/`bcc:` anytime.",
                    "meta":{"mode":"email_flow","stage":"review","provider":provider},
                    "done":False,"sent":False}

        # show updated preview
        shown_to = ", ".join(flow["to"] or [])
        shown_cc = ", ".join(flow["cc"] or []) or "—"
        shown_bcc = ", ".join(flow["bcc"] or []) or "—"
        preview = (f"**Updated preview**\n\n"
                   f"**To:** {shown_to}\n**CC:** {shown_cc}\n**BCC:** {shown_bcc}\n"
                   f"**Subject:** {flow['subject']}\n**Body:**\n{flow['body']}\n\n"
                   "Good to go? **yes** to send, **modify** to adjust, **no** to cancel.")
        return {"reply": preview,
                "meta":{"mode":"email_flow","stage":"review","provider":provider},
                "done":False,"sent":False}

    # ───── NEW: Stage for next-turn modify instructions ─────
    if stage == "awaiting_mod_instructions":
        instr = text.strip()
        if not instr:
            return {"reply":"Tell me what to change (e.g., “make it warmer, add my full name, and ask for a reply by Friday”).",
                    "meta":{"mode":"email_flow","stage":"awaiting_mod_instructions","provider":provider},
                    "done":False,"sent":False}
        new_body = _revise_body(session['email_flow'].get("body",""), instr)
        new_body = _strip_subject_line_in_body(new_body)
        new_body = _apply_sender_name(new_body, _get_user_display_name())
        session['email_flow']["body"] = new_body or session['email_flow'].get("body","")
        session['email_flow']["subject"] = _generate_subject_from_body(session['email_flow']["body"]) or session['email_flow'].get("subject","")
        session['email_flow']["stage"] = "review"; session.modified = True

        shown_to = ", ".join(session['email_flow']["to"] or [])
        shown_cc = ", ".join(session['email_flow']["cc"] or []) or "—"
        shown_bcc = ", ".join(session['email_flow']["bcc"] or []) or "—"
        preview = (f"**Updated draft—take a look**\n\n"
                   f"**To:** {shown_to}\n**CC:** {shown_cc}\n**BCC:** {shown_bcc}\n"
                   f"**Subject:** {session['email_flow']['subject']}\n**Body:**\n{session['email_flow']['body']}\n\n"
                   "Happy with this? **yes** to send, **modify** to tweak again, or **no** to cancel.")
        return {"reply": preview,
                "meta":{"mode":"email_flow","stage":"review","provider":provider},
                "done":False,"sent":False}

    # fallback
    _reset_flow()
    return {"reply":"I reset the email assistant — say “send an email” and I’ll jump back in.",
            "meta":{"mode":"email_flow","stage":"reset"},"done":True,"sent":False}
