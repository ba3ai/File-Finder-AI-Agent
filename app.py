# app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import logging
from functools import wraps
from datetime import timedelta, datetime, UTC, timezone  # >>> CHANGED (added timezone)
from dateutil.parser import isoparse

from flask import Flask, request, jsonify, send_from_directory, session, current_app, redirect
from flask_session import Session
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# 🔐 Auth (custom)
from flask_login import login_required, current_user
from auth import auth_bp, login_manager
from models import db, Prefs  # <-- Prefs used to store per-user flags/tokens
import requests as _rq
# (Optional) App-only flow still available if you need it for admin/background
from graph_app_token import get_app_access_token
# --- at top of file (imports) ---
from services_web_search import web_answer  # <-- add this import

# Email flow orchestration lives in a single module email_service.py
from email_service import is_send_email_intent,continue_email_flow,register_outlook_token_loader, register_gmail_creds_loader


# >>> ADD: chat-flow helpers from openai_api
from openai_api import (
    detect_intent_and_extract,
    answer_general_query,
    detect_chat_flow_rules,         # NEW
    apply_chat_rules_to_text,       # NEW
)

# --- unified email search (optional) ---
try:
    from mail_unified import email_search_unified_smart
    HAS_MAIL_UNIFIED = True
except Exception:
    HAS_MAIL_UNIFIED = False

# unified full-mail fetch + LLM
from mail_all import email_search_all_mailbox, email_search_chatgpt_style

from summarizer import summarize_selected, needs_ms_token

from services_teams import teams_answer

# 📎 Existing modules
from graph_api import (
    search_all_files,
    check_file_access,
    send_multiple_file_email,
)
from db import (
    init_db,
    save_message,
    get_user_chats,
    get_chat_messages,
    delete_old_messages,
    delete_old_chats,
)
from hr_router import handle_query, build_hr_knowledge_json

# Google Drive / Dropbox / Box / MEGA
from google_drive_api import (
    search_drive_files_ranked,   # ranked search helper
    search_drive_files,          # simple search fallback
)
from dropbox_api import (
    search_dropbox_files_ranked,  # ranked search helper
    search_dropbox_files,         # simple search fallback
)
from box_api import (
    search_box_files_ranked,      # Box ranked search
    search_box_files,             # Box simple search fallback
)
from mega_api import search_mega_files_ranked

# Mail & Calendar
from gmail import search_gmail, gmail_smart_search
from ms_outlook_mail import search_outlook_messages, outlook_mail_smart_search
from google_calendar import search_google_calendar, google_calendar_smart_search
from ms_outlook_calendar import search_outlook_events, outlook_calendar_smart_search, outlook_calendar_assistant_reply

import re  # >>> ADDED (ensure available for regex)
from zoneinfo import ZoneInfo  # >>> ADDED

# 🔌 Dashboard connections API
from connections_api import connections_bp
from smart_content import summarize_from_hits

# ─────────────────────────────────────────────────────────────────────────────
# 🌱 Env & logging
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# 🚀 Serve the built React app (Vite build outputs to frontend/dist)
app = Flask(__name__, static_folder="./frontend/dist", static_url_path="/")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.getenv("CLIENT_SECRET", "dev-secret"))

# Sessions (server-side; used for chat state/pagination)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.permanent_session_lifetime = timedelta(hours=1)
CORS(app, supports_credentials=True)

SESSION_DIR = os.path.join(os.getcwd(), "flask_session")
os.makedirs(SESSION_DIR, exist_ok=True)
Session(app)

# DB + Auth
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
login_manager.init_app(app)
with app.app_context():
    db.create_all()

# Blueprints
app.register_blueprint(auth_bp)          # /api/auth/*
app.register_blueprint(connections_bp)   # /api/connections/*

# Initialize your app-specific tables if any
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Trial / subscription guard
# ─────────────────────────────────────────────────────────────────────────────
TRIAL_GATE_ENABLED = os.getenv("TRIAL_ENABLED", "true").lower() == "true"
TRIAL_DAYS = int(os.getenv("TRIAL_DAYS", "7"))

def _get_pref(uid, key):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else None

def _set_pref(uid, key, value):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    if row:
        row.value = value
    else:
        db.session.add(Prefs(user_id=uid, key=key, value=value))
    db.session.commit()

def ensure_trial_window(uid: int):
    """
    Create or adjust the user's trial window. If TRIAL_DAYS shrinks (e.g., 0),
    clamp the stored end time accordingly.
    """
    target_end = (datetime.now(UTC) + timedelta(days=TRIAL_DAYS)).isoformat()
    existing = _get_pref(uid, "trial_end_at")
    if not existing:
        _set_pref(uid, "trial_end_at", target_end)
        return
    # fix naive/corrupt values
    try:
        cur = isoparse(existing)
        if cur.tzinfo is None:
            raise ValueError("naive")
    except Exception:
        _set_pref(uid, "trial_end_at", target_end)
        return
    # clamp if the new window is shorter
    target_dt = isoparse(target_end)
    if cur > target_dt:
        _set_pref(uid, "trial_end_at", target_end)

def get_trial_status(uid: int):
    ends = _get_pref(uid, "trial_end_at")
    if not ends:
        return {"active": True, "expired": False, "days_left": TRIAL_DAYS}
    try:
        end_dt = isoparse(ends)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)
    except Exception:
        end_dt = datetime.now(UTC) + timedelta(days=1)
        _set_pref(uid, "trial_end_at", end_dt.isoformat())
    now = datetime.now(UTC)
    expired = now >= end_dt
    days_left = max(0, (end_dt - now).days)
    return {"active": not expired, "expired": expired, "ends_at": end_dt.isoformat(), "days_left": days_left}

# allow ONLY what the trial page/auth needs
_TRIAL_WHITELIST_PREFIXES = ("/assets", "/static", "/api/auth")
_TRIAL_WHITELIST_EXACT = ("/api/trial/status", "/trial-ended", "/favicon.ico")

@app.before_request
def enforce_trial_guard():
    if not TRIAL_GATE_ENABLED:
        return
    if not getattr(current_user, "is_authenticated", False):
        return

    ensure_trial_window(current_user.id)

    path = request.path or "/"
    if path in _TRIAL_WHITELIST_EXACT or any(path.startswith(p) for p in _TRIAL_WHITELIST_PREFIXES):
        return

    status = get_trial_status(current_user.id)
    if not status.get("expired"):
        return

    # APIs → 402 for the UI; pages → redirect to /trial-ended
    if path.startswith("/api/") or request.accept_mimetypes.best == "application/json":
        return jsonify({"error": "trial_expired", "message": "Your free trial has ended."}), 402
    return redirect("/trial-ended")

@app.route("/api/trial/status")
@login_required
def trial_status():
    ensure_trial_window(current_user.id)
    return jsonify(get_trial_status(current_user.id))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (admin HR + token helpers)
# ─────────────────────────────────────────────────────────────────────────────
def is_hr_admin(user_email: str | None) -> bool:
    allowed_emails = os.getenv("HR_ADMIN_EMAILS", "")
    allowed = [e.strip().lower() for e in allowed_emails.split(",") if e.strip()]
    return bool(user_email and user_email.lower() in allowed)

# ---------- (Optional) app-only token (leave for admin/background if needed)
def require_graph_token():
    """Get app-only Graph token. Only call when needed for admin/background."""
    try:
        return get_app_access_token()
    except Exception:
        logging.exception("Failed to get app-only Graph token")
        return None

# ---------- per-user delegated token refresh (GLOBAL CONNECTOR) --------------
from msal import ConfidentialClientApplication

MS_SCOPES = ["User.Read", "Files.Read.All", "Sites.Read.All", "Mail.Read", "Mail.Send", "Calendars.Read", "Chat.Read", "ChannelMessage.Read.All", "Team.ReadBasic.All", "Channel.ReadBasic.All", "Group.Read.All"]

def _ms_app_delegated():
    return ConfidentialClientApplication(
        client_id=os.getenv("MS_CLIENT_ID"),
        client_credential=os.getenv("MS_CLIENT_SECRET"),
        authority=f"https://login.microsoftonline.com/{os.getenv('MS_TENANT','common')}",
    )

def get_user_ms_token_or_none():
    """
    Return a fresh delegated access_token for the current user, or None if not connected.
    Reads ms_access_token/ms_expires_at/ms_refresh_token from Prefs.
    """
    if not current_user.is_authenticated:
        return None
    uid = current_user.id
    access = _get_pref(uid, "ms_access_token")
    expires_at = _get_pref(uid, "ms_expires_at")  # ISO string
    refresh = _get_pref(uid, "ms_refresh_token")

    try:
        if access and expires_at and isoparse(expires_at) > datetime.now(UTC):
            return access
    except Exception:
        pass

    if refresh:
        appc = _ms_app_delegated()
        result = appc.acquire_token_by_refresh_token(refresh, scopes=MS_SCOPES)
        if "access_token" in result:
            acc = result["access_token"]
            new_refresh = result.get("refresh_token") or refresh
            exp = int(result.get("expires_in", 3600))
            exp_at = (datetime.now(UTC) + timedelta(seconds=max(exp - 60, 0))).isoformat()
            _set_pref(uid, "ms_access_token", acc)
            _set_pref(uid, "ms_refresh_token", new_refresh)
            _set_pref(uid, "ms_expires_at", exp_at)
            return acc
    return None

# ---------- per-user delegated Google token refresh (Gmail/Calendar) ----------
def get_user_google_token_or_none():
    # Return a fresh Google OAuth access_token for the current user, or None if not connected.
    # Looks for google_access_token/google_expires_at/google_refresh_token in Prefs.
    # Tries to refresh via https://oauth2.googleapis.com/token if needed.
    if not current_user.is_authenticated:
        return None
    uid = current_user.id
    access = _get_pref(uid, "google_access_token")
    expires_at = _get_pref(uid, "google_expires_at")  # ISO string
    refresh = _get_pref(uid, "google_refresh_token")

    # still valid?
    try:
        if access and expires_at and isoparse(expires_at) > datetime.now(UTC):
            return access
    except Exception:
        pass

    # refresh if possible
    if refresh and os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
        try:
            data = {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "refresh_token": refresh,
                "grant_type": "refresh_token",
            }
            r = _rq.post("https://oauth2.googleapis.com/token", data=data, timeout=20)
            if r.ok:
                j = r.json()
                acc = j.get("access_token")
                if acc:
                    exp = int(j.get("expires_in", 3600))
                    exp_at = (datetime.now(UTC) + timedelta(seconds=max(exp - 60, 0))).isoformat()
                    _set_pref(uid, "google_access_token", acc)
                    _set_pref(uid, "google_expires_at", exp_at)
                    # some responses omit refresh_token
                    if j.get("refresh_token"):
                        _set_pref(uid, "google_refresh_token", j["refresh_token"])
                    return acc
        except Exception:
            current_app.logger.exception("Google token refresh failed")
    return None

# ---- register Outlook token loader (uses your Prefs-backed refresh) ----
register_outlook_token_loader(get_user_ms_token_or_none)

# ---- register Gmail creds loader (builds google.oauth2 Credentials) ----
def _gmail_creds_loader():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    # pull tokens from Prefs (already used in get_user_google_token_or_none)
    uid = current_user.id if getattr(current_user, "is_authenticated", False) else None
    if not uid:
        return None
    access  = _get_pref(uid, "google_access_token")
    refresh = _get_pref(uid, "google_refresh_token")
    if not (access or refresh):
        return None

    info = {
        "token": access,
        "refresh_token": refresh,
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    try:
        creds = Credentials.from_authorized_user_info(
            info,
            scopes=["https://www.googleapis.com/auth/gmail.send"]
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return creds
    except Exception:
        current_app.logger.exception("Failed to build Gmail credentials")
        return None

register_gmail_creds_loader(_gmail_creds_loader)

# add imports near your email_service imports
from calendar_service import (
    register_outlook_token_loader as cal_register_outlook_token_loader,
    register_gmail_creds_loader as cal_register_gcal_loader,
)

# reuse your existing MS token loader for calendar too
cal_register_outlook_token_loader(get_user_ms_token_or_none)

# separate Google loader with Calendar scope
def _google_calendar_creds_loader():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    uid = current_user.id if getattr(current_user, "is_authenticated", False) else None
    if not uid:
        return None
    access  = _get_pref(uid, "google_access_token")
    refresh = _get_pref(uid, "google_refresh_token")
    if not (access or refresh):
        return None
    info = {
        "token": access,
        "refresh_token": refresh,
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    creds = Credentials.from_authorized_user_info(
        info, scopes=["https://www.googleapis.com/auth/calendar.events"]
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

cal_register_gcal_loader(_google_calendar_creds_loader)



def _first_turn(chat_id: str) -> bool:
    msgs = get_chat_messages(chat_id)
    return len(msgs) == 0

def _save_ai(user_email: str, chat_id: str, is_first_turn: bool, user_text: str | None, ai_text: str):
    if is_first_turn:
        save_message(user_email, chat_id, user_message=(user_text or ""), ai_response=ai_text)
    else:
        save_message(user_email, chat_id, ai_response=ai_text)

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Chat-flow memory helpers (history + per-chat rules)
# ─────────────────────────────────────────────────────────────────────────────
def _load_chat_history(chat_id: str, limit: int = 12) -> list[dict]:
    """
    Return a compact history like:
      [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
    Uses the DB messages for this chat_id.
    """
    try:
        msgs = get_chat_messages(chat_id) or []  # [(sender, message, ts), ...]
    except Exception:
        msgs = []
    msgs = msgs[-limit:]
    hist = []
    for sender, message, ts in msgs:
        if not message:
            continue
        s = (sender or "").lower()
        role = "assistant" if s.startswith("ai") or "assistant" in s else "user"
        hist.append({"role": role, "content": message})
    return hist

def _get_chat_rules(chat_id: str) -> dict:
    store = session.get("chat_rules") or {}
    return store.get(chat_id, {}).copy()

def _set_chat_rules(chat_id: str, rules: dict):
    store = session.get("chat_rules") or {}
    store[chat_id] = rules or {}
    session["chat_rules"] = store

def _apply_if_user_response(text: str, rules: dict, intent_tag: str) -> str:
    """
    Apply chat rules only to natural language answers back to the user.
    We DO NOT apply rules to control/UX prompts like file pickers.
    """
    apply_intents = {"general_response", "mail_search", "calendar_search", "file_content", "file_sent"}
    if intent_tag in apply_intents:
        return apply_chat_rules_to_text(text or "", rules or {})
    return text or ""

# ─────────────────────────────────────────────────────────────────────────────
# Login state for SPA
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/check_login")
def check_login():
    if current_user.is_authenticated:
        return jsonify(logged_in=True, user_email=getattr(current_user, "email", None))
    return jsonify(logged_in=False)

# ─────────────────────────────────────────────────────────────────────────────
# Document APIs
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/hr_documents")
@login_required
def hr_documents():
    docs_path = os.path.join("knowledge_base", "documents")
    metadata_path = os.path.join("knowledge_base", "index_metadata.json")
    metadata = {}

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    files = []
    if os.path.exists(docs_path):
        for fname in os.listdir(docs_path):
            fpath = os.path.join(docs_path, fname)
            if os.path.isfile(fpath):
                size_kb = round(os.path.getsize(fpath) / 1024, 2)
                date_str = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M")
                files.append({
                    "name": fname,
                    "updated": date_str,
                    "size_kb": size_kb,
                    "uploader": metadata.get(fname, {}).get("uploader", "unknown")
                })
    return jsonify({"files": sorted(files, key=lambda f: f["updated"], reverse=True)})

@app.route("/upload_hr_doc", methods=["POST"])
@login_required
def upload_hr_doc():
    user_email = getattr(current_user, "email", None)
    if not is_hr_admin(user_email):
        return jsonify({"error": "❌ Unauthorized"}), 403

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No filename"}), 400

    allowed_exts = (".pdf", ".docx", ".txt")
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(allowed_exts):
        return jsonify({"error": "Unsupported format"}), 400

    save_path = os.path.join("knowledge_base", "documents", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)

    metadata_path = os.path.join("knowledge_base", "index_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    metadata[filename] = {
        "uploader": user_email,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logging.warning(f"⚠️ Failed to write metadata: {e}")

    build_hr_knowledge_json()
    return jsonify({"message": "✅ File uploaded."})

@app.route("/api/hr_documents", methods=["DELETE"])
@login_required
def delete_hr_doc():
    user_email = getattr(current_user, "email", None)
    if not is_hr_admin(user_email):
        return jsonify({"error": "❌ Unauthorized"}), 403

    data = request.json or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    doc_path = os.path.join("knowledge_base", "documents", filename)
    metadata_path = os.path.join("knowledge_base", "index_metadata.json")

    try:
        if os.path.exists(doc_path):
            os.remove(doc_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            metadata.pop(filename, None)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        build_hr_knowledge_json()
        return jsonify({"message": f"✅ '{filename}' deleted and knowledge base updated."})
    except Exception as e:
        logging.exception("❌ Failed to delete document:")
        return jsonify({"error": f"❌ Deletion failed: {e}"}), 500

# ─────────────────────────────────────────────────────────────────────────────
# Chat session APIs
# ─────────────────────────────────────────────────────────────────────────────
# @app.route("/api/skip_selection", methods=["POST"])
# @login_required
# def skip_selection():
#     session["stage"] = "awaiting_query"
#     session["found_files"] = []
#     return jsonify({"message": "Skipped selection"})

@app.route("/api/summarize_selected", methods=["POST"])
@login_required
def api_summarize_selected():
    """
    Summarize the files currently listed in session['found_files'].
    Accepts either selectedFileIds (list of IDs) or selectedIndices (1-based int list or "1,3-5").
    Optional: 'prompt' to customize the summary instruction.
    """
    user_email = getattr(current_user, "email", None)
    if not user_email:
        return jsonify({"error": "missing_user"}), 400

    payload = request.get_json(silent=True) or {}
    selected_ids    = payload.get("selectedFileIds") or []
    selected_indices = payload.get("selectedIndices")
    prompt          = (payload.get("prompt") or "Summarize this file").strip()

    found = session.get("found_files", []) or []
    if not found:
        return jsonify({"error": "no_pending_files", "message": "No file list is active to summarize."}), 400

    # ---- helper: resolve selected files by id or 1-based index
    def _coerce_selected(found_files, selected_indices, selected_ids):
        if not found_files:
            return []
        by_id = {str(f.get("id")): f for f in found_files if f.get("id")}
        picked = []

        # prefer explicit IDs from the UI, if any
        for _id in (selected_ids or []):
            f = by_id.get(str(_id))
            if f:
                picked.append(f)

        # support 1-based indices ("1,3-5") or [1,3]
        if not picked and selected_indices:
            if isinstance(selected_indices, list):
                idxs = [i - 1 for i in selected_indices if isinstance(i, int)]
            else:
                s = str(selected_indices)
                parts = [p.strip() for p in s.split(",") if p.strip()]
                idxs = []
                seen = set()
                for p in parts:
                    if "-" in p:
                        a, b = p.split("-", 1)
                        if a.strip().isdigit() and b.strip().isdigit():
                            start, end = int(a) - 1, int(b) - 1
                            if start > end:
                                start, end = end, start
                            for i in range(start, end + 1):
                                if 0 <= i < len(found_files) and i not in seen:
                                    seen.add(i); idxs.append(i)
                    elif p.isdigit():
                        i = int(p) - 1
                        if 0 <= i < len(found_files) and i not in seen:
                            seen.add(i); idxs.append(i)
            picked = [found_files[i] for i in idxs if 0 <= i < len(found_files)]

        # if nothing explicitly selected and there’s exactly one visible file, pick it
        if not picked and len(found_files) == 1:
            picked = [found_files[0]]

        return picked

    selected = _coerce_selected(found, selected_indices, selected_ids)
    if not selected:
        return jsonify({"error": "no_selection", "message": "Please select one or more files to summarize."}), 400

    # ---- token only if any selected item is a Microsoft file
    ms_token = None
    if needs_ms_token(selected):
        ms_token = get_user_ms_token_or_none()
        if not ms_token:
            return jsonify({
                "error": "need_connection",
                "message": "Please connect SharePoint/OneDrive from the Dashboard first."
            }), 403

    # ---- call summarizer
    try:
        result = summarize_selected(
            user_id=current_user.id,
            selected_files=selected,
            prompt=prompt,
            ms_token=ms_token,
            app_tz=os.getenv("APP_TZ") or "Asia/Dhaka",
            max_tokens=900,
        )
        ai = result.get("answer") or "I couldn't extract a useful summary from those files."
    except Exception as e:
        current_app.logger.exception("summarize_selected failed")
        return jsonify({"error": "summarize_failed", "message": str(e)}), 500

    # ---- save to chat, close picker
    chat_id = session.get("chat_id") or str(int(time.time()))
    session["chat_id"] = chat_id
    # save assistant reply to the thread
    save_message(user_email, chat_id, ai_response=ai)

    # clear current selection so the result window disappears
    session["stage"] = "awaiting_query"
    session["found_files"] = []

    return jsonify({
        "response": ai,
        "intent": "file_content",
        "chat_id": chat_id,
        "dismissPicker": True  # your UI can use this flag to hide the file panel
    })


@app.route("/api/session_state")
@login_required
def session_state():
    return jsonify({
        "stage": session.get("stage"),
        "chat_id": session.get("chat_id"),
        "files": session.get("found_files", [])
    })

@app.route("/api/new_chat")
@login_required
def new_chat():
    # Create an empty chat in session only; do NOT write to DB yet.
    session["chat_id"] = str(int(time.time()))
    session["stage"] = "start"
    session["found_files"] = []
    # NEW: seed empty rules for new chat
    rules_store = session.get("chat_rules") or {}
    rules_store[session["chat_id"]] = {}
    session["chat_rules"] = rules_store
    return jsonify({"chat_id": session["chat_id"]})

@app.route("/api/chats")
@login_required
def api_chats():
    user_email = getattr(current_user, "email", None)
    if not user_email:
        return jsonify([])  # no session, no chats
    delete_old_chats(user_email)
    return jsonify(get_user_chats(user_email))

@app.route("/api/messages/<chat_id>")
@login_required
def get_messages(chat_id):
    messages = get_chat_messages(chat_id)
    return jsonify({
        "messages": [{"sender": m[0], "message": m[1], "timestamp": m[2]} for m in messages]
    })

# ---------- helper to normalize mixed shapes from search_all_files ------------
def _collect_files(obj, out):
    """Recursively pull dict 'file-like' objects from tuples/lists/dicts."""
    if obj is None:
        return
    if isinstance(obj, dict):
        if any(k in obj for k in ("id", "name", "webUrl", "parentReference", "driveId")):
            out.append(obj)
        else:
            for v in obj.values():
                _collect_files(v, out)
    elif isinstance(obj, (list, tuple)):
        for it in obj:
            _collect_files(it, out)

# --- Natural language time windows for mail & calendar ---  # >>> ADDED
def _start_of_day(dt): return dt.replace(hour=0, minute=0, second=0, microsecond=0)
def _end_of_day(dt):   return dt.replace(hour=23, minute=59, second=59, microsecond=999999)

def parse_window_and_keywords(text: str):  # >>> ADDED
    """
    Returns: (time_min_isoZ, time_max_isoZ, cleaned_keywords)
    Understands: today, yesterday, this week, next week, this month, YYYY-MM-DD.
    """
    UTCZ = timezone.utc
    APP_TZV = ZoneInfo(os.getenv("APP_TZ", "Asia/Dhaka"))
    t = (text or "").strip()
    low = t.lower()
    now_local = datetime.now(APP_TZV)

    def Z(dt_local):  # local -> ISO Z
        return dt_local.astimezone(UTCZ).isoformat().replace("+00:00", "Z")

    # explicit date like 2025-08-31 or 2025/08/31
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", low)
    if m:
        y, mo, d = map(int, m.groups())
        s = _start_of_day(datetime(y, mo, d, tzinfo=APP_TZV))
        e = _end_of_day(datetime(y, mo, d, tzinfo=APP_TZV))
        return Z(s), Z(e), t

    if "today" in low:
        return Z(_start_of_day(now_local)), Z(_end_of_day(now_local)), t
    if "yesterday" in low:
        y = now_local - timedelta(days=1)
        return Z(_start_of_day(y)), Z(_end_of_day(y)), t
    if "this week" in low:
        dow = now_local.weekday()
        s = _start_of_day(now_local - timedelta(days=dow))
        e = _end_of_day(s + timedelta(days=6))
        return Z(s), Z(e), t
    if "next week" in low:
        dow = now_local.weekday()
        s = _start_of_day(now_local - timedelta(days=dow) + timedelta(weeks=1))
        e = _end_of_day(s + timedelta(days=6))
        return Z(s), Z(e), t
    if "this month" in low:
        s = _start_of_day(now_local.replace(day=1))
        nm = s.replace(year=s.year + 1, month=1) if s.month == 12 else s.replace(month=s.month + 1)
        e = _end_of_day(nm - timedelta(days=1))
        return Z(s), Z(e), t

    # no window detected
    return None, None, t


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    """Main chat endpoint with source-aware search across MS Graph, Google Drive, Dropbox, Box, MEGA,
    plus Outlook Mail, Gmail, Outlook Calendar, Google Calendar, and Microsoft Teams.
    Now upload-aware: if the user attaches files and asks to 'explain/summarize this', we summarize uploads first.
    """

    # ───────── local imports to keep this drop-in safe ─────────
    import os, re, time, logging, mimetypes
    from datetime import datetime, timezone
    from flask import jsonify, request, session, current_app
    from flask_login import current_user

    # Upload helpers (your module)
    try:
        from uploaded_file_handler import attachments_to_selected, summarize_uploads
    except Exception:
        # lightweight shims if module is unavailable
        def attachments_to_selected(*_, **__): return []
        def summarize_uploads(**_): return {"answer": "", "selected": []}

    # Optional conversational flows (existing)
    try:
        from email_service import is_send_email_intent, continue_email_flow
    except Exception:
        def is_send_email_intent(_): return False
        def continue_email_flow(**_): return {"reply": "Email module is unavailable on the server.", "done": True}
    try:
        from calendar_service import is_calendar_create_intent, continue_calendar_create_flow
    except Exception:
        def is_calendar_create_intent(_): return False
        def continue_calendar_create_flow(*_, **__): return {"reply": "Calendar module is unavailable on the server.", "done": True}

    # Existing helpers
    delete_old_messages(days=3)

    # ───────── small helper blocks ─────────
    def _wants_file_explain(text: str) -> bool:
        q = (text or "").lower()
        triggers = ("explain", "summarize", "summary", "tl;dr", "analyze", "analyse", "describe")
        return any(t in q for t in triggers)

    _SMALLTALK_RX = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|yo|good (morning|afternoon|evening))\b", re.I)
    def _is_smalltalk(text: str) -> bool:
        return bool(_SMALLTALK_RX.match(text or ""))

    _ORDINAL = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10}
    def _indices_from_text(s: str) -> list[int]:
        s = (s or "").lower()
        idxs = []
        for w,n in _ORDINAL.items():
            if re.search(rf"\b{w}\b", s): idxs.append(n)
        idxs += [int(x) for x in re.findall(r"\b(\d{1,2})\b", s)]
        return sorted(set([i for i in idxs if i > 0]))

    def _coerce_selected(found_files, selected_indices, selected_ids):
        """Build a list of selected file dicts from found_files using ids or 1-based indices."""
        if not found_files:
            return []
        by_id = {str(f.get("id")): f for f in found_files if f.get("id")}
        selected = []

        for _id in (selected_ids or []):
            f = by_id.get(str(_id))
            if f: selected.append(f)

        if not selected and selected_indices:
            if isinstance(selected_indices, list):
                idxs = [i - 1 for i in selected_indices if isinstance(i, int)]
            else:
                try:
                    idxs = [int(s.strip()) - 1 for s in str(selected_indices).split(",") if s.strip().isdigit()]
                except Exception:
                    idxs = []
            selected = [found_files[i] for i in idxs if 0 <= i < len(found_files)]

        if not selected and len(found_files) == 1:
            selected = [found_files[0]]
        return selected

    def _remember_selected_files(chat_id: str, files: list[dict]):
        if not files: return
        bag = session.get("last_selected_files") or {}
        compact = []
        for f in files:
            compact.append({
                "id": f.get("id"),
                "name": f.get("name"),
                "source": f.get("source"),
                "webUrl": f.get("webUrl") or f.get("webViewLink") or f.get("url") or f.get("link") or f.get("preview_url"),
                "parentReference": f.get("parentReference") or {},
            })
        bag[chat_id] = compact
        session["last_selected_files"] = bag
        session.modified = True

    def _recall_selected_files(chat_id: str) -> list[dict]:
        bag = session.get("last_selected_files") or {}
        return bag.get(chat_id) or []

    # ───────── request + session bootstrap ─────────
    user_email = getattr(current_user, "email", None)
    if not user_email:
        return jsonify(response="❌ Missing user context", intent="error"), 400

    payload              = request.json or {}
    user_input           = (payload.get("message") or "").trim() if hasattr(str, "trim") else (payload.get("message") or "").strip()
    is_selection         = bool(payload.get("selectionStage"))
    selected_indices     = payload.get("selectedIndices")
    selected_ids_in      = payload.get("selectedFileIds") or []
    sources              = payload.get("sources") or []
    attachments          = payload.get("attachments") or []           # <— upload chips from UI
    if not attachments and payload.get("upload_ids"):                 # back-compat: map ids -> minimal chips
        attachments = [{"id": i, "name": i, "path": i} for i in payload["upload_ids"] if i]

    # normalize source pills
    sources = [("google_drive" if isinstance(s, str) and s.lower() == "google" else (s or "")).lower() for s in sources]

    incoming_chat_id = (payload.get("chat_id") or "").strip()
    if incoming_chat_id and incoming_chat_id != session.get("chat_id"):
        session["chat_id"]      = incoming_chat_id
        session["stage"]        = "awaiting_query"
        session["found_files"]  = []
        session["last_query"]   = ""

    chat_id = session.get("chat_id") or incoming_chat_id or str(int(time.time()))
    session["chat_id"] = chat_id
    if not session.get("stage"):
        session["stage"] = "start"
    if session["stage"] == "start":
        session["stage"] = "awaiting_query"
        if not user_input:
            return jsonify(intent="ready", chat_id=chat_id)

    print(f"🔍 User query: {user_input} | Sources: {sources} | Selection stage: {is_selection} | Selected indices: {selected_indices} | Selected IDs: {selected_ids_in} | attachments: {len(attachments)} -> {attachments}")

    # chat rules / history (existing)
    rules = _get_chat_rules(chat_id)
    if user_input:
        try:
            upd = detect_chat_flow_rules(user_input)
            if upd.get("reset"): rules = {}
            else: rules.update({k: v for k, v in upd.items() if k != "reset"})
            _set_chat_rules(chat_id, rules)
        except Exception:
            pass
    history = _load_chat_history(chat_id, limit=12)
    first_turn = _first_turn(chat_id)
    if user_input and not first_turn:
        save_message(user_email, chat_id, user_message=user_input)

    # utility
    def _need_ms_token_or_error(need_ms: bool):
        if not need_ms: return None, None
        token = get_user_ms_token_or_none()
        if not token:
            ai = "Please connect SharePoint/OneDrive from the Dashboard first."
            ai = _apply_if_user_response(ai, rules, "general_response")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            return None, (jsonify({"response": ai, "intent": "need_connection", "chat_id": chat_id}), 403)
        return token, None

    def _looks_like_selection(s: str) -> bool:
        return bool(s) and all(part.strip().isdigit() for part in s.split(","))

    # selection -> new question cancels unless explaining
    if (
        session.get("stage") == "awaiting_selection"
        and user_input
        and not is_selection
        and not _looks_like_selection(user_input)
        and not _wants_file_explain(user_input)
    ):
        session["stage"] = "awaiting_query"
        session["found_files"] = []

    # ───────── smalltalk quick path ─────────
    if _is_smalltalk(user_input):
        ai = answer_general_query(user_input, history=history, chat_rules=rules)
        _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="general_response", chat_id=chat_id)

    # ───────── email/calendar flows (existing) ─────────
    if session.get("email_flow"):
        res = continue_email_flow(user_id=current_user.id, user_text=user_input)
        ai = res.get("reply") or ""
        if ai: _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="send_email_flow", chat_id=chat_id, meta=res.get("meta", {}), done=res.get("done", False), sent=res.get("sent", False))
    if is_send_email_intent(user_input):
        res = continue_email_flow(user_id=current_user.id, user_text=user_input)
        ai = res.get("reply") or ""
        if ai: _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="send_email_flow", chat_id=chat_id, meta=res.get("meta", {}), done=res.get("done", False), sent=res.get("sent", False))

    if session.get("cal_flow"):
        ans = continue_calendar_create_flow(current_user.id, user_input)
        ai  = ans.get("reply") or ""
        if ai:
            ai = _apply_if_user_response(ai, rules, "calendar_search")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="calendar_create", chat_id=chat_id, meta=ans.get("meta", {}), done=ans.get("done", False))
    if is_calendar_create_intent(user_input):
        ans = continue_calendar_create_flow(current_user.id, user_input)
        ai  = _apply_if_user_response(ans.get("reply") or "", rules, "calendar_search")
        _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="calendar_create", chat_id=chat_id, meta=ans.get("meta", {}), done=ans.get("done", False))

    # ───────── NEW: upload fast-paths ─────────
    UPLOAD_ROOT = current_app.config.get("UPLOAD_ROOT", os.path.join(current_app.root_path, "uploads"))

    # If files are attached and the prompt refers to "this file" OR is an explain/summarize prompt -> summarize uploads directly
    if attachments and (_wants_file_explain(user_input) or "this file" in (user_input or "").lower() or "attached" in (user_input or "").lower()):
        try:
            res = summarize_uploads(
                user_id=current_user.id,
                chat_id=chat_id,
                attachments=attachments,
                prompt=user_input or "Summarize the uploaded file(s).",
                upload_root=UPLOAD_ROOT,
                app_tz=os.getenv("APP_TZ") or "Asia/Dhaka",
                max_tokens=900,
            )
            ai = res.get("answer") or "I couldn't extract a useful answer from the uploaded file(s)."
            ai = _apply_if_user_response(ai, rules, "file_content")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            # remember selection so follow-ups like "now extract the emails" work
            _remember_selected_files(chat_id, res.get("selected") or [])
            session["stage"] = "awaiting_query"
            session["found_files"] = []
            return jsonify(response=ai, intent="file_content", chat_id=chat_id)
        except Exception as e:
            current_app.logger.exception("upload summarization failed: %s", e)

    # If picker is open and the user says "explain..." use current selection
    if session.get("stage") == "awaiting_selection" and user_input and _wants_file_explain(user_input):
        found = session.get("found_files", []) or []
        selected = _coerce_selected(found, selected_indices, selected_ids_in)
        if not selected:
            msg = "Please select one or more files to summarize (use the checkboxes)."
            msg = _apply_if_user_response(msg, rules, "general_response")
            return jsonify({"response": msg, "intent": "file_search", "chat_id": chat_id})

        try:
            from summarizer import summarize_selected as _summarize_selected, needs_ms_token as _summarize_needs_ms
        except Exception:
            _summarize_selected = None
            _summarize_needs_ms = None
        if _summarize_selected is None:
            ai = "Summarizer module is unavailable on the server."
            ai = _apply_if_user_response(ai, rules, "general_response")
            return jsonify({"response": ai, "intent": "general_response", "chat_id": chat_id}), 500

        ms_needed = _summarize_needs_ms(selected) if _summarize_needs_ms else any(
            f.get("source") not in ("google_drive","dropbox","box","mega","upload") for f in selected
        )
        if ms_needed:
            tok, err = _need_ms_token_or_error(True)
            if err: return err
            ms_token2 = tok
        else:
            ms_token2 = None

        try:
            result = _summarize_selected(
                user_id=current_user.id,
                selected_files=selected,
                prompt=user_input,
                ms_token=ms_token2,
                app_tz=os.getenv("APP_TZ") or "Asia/Dhaka",
                max_tokens=900,
            )
            ai = result.get("answer") or "I couldn't extract a useful summary from the selected files."
        except Exception as e:
            current_app.logger.exception("inline explain/summarize failed")
            ai = f"Sorry, I couldn't summarize those files right now: {e}"

        ai = _apply_if_user_response(ai, rules, "file_content")
        session["stage"] = "awaiting_query"
        session["found_files"] = []
        _remember_selected_files(chat_id, selected)
        _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="file_content", chat_id=chat_id)

    # ───────── intent routing ─────────
    try:
        det = detect_intent_and_extract(user_input or "")
    except Exception:
        det = {}
    print(f"🧭 Intent detection result: {det}")
    intent     = (det.get("intent") or "").lower()
    confidence = float(det.get("confidence") or 0.0)
    tmin       = det.get("time_min_iso")
    tmax       = det.get("time_max_iso")
    logging.info(f"Detected intent: {intent} (confidence {confidence}) | time_min: {tmin} | time_max: {tmax}")

    # handoffs to flows
    if intent == "send_email":
        res = continue_email_flow(user_id=current_user.id, user_text=user_input)
        ai = res.get("reply") or ""
        if ai: _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="send_email_flow", chat_id=chat_id, meta=res.get("meta", {}), done=res.get("done", False), sent=res.get("sent", False))

    if intent == "calendar_create":
        ans = continue_calendar_create_flow(current_user.id, user_input)
        ai  = _apply_if_user_response(ans.get("reply") or "", rules, "calendar_search")
        _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="calendar_create", chat_id=chat_id, meta=ans.get("meta", {}), done=ans.get("done", False))

    # fallback time window parse
    if not tmin and not tmax:
        try:
            tmin, tmax, _ = parse_window_and_keywords(user_input)
        except Exception:
            tmin, tmax = None, None

    MC_KEYS = ("outlook", "gmail", "outlook_calendar", "google_calendar", "teams")
    scope = next((s for s in (sources or []) if s in MC_KEYS), None)
    ms_token = get_user_ms_token_or_none()
    g_token  = get_user_google_token_or_none()

    # ===== EMAIL SEARCH =====
    if intent == "email_search" and confidence >= 0.5:
        allowed = []
        if scope in (None, "outlook") and ms_token: allowed.append("outlook")
        if scope in (None, "gmail")   and g_token:  allowed.append("gmail")
        if not allowed:
            ai = "Please connect Outlook or Gmail (Dashboard → Connections)."
            ai = _apply_if_user_response(ai, rules, "general_response")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            return jsonify(response=ai, intent="need_connection", chat_id=chat_id), 403

        ai = email_search_chatgpt_style(
            user_query=user_input,
            sources=allowed,
            ms_token=ms_token,
            g_token=g_token,
            user_tz=os.getenv("APP_TZ") or "Asia/Dhaka",
            default_if_no_window="today",
            llm_time_min_iso=det.get("time_min_iso"),
            llm_time_max_iso=det.get("time_max_iso"),
        )
        ai = _apply_if_user_response(ai, rules, "mail_search")
        _save_ai(user_email, chat_id, first_turn, user_input, ai)
        session["stage"] = "awaiting_query"
        session["found_files"] = []
        return jsonify(response=ai, intent="mail_search", chat_id=chat_id)

    # ===== CALENDAR SEARCH =====
    if intent == "calendar_search" and confidence >= 0.5:
        allowed = []
        if scope in (None, "outlook_calendar") and ms_token: allowed.append("outlook_calendar")
        if scope in (None, "google_calendar") and g_token: allowed.append("google_calendar")
        if not allowed:
            ai = "Please connect your calendar (Outlook or Google) in Dashboard → Connections."
            ai = _apply_if_user_response(ai, rules, "general_response")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            return jsonify(response=ai, intent="need_connection", chat_id=chat_id), 403

        if "outlook_calendar" in allowed:
            ai = outlook_calendar_assistant_reply(ms_token, user_input, limit=20, time_min=tmin, time_max=tmax, user_tz=os.getenv("APP_TZ") or "Asia/Dhaka")
        else:
            ai = google_calendar_smart_search(g_token, user_input, limit=20, time_min=tmin, time_max=tmax)

        ai = _apply_if_user_response(ai, rules, "calendar_search")
        _save_ai(user_email, chat_id, first_turn, user_input, ai)
        session["stage"] = "awaiting_query"
        session["found_files"] = []
        return jsonify(response=ai, intent="calendar_search", chat_id=chat_id)

    # ===== TEAMS MESSAGES =====
    from services_teams import teams_answer
    if intent == "message_search" and confidence >= 0.5:
        if not ms_token:
            ai = "Please connect Microsoft 365 first (Dashboard → Connections)."
            ai = _apply_if_user_response(ai, rules, "general_response")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            return jsonify(response=ai, intent="need_connection", chat_id=chat_id), 403
        try:
            ans = teams_answer(ms_token, user_input, size=25, time_min=tmin, time_max=tmax)
            ai  = ans.get("text") or "I couldn’t find any Teams messages matching that request."
            ai  = _apply_if_user_response(ai, rules, "general_response")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            session["stage"] = "awaiting_query"
            session["found_files"] = []
            return jsonify(response=ai, intent="teams_search", chat_id=chat_id, items=ans.get("items", []), mode=ans.get("mode"))
        except Exception as e:
            ai = f"Teams lookup failed: {e}"
            ai = _apply_if_user_response(ai, rules, "general_response")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            return jsonify(response=ai, intent="general_response", chat_id=chat_id)

    # ===== WEB SEARCH =====
    if intent == "web_search" and confidence >= 0.5:
        try:
            ans = web_answer(user_input=user_input, count=7, include_pages=True, max_tokens=900)
            ai = ans.get("text") or "I couldn't find enough reliable information on the web."
            ai = _apply_if_user_response(ai, rules, "general_response")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            session["stage"] = "awaiting_query"
            session["found_files"] = []
            return jsonify(response=ai, intent="general_response", sources=ans.get("sources", []), chat_id=chat_id)
        except Exception as e:
            ai = f"Web lookup failed: {e}"
            ai = _apply_if_user_response(ai, rules, "general_response")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            return jsonify(response=ai, intent="general_response", chat_id=chat_id)

    # ===== Selection flow (remember) =====
    if is_selection and selected_indices:
        files = session.get("found_files", []) or []
        if isinstance(selected_indices, list):
            idxs = [i - 1 for i in selected_indices if isinstance(i, int)]
        else:
            try:
                idxs = [int(s.strip()) - 1 for s in str(selected_indices).split(",") if s.strip().isdigit()]
            except Exception:
                idxs = []
        selected_files_mem = [files[i] for i in idxs if 0 <= i < len(files)]
        _remember_selected_files(chat_id, selected_files_mem)

        need_ms = any(0 <= i < len(files) and files[i].get("source") not in ("google_drive", "dropbox", "box", "mega") for i in idxs)
        ms_token2, err = _need_ms_token_or_error(need_ms)
        if err: return err

        resp = handle_file_selection(selected_indices, ms_token2, user_email, chat_id)
        if isinstance(resp, tuple):
            body, code = resp
            if isinstance(body, dict):
                body.setdefault("chat_id", chat_id)
                return jsonify(body), code
            return resp
        if isinstance(resp, dict):
            resp.setdefault("chat_id", chat_id)
            return jsonify(resp)
        return resp

    if session.get("stage") == "awaiting_selection" and _looks_like_selection(user_input):
        files = session.get("found_files", []) or []
        idxs  = [int(s.strip()) - 1 for s in user_input.split(",") if s.strip().isdigit()]
        selected_files_mem = [files[i] for i in idxs if 0 <= i < len(files)]
        _remember_selected_files(chat_id, selected_files_mem)

        need_ms = any(0 <= i < len(files) and files[i].get("source") not in ("google_drive", "dropbox", "box", "mega") for i in idxs)
        ms_token2, err = _need_ms_token_or_error(need_ms)
        if err: return err

        resp = handle_file_selection(user_input, ms_token2, user_email, chat_id)
        if isinstance(resp, tuple):
            body, code = resp
            if isinstance(body, dict):
                body.setdefault("chat_id", chat_id)
                return jsonify(body), code
            return resp
        if isinstance(resp, dict):
            resp.setdefault("chat_id", chat_id)
            return jsonify(resp)
        return resp

    # ===== Main LLM-first pipeline =====
    def _run_intent_pipeline():
        gpt_result = detect_intent_and_extract(user_input)
        intent2 = (gpt_result.get("intent") or "").lower()
        query   = (gpt_result.get("data") or "").strip()
        extracted_keywords = gpt_result.get("extracted_keywords") or []
        if not extracted_keywords and query:
            extracted_keywords = [t for t in query.split() if t]

        if intent2 == "hr_admin":
            ai = handle_query(user_input, intent=intent2)
            ai = _apply_if_user_response(ai, rules, "general_response")
            if ai and not ai.startswith("⚠️ No readable HR documents"):
                _save_ai(user_email, chat_id, first_turn, user_input, ai)
                return jsonify(response=ai, intent="hr_admin", chat_id=chat_id)

        if intent2 == "file_search_prompt":
            ai = "Can you tell me what kind of file you're trying to find?"
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            return jsonify(response=ai, intent="general_response", chat_id=chat_id)

        # ── file_content: prefer attached uploads first ──
        if intent2 == "file_content":
            if attachments:
                up = summarize_uploads(
                    user_id=current_user.id,
                    chat_id=chat_id,
                    attachments=attachments,
                    prompt=user_input or query,
                    upload_root=UPLOAD_ROOT,
                    app_tz=os.getenv("APP_TZ") or "Asia/Dhaka",
                    max_tokens=900,
                )
                ai_up = (up or {}).get("answer") or "I couldn't extract the requested information."
                ai_up = _apply_if_user_response(ai_up, rules, "file_content")
                _save_ai(user_email, chat_id, first_turn, user_input, ai_up)
                _remember_selected_files(chat_id, up.get("selected") or [])
                session["stage"] = "awaiting_query"
                session["found_files"] = []
                return jsonify(response=ai_up, intent="file_content", chat_id=chat_id)

            # … else fall back to connector search + summarize (existing code path) …
            search_str = " ".join(extracted_keywords[:8]) if extracted_keywords else (query or user_input)

            use_ms   = True if not sources else any(s in ("sharepoint", "onedrive") for s in sources)
            use_gd   = True if not sources else ("google_drive" in sources)
            use_dbx  = True if not sources else ("dropbox" in sources)
            use_box  = True if not sources else ("box" in sources)
            use_mega = True if not sources else ("mega" in sources)

            all_hits, ms_hits, ms_token3 = [], [], None

            if use_ms:
                ms_token3, err = _need_ms_token_or_error(True)
                if err: return err
                session["last_query"] = search_str
                try:
                    raw_ms = search_all_files(ms_token3, search_str, user_input) or []
                    ms_hits = []
                    _collect_files(raw_ms, ms_hits)
                    ms_hits = [x for x in ms_hits if isinstance(x, dict)]
                    for f in ms_hits: f["source"] = f.get("source") or "ms_graph"
                except Exception as e:
                    current_app.logger.exception("MS search failed: %s", e)
                    ms_hits = []

            gd_hits = []
            if use_gd:
                try:
                    gd_hits = search_drive_files_ranked(uid=current_user.id, query=search_str, fetch_pages=2, page_size=25, mime_type=None) or []
                except NameError:
                    try:
                        gd_raw = search_drive_files(uid=current_user.id, query=search_str) or []
                        gd_hits = gd_raw[:10]
                    except Exception:
                        gd_hits = []
                except Exception:
                    gd_hits = []
                for g in gd_hits:
                    g["source"] = "google_drive"
                    if "webUrl" not in g and "webViewLink" in g: g["webUrl"] = g["webViewLink"]
                gd_hits = gd_hits[:10]

            dbx_hits = []
            if use_dbx:
                try:
                    dbx_hits = search_dropbox_files_ranked(uid=current_user.id, query=search_str, limit=50, candidate_multiplier=3) or []
                except NameError:
                    try:
                        dbx_raw = search_dropbox_files(uid=current_user.id, query=search_str) or []
                        dbx_hits = dbx_raw[:10]
                    except Exception:
                        dbx_hits = []
                except Exception:
                    dbx_hits = []
                for d in dbx_hits:
                    d["source"] = "dropbox"
                    if "webUrl" not in d:
                        if "url" in d:           d["webUrl"] = d["url"]
                        elif "link" in d:        d["webUrl"] = d["link"]
                        elif "preview_url" in d: d["webUrl"] = d["preview_url"]

            box_hits = []
            if use_box:
                try:
                    box_hits = search_box_files_ranked(uid=current_user.id, query=search_str, limit=50, candidate_multiplier=3) or []
                except NameError:
                    try:
                        box_raw = search_box_files(access_token=None, query=search_str) or []
                        box_hits = box_raw[:10]
                    except Exception:
                        box_hits = []
                except Exception:
                    box_hits = []
                for b in box_hits:
                    b["source"] = "box"
                    if "webUrl" not in b and b.get("id"): b["webUrl"] = f"https://app.box.com/file/{b['id']}"

            mega_hits = []
            if use_mega:
                try:
                    mega_hits = search_mega_files_ranked(uid=current_user.id, query=search_str, limit=50) or []
                except Exception:
                    mega_hits = []
                for m in mega_hits:
                    m["source"] = "mega"
                    if "webUrl" not in m or not m.get("webUrl"): m["webUrl"] = "https://mega.nz/fm"

            all_hits = ms_hits + gd_hits + dbx_hits + box_hits + mega_hits

            if ms_hits:
                perform_access_check = os.getenv("PERFORM_ACCESS_CHECK", "true").lower() == "true"
                if perform_access_check:
                    filtered = []
                    for f in ms_hits:
                        try:
                            if check_file_access(ms_token3, f["id"], user_email, f.get("parentReference", {}).get("siteId")):
                                filtered.append(f)
                        except Exception:
                            pass
                    all_hits = filtered + gd_hits + dbx_hits + box_hits + mega_hits

            if not all_hits:
                ai = "📁 No files found for that request."
                ai = _apply_if_user_response(ai, rules, "file_content")
                _save_ai(user_email, chat_id, first_turn, user_input, ai)
                return jsonify(response=ai, intent="file_content", chat_id=chat_id)

            if any(h.get("source") not in ("google_drive", "dropbox", "box", "mega") for h in all_hits) and not ms_token3:
                ms_token3, err = _need_ms_token_or_error(True)
                if err: return err

            result = summarize_from_hits(current_user.id, ms_token3, all_hits, user_input)
            ai = result.get("answer") or "I couldn't extract the requested information."
            ai = _apply_if_user_response(ai, rules, "file_content")
            _save_ai(user_email, chat_id, first_turn, user_input, ai)
            session["stage"] = "awaiting_query"
            session["found_files"] = []
            return jsonify(response=ai, intent="file_content", chat_id=chat_id)

# ---------- file search (existing selection flow) ----------
        if intent2 == "file_search" and query and len(query) >= 2:
            # default to all connected if no pill selected
            use_ms   = True if not sources else any(s in ("sharepoint", "onedrive") for s in sources)
            use_gd   = True if not sources else ("google_drive" in sources)
            use_dbx  = True if not sources else ("dropbox" in sources)
            use_box  = True if not sources else ("box" in sources)
            use_mega = True if not sources else ("mega" in sources)

            all_hits = []

            # MS Graph
            ms_hits = []
            if use_ms:
                ms_token4, err = _need_ms_token_or_error(True)
                if err: return err
                session["last_query"] = query
                try:
                    raw_ms = search_all_files(ms_token4, query, user_input) or []
                    ms_hits = []
                    _collect_files(raw_ms, ms_hits)
                    ms_hits = [x for x in ms_hits if isinstance(x, dict)]
                    for f in ms_hits:
                        f["source"] = f.get("source") or "ms_graph"
                except Exception as e:
                    current_app.logger.exception("MS search failed: %s", e)
                    ms_hits = []

            # Google Drive
            gd_hits = []
            if use_gd:
                try:
                    gd_hits = search_drive_files_ranked(
                        uid=current_user.id,
                        query=query,
                        fetch_pages=2,
                        page_size=25,
                        mime_type=None,
                    ) or []
                except NameError:
                    try:
                        gd_raw = search_drive_files(uid=current_user.id, query=query) or []
                        gd_hits = gd_raw[:10]
                    except Exception:
                        gd_hits = []
                except Exception:
                    gd_hits = []
                for g in gd_hits:
                    g["source"] = "google_drive"
                    if "webUrl" not in g and "webViewLink" in g:
                        g["webUrl"] = g["webViewLink"]
                gd_hits = gd_hits[:10]

            # Dropbox
            dbx_hits = []
            if use_dbx:
                try:
                    dbx_hits = search_dropbox_files_ranked(
                        uid=current_user.id,
                        query=query,
                        limit=50,
                        candidate_multiplier=3
                    ) or []
                except NameError:
                    try:
                        dbx_raw = search_dropbox_files(uid=current_user.id, query=query) or []
                        dbx_hits = dbx_raw[:10]
                    except Exception:
                        dbx_hits = []
                except Exception:
                    dbx_hits = []
                for d in dbx_hits:
                    d["source"] = "dropbox"
                    if "webUrl" not in d:
                        if "url" in d:           d["webUrl"] = d["url"]
                        elif "link" in d:        d["webUrl"] = d["link"]
                        elif "preview_url" in d: d["webUrl"] = d["preview_url"]

            # Box
            box_hits = []
            if use_box:
                try:
                    box_hits = search_box_files_ranked(
                        uid=current_user.id,
                        query=query,
                        limit=50,
                        candidate_multiplier=3
                    ) or []
                except NameError:
                    try:
                        box_raw = search_box_files(access_token=None, query=query) or []
                        box_hits = box_raw[:10]
                    except Exception:
                        box_hits = []
                except Exception:
                    box_hits = []
                for b in box_hits:
                    b["source"] = "box"
                    if "webUrl" not in b and b.get("id"):
                        b["webUrl"] = f"https://app.box.com/file/{b['id']}"

            # MEGA
            mega_hits = []
            if use_mega:
                try:
                    mega_hits = search_mega_files_ranked(
                        uid=current_user.id,
                        query=query,
                        limit=50
                    ) or []
                except Exception:
                    mega_hits = []
                for m in mega_hits:
                    m["source"] = "mega"
                    if "webUrl" not in m or not m.get("webUrl"):
                        m["webUrl"] = "https://mega.nz/fm"

            # Gather all
            all_hits.extend(ms_hits)
            all_hits.extend(gd_hits)
            all_hits.extend(dbx_hits)
            all_hits.extend(box_hits)
            all_hits.extend(mega_hits)

            # optional access check for MS results only
            if ms_hits:
                perform_access_check = os.getenv("PERFORM_ACCESS_CHECK", "true").lower() == "true"
                if perform_access_check:
                    filtered = []
                    for f in ms_hits:
                        try:
                            if check_file_access(
                                ms_token4,
                                f["id"],
                                user_email,
                                f.get("parentReference", {}).get("siteId"),
                            ):
                                filtered.append(f)
                        except Exception:
                            pass
                    ms_hits = filtered

            if not all_hits:
                ai = "📁 No files found."
                # Do NOT apply rules (control prompt)
                _save_ai(user_email, chat_id, first_turn, user_input, ai)
                return jsonify(response=ai, intent="file_search", chat_id=chat_id)

            session["stage"] = "awaiting_selection"
            session["found_files"] = all_hits

            ai = "Please select file (e.g., 1,3):"
            # Do NOT apply rules to this structural UI message
            _save_ai(user_email, chat_id, first_turn, user_input, ai)

            per_page   = 5
            page       = 1
            paginated  = all_hits[:per_page]
            file_types = sorted(list(set([
                os.path.splitext(f["name"])[1].lower()
                for f in all_hits if "." in f["name"]
            ])))
            selected_file_ids = [f["id"] for f in all_hits]

            return jsonify({
                "response": ai,
                "pauseGPT": True,
                "files": paginated,
                "page": page,
                "total": len(all_hits),
                "file_types": file_types,
                "selectedFileIds": selected_file_ids,
                "allFileIds": [f["id"] for f in all_hits],
                "chat_id": chat_id,
            })

        # general Q&A / fallback (NOW with history & chat rules)
        ai = answer_general_query(user_input, history=history, chat_rules=rules)
        _save_ai(user_email, chat_id, first_turn, user_input, ai)
        return jsonify(response=ai, intent="general_response", chat_id=chat_id)

    if session.get("stage") == "awaiting_query":
        return _run_intent_pipeline()

    session["stage"] = "awaiting_query"
    session["found_files"] = []
    return _run_intent_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/paginate_files")
@login_required
def paginate_files():
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1

    filter_type = (request.args.get("type", "") or "").lower().strip()
    files = session.get("found_files", [])

    file_types = sorted(list(set([
        os.path.splitext(f["name"])[1].lower()
        for f in (session.get("found_files", [])) if "." in f["name"]
    ])))

    if filter_type:
        files = [f for f in files if f["name"].lower().endswith(filter_type)]

    per_page = 5
    total = len(files)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = files[start:end]

    return jsonify({
        "files": paginated,
        "page": page,
        "total": total,
        "file_types": file_types
    })

# ─────────────────────────────────────────────────────────────────────────────
# Profile & Security APIs used by the ProfileModal (drop-in)
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, io, zipfile, base64, hmac, hashlib, struct, secrets
from datetime import datetime, timezone as _tz
from flask import request, jsonify, send_file, send_from_directory, current_app, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

# Optional: real QR (falls back to a readable SVG if segno is missing)
try:
    import segno
except Exception:
    segno = None

# ── Storage for avatars ──────────────────────────────────────────────────────
AVATAR_DIR = os.path.join(os.getcwd(), "uploads", "avatars")
os.makedirs(AVATAR_DIR, exist_ok=True)

# ── Pref helpers (use your existing ones if present) ─────────────────────────
if "_get_pref" not in globals():
    def _get_pref(user_id, key):
        try:
            if "Prefs" in globals() and "db" in globals():
                row = Prefs.query.filter_by(user_id=user_id, key=key).first()
                return row.value if row else None
        except Exception:
            pass
        return (session.get("prefs", {}) or {}).get(str(user_id), {}).get(key)

if "_set_pref" not in globals():
    def _set_pref(user_id, key, value):
        try:
            if "Prefs" in globals() and "db" in globals():
                row = Prefs.query.filter_by(user_id=user_id, key=key).first()
                if row:
                    row.value = value
                else:
                    row = Prefs(user_id=user_id, key=key, value=value)
                    db.session.add(row)
                db.session.commit()
                return
        except Exception:
            current_app.logger.debug("Prefs DB not available; using session fallback.")
        prefs = session.get("prefs", {})
        u = prefs.setdefault(str(user_id), {})
        u[key] = value
        session["prefs"] = prefs

def _get_json_pref(uid, key, default=None):
    raw = _get_pref(uid, key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default

def _set_json_pref(uid, key, obj):
    _set_pref(uid, key, json.dumps(obj or {}))

# ── Helpers: Base32 secret & QR ──────────────────────────────────────────────
def make_base32_secret(num_bytes: int = 20) -> str:
    """RFC 4648 Base32 (uppercase) without padding."""
    return base64.b32encode(os.urandom(num_bytes)).decode("utf-8").rstrip("=")

def make_qr_svg(text: str) -> str:
    if segno:
        buf = io.BytesIO()
        qr = segno.make(text, error="m")
        qr.save(buf, kind="svg", scale=4, border=0)
        return buf.getvalue().decode("utf-8")  # return SVG as string
    # fallback
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'>
  <rect width='100%' height='100%' fill='#fff'/>
  <foreignObject x='8' y='8' width='164' height='164'>
    <body xmlns='http://www.w3.org/1999/xhtml' style="font:12px/1.35 system-ui;color:#111">
      <div style="font-weight:700;margin-bottom:4px">Scan unavailable</div>
      <div>Copy this into your authenticator:</div>
      <div style="word-break:break-all;color:#444;margin-top:6px">{esc}</div>
    </body>
  </foreignObject>
</svg>"""

# ── TOTP (HOTP/TOTP) ─────────────────────────────────────────────────────────
def _b32decode(s):
    s = s.strip().replace(" ", "")
    pad = "=" * ((8 - len(s) % 8) % 8)
    return base64.b32decode((s + pad).upper(), casefold=True)

def _hotp(secret_b32, counter, digits=6):
    key = _b32decode(secret_b32)
    msg = struct.pack(">Q", counter)
    h = hmac.new(key, msg, hashlib.sha1).digest()
    o = h[-1] & 0x0F
    code = (struct.unpack(">I", h[o:o+4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)

def _totp(secret_b32, ts=None, step=30, digits=6):
    if ts is None:
        ts = int(time.time())
    return _hotp(secret_b32, int(ts // step), digits)

def _verify_totp(secret_b32, code, window=1, step=30):
    try:
        code = str(int(code)).zfill(6)
    except Exception:
        return False
    now = int(time.time())
    for w in range(-window, window + 1):
        if _totp(secret_b32, now + w * step) == code:
            return True
    return False

# ── Profile ──────────────────────────────────────────────────────────────────
@app.route("/api/profile", methods=["GET", "PATCH"])
@login_required
def api_profile():
    uid = current_user.id
    if request.method == "GET":
        name = getattr(current_user, "name", None) or _get_pref(uid, "display_name") or ""
        avatar_url = _get_pref(uid, "avatar_url") or ""
        return jsonify({
            "id": uid,
            "email": getattr(current_user, "email", None),
            "name": name,
            "avatar_url": avatar_url
        })

    data = request.get_json(silent=True) or {}
    if "name" not in data:
        return jsonify({"ok": True})  # nothing to change
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name_required"}), 400
    try:
        if hasattr(current_user, "name"):
            current_user.name = name
            if "db" in globals():
                db.session.commit()
        else:
            _set_pref(uid, "display_name", name)
        return jsonify({"ok": True})
    except Exception as e:
        current_app.logger.exception("Save profile failed: %s", e)
        return jsonify({"error": "save_failed"}), 500

@app.route("/api/profile/avatar", methods=["POST"])
@login_required
def api_profile_avatar():
    uid = current_user.id
    f = request.files.get("avatar")
    if not f or f.filename == "":
        return jsonify({"error": "no_file"}), 400
    ext = (os.path.splitext(f.filename)[1] or ".png").lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    fname = secure_filename(f"{uid}-{int(time.time())}{ext}")
    path = os.path.join(AVATAR_DIR, fname)
    f.save(path)
    public_url = f"/uploads/avatars/{fname}"
    _set_pref(uid, "avatar_url", public_url)
    return jsonify({"ok": True, "avatar_url": public_url})

@app.route("/uploads/avatars/<path:filename>")
def serve_avatar(filename):
    return send_from_directory(AVATAR_DIR, filename)

# ── Password change ──────────────────────────────────────────────────────────
@app.route("/api/security/change_password", methods=["POST"])
@login_required
def api_change_password():
    data = request.get_json(silent=True) or {}
    cur = (data.get("current_password") or "").strip()
    new = (data.get("new_password") or "").strip()
    if not new:
        return jsonify({"error": "new_password_required"}), 400
    try:
        if hasattr(current_user, "check_password") and hasattr(current_user, "set_password"):
            if not current_user.check_password(cur):
                return jsonify({"error": "invalid_current_password"}), 400
            current_user.set_password(new)
            if "db" in globals():
                db.session.commit()
            return jsonify({"ok": True})
        return jsonify({"error": "not_supported"}), 501
    except Exception as e:
        current_app.logger.exception("change_password failed: %s", e)
        return jsonify({"error": "server_error"}), 500

# ── 2FA: status/setup/verify/disable ─────────────────────────────────────────
@app.route("/api/security/2fa/status")
@login_required
def api_2fa_status():
    uid = current_user.id
    enabled = (_get_pref(uid, "2fa_enabled") == "1")
    return jsonify({"enabled": enabled})

@app.route("/api/security/2fa/setup", methods=["POST"])
@login_required
def api_2fa_setup():
    uid = current_user.id
    secret_b32 = make_base32_secret(20)
    _set_pref(uid, "2fa_secret", secret_b32)
    _set_pref(uid, "2fa_enabled", "0")

    label  = getattr(current_user, "email", f"user-{uid}")
    issuer = "FileFinder"
    otpauth_url = (
        f"otpauth://totp/{issuer}:{label}"
        f"?secret={secret_b32}&issuer={issuer}&algorithm=SHA1&digits=6&period=30"
    )

    qr_svg = make_qr_svg(otpauth_url)
    return jsonify({"secret": secret_b32, "otpauth_url": otpauth_url, "qr_svg": qr_svg})

@app.route("/api/security/2fa/verify", methods=["POST"])
@login_required
def api_2fa_verify():
    uid = current_user.id
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip()
    secret = _get_pref(uid, "2fa_secret")
    if not secret:
        return jsonify({"error": "2fa_not_started"}), 400
    if not code:
        return jsonify({"error": "code_required"}), 400
    if not _verify_totp(secret, code, window=1, step=30):
        return jsonify({"error": "invalid_code"}), 400
    _set_pref(uid, "2fa_enabled", "1")
    return jsonify({"ok": True})

@app.route("/api/security/2fa/disable", methods=["POST"])
@login_required
def api_2fa_disable():
    uid = current_user.id
    _set_pref(uid, "2fa_enabled", "0")
    return jsonify({"ok": True})

# ── API token (store in Prefs) ───────────────────────────────────────────────
@app.route("/api/profile/api_token")
@login_required
def api_token_get():
    uid = current_user.id
    tok = _get_pref(uid, "api_token")
    if tok:
        return jsonify({"token": tok, "exists": True})
    return jsonify({"exists": False})

@app.route("/api/profile/api_token/rotate", methods=["POST"])
@login_required
def api_token_rotate():
    uid = current_user.id
    tok = secrets.token_hex(20)
    _set_pref(uid, "api_token", tok)
    return jsonify({"token": tok})

# ── Sessions (basic implementation) ──────────────────────────────────────────
@app.route("/api/auth/sessions")
@login_required
def api_sessions():
    sess = [{
        "id": "this",
        "user_agent": request.user_agent.string,
        "last_active": datetime.now(_tz.utc).isoformat(),
    }]
    return jsonify({"sessions": sess})

@app.route("/api/auth/sessions/revoke_others", methods=["POST"])
@login_required
def api_sessions_revoke_others():
    return jsonify({"ok": True})

# ── Export & Delete ──────────────────────────────────────────────────────────
@app.route("/api/account/export")
@login_required
def api_account_export():
    uid = current_user.id
    bundle = {
        "user": {
            "id": uid,
            "email": getattr(current_user, "email", None),
            "name": getattr(current_user, "name", None),
        },
        "prefs": {
            "display_name": _get_pref(uid, "display_name"),
            "avatar_url": _get_pref(uid, "avatar_url"),
            "api_token": _get_pref(uid, "api_token"),
            "2fa_enabled": _get_pref(uid, "2fa_enabled"),
            "2fa_secret": _get_pref(uid, "2fa_secret"),
        },
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("export.json", json.dumps(bundle, indent=2))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="export.zip")

@app.route("/api/account", methods=["DELETE"])
@login_required
def api_account_delete():
    uid = current_user.id
    # Clear prefs for this user
    try:
        if "Prefs" in globals() and "db" in globals():
            Prefs.query.filter_by(user_id=uid).delete()
            db.session.commit()
    except Exception:
        pass
    # Try deleting the user record if possible
    try:
        from models import User  # adjust to your project layout if needed
        u = User.query.get(uid)
        if u:
            db.session.delete(u)
            db.session.commit()
    except Exception:
        session.clear()
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────────────────────
# File selection handler
# ─────────────────────────────────────────────────────────────────────────────
def handle_file_selection(user_input, token, user_email, chat_id):
    """
    Handle file selection from the list stored in session["found_files"].

    user_input: list of 1-based indices OR string like "1,3-5"
                special string "cancel" to abort.
    token:      MS Graph access token (may be None if not needed)
    """
    files = session.get("found_files", []) or []
    if not files:
        session["stage"] = "awaiting_query"
        session["found_files"] = []
        return jsonify(response="⚠️ File list expired", intent="error", chat_id=chat_id)

    # ---- parse indices helper (supports "1,3-5" and [1,3])
    def _parse_indices(inp):
        if isinstance(inp, list):
            ordered = []
            seen = set()
            for v in inp:
                try:
                    i = int(v) - 1
                except Exception:
                    continue
                if 0 <= i < len(files) and i not in seen:
                    seen.add(i); ordered.append(i)
            return ordered

        s = str(inp or "").strip().lower()
        if s == "cancel":
            return "CANCEL"

        parts = [p.strip() for p in s.split(",") if p.strip()]
        ordered, seen = [], set()
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                if a.strip().isdigit() and b.strip().isdigit():
                    start, end = int(a) - 1, int(b) - 1
                    if start > end:
                        start, end = end, start
                    for i in range(start, end + 1):
                        if 0 <= i < len(files) and i not in seen:
                            seen.add(i); ordered.append(i)
            elif p.isdigit():
                i = int(p) - 1
                if 0 <= i < len(files) and i not in seen:
                    seen.add(i); ordered.append(i)
        return ordered

    parsed = _parse_indices(user_input)
    if parsed == "CANCEL":
        session["stage"] = "awaiting_query"
        session["found_files"] = []
        return jsonify(response="❌ Cancelled", intent="general_response", chat_id=chat_id)

    if not isinstance(parsed, list) or not parsed:
        return jsonify(response="❌ Invalid selection", intent="error", chat_id=chat_id)

    selected = [files[i] for i in parsed if 0 <= i < len(files)]
    if not selected:
        return jsonify(response="⚠️ No matching files found", intent="error", chat_id=chat_id)

    # ---- helpers
    def _link_of(f):
        return (
            f.get("webUrl")
            or f.get("webViewLink")
            or f.get("preview_url")
            or f.get("url")
            or f.get("link")
            or ""
        )

    def _line_for(f):
        name = f.get("name") or f.get("displayName") or "(unnamed)"
        url  = _link_of(f)
        return f"- [{name}]({url})" if url else f"- {name}"

    # ---- split by source
    gd_files   = [f for f in selected if f.get("source") == "google_drive"]
    dbx_files  = [f for f in selected if f.get("source") == "dropbox"]
    box_files  = [f for f in selected if f.get("source") == "box"]
    mega_files = [f for f in selected if f.get("source") == "mega"]
    ms_files   = [f for f in selected if f.get("source") not in ("google_drive", "dropbox", "box", "mega")]

    # ---- Microsoft access check + email send (if any MS items)
    ms_lines = []
    ms_header = None
    if ms_files:
        if not token:
            ms_header = "⚠️ Microsoft items could not be sent (not connected)."
            ms_lines = [_line_for(f) for f in ms_files]
        else:
            accessible = []
            for f in ms_files:
                try:
                    if check_file_access(
                        token,
                        f["id"],
                        user_email,
                        f.get("parentReference", {}).get("siteId"),
                    ):
                        accessible.append(f)
                except Exception:
                    pass

            if accessible:
                try:
                    send_multiple_file_email(token, user_email, accessible)
                    ms_header = "✅ Sent via email (Microsoft):"
                except Exception:
                    current_app.logger.exception("send_multiple_file_email failed")
                    ms_header = "⚠️ Could not send Microsoft files via email."
                ms_lines = [_line_for(f) for f in accessible]
            else:
                ms_header = "⚠️ No accessible Microsoft items."
                ms_lines = []

    # ---- Compose provider sections
    parts = []

    if ms_files:
        parts.append(ms_header)
        if ms_lines:
            parts.extend(ms_lines)

    if gd_files:
        parts.append("\n🔗 Google Drive links:")
        parts.extend([_line_for(f) for f in gd_files])

    if dbx_files:
        parts.append("\n🔗 Dropbox links:")
        parts.extend([_line_for(f) for f in dbx_files])

    if box_files:
        parts.append("\n🔗 Box links:")
        parts.extend([_line_for(f) for f in box_files])

    if mega_files:
        parts.append("\n🔗 MEGA links:")
        parts.extend([_line_for(f) for f in mega_files])

    if not parts:
        parts.append("Nothing to send.")

    parts.append("\nNeed anything else?")
    confirmation_message = "\n".join(parts)

    save_message(user_email, chat_id, ai_response=confirmation_message)
    session["stage"] = "awaiting_query"
    session["found_files"] = []

    return jsonify(response=confirmation_message, intent="file_sent", chat_id=chat_id)

def is_number_selection(text):
    try:
        return all(s.strip().isdigit() for s in text.split(','))
    except Exception:
        return False
    
# --- Uploads API (ChatGPT-like) ---------------------------------------------
import os, time, uuid
from flask import send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_ROOT = os.path.join(os.getcwd(), "uploads", "chat")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB

def _user_chat_dir():
    uid = getattr(current_user, "id", "anon")
    chat_id = session.get("chat_id", "default")
    d = os.path.join(UPLOAD_ROOT, str(uid), str(chat_id))
    os.makedirs(d, exist_ok=True)
    return d

@app.route("/api/uploads/files", methods=["POST"])
@login_required
def api_upload_files():
    if "files" not in request.files:
        return jsonify({"error": "no_files"}), 400

    folder = _user_chat_dir()
    out = []
    for f in request.files.getlist("files"):
        if not f or not f.filename:
            continue
        name = secure_filename(f.filename)
        file_id = uuid.uuid4().hex
        fname = f"{int(time.time())}-{file_id}-{name}"
        path = os.path.join(folder, fname)
        f.save(path)
        out.append({
            "id": file_id,                 # <= stable id for chips
            "name": name,
            "size": os.path.getsize(path),
            "url": f"/uploads/chat/{current_user.id}/{session.get('chat_id','default')}/{fname}",
            "path": fname                  # <= token we can safely delete by
        })
    return jsonify({"files": out})

@app.route("/api/uploads/files", methods=["DELETE"])
@login_required
def api_delete_upload():
    data = request.get_json(silent=True) or {}
    token = os.path.basename(data.get("path") or "")
    if not token:
        return jsonify({"ok": False, "error": "missing_path"}), 400

    folder = _user_chat_dir()
    file_path = os.path.join(folder, token)
    if not os.path.isfile(file_path):
        return jsonify({"ok": False, "error": "not_found"}), 404

    try:
        os.remove(file_path)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True})

@app.route("/uploads/chat/<int:uid>/<chat_id>/<path:filename>")
def serve_chat_upload(uid, chat_id, filename):
    folder = os.path.join(UPLOAD_ROOT, str(uid), str(chat_id))
    return send_from_directory(folder, filename)


# ─────────────────────────────────────────────────────────────────────────────
# SPA routes (serve React index.html so refresh works)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/login")
@app.route("/signup")
@app.route("/dashboard")
@app.route("/admin")
@app.route("/admin/upload")
@app.route("/trial-ended")  # route the SPA will render as the “trial ended” page
def spa_routes():
    return send_from_directory(app.static_folder, "index.html")

# Static serving & catch-all
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# 🏁 Startup
if __name__ == "__main__":
    app.run(debug=True)
