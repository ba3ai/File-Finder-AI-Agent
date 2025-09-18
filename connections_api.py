# connections_api.py
import os
import json
import time
import secrets
import requests
from datetime import datetime, timedelta
from urllib.parse import urlencode

from flask import Blueprint, make_response, request, redirect, jsonify, current_app
from flask_login import login_required, current_user
from itsdangerous import URLSafeSerializer, BadSignature

from models import db, Prefs  # Prefs: id, user_id, key, value, created_at, updated_at

connections_bp = Blueprint("connections", __name__, url_prefix="/api/connections")

# =============================================================================
# Config helpers (single public base URL keeps redirect URIs consistent)
# =============================================================================
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:5000").rstrip("/")
SECRET_KEY      = os.getenv("SECRET_KEY", "dev-secret")

def _serializer():
    # Used to sign/verify OAuth state payloads
    return URLSafeSerializer(SECRET_KEY, salt="oauth-state")

def _make_state(uid: int) -> str:
    payload = {"uid": int(uid), "ts": int(time.time()), "nonce": secrets.token_urlsafe(8)}
    return _serializer().dumps(payload)

def _parse_state(state: str) -> dict:
    return _serializer().loads(state)

# Microsoft redirect (computed from PUBLIC_BASE_URL)
MS_REDIRECT_PATH = "/api/connections/microsoft/callback"
MS_REDIRECT_URI  = f"{PUBLIC_BASE_URL}{MS_REDIRECT_PATH}"
print(f"Microsoft redirect URI: {MS_REDIRECT_URI}")

# Google/Dropbox redirects come from env to match the console config
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI")
DROPBOX_REDIRECT_URI = os.getenv("DROPBOX_REDIRECT_URI")

# Box redirect computed from PUBLIC_BASE_URL (no hardcoded 127.0.0.1)
BOX_REDIRECT_PATH = "/api/connections/box/callback"
BOX_REDIRECT_URI  = f"{PUBLIC_BASE_URL}{BOX_REDIRECT_PATH}"

# Slack redirect computed from PUBLIC_BASE_URL
SLACK_REDIRECT_PATH = "/api/connections/slack/callback"
SLACK_REDIRECT_URI  = f"{PUBLIC_BASE_URL}{SLACK_REDIRECT_PATH}"

# =============================================================================
# Utilities
# =============================================================================
def _json(data, status=200):
    """JSON response with no-store so one user's status never shows for another."""
    resp = make_response(json.dumps(data), status)
    resp.mimetype = "application/json"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

def _get_pref(uid, key, default=None):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else default

def _set_pref(uid, key, value):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    if row:
        row.value = value
        row.updated_at = datetime.utcnow()
    else:
        row = Prefs(user_id=uid, key=key, value=value)
        db.session.add(row)
    db.session.commit()

def _del_pref(uid, key):
    Prefs.query.filter_by(user_id=uid, key=key).delete()
    db.session.commit()

# =============================================================================
# Per-user preference keys
# =============================================================================
# Microsoft
MS_ACCESS_TOKEN  = "ms_access_token"
MS_REFRESH_TOKEN = "ms_refresh_token"
MS_EXPIRES_AT    = "ms_expires_at"
MS_ACCOUNT_EMAIL = "ms_account_email"
MS_OAUTH_STATE   = "ms_oauth_state"

# Google
GOOGLE_ACCESS_TOKEN  = "google_access_token"
GOOGLE_REFRESH_TOKEN = "google_refresh_token"
GOOGLE_EXPIRES_AT    = "google_expires_at"
GOOGLE_ACCOUNT_EMAIL = "google_account_email"
GOOGLE_OAUTH_STATE   = "google_oauth_state"

# Dropbox
DBX_ACCESS_TOKEN  = "dbx_access_token"
DBX_REFRESH_TOKEN = "dbx_refresh_token"
DBX_EXPIRES_AT    = "dbx_expires_at"
DBX_ACCOUNT_EMAIL = "dbx_account_email"
DBX_OAUTH_STATE   = "dbx_oauth_state"

# Box
BOX_ACCESS_TOKEN  = "box_access_token"
BOX_REFRESH_TOKEN = "box_refresh_token"
BOX_EXPIRES_AT    = "box_expires_at"
BOX_ACCOUNT_EMAIL = "box_account_email"
BOX_ACCOUNT_NAME  = "box_account_name"
BOX_OAUTH_STATE   = "box_oauth_state"

# MEGA
MEGA_SESSION_KEY = "mega_session"
MEGA_LINKS_KEY   = "mega_links"

# Slack
SLACK_BOT_TOKEN   = "slack_bot_token"
SLACK_USER_TOKEN  = "slack_user_token"
SLACK_TEAM_ID     = "slack_team_id"
SLACK_TEAM_NAME   = "slack_team_name"
SLACK_OAUTH_STATE = "slack_oauth_state"

# =============================================================================
# GET /api/connections  ->  current user's connection status
# =============================================================================
@connections_bp.route("", methods=["GET"])
@login_required
def list_connections():
    uid = current_user.id

    ms_connected = bool(_get_pref(uid, MS_ACCESS_TOKEN))
    ms_email     = _get_pref(uid, MS_ACCOUNT_EMAIL)

    g_connected  = bool(_get_pref(uid, GOOGLE_ACCESS_TOKEN))
    g_email      = _get_pref(uid, GOOGLE_ACCOUNT_EMAIL)

    dbx_connected = bool(_get_pref(uid, DBX_ACCESS_TOKEN))
    dbx_email     = _get_pref(uid, DBX_ACCOUNT_EMAIL)

    box_connected = bool(_get_pref(uid, BOX_ACCESS_TOKEN))
    box_email     = _get_pref(uid, BOX_ACCOUNT_EMAIL)

    # Slack: connected if either a bot token or user token is present
    slack_connected = bool(_get_pref(uid, SLACK_BOT_TOKEN) or _get_pref(uid, SLACK_USER_TOKEN))
    slack_team      = _get_pref(uid, SLACK_TEAM_NAME) or ""

    result = [
        # Files
        {"provider": "sharepoint",   "connected": ms_connected,    "account_email": ms_email},
        {"provider": "onedrive",     "connected": ms_connected,    "account_email": ms_email},
        {"provider": "google_drive", "connected": g_connected,     "account_email": g_email},
        {"provider": "dropbox",      "connected": dbx_connected,   "account_email": dbx_email},
        {"provider": "box",          "connected": box_connected,   "account_email": box_email},

        # Mail (virtual)
        {"provider": "outlook", "connected": ms_connected, "account_email": ms_email},
        {"provider": "gmail",   "connected": g_connected,  "account_email": g_email},

        # Calendar (virtual)
        {"provider": "outlook_calendar", "connected": ms_connected, "account_email": ms_email},
        {"provider": "google_calendar",  "connected": g_connected,  "account_email": g_email},

        # Teams (shares Microsoft token)
        {"provider": "teams", "connected": ms_connected, "account_email": ms_email},

        # Slack (own OAuth)
        {"provider": "slack", "connected": slack_connected, "account_email": slack_team},
    ]

    mega_connected = bool(_get_pref(uid, MEGA_SESSION_KEY))
    result.append({"provider": "mega", "connected": mega_connected, "account_email": None})

    return _json({"connections": result})

# =============================================================================
# Microsoft (delegated) OAuth – SharePoint & OneDrive (+ Teams scopes)
# =============================================================================
from msal import ConfidentialClientApplication

def _ms_app():
    client_id = os.getenv("MS_CLIENT_ID")
    client_secret = os.getenv("MS_CLIENT_SECRET")
    tenant = os.getenv("MS_TENANT_ID", "common")
    if not client_id or not client_secret:
        raise RuntimeError("MS_CLIENT_ID / MS_CLIENT_SECRET not configured")
    return ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_secret,
        authority=f"https://login.microsoftonline.com/{tenant}",
    )

# Do NOT add openid/profile/offline_access here; MSAL handles ID token.
MS_SCOPES = os.getenv(
    "MS_SCOPES",
    # Added Chat.Read and ChannelMessage.Read.All for Teams/Chats
    "User.Read Calendars.Read Mail.Read Files.Read.All Sites.Read.All "
    "Team.ReadBasic.All Channel.ReadBasic.All Chat.Read ChannelMessage.Read.All"
).split()

@connections_bp.route("/microsoft/authurl", methods=["GET"])
@login_required
def microsoft_authurl():
    app = _ms_app()
    state = _make_state(current_user.id)
    # Save state to enforce one-time use for this user (extra CSRF guard)
    _set_pref(current_user.id, MS_OAUTH_STATE, state)
    url = app.get_authorization_request_url(
        scopes=MS_SCOPES,
        redirect_uri=MS_REDIRECT_URI,
        state=state,
    )
    return _json({"auth_url": url})

@connections_bp.route("/microsoft/callback")
def microsoft_callback():
    # NOTE: no @login_required — we verify signed state instead.
    state = request.args.get("state")
    code  = request.args.get("code")
    if not state or not code:
        return _json({"error": "Missing state or code"}, 400)

    try:
        payload = _parse_state(state)
        uid = int(payload.get("uid"))
    except (BadSignature, ValueError, TypeError):
        return _json({"error": "Invalid state"}, 400)

    # Ensure this state was issued for that user (and consume it)
    saved = _get_pref(uid, MS_OAUTH_STATE)
    if not saved or saved != state:
        return _json({"error": "State mismatch or expired"}, 400)
    _del_pref(uid, MS_OAUTH_STATE)

    token_result = _ms_app().acquire_token_by_authorization_code(
        code, scopes=MS_SCOPES, redirect_uri=MS_REDIRECT_URI
    )
    if "access_token" not in token_result:
        return _json({"error": token_result.get("error_description", "Microsoft auth failed")}, 400)

    access_token = token_result["access_token"]
    refresh_token = token_result.get("refresh_token")
    expires_in = int(token_result.get("expires_in", 3600))
    expires_at = (datetime.utcnow() + timedelta(seconds=max(expires_in - 60, 0))).isoformat()

    # Email from id token claims (MSAL adds 'openid' internally)
    id_claims = token_result.get("id_token_claims") or {}
    email = id_claims.get("preferred_username") or id_claims.get("upn") or id_claims.get("email")

    _set_pref(uid, MS_ACCESS_TOKEN, access_token)
    if refresh_token:
        _set_pref(uid, MS_REFRESH_TOKEN, refresh_token)
    _set_pref(uid, MS_EXPIRES_AT, expires_at)
    if email:
        _set_pref(uid, MS_ACCOUNT_EMAIL, email)

    return redirect("/dashboard?connected=ms")

@connections_bp.route("/microsoft/disconnect/<provider>", methods=["DELETE"])
@login_required
def microsoft_disconnect(provider):
    uid = current_user.id
    for k in (MS_ACCESS_TOKEN, MS_REFRESH_TOKEN, MS_EXPIRES_AT, MS_ACCOUNT_EMAIL, MS_OAUTH_STATE):
        _del_pref(uid, k)
    return _json({"message": "Microsoft disconnected"})

# =============================================================================
# Google Drive OAuth (read-only) + Gmail + Calendar
# =============================================================================
def _google_client_config():
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not configured")
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "userinfo_endpoint": "https://www.googleapis.com/oauth2/v2/userinfo",
    }

# Request Drive + Gmail + Calendar in one Google consent
GOOGLE_SCOPES = (
    "openid email profile "
    "https://www.googleapis.com/auth/drive.readonly "
    "https://www.googleapis.com/auth/gmail.readonly "
    "https://www.googleapis.com/auth/calendar.readonly"
)

@connections_bp.route("/google/authurl", methods=["GET"])
@login_required
def google_authurl():
    cfg = _google_client_config()
    state = _make_state(current_user.id)
    _set_pref(current_user.id, GOOGLE_OAUTH_STATE, state)
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "access_type": "offline",              # request refresh_token
        "include_granted_scopes": "true",
        "prompt": "consent",                   # ensures refresh_token on first consent
        "state": state,
    }
    url = cfg["auth_uri"] + "?" + urlencode(params)
    return _json({"auth_url": url})

@connections_bp.route("/google/callback")
def google_callback():
    state = request.args.get("state")
    code  = request.args.get("code")
    if not state or not code:
        return _json({"error": "Missing state or code"}, 400)

    try:
        payload = _parse_state(state)
        uid = int(payload.get("uid"))
    except (BadSignature, ValueError, TypeError):
        return _json({"error": "Invalid state"}, 400)

    saved = _get_pref(uid, GOOGLE_OAUTH_STATE)
    if not saved or saved != state:
        return _json({"error": "State mismatch or expired"}, 400)
    _del_pref(uid, GOOGLE_OAUTH_STATE)

    cfg = _google_client_config()
    tok_res = requests.post(
        cfg["token_uri"],
        data={
            "code": code,
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    if not tok_res.ok:
        return _json({"error": f"Token exchange failed: {tok_res.text}"}, 400)
    tok = tok_res.json()

    access_token  = tok.get("access_token")
    refresh_token = tok.get("refresh_token")   # may be None on re-consent
    expires_in    = int(tok.get("expires_in", 0))
    exp_ts        = int(time.time()) + expires_in if expires_in else 0

    # Fetch profile (email) for “Connected as …”
    email = None
    try:
        ui = requests.get(
            cfg["userinfo_endpoint"],
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20,
        )
        if ui.ok:
            email = ui.json().get("email")
    except Exception:
        pass

    _set_pref(uid, GOOGLE_ACCESS_TOKEN, access_token)
    if refresh_token:
        _set_pref(uid, GOOGLE_REFRESH_TOKEN, refresh_token)
    if exp_ts:
        _set_pref(uid, GOOGLE_EXPIRES_AT, str(exp_ts))
    if email:
        _set_pref(uid, GOOGLE_ACCOUNT_EMAIL, email)

    return redirect("/dashboard?connected=google")

@connections_bp.route("/google/disconnect", methods=["DELETE"])
@login_required
def google_disconnect():
    uid = current_user.id
    access = _get_pref(uid, GOOGLE_ACCESS_TOKEN)
    if access:
        try:
            requests.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": access},
                timeout=10,
            )
        except Exception:
            pass

    for k in (GOOGLE_ACCESS_TOKEN, GOOGLE_REFRESH_TOKEN, GOOGLE_EXPIRES_AT, GOOGLE_ACCOUNT_EMAIL, GOOGLE_OAUTH_STATE):
        _del_pref(uid, k)
    return _json({"message": "Google Drive disconnected"})

# =============================================================================
# Dropbox OAuth (scoped)
# =============================================================================
def _dbx_client_config():
    cid = os.getenv("DROPBOX_CLIENT_ID")
    csec = os.getenv("DROPBOX_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("DROPBOX_CLIENT_ID / DROPBOX_CLIENT_SECRET not configured")
    return {
        "client_id": cid,
        "client_secret": csec,
        "auth_uri": "https://www.dropbox.com/oauth2/authorize",
        "token_uri": "https://api.dropboxapi.com/oauth2/token",
        "redirect_uri": DROPBOX_REDIRECT_URI,
        "userinfo": "https://api.dropboxapi.com/2/users/get_current_account",
        "revoke": "https://api.dropboxapi.com/2/auth/token/revoke",
    }

# Minimum scopes for listing/searching metadata + account email + temp links
DBX_SCOPES = "files.metadata.read account_info.read files.content.read"

@connections_bp.route("/dropbox/authurl", methods=["GET"])
@login_required
def dropbox_authurl():
    cfg = _dbx_client_config()
    state = _make_state(current_user.id)
    _set_pref(current_user.id, DBX_OAUTH_STATE, state)
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "token_access_type": "offline",  # refresh_token
        "scope": DBX_SCOPES,
        "state": state,
    }
    url = cfg["auth_uri"] + "?" + urlencode(params)
    return _json({"auth_url": url})

@connections_bp.route("/dropbox/callback")
def dropbox_callback():
    state = request.args.get("state")
    code  = request.args.get("code")
    if not state or not code:
        return _json({"error": "Missing state or code"}, 400)

    try:
        payload = _parse_state(state)
        uid = int(payload.get("uid"))
    except (BadSignature, ValueError, TypeError):
        return _json({"error": "Invalid state"}, 400)

    saved = _get_pref(uid, DBX_OAUTH_STATE)
    if not saved or saved != state:
        return _json({"error": "State mismatch or expired"}, 400)
    _del_pref(uid, DBX_OAUTH_STATE)

    cfg = _dbx_client_config()
    tok = requests.post(
        cfg["token_uri"],
        data={
            "code": code,
            "grant_type": "authorization_code",
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uri": cfg["redirect_uri"],
        },
        timeout=30,
    )
    if not tok.ok:
        return _json({"error": f"Token exchange failed: {tok.text}"}, 400)
    t = tok.json()

    access_token  = t.get("access_token")
    refresh_token = t.get("refresh_token")
    expires_in    = int(t.get("expires_in", 0))
    exp_ts        = int(time.time()) + expires_in if expires_in else 0

    # Get account info for “Connected as …”
    email = None
    try:
        ui = requests.post(
            cfg["userinfo"],
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20,
        )
        if ui.ok:
            email = ui.json().get("email")
    except Exception:
        pass

    _set_pref(uid, DBX_ACCESS_TOKEN, access_token)
    if refresh_token:
        _set_pref(uid, DBX_REFRESH_TOKEN, refresh_token)
    if exp_ts:
        _set_pref(uid, DBX_EXPIRES_AT, str(exp_ts))
    if email:
        _set_pref(uid, DBX_ACCOUNT_EMAIL, email)

    return redirect("/dashboard?connected=dropbox")

@connections_bp.route("/dropbox/disconnect", methods=["DELETE"])
@login_required
def dropbox_disconnect():
    uid = current_user.id
    access = _get_pref(uid, DBX_ACCESS_TOKEN)
    cfg = _dbx_client_config()
    if access:
        try:
            requests.post(cfg["revoke"], headers={"Authorization": f"Bearer {access}"}, timeout=10)
        except Exception:
            pass
    for k in (DBX_ACCESS_TOKEN, DBX_REFRESH_TOKEN, DBX_EXPIRES_AT, DBX_ACCOUNT_EMAIL, DBX_OAUTH_STATE):
        _del_pref(uid, k)
    return _json({"message": "Dropbox disconnected"})

# =============================================================================
# Box OAuth (User Authentication – OAuth 2.0)
# =============================================================================
BOX_AUTH_URL   = "https://account.box.com/api/oauth2/authorize"
BOX_TOKEN_URL  = "https://api.box.com/oauth2/token"
BOX_REVOKE_URL = "https://api.box.com/oauth2/revoke"
BOX_API_ME     = "https://api.box.com/2.0/users/me"

@connections_bp.route("/box/authurl", methods=["GET"])
@login_required
def box_authurl():
    state = _make_state(current_user.id)
    _set_pref(current_user.id, BOX_OAUTH_STATE, state)
    params = {
        "response_type": "code",
        "client_id": os.getenv("BOX_CLIENT_ID"),
        "redirect_uri": BOX_REDIRECT_URI,
        "state": state,
    }
    url = f"{BOX_AUTH_URL}?{urlencode(params)}"
    return _json({"auth_url": url})

@connections_bp.route("/box/callback", methods=["GET"])
def box_callback():
    code  = request.args.get("code")
    state = request.args.get("state")
    if not state or not code:
        return redirect("/dashboard?connected=box&error=state")

    try:
        payload = _parse_state(state)
        uid = int(payload.get("uid"))
    except (BadSignature, ValueError, TypeError):
        return redirect("/dashboard?connected=box&error=state")

    saved = _get_pref(uid, BOX_OAUTH_STATE)
    if not saved or saved != state:
        return redirect("/dashboard?connected=box&error=state")
    _del_pref(uid, BOX_OAUTH_STATE)

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": os.getenv("BOX_CLIENT_ID"),
        "client_secret": os.getenv("BOX_CLIENT_SECRET"),
        "redirect_uri": BOX_REDIRECT_URI,
    }
    tok = requests.post(BOX_TOKEN_URL, data=data, timeout=20)
    if not tok.ok:
        return redirect("/dashboard?connected=box&error=token")
    t = tok.json()

    at = t.get("access_token")
    rt = t.get("refresh_token")
    exp = int(time.time() + int(t.get("expires_in", 3600)))
    if not at or not rt:
        return redirect("/dashboard?connected=box&error=token")

    _set_pref(uid, BOX_ACCESS_TOKEN, at)
    _set_pref(uid, BOX_REFRESH_TOKEN, rt)
    _set_pref(uid, BOX_EXPIRES_AT, str(exp))

    # Fetch account profile for “Connected as …”
    try:
        me = requests.get(BOX_API_ME, headers={"Authorization": f"Bearer {at}"}, timeout=15).json()
        _set_pref(uid, BOX_ACCOUNT_EMAIL, me.get("login") or "")
        _set_pref(uid, BOX_ACCOUNT_NAME,  me.get("name") or "")
    except Exception:
        pass

    return redirect("/dashboard?connected=box")

@connections_bp.route("/box/disconnect", methods=["DELETE"])
@login_required
def box_disconnect():
    uid = current_user.id
    cid = os.getenv("BOX_CLIENT_ID")
    csec = os.getenv("BOX_CLIENT_SECRET")
    at = _get_pref(uid, BOX_ACCESS_TOKEN)
    rt = _get_pref(uid, BOX_REFRESH_TOKEN)

    # best-effort revoke
    try:
        if at:
            requests.post(BOX_REVOKE_URL, data={"client_id": cid, "client_secret": csec, "token": at}, timeout=10)
        if rt:
            requests.post(BOX_REVOKE_URL, data={"client_id": cid, "client_secret": csec, "token": rt}, timeout=10)
    except Exception:
        pass

    for k in (BOX_ACCESS_TOKEN, BOX_REFRESH_TOKEN, BOX_EXPIRES_AT, BOX_ACCOUNT_EMAIL, BOX_ACCOUNT_NAME, BOX_OAUTH_STATE):
        _del_pref(uid, k)
    return _json({"message": "Box disconnected"})

# ── MEGA routes (session-only, named endpoints to avoid collisions)
@connections_bp.route("/mega/connect", methods=["POST"], endpoint="mega_connect_api")
@login_required
def mega_connect_api():
    """
    Body: { "session": "<MEGAcmd exported session string>" }
    We store only the session, not the password.
    """
    uid = current_user.id
    data = request.get_json(force=True, silent=True) or {}
    session_str = (data.get("session") or "").strip()
    if not session_str:
        return _json({"error": "Missing session"}, 400)
    _set_pref(uid, MEGA_SESSION_KEY, session_str)
    return _json({"message": "MEGA connected (session saved)"})

@connections_bp.route("/mega/add_link", methods=["POST"], endpoint="mega_add_link_api")
@login_required
def mega_add_link_api():
    uid = current_user.id
    data = request.get_json(force=True, silent=True) or {}
    link = (data.get("link") or "").strip()
    if not link:
        return _json({"error": "Missing link"}, 400)
    from mega_api import add_user_link
    links = add_user_link(uid, link)
    return _json({"links": links})

@connections_bp.route("/mega/remove_link", methods=["POST"], endpoint="mega_remove_link_api")
@login_required
def mega_remove_link_api():
    uid = current_user.id
    data = request.get_json(force=True, silent=True) or {}
    link = (data.get("link") or "").strip()
    from mega_api import remove_user_link
    links = remove_user_link(uid, link)
    return _json({"links": links})

@connections_bp.route("/mega/disconnect", methods=["DELETE"], endpoint="mega_disconnect_api")
@login_required
def mega_disconnect_api():
    uid = current_user.id
    for key in (MEGA_SESSION_KEY, MEGA_LINKS_KEY):
        _del_pref(uid, key)
    return _json({"message": "MEGA disconnected"})

# =============================================================================
# Teams routes reuse Microsoft consent/disconnect
# =============================================================================
@connections_bp.route("/teams/authurl", methods=["GET"])
@login_required
def teams_authurl():
    # Reuse Microsoft consent (includes Teams scopes)
    return microsoft_authurl()

@connections_bp.route("/teams/disconnect", methods=["DELETE"])
@login_required
def teams_disconnect():
    # Reuse Microsoft disconnect
    return microsoft_disconnect("teams")

# =============================================================================
# Slack OAuth (bot + optional user token for search.messages)
# =============================================================================
@connections_bp.route("/slack/start", methods=["GET"])
@login_required
def slack_start():
    client_id = os.getenv("SLACK_CLIENT_ID")
    client_secret = os.getenv("SLACK_CLIENT_SECRET")
    if not client_id or not client_secret:
        return _json({"error": "Slack not configured. Set SLACK_CLIENT_ID/SLACK_CLIENT_SECRET and PUBLIC_BASE_URL."}, 400)

    state = _make_state(current_user.id)
    _set_pref(current_user.id, SLACK_OAUTH_STATE, state)

    params = {
        "client_id": client_id,
        # Bot scopes (channel list/history + users + files) for fallback scanning
        "scope": ",".join([
            "channels:read",
            "groups:read",
            "im:history",
            "mpim:history",
            "channels:history",
            "groups:history",
            "users:read",
            "files:read",
        ]),
        # User scopes (optional): workspace-wide search via search.messages
        "user_scope": ",".join([
            "search:read",
            "users:read",
            "channels:read",
            "groups:read",
            "im:history",
            "mpim:history",
        ]),
        "redirect_uri": SLACK_REDIRECT_URI,
        "state": state,
    }
    url = "https://slack.com/oauth/v2/authorize?" + urlencode(params)
    return _json({"redirect": url})

@connections_bp.route("/slack/callback", methods=["GET"])
def slack_callback():
    code  = request.args.get("code")
    state = request.args.get("state") or ""
    if not code or not state:
        return redirect("/?slack=error")

    # Verify signed state and bind to user
    try:
        payload = _parse_state(state)
        uid = int(payload.get("uid"))
    except (BadSignature, ValueError, TypeError):
        return redirect("/?slack=error")

    saved = _get_pref(uid, SLACK_OAUTH_STATE)
    if not saved or saved != state:
        return redirect("/?slack=error")
    _del_pref(uid, SLACK_OAUTH_STATE)

    data = {
        "client_id": os.getenv("SLACK_CLIENT_ID"),
        "client_secret": os.getenv("SLACK_CLIENT_SECRET"),
        "code": code,
        "redirect_uri": SLACK_REDIRECT_URI,
    }
    try:
        r = requests.post("https://slack.com/api/oauth.v2.access", data=data, timeout=20)
        j = r.json() if r.ok else {}
    except Exception:
        j = {}

    if not j.get("ok"):
        current_app.logger.error("Slack oauth failed: %s", j)
        return redirect("/?slack=error")

    bot_token  = j.get("access_token")
    user_token = (j.get("authed_user") or {}).get("access_token")
    team_id    = (j.get("team") or {}).get("id")
    team_name  = (j.get("team") or {}).get("name")

    if bot_token:  _set_pref(uid, SLACK_BOT_TOKEN, bot_token)
    if user_token: _set_pref(uid, SLACK_USER_TOKEN, user_token)
    if team_id:    _set_pref(uid, SLACK_TEAM_ID, team_id)
    if team_name:  _set_pref(uid, SLACK_TEAM_NAME, team_name)

    return redirect("/dashboard?connected=slack")

@connections_bp.route("/slack/disconnect", methods=["DELETE"])
@login_required
def slack_disconnect():
    uid = current_user.id
    tokens = [_get_pref(uid, SLACK_USER_TOKEN), _get_pref(uid, SLACK_BOT_TOKEN)]
    # best-effort revoke
    for t in tokens:
        if not t:
            continue
        try:
            requests.post("https://slack.com/api/auth.revoke",
                          headers={"Authorization": f"Bearer {t}"},
                          timeout=10)
        except Exception:
            pass
    for key in (SLACK_USER_TOKEN, SLACK_BOT_TOKEN, SLACK_TEAM_ID, SLACK_TEAM_NAME, SLACK_OAUTH_STATE):
        _del_pref(uid, key)
    return _json({"message": "Slack disconnected"})

# =============================================================================
# (Optional) Connect-all helper – UI can ignore this if not needed
# =============================================================================
@connections_bp.route("/connect_all", methods=["POST"])
@login_required
def connect_all():
    """
    Return the Google auth URL (absolute) so the UI can redirect immediately.
    We compute it here instead of returning our own /google/authurl endpoint.
    """
    def _google_client_config():
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not configured")
        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
        }

    cfg = _google_client_config()
    state = _make_state(current_user.id)
    _set_pref(current_user.id, GOOGLE_OAUTH_STATE, state)
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile https://www.googleapis.com/auth/drive.readonly",
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
    }
    url = cfg["auth_uri"] + "?" + urlencode(params)
    return _json({"redirect": url})
