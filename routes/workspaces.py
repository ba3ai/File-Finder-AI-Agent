# routes/workspaces.py
import os
import time
import base64
import smtplib
from email.message import EmailMessage
from urllib.parse import urlencode

import requests
from datetime import datetime, timedelta
from sqlalchemy import func

from flask import Blueprint, jsonify, request, current_app, redirect, session
from flask_login import current_user

from models import db, User, Workspace, WorkspaceMember, WorkspaceInvite, Prefs, WorkspaceJoinRequest
from guards import login_required_if_not_testing

from connections_api import (
    MS_SCOPES, MS_ACCESS_TOKEN, MS_REFRESH_TOKEN, MS_EXPIRES_AT, MS_ACCOUNT_EMAIL,
    GOOGLE_ACCESS_TOKEN, GOOGLE_REFRESH_TOKEN, GOOGLE_EXPIRES_AT, GOOGLE_ACCOUNT_EMAIL,
)

workspaces_bp = Blueprint("workspaces", __name__, url_prefix="/api/workspaces")

# ----------------------------- helpers -----------------------------

def _get_or_create_workspace_for_owner(user: User) -> Workspace:
    ws = Workspace.query.filter_by(owner_user_id=user.id).first()
    if ws:
        if current_app.config.get("TEAM_FEATURES_DISABLED", False):
            lim = int(current_app.config.get("TEAM_DEV_SEATS_LIMIT", ws.seats_limit or 50))
            if not ws.seats_limit or ws.seats_limit < lim:
                ws.seats_limit = lim
                db.session.commit()
        return ws

    ws = Workspace(
        name=(getattr(user, "display_name", None) or user.email.split("@")[0] or "Workspace"),
        owner_user_id=user.id,
        seats_limit=int(
            current_app.config.get("TEAM_DEV_SEATS_LIMIT", 3)
            if current_app.config.get("TEAM_FEATURES_DISABLED", False)
            else current_app.config.get("TEAM_DEFAULT_SEATS", 3)
        ),
    )
    db.session.add(ws); db.session.commit()
    db.session.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="owner", status="active"))
    db.session.commit()
    return ws


def _pref(uid: int, key: str, default=None):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else default

def _set_pref(uid: int, key: str, value: str):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    if row:
        row.value = value; row.updated_at = datetime.utcnow()
    else:
        row = Prefs(user_id=uid, key=key, value=value); db.session.add(row)
    db.session.commit()

def build_email(from_addr: str, to_addr: str, subject: str, html_body: str, text_body: str | None = None) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = from_addr; msg["To"] = to_addr; msg["Subject"] = subject
    msg.set_content(text_body or "Open this email in an HTML-capable client.")
    msg.add_alternative(html_body, subtype="html")
    return msg

def rfc822_b64url(msg: EmailMessage) -> str:
    raw = msg.as_bytes()
    return base64.urlsafe_b64encode(raw).decode("ascii")

# ------------------------- MS token refresh -------------------------

def _ms_oauth_config():
    return {
        "client_id": current_app.config.get("MS_CLIENT_ID") or os.getenv("MS_CLIENT_ID"),
        "client_secret": current_app.config.get("MS_CLIENT_SECRET") or os.getenv("MS_CLIENT_SECRET"),
        "tenant": current_app.config.get("MS_TENANT_ID") or os.getenv("MS_TENANT_ID", "common"),
        "token_url": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
    }

def get_ms_access_token_for_user(user: User) -> str | None:
    uid = user.id
    at = _pref(uid, MS_ACCESS_TOKEN)
    exp_iso = _pref(uid, MS_EXPIRES_AT)
    rt = _pref(uid, MS_REFRESH_TOKEN)

    try:
        if at and exp_iso and datetime.fromisoformat(exp_iso) > datetime.utcnow() + timedelta(seconds=30):
            return at
    except Exception:
        pass

    if not rt:
        return None

    cfg = _ms_oauth_config()
    data = {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "grant_type": "refresh_token",
        "refresh_token": rt,
        "scope": " ".join(MS_SCOPES),
    }
    token_url = cfg["token_url"].format(tenant=cfg["tenant"])
    resp = requests.post(token_url, data=data, timeout=20)
    if not resp.ok:
        current_app.logger.warning("MS refresh failed %s: %s", resp.status_code, resp.text)
        return None

    j = resp.json()
    new_at = j.get("access_token")
    new_rt = j.get("refresh_token") or rt
    expires_in = int(j.get("expires_in", 3600))
    if new_at:
        _set_pref(uid, MS_ACCESS_TOKEN, new_at)
        _set_pref(uid, MS_REFRESH_TOKEN, new_rt)
        _set_pref(uid, MS_EXPIRES_AT, (datetime.utcnow() + timedelta(seconds=max(expires_in-60, 0))).isoformat())
        return new_at
    return None

# ------------------------ Gmail token refresh -----------------------

def _google_oauth_config():
    return {
        "client_id": current_app.config.get("GOOGLE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": current_app.config.get("GOOGLE_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }

def get_gmail_access_token_for_user(user: User) -> str | None:
    uid = user.id
    at = _pref(uid, GOOGLE_ACCESS_TOKEN)
    exp_epoch = _pref(uid, GOOGLE_EXPIRES_AT)
    rt = _pref(uid, GOOGLE_REFRESH_TOKEN)

    try:
        if at and exp_epoch and int(exp_epoch) > int(time.time()) + 30:
            return at
    except Exception:
        pass

    if not rt:
        return None

    cfg = _google_oauth_config()
    data = {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "refresh_token": rt,
        "grant_type": "refresh_token",
    }
    resp = requests.post(cfg["token_uri"], data=data, timeout=20)
    if not resp.ok:
        current_app.logger.warning("Google refresh failed %s: %s", resp.status_code, resp.text)
        return None

    j = resp.json()
    new_at = j.get("access_token")
    expires_in = int(j.get("expires_in", 3600))
    if new_at:
        _set_pref(uid, GOOGLE_ACCESS_TOKEN, new_at)
        _set_pref(uid, GOOGLE_EXPIRES_AT, str(int(time.time()) + max(expires_in-60, 0)))
        return new_at
    return None

# ---------------------------- senders --------------------------------

def send_via_outlook(user: User, to_addr: str, subject: str, html: str, text: str, fallback_from: str) -> bool:
    token = get_ms_access_token_for_user(user)
    if not token: return False
    payload = {
        "message": {
            "subject": subject,
            "body": {"contentType": "HTML", "content": html},
            "toRecipients": [{"emailAddress": {"address": to_addr}}],
        },
        "saveToSentItems": True,
    }
    resp = requests.post(
        "https://graph.microsoft.com/v1.0/me/sendMail",
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=20,
    )
    return resp.status_code in (202, 200, 204)

def send_via_gmail(user: User, to_addr: str, subject: str, html: str, text: str, fallback_from: str) -> bool:
    token = get_gmail_access_token_for_user(user)
    if not token: return False
    msg = build_email(fallback_from or "me", to_addr, subject, html, text)
    raw = rfc822_b64url(msg)
    resp = requests.post(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        json={"raw": raw},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=20,
    )
    return resp.status_code == 200

def send_via_smtp(to_addr: str, subject: str, html_body: str, text_body: str = None) -> bool:
    server   = current_app.config.get("MAIL_SERVER")
    port     = int(current_app.config.get("MAIL_PORT", 587))
    user     = current_app.config.get("MAIL_USERNAME")
    pwd      = current_app.config.get("MAIL_PASSWORD")
    use_tls  = bool(str(current_app.config.get("MAIL_USE_TLS", True)).lower() in ("1","true","yes"))
    sender   = current_app.config.get("MAIL_SENDER", user)
    if not server or not sender:
        current_app.logger.warning("SMTP not configured; cannot send invite email.")
        return False
    msg = build_email(sender, to_addr, subject, html_body, text_body)
    with smtplib.SMTP(server, port) as s:
        if use_tls: s.starttls()
        if user and pwd: s.login(user, pwd)
        s.send_message(msg)
    return True

def send_invite_from_user_or_fallback(user: User, to_addr: str, subject: str, html: str, text: str) -> str:
    try:
        if send_via_outlook(user, to_addr, subject, html, text, getattr(user, "email", None)): return "outlook"
    except Exception: current_app.logger.exception("Outlook send error")
    try:
        if send_via_gmail(user, to_addr, subject, html, text, getattr(user, "email", None)): return "gmail"
    except Exception: current_app.logger.exception("Gmail send error")
    try:
        if send_via_smtp(to_addr, subject, html, text): return "smtp"
    except Exception: current_app.logger.exception("SMTP send error")
    return "none"

# ----------------------------- routes --------------------------------

@workspaces_bp.get("/me")
@login_required_if_not_testing
def my_workspace():
    ws = Workspace.query.join(
        WorkspaceMember, Workspace.id == WorkspaceMember.workspace_id
    ).filter(
        WorkspaceMember.user_id == current_user.id,
        WorkspaceMember.status == "active"
    ).first()
    if not ws:
        ws = _get_or_create_workspace_for_owner(current_user)
    members = [m.as_dict() for m in WorkspaceMember.query.filter_by(workspace_id=ws.id, status="active").all()]
    return jsonify({"workspace": ws.as_dict(), "members": members})

@workspaces_bp.post("/members/add")
@login_required_if_not_testing
def add_member():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email: return jsonify({"error": "email_required"}), 400
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first() or _get_or_create_workspace_for_owner(current_user)
    active_count = WorkspaceMember.query.filter_by(workspace_id=ws.id, status="active").count()
    if active_count >= (ws.seats_limit or 0):
        return jsonify({"error": "seats_exhausted", "seats_limit": ws.seats_limit}), 409
    u = User.query.filter(User.email.ilike(email)).first()
    if not u: return jsonify({"error": "user_not_found"}), 404
    link = WorkspaceMember.query.filter_by(workspace_id=ws.id, user_id=u.id).first()
    if link and link.status == "active":
        return jsonify({"ok": True, "member": link.as_dict(), "already_member": True})
    if link:
        link.status = "active"; link.role = link.role or "member"
    else:
        db.session.add(WorkspaceMember(workspace_id=ws.id, user_id=u.id, role="member", status="active"))
    db.session.commit()
    link = WorkspaceMember.query.filter_by(workspace_id=ws.id, user_id=u.id).first()
    return jsonify({"ok": True, "member": link.as_dict(), "workspace": ws.as_dict()})

@workspaces_bp.get("/invites")
@login_required_if_not_testing
def list_invites():
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws: return jsonify({"invites": []})
    invites = [i.as_dict() for i in WorkspaceInvite.query.filter_by(workspace_id=ws.id).order_by(WorkspaceInvite.created_at.desc()).all()]
    return jsonify({"invites": invites})

@workspaces_bp.post("/invites/send")
@login_required_if_not_testing
def send_invites():
    data = request.get_json(silent=True) or {}
    items = data.get("invites") or []
    if not isinstance(items, list) or not items: return jsonify({"error": "no_invites"}), 400

    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws: return jsonify({"error": "workspace_not_found"}), 404

    cleaned, seen = [], set()
    for it in items:
        email = (it.get("email") or "").strip().lower()
        role  = (it.get("role") or "member").lower()
        role = role if role in {"member", "admin", "owner"} else "member"
        if email and email not in seen:
            seen.add(email); cleaned.append({"email": email, "role": role})
    if not cleaned: return jsonify({"error": "invalid_emails"}), 400

    active_members = WorkspaceMember.query.filter_by(workspace_id=ws.id, status="active").count()
    capacity_left  = max(0, (ws.seats_limit or 0) - active_members)
    if ws.seats_limit and len(cleaned) > capacity_left:
        return jsonify({"error": "seats_exhausted", "capacity_left": capacity_left}), 409

    created = []
    for it in cleaned:
        inv = WorkspaceInvite.query.filter_by(workspace_id=ws.id, email=it["email"], status="pending").first()
        if not inv:
            inv = WorkspaceInvite(
                workspace_id=ws.id, email=it["email"], role=it["role"],
                token=WorkspaceInvite.new_token(), status="pending",
                expires_at=datetime.utcnow() + timedelta(days=14),
            )
            db.session.add(inv)
        created.append(inv)
    db.session.commit()

    base = current_app.config.get("FRONTEND_BASE_URL", "http://localhost:5000")
    results = []
    for inv in created:
        accept_url = f"{base}/workspace/accept?{urlencode({'token': inv.token})}"
        subj = f"You're invited to join {ws.name}"
        html = f"""
        <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;font-size:14px;color:#111">
          <p>Hello,</p>
          <p>You’ve been invited to join the <strong>{ws.name}</strong> workspace as <strong>{inv.role}</strong>.</p>
          <p><a href="{accept_url}" style="background:#10b981;color:#000;padding:10px 14px;border-radius:8px;text-decoration:none;display:inline-block">Accept invite</a></p>
          <p>If the button doesn't work, copy this link:</p>
          <p><a href="{accept_url}">{accept_url}</a></p>
          <p style="color:#666">This invite may expire after 14 days.</p>
        </div>
        """
        text = f"You're invited to join {ws.name} as {inv.role}. Open: {accept_url}"
        method = send_invite_from_user_or_fallback(current_user, inv.email, subj, html, text)
        results.append({"email": inv.email, "method": method, "ok": method != "none"})
    return jsonify({"ok": True, "invites": [i.as_dict() for i in created], "delivery": results})

@workspaces_bp.post("/invites/accept")
def accept_invite_post():
    """
    Accept by token.
    - If not authenticated: stash token in session and return whether that email
      already has an account so the client can choose Login vs Signup.
    - If authenticated: accept the invite and add/activate the member.
    """
    data = request.get_json(silent=True) or {}
    token = (data.get("token") or session.get("pending_ws_invite") or "").strip()
    if not token: return jsonify({"error": "token_required"}), 400

    inv = WorkspaceInvite.query.filter_by(token=token, status="pending").first()
    if not inv: return jsonify({"error": "invalid_or_used"}), 400
    if inv.expires_at and datetime.utcnow() > inv.expires_at:
        inv.status = "expired"; db.session.commit()
        return jsonify({"error": "expired"}), 400

    # If not logged in yet, record token + tell the client where to go
    if not getattr(current_user, "is_authenticated", False):
        session["pending_ws_invite"] = token
        has_account = User.query.filter(User.email.ilike(inv.email)).count() > 0
        return jsonify({
            "ok": True,
            "requires_login": True,
            "has_account": has_account,
            "invite_email": inv.email,
        })

    # Must be the same email that was invited
    if (current_user.email or "").lower() != inv.email.lower():
        return jsonify({"error": "email_mismatch"}), 403

    ws = Workspace.query.get(inv.workspace_id)
    if not ws: return jsonify({"error": "workspace_not_found"}), 404

    # seats check
    active = WorkspaceMember.query.filter_by(workspace_id=ws.id, status="active").count()
    if active >= (ws.seats_limit or 0): return jsonify({"error": "seats_exhausted"}), 409

    link = WorkspaceMember.query.filter_by(workspace_id=ws.id, user_id=current_user.id).first()
    if link:
        link.status = "active"; link.role = link.role or inv.role
    else:
        db.session.add(WorkspaceMember(workspace_id=ws.id, user_id=current_user.id, role=inv.role, status="active"))

    inv.status = "accepted"
    inv.accepted_at = datetime.utcnow()
    inv.accepted_by_user_id = current_user.id
    db.session.commit()
    session.pop("pending_ws_invite", None)
    return jsonify({"ok": True, "workspace": ws.as_dict()})

@workspaces_bp.get("/invites/accept")
@login_required_if_not_testing
def accept_invite_get():
    token = (request.args.get("token") or "").strip()
    if not token: return jsonify({"error": "token_required"}), 400

    inv = WorkspaceInvite.query.filter_by(token=token).first()
    if not inv or inv.status != "pending": return jsonify({"error": "invalid_or_used"}), 400
    if inv.expires_at and datetime.utcnow() > inv.expires_at:
        inv.status = "expired"; db.session.commit()
        return jsonify({"error": "expired"}), 400
    if (current_user.email or "").lower() != inv.email.lower():
        return jsonify({"error": "email_mismatch"}), 403

    ws = Workspace.query.get(inv.workspace_id)
    if not ws: return jsonify({"error": "workspace_not_found"}), 404

    active = WorkspaceMember.query.filter_by(workspace_id=ws.id, status="active").count()
    if active >= (ws.seats_limit or 0): return jsonify({"error": "seats_exhausted"}), 409

    link = WorkspaceMember.query.filter_by(workspace_id=ws.id, user_id=current_user.id).first()
    if link:
        link.status = "active"; link.role = link.role or inv.role
    else:
        db.session.add(WorkspaceMember(workspace_id=ws.id, user_id=current_user.id, role=inv.role, status="active"))

    inv.status = "accepted"
    inv.accepted_at = datetime.utcnow()
    inv.accepted_by_user_id = current_user.id
    db.session.commit()
    return redirect("/workspace")

# Optional helper for UX (not required)
@workspaces_bp.get("/invites/info")
def invite_info():
    token = (request.args.get("token") or "").strip()
    if not token: return jsonify({"error": "token_required"}), 400
    inv = WorkspaceInvite.query.filter_by(token=token).first()
    if not inv:
        return jsonify({"valid": False})
    has_account = User.query.filter(User.email.ilike(inv.email)).count() > 0
    return jsonify({
        "valid": inv.status == "pending",
        "email": inv.email,
        "has_account": has_account,
        "status": inv.status
    })

@workspaces_bp.get("/invites/pending-token")
def pending_token():
    return jsonify({"token": session.get("pending_ws_invite")})

@workspaces_bp.get("/invites/pending")
@login_required_if_not_testing
def invites_pending():
    # Owner’s workspace is the “admin view” we expose here
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws:
        return jsonify({"invites": []})
    rows = (WorkspaceInvite.query
            .filter_by(workspace_id=ws.id, status="pending")
            .order_by(WorkspaceInvite.created_at.desc())
            .all())
    # UI expects: { id, email, invited_at, role }
    out = []
    for i in rows:
        out.append({
            "id": i.id,
            "email": (i.email or "").lower(),
            "role": i.role or "member",
            "invited_at": i.created_at.isoformat() if getattr(i, "created_at", None) else None
        })
    return jsonify({"invites": out})


@workspaces_bp.delete("/invites/<int:invite_id>")
@login_required_if_not_testing
def invites_cancel(invite_id):
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws:
        return jsonify({"error": "workspace_not_found"}), 404
    inv = WorkspaceInvite.query.filter_by(id=invite_id, workspace_id=ws.id).first()
    if not inv:
        return jsonify({"error": "not_found"}), 404
    if inv.status == "accepted":
        return jsonify({"error": "already_accepted"}), 409
    inv.status = "cancelled"
    db.session.commit()
    return jsonify({"ok": True})

@workspaces_bp.post("/invites/<int:invite_id>/resend")
@login_required_if_not_testing
def invites_resend(invite_id):
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws:
        return jsonify({"error": "workspace_not_found"}), 404
    inv = WorkspaceInvite.query.filter_by(id=invite_id, workspace_id=ws.id).first()
    if not inv or inv.status != "pending":
        return jsonify({"error": "not_found_or_not_pending"}), 404

    base = current_app.config.get("FRONTEND_BASE_URL", "http://localhost:5000")
    accept_url = f"{base}/workspace/accept?token={inv.token}"

    subj = f"You're invited to join {ws.name}"
    html = f"""
      <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;font-size:14px;color:#111">
        <p>Hello,</p>
        <p>You’ve been invited to join the <strong>{ws.name}</strong> workspace as <strong>{inv.role}</strong>.</p>
        <p><a href="{accept_url}" style="background:#10b981;color:#000;padding:10px 14px;border-radius:8px;text-decoration:none;display:inline-block">
          Accept invite</a></p>
        <p>If the button doesn't work, copy this link:</p>
        <p><a href="{accept_url}">{accept_url}</a></p>
      </div>
    """
    text = f"You're invited to join {ws.name} as {inv.role}. Open: {accept_url}"
    method = send_invite_from_user_or_fallback(current_user, inv.email, subj, html, text)
    return jsonify({"ok": method != "none", "method": method})


@workspaces_bp.get("/requests/pending")
@login_required_if_not_testing
def requests_pending():
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws:
        return jsonify({"requests": []})
    rows = (WorkspaceJoinRequest.query
            .filter_by(workspace_id=ws.id)
            .order_by(WorkspaceJoinRequest.created_at.desc())
            .all())
    out = [r.as_dict() for r in rows]
    return jsonify({"requests": out})


@workspaces_bp.post("/requests/<int:req_id>/accept")
@login_required_if_not_testing
def requests_accept(req_id):
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws:
        return jsonify({"error": "workspace_not_found"}), 404
    r = WorkspaceJoinRequest.query.filter_by(id=req_id, workspace_id=ws.id).first()
    if not r:
        return jsonify({"error": "not_found"}), 404

    # seat check
    active_count = WorkspaceMember.query.filter_by(workspace_id=ws.id, status="active").count()
    if active_count >= (ws.seats_limit or 0):
        return jsonify({"error": "seats_exhausted"}), 409

    # add or activate member (if user exists)
    user = User.query.filter(User.email.ilike(r.email)).first()
    if user:
        link = WorkspaceMember.query.filter_by(workspace_id=ws.id, user_id=user.id).first()
        if link:
            link.status = "active"; link.role = link.role or (r.role or "member")
        else:
            db.session.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id,
                                           role=r.role or "member", status="active"))
    # delete the request either way (we treat accept as “handled”)
    db.session.delete(r)
    db.session.commit()
    return jsonify({"ok": True})

@workspaces_bp.delete("/requests/<int:req_id>")
@login_required_if_not_testing
def requests_reject(req_id):
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws:
        return jsonify({"error": "workspace_not_found"}), 404
    r = WorkspaceJoinRequest.query.filter_by(id=req_id, workspace_id=ws.id).first()
    if not r:
        return jsonify({"error": "not_found"}), 404
    db.session.delete(r)
    db.session.commit()
    return jsonify({"ok": True})

@workspaces_bp.delete("/members/<int:member_id>")
@login_required_if_not_testing
def remove_member(member_id):
    # Only allow owner (or admin, if you support it) to remove members
    ws = Workspace.query.filter_by(owner_user_id=current_user.id).first()
    if not ws:
        return jsonify({"error": "workspace_not_found"}), 404

    m = WorkspaceMember.query.filter_by(id=member_id, workspace_id=ws.id).first()
    if not m:
        return jsonify({"error": "member_not_found"}), 404

    # Don’t allow removing the owner
    if m.user_id == ws.owner_user_id:
        return jsonify({"error": "cannot_remove_owner"}), 403

    # Either hard delete or soft delete. Pick ONE approach:

    # Hard delete:
    db.session.delete(m)

    # Soft delete (if you track statuses):
    # m.status = "removed"

    db.session.commit()
    return jsonify({"ok": True})

