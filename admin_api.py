from __future__ import annotations
from datetime import datetime, timedelta
import secrets

from flask import Blueprint, request, jsonify, session, g
from sqlalchemy import func, or_
from werkzeug.security import generate_password_hash

from models import db, User
from admin_models import (
    StaffUser, normalize_role,
    Subscription, Ticket, Revenue,
)

admin_bp = Blueprint("admin_api", __name__, url_prefix="/api/admin")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers & Guards
# ─────────────────────────────────────────────────────────────────────────────
def _row_user(u: User) -> dict:
    return {
        "id": u.id,
        "email": u.email,
        "display_name": getattr(u, "display_name", None) or getattr(u, "name", None),
        "name": getattr(u, "display_name", None) or getattr(u, "name", None) or (u.email.split("@")[0] if u.email else "User"),
        "role": (u.role or "user"),
        "is_active": bool(getattr(u, "is_active", True)),
        "status": "active" if getattr(u, "is_active", True) else "inactive",
        "plan": getattr(u, "plan", None) or "none",
        "subscription_active": bool(getattr(u, "subscription_active", False)),
        "created_at": u.created_at.isoformat() if getattr(u, "created_at", None) else None,
        "last_login_at": u.last_login_at.isoformat() if getattr(u, "last_login_at", None) else None,

        # ✅ send trial info so the table can render it
        "trial_started_at": u.trial_started_at.isoformat() if getattr(u, "trial_started_at", None) else None,
        "trial_ends_at": u.trial_ends_at.isoformat() if getattr(u, "trial_ends_at", None) else None,
        "trial_active": getattr(u, "trial_active", False),
    }


def _row_staff(su: StaffUser) -> dict:
    return {
        "id": su.id,
        "email": su.email,
        "display_name": su.display_name,
        "name": su.display_name or (su.email.split("@")[0] if su.email else "Staff"),
        "role": su.role,                   # superadmin | admin | client_support
        "is_active": bool(su.is_active),
        "status": "active" if su.is_active else "inactive",
        "plan": None,                      # staff don’t have plans
        "subscription_active": False,
        "created_at": su.created_at.isoformat() if su.created_at else None,
        "last_login_at": su.last_login_at.isoformat() if su.last_login_at else None,
    }

def _page_and_size():
    page_raw = request.args.get("page") or "1"
    size_raw = request.args.get("size") or "10"
    try:
        page = max(1, int(page_raw))
    except ValueError:
        page = 1
    if size_raw.strip().lower() == "all":
        return page, "all", 0
    try:
        size = max(1, min(2000, int(size_raw)))
    except ValueError:
        size = 10
    return page, size, (page - 1) * size

def staff_session_required(view):
    """Require a staff session (stored in Flask session)."""
    from functools import wraps
    @wraps(view)
    def wrapper(*args, **kwargs):
        sid = session.get("staff_user_id")
        if not sid:
            return jsonify({"error": "auth_required"}), 401
        su = StaffUser.query.get(sid)
        if not su or not su.is_active:
            session.pop("staff_user_id", None)
            return jsonify({"error": "forbidden", "message": "Staff only"}), 403
        g.staff_user = su
        return view(*args, **kwargs)
    return wrapper

# ─────────────────────────────────────────────────────────────────────────────
# Staff identity & login
# ─────────────────────────────────────────────────────────────────────────────
@admin_bp.post("/staff/login")
def staff_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "missing_fields"}), 400

    su = StaffUser.query.filter(func.lower(StaffUser.email) == email).first()
    if not su or not su.check_password(password):
        return jsonify({"error": "invalid_credentials"}), 401
    if not su.is_active:
        return jsonify({"error": "inactive_account"}), 403

    su.last_login_at = datetime.utcnow()
    db.session.commit()
    session["staff_user_id"] = su.id

    return jsonify({"ok": True, "redirect": "/superadmin/userlist", "role": su.role})

@admin_bp.get("/staff/me")
def staff_me():
    sid = session.get("staff_user_id")
    if not sid:
        return jsonify({"is_staff": False})
    su = StaffUser.query.get(sid)
    if not su or not su.is_active:
        session.pop("staff_user_id", None)
        return jsonify({"is_staff": False})
    return jsonify({"is_staff": True, "role": su.role, "user": su.as_dict()})

@admin_bp.post("/staff/logout")
def staff_logout():
    session.pop("staff_user_id", None)
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────────────────────
# Staff users CRUD / LIST (operate on staff_users table)
# ─────────────────────────────────────────────────────────────────────────────
@admin_bp.post("/staff/users")
@staff_session_required
def create_staff_user():
    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip().lower()
    display_name = (data.get("display_name") or data.get("name") or "").strip()
    role_in = normalize_role(data.get("role"))
    status_in = (data.get("status") or "active").strip().lower()
    pwd = (data.get("password") or "").strip()

    if not email or not display_name or not role_in:
        return jsonify({"error": "email/name/role_required"}), 400
    if StaffUser.query.filter(func.lower(StaffUser.email) == email).first():
        return jsonify({"error": "email_already_exists"}), 409

    if not pwd:
        pwd = "Temp@" + secrets.token_urlsafe(8)
        temp_used = True
    else:
        temp_used = False

    su = StaffUser(
        email=email,
        display_name=display_name,
        role=role_in,
        is_active=(status_in != "inactive"),
        created_at=datetime.utcnow(),
    )
    su.set_password(pwd)
    db.session.add(su)
    db.session.commit()

    payload = {"ok": True, "item": su.as_dict()}
    if temp_used:
        payload["temp_password"] = pwd
    return jsonify(payload), 201

@admin_bp.get("/staff/users")
@staff_session_required
def list_staff_users():
    q = (request.args.get("q") or "").strip()
    status = (request.args.get("status") or "").strip().lower()       # "", active, inactive
    role = normalize_role(request.args.get("role"))                    # superadmin | admin | client_support | None
    page, size, offset = _page_and_size()

    base = StaffUser.query
    if q:
        like = f"%{q}%"
        base = base.filter(or_(StaffUser.email.ilike(like),
                               StaffUser.display_name.ilike(like)))
    if status == "active":
        base = base.filter(StaffUser.is_active.is_(True))
    elif status == "inactive":
        base = base.filter(StaffUser.is_active.is_(False))
    if role:
        base = base.filter(StaffUser.role == role)

    total = base.count()
    base = base.order_by(StaffUser.created_at.desc().nullslast())
    items = base.all() if size == "all" else base.offset(offset).limit(size).all()
    page_out, size_out = (1, "all") if size == "all" else (page, size)

    cutoff = datetime.utcnow() - timedelta(days=7)
    new_last_7 = StaffUser.query.filter(StaffUser.created_at >= cutoff).count()
    active_cnt = StaffUser.query.filter(StaffUser.is_active.is_(True)).count()
    inactive_cnt = StaffUser.query.filter(StaffUser.is_active.is_(False)).count()

    return jsonify({
        "items": [_row_staff(s) for s in items],
        "total": total,
        "page": page_out,
        "size": size_out,
        "stats": {
            "total": total,
            "active": active_cnt,
            "inactive": inactive_cnt,
            "newLast7Days": new_last_7,
            "scope": "filtered",
        },
    })

# ─────────────────────────────────────────────────────────────────────────────
# End-user management (regular `users` table)
# ─────────────────────────────────────────────────────────────────────────────
@admin_bp.get("/users")
@staff_session_required
def list_users():
    q = (request.args.get("q") or "").strip()
    status = (request.args.get("status") or "").strip().lower()    # "", active, inactive
    # role param kept for UI compatibility (most end users are "user")
    page, size, offset = _page_and_size()

    base = User.query
    if q:
        like = f"%{q}%"
        base = base.filter(or_(User.email.ilike(like),
                               getattr(User, "display_name", User.email).ilike(like)))

    total = base.count()
    if status == "active":
        base = base.filter(User.is_active.is_(True))
    elif status == "inactive":
        base = base.filter(User.is_active.is_(False))

    base = base.order_by(User.created_at.desc().nullslast())
    items = base.all() if size == "all" else base.offset(offset).limit(size).all()
    page_out, size_out = (1, "all") if size == "all" else (page, size)

    cutoff = datetime.utcnow() - timedelta(days=7)
    new_last_7 = User.query.filter(User.created_at >= cutoff).count()

    return jsonify({
        "items": [_row_user(u) for u in items],
        "total": total,
        "page": page_out,
        "size": size_out,
        "stats": {
            "total": total,
            "active": User.query.filter(User.is_active.is_(True)).count(),
            "inactive": User.query.filter(User.is_active.is_(False)).count(),
            "newLast7Days": new_last_7,
            "scope": "filtered",
        },
    })

@admin_bp.post("/users")
@staff_session_required
def create_end_user():
    data = request.get_json(force=True) or {}
    display_name = (data.get("display_name") or data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    status_in = (data.get("status") or "active").strip().lower()
    plan = (data.get("plan") or "free_trial").strip()
    plain_password = (data.get("password") or "").strip()

    if not display_name or not email:
        return jsonify({"error": "name/display_name and email are required"}), 400
    if User.query.filter(func.lower(User.email) == email.lower()).first():
        return jsonify({"error": "email_already_exists"}), 409

    is_active = (status_in != "inactive")

    u = User(
        email=email,
        display_name=display_name,
        is_active=is_active,
        role="user",
        created_at=datetime.utcnow(),
        plan=plan,
        subscription_active=False,
    )

    # ensure password is set (schema requires it)
    temp_used = False
    if not plain_password:
        plain_password = "Temp@" + secrets.token_urlsafe(8)
        temp_used = True

    if hasattr(u, "set_password") and callable(getattr(u, "set_password")):
        u.set_password(plain_password)
    else:
        u.password_hash = generate_password_hash(plain_password)

    db.session.add(u)
    db.session.commit()

    payload = {"ok": True, "item": _row_user(u)}
    if temp_used:
        payload["temp_password"] = plain_password
    return jsonify(payload), 201

# ─────────────────────────────────────────────────────────────────────────────
# Subscriptions / Tickets / Revenue
# ─────────────────────────────────────────────────────────────────────────────
@admin_bp.get("/subscriptions")
@staff_session_required
def list_subscriptions():
    page, size, offset = _page_and_size()
    q = db.session.query(Subscription, User).join(User, Subscription.user_id == User.id)
    total = q.count()
    rows = q.order_by(Subscription.start_date.desc()).all() if size == "all" else q.order_by(
        Subscription.start_date.desc()).offset(offset).limit(size).all()
    items = []
    for s, u in rows:
        items.append(dict(
            id=s.id, user_id=u.id, user_name=u.display_name or (u.email.split("@")[0] if u.email else "User"),
            user_email=u.email, plan=s.plan, price=s.price, status=s.status,
            start_date=s.start_date.isoformat() if s.start_date else None,
            end_date=s.end_date.isoformat() if s.end_date else None,
        ))
    return jsonify({"items": items, "page": 1 if size == "all" else page, "size": size, "total": total})

@admin_bp.get("/tickets")
@staff_session_required
def list_tickets():
    status = (request.args.get("status") or "").strip().lower()
    assignee = request.args.get("assignee")
    page, size, offset = _page_and_size()

    q = db.session.query(Ticket, User).outerjoin(User, Ticket.assigned_to == User.id)
    if status in ("open", "in_progress", "closed"):
        q = q.filter(Ticket.status == status)
    if assignee:
        try:
            q = q.filter(Ticket.assigned_to == int(assignee))
        except ValueError:
            pass

    total = q.count()
    rows = q.order_by(Ticket.created_at.desc()).all() if size == "all" else q.order_by(
        Ticket.created_at.desc()).offset(offset).limit(size).all()
    items = []
    for t, u in rows:
        items.append(dict(
            id=t.id, title=t.title, priority=t.priority, status=t.status,
            assigned_to=t.assigned_to,
            user_name=(u.display_name if u and u.display_name else (u.email.split("@")[0] if u and u.email else None)),
            created_at=t.created_at.isoformat() if t.created_at else None,
        ))
    return jsonify({"items": items, "page": 1 if size == "all" else page, "size": size, "total": total})

@admin_bp.get("/revenue")
@staff_session_required
def revenue_summary():
    total_paid = db.session.query(
        db.func.coalesce(db.func.sum(db.case((Revenue.status == "paid", Revenue.amount), else_=0.0)), 0.0)
    ).scalar()
    total_due = db.session.query(
        db.func.coalesce(db.func.sum(db.case((Revenue.status == "due", Revenue.amount), else_=0.0)), 0.0)
    ).scalar()
    invoices = db.session.query(db.func.count(Revenue.id)).scalar()
    latest = Revenue.query.order_by(Revenue.invoice_date.desc()).limit(12).all()
    return jsonify({
        "summary": dict(
            total_paid=float(total_paid or 0.0),
            total_due=float(total_due or 0.0),
            invoices=int(invoices or 0),
        ),
        "latest": [dict(
            id=r.id, invoice_no=r.invoice_no, amount=r.amount, status=r.status,
            invoice_date=r.invoice_date.isoformat() if r.invoice_date else None
        ) for r in latest]
    })
