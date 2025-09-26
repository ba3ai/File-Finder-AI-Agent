# routes/admin_trial_routes.py
from datetime import datetime, timedelta, timezone
from flask import current_app, jsonify, request, Blueprint

# Reuse the existing admin blueprint/auth so the route shares the same guard as /api/admin/staff/me
from admin_api import admin_bp

# Try to import your SQLAlchemy models (User for direct column update; Prefs for key/value)
try:
    from models import db, Prefs, User
except Exception:
    db = Prefs = User = None  # fallback if import path differs

UTC = timezone.utc

# Keep your original blueprint so existing app.register_blueprint(...) lines do not break
admin_trial_bp = Blueprint("admin_trial", __name__, url_prefix="/api/admin/trial")


# -------------------------- helpers --------------------------

def _now() -> datetime:
    return datetime.now(UTC)

def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()

def _parse_date(s: str):
    """
    Accepts 'YYYY-MM-DD' or ISO8601 (with 'Z' or offset).
    Returns aware UTC datetime or None on failure.
    """
    s = (s or "").strip()
    if not s:
        return None
    # YYYY-MM-DD
    try:
        if len(s) == 10:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
    except Exception:
        pass
    # ISO with/without Z
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None

def _get_pref(uid: int, key: str):
    if not (Prefs and db):
        return None
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else None

def _set_pref(uid: int, key: str, value: str):
    if not (Prefs and db):
        raise RuntimeError("Prefs model is unavailable")
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    if row:
        row.value = value
    else:
        row = Prefs(user_id=uid, key=key, value=value)
        db.session.add(row)
    db.session.commit()

def _set_user_trial_end(uid: int, dt: datetime) -> bool:
    """
    Best-effort: if SQLAlchemy User model is present, update users.trial_ends_at.
    Returns True if updated via User, else False.
    """
    if not (User and db):
        return False
    u = User.query.get(uid)
    if not u:
        return False
    # If they’re not yet subscribed, keep plan coherent
    try:
        if not getattr(u, "subscription_active", False):
            if not getattr(u, "trial_started_at", None):
                u.trial_started_at = _now()
            if not getattr(u, "plan", None):
                u.plan = "free_trial"
    except Exception:
        pass
    u.trial_ends_at = dt
    db.session.commit()
    return True


# If your admin_api exposes a decorator like staff_required, use it on both blueprints.
try:
    from admin_api import staff_required as _admin_guard
except Exception:
    def _admin_guard(fn):  # no-op fallback; admin_bp likely has its own before_request guard
        return fn


# ---------------------- core implementation ----------------------

def _extend_trial_impl(payload: dict):
    user_id = payload.get("user_id")
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400
    try:
        user_id = int(user_id)
    except Exception:
        return jsonify({"error": "user_id must be integer"}), 400

    days  = payload.get("days")
    until = payload.get("until")
    if days is None and not until:
        return jsonify({"error": "Provide either days or until"}), 400

    MAX_EXTEND = int(current_app.config.get("TRIAL_MAX_EXTEND_DAYS", 60))
    now = _now()

    # Compute new_end
    if days is not None:
        try:
            days = int(days)
        except Exception:
            return jsonify({"error": "days must be integer"}), 400
        if days <= 0:
            return jsonify({"error": "days must be > 0"}), 400
        new_end = now + timedelta(days=days)
    else:
        dt = _parse_date(str(until))
        if not dt:
            return jsonify({"error": "until must be YYYY-MM-DD or ISO8601"}), 400
        if dt <= now:
            return jsonify({"error": "until must be in the future"}), 400
        new_end = dt

    if (new_end - now).days > MAX_EXTEND:
        return jsonify({"error": f"Cannot extend beyond {MAX_EXTEND} days from today"}), 400

    # Write to Prefs (many parts of your app read trial_end_at from Prefs)
    try:
        _set_pref(user_id, "trial_end_at", _to_iso(new_end))
    except Exception as e:
        current_app.logger.exception("Failed to save trial_end_at to Prefs")
        return jsonify({"error": "save_failed", "detail": str(e)}), 500

    # Also try to write to User.trial_ends_at (so admin lists can render it directly)
    user_col_updated = False
    try:
        user_col_updated = _set_user_trial_end(user_id, new_end)
    except Exception:
        current_app.logger.warning("Could not update users.trial_ends_at (table/model may not exist)")

    # Optional: audit (best-effort; do not block)
    try:
        if db:
            db.session.execute("""
                CREATE TABLE IF NOT EXISTS admin_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_id TEXT,
                    action TEXT,
                    target_id INTEGER,
                    meta TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            db.session.execute(
                "INSERT INTO admin_audits (actor_id, action, target_id, meta) VALUES (:a,:b,:c,:m)",
                {"a": "staff", "b": "trial_extend", "c": user_id, "m": f'{{"new_trial_ends_at":"{_to_iso(new_end)}"}}'}
            )
            db.session.commit()
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "user_id": user_id,
        "trial_ends_at": _to_iso(new_end),
        "persisted": {
            "prefs": True,
            "user_column": bool(user_col_updated)
        }
    })


# ----------------------- routes (both paths) -----------------------
# Mounted on admin_bp so it uses the SAME Super Admin auth as the rest of /api/admin/*
@admin_bp.post("/trial/extend")
@_admin_guard
def extend_trial_admin_bp():
    payload = request.get_json(silent=True) or {}
    return _extend_trial_impl(payload)

# Kept on your original blueprint as well (in case app.py registers it)
# We still guard it with the same admin decorator to avoid bypass.
@admin_trial_bp.post("/extend")
@_admin_guard
def extend_trial_admin_trial_bp():
    payload = request.get_json(silent=True) or {}
    return _extend_trial_impl(payload)
