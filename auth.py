# auth.py
import os
import re
from datetime import datetime

from flask import Blueprint, request, jsonify, session, current_app
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from sqlalchemy.exc import IntegrityError

from models import db, User, PasswordReset
from security import hash_password, verify_password

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
login_manager = LoginManager()

# SPA: return JSON instead of redirecting to a login view
login_manager.login_view = None

# ---- trial config ----
TRIAL_DAYS = int(os.getenv("TRIAL_DAYS", "7"))


@login_manager.unauthorized_handler
def _unauthorized():
    resp = jsonify({"error": "Unauthorized"})
    _nocache(resp)
    return resp, 401


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


# ------------------------- helpers -------------------------

def _norm(e: str) -> str:
    return (e or "").strip().lower()


def _nocache(resp):
    """Mark responses as uncacheable so the browser never reuses another user's state."""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["Vary"] = "Cookie"


def _maybe_start_trial(user: User):
    """
    Start a free trial for this user if:
      - they don't have an active subscription, and
      - they haven't already consumed a trial.
    """
    if not user.subscription_active and not user.trial_consumed:
        user.start_trial(days=TRIAL_DAYS)


def _public_user(user: User):
    """
    Minimal public shape used by the frontend. Uses the helper on User when available.
    """
    try:
        return user.to_public_dict()
    except AttributeError:
        # Fallback if someone swaps models.py
        return {
            "id": user.id,
            "email": user.email,
            "role": getattr(user, "role", "user"),
        }


# ------------------------- JSON API -------------------------

@auth_bp.post("/register")
def api_register():
    """
    Sign up a new user.
    Option B: start the trial immediately upon successful signup.
    (If you prefer trial on first login only, delete the _maybe_start_trial() call here.)
    """
    data = request.get_json(silent=True) or {}
    email = _norm(data.get("email"))
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify(error="Email and password are required"), 400
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return jsonify(error="Invalid email"), 400
    if len(password) < 8:
        return jsonify(error="Password must be at least 8 characters"), 400

    try:
        user = User(email=email, password_hash=hash_password(password), role="user")
        # Start the trial at signup (Option B)
        _maybe_start_trial(user)

        db.session.add(user)
        db.session.commit()

        resp = jsonify(message="Registered", user=_public_user(user))
        _nocache(resp)
        return resp, 201
    except IntegrityError:
        db.session.rollback()
        return jsonify(error="Email already registered"), 409


@auth_bp.post("/login")
def api_login():
    """
    Log in an existing user.
    Option B: if the user hasn't consumed a trial yet and isn't paid, start it now.
    """
    data = request.get_json(silent=True) or {}
    email = _norm(data.get("email"))
    password = (data.get("password") or "").strip()
    remember = bool(data.get("remember", False))  # default: no persistent login

    user = User.query.filter_by(email=email).first()
    if not user or not user.is_active or not verify_password(password, user.password_hash):
        resp = jsonify(error="Invalid credentials")
        _nocache(resp)
        return resp, 401

    # Clear any previous identity/session, then log in fresh
    if current_user.is_authenticated:
        try:
            logout_user()
        except Exception:
            pass
    session.clear()
    # Force a new session cookie by touching server-side session
    session["sid_nonce"] = os.urandom(8).hex()
    session.permanent = False  # keep short-lived unless you want permanent sessions

    # Log the user in
    login_user(user, remember=remember)
    user.last_login_at = datetime.utcnow()

    # Start trial on first successful login if applicable (Option B)
    _maybe_start_trial(user)

    db.session.commit()

    resp = jsonify(message="Logged in", user=_public_user(user))
    _nocache(resp)
    return resp


@auth_bp.post("/logout")
@login_required
def api_logout():
    try:
        logout_user()
    except Exception:
        pass
    session.clear()

    # Proactively delete the remember cookie so the browser won't re-auth automatically
    resp = jsonify(message="Logged out")
    _nocache(resp)
    cookie_name = current_app.config.get("REMEMBER_COOKIE_NAME", "remember_token")
    resp.delete_cookie(
        cookie_name,
        path="/",
        secure=current_app.config.get("SESSION_COOKIE_SECURE", False),
        samesite=current_app.config.get("SESSION_COOKIE_SAMESITE", "Lax"),
        domain=current_app.config.get("SESSION_COOKIE_DOMAIN", None),
    )
    return resp, 200


@auth_bp.get("/me")
def api_me():
    """
    Returns the current authentication state plus plan/trial info
    so the frontend can show a banner and gate features.
    """
    if current_user.is_authenticated:
        user = User.query.get(current_user.id)
        resp = jsonify(
            authenticated=True,
            user=_public_user(user),
        )
        _nocache(resp)
        return resp
    resp = jsonify(authenticated=False)
    _nocache(resp)
    return resp


@auth_bp.post("/forgot")
def api_forgot():
    data = request.get_json(silent=True) or {}
    email = _norm(data.get("email"))
    if not email:
        return jsonify(error="Email required"), 400

    user = User.query.filter_by(email=email).first()
    if user:
        pr = PasswordReset(user_id=user.id, token=os.urandom(16).hex())
        db.session.add(pr)
        db.session.commit()
        current_app.logger.info(f"[Password reset] {email}: token={pr.token}")
        # TODO: send email with token link

    resp = jsonify(message="If the account exists, a reset link was sent")
    _nocache(resp)
    return resp
