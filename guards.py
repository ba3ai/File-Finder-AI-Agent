# guards.py
from functools import wraps
from flask import jsonify, current_app
from flask_login import current_user
from flask import jsonify
from flask_login import current_user, login_required



def require_team_workspace(view):
    @login_required
    def wrapped(*args, **kwargs):
        # if you want to allow during local dev:
        if current_app.config.get("TEAM_WS_DEV_BYPASS"):
            return view(*args, **kwargs)

        if not current_user or not current_user.has_team_plan:
            # IMPORTANT: use 402 so the UI knows it's a paywall
            return jsonify({"error": "team_required"}), 402
        return view(*args, **kwargs)
    return wrapped

def _is_team_plan(plan: str) -> bool:
    """Return True if user's plan is one of the configured Team plan codes."""
    plan = (plan or "").strip()
    codes = current_app.config.get("TEAM_PLAN_CODES", set())
    return plan in codes

def require_team_plan(fn):
    """
    Decorator to restrict an endpoint to Team-plan users.

    Testing shortcut:
      If app.config["TEAM_FEATURES_DISABLED"] is True, bypass the plan check
      entirely so any authenticated user can access team/workspace endpoints.
    """
    @wraps(fn)
    def _wrap(*args, **kwargs):
        # ✅ Bypass in testing mode
        if current_app.config.get("TEAM_FEATURES_DISABLED", False):
            return fn(*args, **kwargs)

        # Normal enforcement
        # (Assumes routes also use @login_required. If not, we still sanity-check.)
        if not getattr(current_user, "is_authenticated", False):
            return jsonify({"error": "unauthorized"}), 401

        has_active_sub = bool(getattr(current_user, "subscription_active", False))
        on_team_plan = _is_team_plan(getattr(current_user, "plan", ""))

        if not (has_active_sub and on_team_plan):
            return jsonify({"error": "team_plan_required"}), 402

        return fn(*args, **kwargs)
    return _wrap



def login_required_if_not_testing(fn):
    """
    During local testing (TEAM_FEATURES_DISABLED=true) let requests through
    without Flask-Login auth. Otherwise enforce authentication like usual.
    """
    @wraps(fn)
    def _wrap(*args, **kwargs):
        if current_app.config.get("TEAM_FEATURES_DISABLED", False):
            return fn(*args, **kwargs)
        if not getattr(current_user, "is_authenticated", False):
            return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return _wrap
