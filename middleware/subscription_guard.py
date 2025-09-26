# middleware/subscription_guard.py
import sqlite3
from functools import wraps
from datetime import datetime, timezone
from flask import jsonify
from flask_login import current_user
from services.billing_service import get_db

def parse_iso(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except:
        return None

def require_active_subscription(allow_trial=True):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return jsonify({"error": "Unauthorized"}), 401

            # Trial check
            if allow_trial and getattr(current_user, "trial_ends_at", None):
                trial_end = parse_iso(current_user.trial_ends_at)
                if trial_end and datetime.now(timezone.utc) <= trial_end:
                    return fn(*args, **kwargs)

            # Subscription check
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT status FROM subscriptions WHERE user_id=?", (current_user.id,))
            row = cur.fetchone()
            conn.close()
            if not row or row["status"] not in ("trialing","active","past_due"):
                return jsonify({"error": "Subscription required"}), 402

            return fn(*args, **kwargs)
        return wrapper
    return decorator
