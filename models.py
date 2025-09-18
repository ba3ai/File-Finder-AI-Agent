# models.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


# -------------------------
# User + Trial/Subscription
# -------------------------
class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    display_name = db.Column(db.String(255))
    role = db.Column(db.String(32), default="user")
    is_active = db.Column(db.Boolean, default=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime)

    # ---- Trial / Subscription fields ----
    # plan: "none" | "free_trial" | "pro" (add more plans later as needed)
    plan = db.Column(db.String(32), default="none", nullable=False)

    # Trial lifecycle: these are set when a trial is started
    trial_started_at = db.Column(db.DateTime, nullable=True)
    trial_ends_at = db.Column(db.DateTime, nullable=True)

    # Prevents issuing multiple trials to the same user
    trial_consumed = db.Column(db.Boolean, default=False, nullable=False)

    # True if the user has an active paid subscription (regardless of trial)
    subscription_active = db.Column(db.Boolean, default=False, nullable=False)

    # Relationships
    connected_accounts = db.relationship(
        "ConnectedAccount",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
    )

    

    def get_id(self) -> str:  # explicit is fine; UserMixin also provides one
        return str(self.id)

    # ---------- Trial helpers ----------
    def start_trial(self, days: int = 7, *, force: bool = False) -> None:
        """
        Initialize a free trial. By default it will *not* reissue a trial if one
        has already been consumed unless force=True (admin usage).
        """
        if self.subscription_active and not force:
            # Already paid; no need for a trial
            return

        if self.trial_consumed and not force:
            # Trial was already used at some point; don't reissue
            return

        now = datetime.utcnow()
        self.plan = "free_trial"
        self.trial_started_at = now
        self.trial_ends_at = now + timedelta(days=days)
        self.trial_consumed = True

    @property
    def trial_active(self) -> bool:
        if self.plan != "free_trial" or not self.trial_ends_at:
            return False
        return datetime.utcnow() < self.trial_ends_at

    @property
    def remaining_trial_seconds(self) -> int:
        """Seconds left on trial (0 if none/expired)."""
        if not self.trial_active:
            return 0
        return max(0, int((self.trial_ends_at - datetime.utcnow()).total_seconds()))

    @property
    def account_active(self) -> bool:
        """
        Single source of truth for entitlement:
        - Active subscription OR
        - Active free trial
        """
        return bool(self.subscription_active or self.trial_active)

    def to_public_dict(self) -> Dict[str, Any]:
        """
        Minimal info the frontend needs for banners/gating.
        """
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "plan": self.plan,
            "subscription_active": self.subscription_active,
            "trial_started_at": self.trial_started_at.isoformat() if self.trial_started_at else None,
            "trial_ends_at": self.trial_ends_at.isoformat() if self.trial_ends_at else None,
            "trial_active": self.trial_active,
            "remaining_trial_seconds": self.remaining_trial_seconds,
            "account_active": self.account_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }

    # Optional: convenience helpers for subscription transitions
    def activate_subscription(self, plan: str = "pro") -> None:
        self.subscription_active = True
        self.plan = plan

    def cancel_subscription(self) -> None:
        self.subscription_active = False
        # If no trial is active, mark plan accordingly
        if not self.trial_active:
            self.plan = "none"


# -------------------------
# Password reset tokens
# -------------------------
class PasswordReset(db.Model):
    __tablename__ = "password_resets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    token = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# -------------------------
# Connected OAuth accounts
# -------------------------
class ConnectedAccount(db.Model):
    """
    Stores delegated OAuth tokens (e.g., Google Drive) or metadata for
    app-only tokens where applicable (e.g., Microsoft Graph in some flows).
    """
    __tablename__ = "connected_accounts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), index=True, nullable=False)

    # "sharepoint", "onedrive", "google_drive", "dropbox", etc.
    provider = db.Column(db.String(50), nullable=False)

    # Delegated OAuth tokens
    access_token = db.Column(db.Text, nullable=True)
    refresh_token = db.Column(db.Text, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=True)

    # Optional: scope / metadata / provider email
    account_email = db.Column(db.String(255), nullable=True)
    scope = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (
        db.UniqueConstraint("user_id", "provider", name="uq_user_provider"),
    )

    # Helpful validity checks for access tokens
    def token_is_valid(self, skew_minutes: int = 2) -> bool:
        """
        Returns True if there is a non-expired access token, considering a small clock skew.
        """
        if not self.access_token:
            return False
        if not self.expires_at:
            # Some providers don't return expires_at; treat as present but unverifiable
            return True
        return self.expires_at > (datetime.utcnow() + timedelta(minutes=skew_minutes))

    @property
    def seconds_until_expiry(self) -> int:
        if not self.expires_at:
            return 0
        return max(0, int((self.expires_at - datetime.utcnow()).total_seconds()))


# -------------------------
# Simple per-user preferences / KV store
# -------------------------
class Prefs(db.Model):
    """
    Generic per-user key/value store. Useful for small settings or
    provider-specific extras you don't want to model explicitly.
    """
    __tablename__ = "prefs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    key = db.Column(db.String(120), nullable=False, index=True)
    value = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (
        db.UniqueConstraint("user_id", "key", name="uq_user_key"),
    )

    def __repr__(self) -> str:
        return f"<Prefs user_id={self.user_id} key={self.key!r}>"
    


# models.py (add imports if missing)
from datetime import datetime

class EmailLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)
    provider = db.Column(db.String(20))
    to = db.Column(db.String(255))
    cc = db.Column(db.Text, nullable=True)
    bcc = db.Column(db.Text, nullable=True)
    subject = db.Column(db.String(998))
    body_preview = db.Column(db.Text)
    status = db.Column(db.String(20))  # 'sent' | 'failed'
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

