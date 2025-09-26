# models.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import uuid
from datetime import datetime, timedelta

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


# ------------- Workspaces (Team Seats) -------------

class Workspace(db.Model):
    __tablename__ = "workspaces"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(160), nullable=False)
    owner_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    seats_limit = db.Column(db.Integer, nullable=False, default=3)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    owner = db.relationship("User", foreign_keys=[owner_user_id])
    members = db.relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan")

    @property
    def seats_used(self) -> int:
        # count active members
        return len([m for m in self.members if m.status == "active"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "owner_user_id": self.owner_user_id,
            "seats_limit": self.seats_limit,
            "seats_used": self.seats_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class WorkspaceMember(db.Model):
    __tablename__ = "workspace_members"
    id = db.Column(db.Integer, primary_key=True)
    workspace_id = db.Column(db.Integer, db.ForeignKey("workspaces.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    role = db.Column(db.String(20), default="member")  # 'owner' | 'admin' | 'member'
    status = db.Column(db.String(20), default="active")  # 'active' | 'invited' | 'removed'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    workspace = db.relationship("Workspace", back_populates="members")
    user = db.relationship("User")

    __table_args__ = (
        db.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_user"),
    )

    def as_dict(self) -> Dict[str, Any]:
        u = self.user
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user": {
                "id": u.id,
                "email": u.email,
                "name": getattr(u, "display_name", None) or getattr(u, "name", None) or (u.email.split("@")[0] if u.email else "User"),
            } if u else None
        }


# ---- Workspace Invitations ----------------------------------------------------

class WorkspaceInvite(db.Model):
    __tablename__ = "workspace_invites"
    id = db.Column(db.Integer, primary_key=True)
    workspace_id = db.Column(db.Integer, db.ForeignKey("workspaces.id"), nullable=False)
    email = db.Column(db.String(255), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False, default="member")   # 'member' | 'admin' | 'owner'
    token = db.Column(db.String(64), unique=True, nullable=False, index=True)
    status = db.Column(db.String(20), nullable=False, default="pending")  # 'pending' | 'accepted' | 'revoked' | 'expired'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=14))

    workspace = db.relationship("Workspace", backref="invites")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "email": self.email,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @staticmethod
    def new_token() -> str:
        return uuid.uuid4().hex + uuid.uuid4().hex
    
class WorkspaceJoinRequest(db.Model):
    __tablename__ = "workspace_join_requests"
    id = db.Column(db.Integer, primary_key=True)
    workspace_id = db.Column(db.Integer, db.ForeignKey("workspaces.id"), nullable=False)
    email = db.Column(db.String(255), nullable=False, index=True)
    role = db.Column(db.String(32), default="member")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def as_dict(self):
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "email": (self.email or "").lower(),
            "role": self.role or "member",
            "created_at": (self.created_at.isoformat() if self.created_at else None),
        }
