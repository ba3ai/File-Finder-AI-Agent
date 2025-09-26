from __future__ import annotations
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User  # keep your existing "users" table for end users

# ─────────────────────────────────────────────────────────────────────────────
# Staff users live in their OWN table now
#   role ∈ {"superadmin","admin","client_support"}
# ─────────────────────────────────────────────────────────────────────────────
ROLE_SUPERADMIN = "superadmin"
ROLE_ADMIN = "admin"
ROLE_CLIENTSUPP = "client_support"
ALL_STAFF_ROLES = {ROLE_SUPERADMIN, ROLE_ADMIN, ROLE_CLIENTSUPP}

def normalize_role(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    if v in {"client_support_team", "client_supports"}:
        v = ROLE_CLIENTSUPP
    return v if v in ALL_STAFF_ROLES else None

class StaffUser(db.Model):
    __tablename__ = "staff_users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(255), nullable=True)
    role = db.Column(db.String(32), nullable=False, default=ROLE_ADMIN)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime, nullable=True)

    # helpers
    def set_password(self, pwd: str) -> None:
        self.password_hash = generate_password_hash(pwd)

    def check_password(self, pwd: str) -> bool:
        try:
            return check_password_hash(self.password_hash, pwd or "")
        except Exception:
            return False

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Business objects (still tied to end-users in `users` table)
# ─────────────────────────────────────────────────────────────────────────────
class Subscription(db.Model):
    __tablename__ = "subscriptions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    plan = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False, default=0.0)
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default="active")  # active|expired|canceled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("subscriptions", lazy="dynamic"))

class Ticket(db.Model):
    __tablename__ = "tickets"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    priority = db.Column(db.String(20), default="medium")  # low|medium|high
    status = db.Column(db.String(20), default="open")      # open|in_progress|closed
    assigned_to = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    assignee = db.relationship("User", foreign_keys=[assigned_to])

class Revenue(db.Model):
    __tablename__ = "revenue"
    id = db.Column(db.Integer, primary_key=True)
    invoice_no = db.Column(db.String(50), unique=True)
    amount = db.Column(db.Float, nullable=False, default=0.0)
    status = db.Column(db.String(20), default="paid")  # paid|due
    invoice_date = db.Column(db.DateTime, default=datetime.utcnow)
