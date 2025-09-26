# services/billing_2co.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import hmac, hashlib
from urllib.parse import urlencode

from flask import current_app, request
from models import db, User

UTC = timezone.utc

@dataclass
class SubscriptionDTO:
    provider: str = "2checkout"
    plan_code: Optional[str] = None
    status: str = "none"        # active|canceled|past_due|none
    next_renewal_at: Optional[str] = None
    provider_customer_id: Optional[str] = None
    provider_subscription_id: Optional[str] = None

def _now() -> datetime: return datetime.now(UTC)

def list_public_prices() -> Dict[str, Any]:
    # ✅ expose features/highlight to the UI
    return {
        "provider": "2checkout",
        "plans": current_app.config.get("BILLING_PLANS", []),
        "buy_base": current_app.config.get("TWOCHECKOUT_BUY_URL"),
    }

def build_buy_link(user: User, plan_code: str, return_url: str) -> str:
    buy_base = current_app.config.get("TWOCHECKOUT_BUY_URL")
    qs = {
        "prod": plan_code,
        "return-url": return_url,
        "return-type": "link",
        "x_user_id": str(user.id),                 # used to match INS to a user
        "customer-email": user.email or "",
        "customer-name": user.display_name or (user.email.split("@")[0] if user.email else "User"),
    }
    return f"{buy_base}?{urlencode(qs)}"

def read_subscription(user_id: int) -> SubscriptionDTO:
    u = User.query.get(user_id)
    if not u:
        return SubscriptionDTO()
    status = "active" if getattr(u, "subscription_active", False) else "none"
    next_renewal = getattr(u, "expires_at", None)
    return SubscriptionDTO(
        plan_code=u.plan or None,
        status=status,
        next_renewal_at=next_renewal.isoformat() if next_renewal else None,
        provider_customer_id=getattr(u, "provider_customer_id", None),
        provider_subscription_id=getattr(u, "provider_subscription_id", None),
    )

# ---------------- INS webhook handling ----------------

def _verify_ins(raw_body: bytes, signature: str) -> bool:
    secret = (current_app.config.get("TWOCHECKOUT_SECRET") or "").encode()
    if not secret or not signature:
        return False
    expected = hmac.new(secret, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.lower())


def handle_ins(payload: Dict[str, Any], raw: bytes) -> Dict[str, Any]:
    # signature can be in a header or field; if you use a header, adjust reading here
    signature = payload.get("SIGNATURE") or payload.get("SIGNATURE_SHA2_256") or request.headers.get("X-2CO-Signature", "")
    if not _verify_ins(raw, signature):
        return {"ok": False, "error": "bad_signature"}

    user_id = payload.get("x_user_id")
    u = User.query.get(int(user_id)) if user_id and str(user_id).isdigit() else None
    if not u:
        return {"ok": False, "error": "user_not_found"}

    event = (payload.get("MESSAGE_TYPE") or "").lower()
    plan_code = payload.get("ITEM_CODE") or payload.get("PROD_CODE") or u.plan

    if event in {"order_created", "recurring_started", "recurring_charge_success"}:
        u.subscription_active = True
        u.plan = plan_code or u.plan  # ✅ sets Pro when the Pro product code is purchased
        db.session.commit()
        return {"ok": True, "status": "active", "plan": u.plan}

    if event in {"recurring_stopped", "refund_issued", "chargeback"}:
        u.subscription_active = False
        db.session.commit()
        return {"ok": True, "status": "canceled", "plan": u.plan}

    return {"ok": True, "event": event}
