# services/billing_service.py
import os, sqlite3, json, time
from datetime import datetime
import stripe
from flask import current_app

def get_db():
    # Replace with your existing sqlite helper if you have one
    conn = sqlite3.connect(current_app.config.get("SQLITE_PATH", "app.db"))
    conn.row_factory = sqlite3.Row
    return conn

def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def init_stripe():
    stripe.api_key = current_app.config["STRIPE_SECRET_KEY"]

# Optional: Hardcode your “product model” and map to Stripe price ids.
# Keep in ENV or DB if you prefer runtime edits from Super Admin
PLANS = {
    "STARTER": {
        "name": "Starter",
        "features": [
            "Up to 3 connectors",
            "5,000 search credits / month",
            "Community support"
        ],
        "monthly_price_id": os.getenv("STRIPE_PRICE_STARTER_MONTHLY", ""),
        "yearly_price_id": os.getenv("STRIPE_PRICE_STARTER_YEARLY", "")
    },
    "PRO": {
        "name": "Pro",
        "features": [
            "Up to 10 connectors",
            "50,000 search credits / month",
            "Priority email support",
            "Team seats (up to 10)"
        ],
        "monthly_price_id": os.getenv("STRIPE_PRICE_PRO_MONTHLY", ""),
        "yearly_price_id": os.getenv("STRIPE_PRICE_PRO_YEARLY", "")
    },
    "ENTERPRISE": {
        "name": "Enterprise",
        "features": [
            "Unlimited connectors",
            "Custom credit pools",
            "SAML/SCIM",
            "Dedicated success manager"
        ],
        "monthly_price_id": os.getenv("STRIPE_PRICE_ENT_MONTHLY", ""),
        "yearly_price_id": os.getenv("STRIPE_PRICE_ENT_YEARLY", "")
    }
}

def list_public_prices():
    out = []
    for code, p in PLANS.items():
        out.append({
            "code": code,
            "name": p["name"],
            "features": p["features"],
            "monthly_price_id": p["monthly_price_id"],
            "yearly_price_id": p["yearly_price_id"]
        })
    return out

def upsert_subscription(user_id, stripe_customer_id, stripe_subscription, plan_code, price_id):
    conn = get_db()
    cur = conn.cursor()
    # Check existing
    cur.execute("SELECT id FROM subscriptions WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    fields = (
        stripe_customer_id,
        stripe_subscription["id"],
        plan_code,
        price_id,
        stripe_subscription["status"],
        datetime.utcfromtimestamp(stripe_subscription["current_period_start"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        datetime.utcfromtimestamp(stripe_subscription["current_period_end"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        1 if stripe_subscription.get("cancel_at_period_end") else 0,
        now_iso(),
        user_id,
    )
    if row:
        cur.execute("""
            UPDATE subscriptions
              SET stripe_customer_id=?,
                  stripe_subscription_id=?,
                  plan_id=?,
                  price_id=?,
                  status=?,
                  current_period_start=?,
                  current_period_end=?,
                  cancel_at_period_end=?,
                  updated_at=?
            WHERE user_id=?""", fields)
    else:
        cur.execute("""
            INSERT INTO subscriptions (stripe_customer_id, stripe_subscription_id, plan_id, price_id, status,
                                       current_period_start, current_period_end, cancel_at_period_end, created_at, updated_at, user_id)
            VALUES (?,?,?,?,?,?,?,?,?, ?, ?)""",
            (stripe_customer_id,
             stripe_subscription["id"],
             plan_code,
             price_id,
             stripe_subscription["status"],
             datetime.utcfromtimestamp(stripe_subscription["current_period_start"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
             datetime.utcfromtimestamp(stripe_subscription["current_period_end"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
             1 if stripe_subscription.get("cancel_at_period_end") else 0,
             now_iso(), now_iso(), user_id)
        )
    conn.commit()
    conn.close()

def read_subscription(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM subscriptions WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def create_checkout_session(user, price_id, plan_code, success_url, cancel_url):
    init_stripe()
    # Find or create customer
    customer = stripe.Customer.search(query=f"email:'{user.email}'").data
    if customer:
        cust_id = customer[0].id
    else:
        cust = stripe.Customer.create(email=user.email, metadata={"user_id": user.id})
        cust_id = cust.id

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=cust_id,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=cancel_url,
        allow_promotion_codes=True,
        subscription_data={
            "metadata": {"user_id": str(user.id), "plan_code": plan_code}
        },
        metadata={"user_id": str(user.id), "plan_code": plan_code}
    )
    return session.url

def create_portal_link(user, return_url):
    init_stripe()
    customer = stripe.Customer.search(query=f"email:'{user.email}'").data
    if not customer:
        # No customer yet
        return None
    portal = stripe.billing_portal.Session.create(
        customer=customer[0].id,
        return_url=return_url
    )
    return portal.url

def record_webhook(event):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO webhook_events (stripe_event_id, type, payload_json)
        VALUES (?, ?, ?)""",
        (event["id"], event["type"], json.dumps(event)))
    conn.commit()
    conn.close()
