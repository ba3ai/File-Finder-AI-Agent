# routes/stripe_webhooks.py
import stripe
from flask import Blueprint, request, current_app, jsonify
from services.billing_service import init_stripe, record_webhook, upsert_subscription

stripe_wh_bp = Blueprint("stripe_wh", __name__, url_prefix="/webhooks")

@stripe_wh_bp.post("/stripe")
def stripe_webhook():
    init_stripe()
    payload = request.data
    sig = request.headers.get("stripe-signature")
    wh_secret = current_app.config["STRIPE_WEBHOOK_SECRET"]
    try:
        event = stripe.Webhook.construct_event(payload, sig, wh_secret)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    record_webhook(event)

    t = event["type"]
    data = event["data"]["object"]

    # When a subscription is created or updated
    if t in ("customer.subscription.created", "customer.subscription.updated"):
        sub = data
        price_id = sub["items"]["data"][0]["price"]["id"]
        plan_code = sub["metadata"].get("plan_code") if sub.get("metadata") else None

        # Map back to user_id: metadata at subscription or customer level
        user_id = None
        if sub.get("metadata") and sub["metadata"].get("user_id"):
            user_id = int(sub["metadata"]["user_id"])
        elif sub.get("customer"):
            try:
                cust = stripe.Customer.retrieve(sub["customer"])
                if cust and cust.get("metadata", {}).get("user_id"):
                    user_id = int(cust["metadata"]["user_id"])
            except Exception:
                pass

        if user_id:
            upsert_subscription(
                user_id=user_id,
                stripe_customer_id=sub["customer"],
                stripe_subscription=sub,
                plan_code=plan_code or "UNKNOWN",
                price_id=price_id
            )

    # Handle cancellation
    if t == "customer.subscription.deleted":
        sub = data
        # Mark status canceled
        try:
            from services.billing_service import read_subscription, get_db, now_iso
            existing = read_subscription(None)  # placeholder to import utils
        except:
            pass
        # Quick update by subscription_id
        from services.billing_service import get_db, now_iso
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            UPDATE subscriptions
               SET status='canceled', updated_at=?
             WHERE stripe_subscription_id=?""",
            (now_iso(), sub["id"]))
        conn.commit()
        conn.close()

    return jsonify({"received": True}), 200
