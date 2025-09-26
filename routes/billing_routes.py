# routes/billing_routes.py
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from services.billing_service import (
    list_public_prices, create_checkout_session, create_portal_link, read_subscription
)

billing_bp = Blueprint("billing", __name__, url_prefix="/api/billing")

@billing_bp.get("/prices")
def get_prices():
    return jsonify({
        "publishable_key": current_app.config["STRIPE_PUBLISHABLE_KEY"],
        "plans": list_public_prices()
    })

@billing_bp.post("/checkout")
@login_required
def start_checkout():
    data = request.get_json() or {}
    price_id = data.get("price_id")
    plan_code = data.get("plan_code")
    if not price_id or not plan_code:
        return jsonify({"error": "Missing price_id or plan_code"}), 400

    success_url = f"{current_app.config['FRONTEND_BASE_URL']}/billing/success"
    cancel_url = f"{current_app.config['FRONTEND_BASE_URL']}/pricing"
    url = create_checkout_session(current_user, price_id, plan_code, success_url, cancel_url)
    return jsonify({"url": url})

@billing_bp.post("/portal")
@login_required
def portal():
    return_url = f"{current_app.config['FRONTEND_BASE_URL']}/account"
    url = create_portal_link(current_user, return_url)
    if not url:
        return jsonify({"error": "No customer found"}), 404
    return jsonify({"url": url})

@billing_bp.get("/me")
@login_required
def my_subscription():
    sub = read_subscription(current_user.id)
    return jsonify({"subscription": sub})
