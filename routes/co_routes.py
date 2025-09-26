# routes/2co_routes.py
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from services.billing_2co import list_public_prices, build_buy_link, read_subscription

billing2co_bp = Blueprint("billing2co", __name__, url_prefix="/api/billing")

@billing2co_bp.get("/prices")
def prices():
    return jsonify(list_public_prices())

@billing2co_bp.post("/checkout")
@login_required
def checkout():
    data = request.get_json(silent=True) or {}
    plan_code = (data.get("plan_code") or "").strip()
    if not plan_code:
        return jsonify({"error": "plan_code required"}), 400
    return_url = f"{current_app.config['FRONTEND_BASE_URL']}/billing/return"
    url = build_buy_link(current_user, plan_code, return_url)
    return jsonify({"url": url})

@billing2co_bp.post("/portal")
@login_required
def portal():
    # Use 2CO hosted "My account" portal, or your own account page
    return jsonify({"url": "https://secure.2checkout.com/myaccount/"})

@billing2co_bp.get("/me")
@login_required
def me():
    sub = read_subscription(current_user.id)
    return jsonify({"subscription": sub.__dict__})
