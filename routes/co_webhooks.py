# routes/2co_webhooks.py
from flask import Blueprint, request, jsonify
from services.billing_2co import handle_ins

ins2co_bp = Blueprint("ins2co", __name__, url_prefix="/webhooks/2co")

@ins2co_bp.post("/ins")
def ins():
    payload = request.form.to_dict(flat=True) or {}
    raw = request.get_data()  # raw body for HMAC
    result = handle_ins(payload, raw)
    code = 200 if result.get("ok") else 400
    return jsonify(result), code
