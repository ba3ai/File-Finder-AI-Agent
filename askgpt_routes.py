# backend/routes/askgpt_routes.py
from flask import Blueprint, request, jsonify
from askgpt_suggester import get_suggestions

askgpt_bp = Blueprint("askgpt", __name__, url_prefix="/askgpt")

@askgpt_bp.route("/suggestions", methods=["POST"])
def suggestions():
    """
    Body: { "history": [ { "role":"user"|"assistant", "content":"..." }, ... ],
            "max": 3 }
    Returns: { "suggestions": ["...", "..."] }
    """
    data = request.get_json(silent=True) or {}
    history = data.get("history", [])
    max_suggestions = int(data.get("max", 3))
    suggestions = get_suggestions(history, max_suggestions)
    return jsonify({"suggestions": suggestions})
