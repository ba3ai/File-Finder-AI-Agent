# app_askgpt.py — Ask-GPT style endpoints backed by Ollama
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime

# uses the wrapper we added above
from openai_api import call_chat_model

bp = Blueprint("askgpt", __name__)

SYSTEM_SUGGEST = """You generate 3 concise follow-up questions a user might ask about the provided selection.
- Be specific to the selection content.
- 5-12 words each.
- No numbering, no punctuation at the end.
Return as a JSON list of strings."""

SYSTEM_ANSWER = """You are a helpful assistant responding about a user-selected passage.
Rules:
- Focus ONLY on the selection; if unclear, say what is missing.
- Be concise and structured (bullets or short paragraphs).
- If asked to edit/rewrite, keep meaning and tone.
- If asked for code, give the smallest working example.
- If the selection looks like confidential data, remind about privacy.
When useful, cite exact phrases from the selection with quotes."""

@bp.route("/suggestions", methods=["POST"])
@login_required
def askgpt_suggest():
    data = request.json or {}
    sel = (data.get("selection_text") or "").strip()
    ctx = data.get("context") or {}
    if not sel:
        return jsonify({"suggestions": []})
    user = f"Selection:\n{sel}\n\nContext (optional): {ctx}"
    out = call_chat_model(system=SYSTEM_SUGGEST, user=user, temperature=0.4, json_mode=True)
    suggestions = out if isinstance(out, list) else []
    # cap to 5 just in case model returns more
    return jsonify({"suggestions": [s for s in suggestions if isinstance(s, str)][:5]})

@bp.route("/askgpt/ask", methods=["POST"])
@login_required
def askgpt_ask():
    data = request.json or {}
    q   = (data.get("question") or "").strip()
    sel = (data.get("selection") or "").strip()
    ctx = data.get("context") or {}
    if not q or not sel:
        return jsonify({"error": "Missing question or selection"}), 400

    user = f"""User question: {q}

Selection:
\"\"\"{sel}\"\"\"


Context (optional): {ctx}
User: {getattr(current_user, 'email', 'anon')} at {datetime.utcnow().isoformat()}Z
"""
    text = call_chat_model(system=SYSTEM_ANSWER, user=user, temperature=0.2, json_mode=False)
    return jsonify({"answer": text})
