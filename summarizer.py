# services/summarizer.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

# LLM helper (optional)
try:
    from openai_api import answer_general_query  # (user_input, history=None, system_prompt=None, max_tokens=900)
except Exception:
    answer_general_query = None

# ✅ Correct extractor import for your repo:
try:
    from smart_content import summarize_from_hits as _summarize_from_hits
except Exception:
    _summarize_from_hits = None

FILE_SOURCES_NON_MS = {"google_drive", "dropbox", "box", "mega"}

def needs_ms_token(files: List[Dict[str, Any]]) -> bool:
    for f in files or []:
        src = (f.get("source") or "").lower()
        if src not in FILE_SOURCES_NON_MS:
            return True
    return False

def _default_instructions() -> str:
    return (
        "Summarize clearly and concisely.\n"
    )

def summarize_selected(
    user_id: int,
    selected_files: List[Dict[str, Any]],
    *,
    prompt: Optional[str] = None,
    ms_token: Optional[str] = None,
    app_tz: Optional[str] = None,
    max_tokens: int = 900,
) -> Dict[str, Any]:
    if not selected_files:
        return {"answer": "No files were selected.", "files": [], "used_ms": False}

    instructions = prompt or _default_instructions()

    # ✅ Use your real, content-aware summarizer
    if _summarize_from_hits:
        result = _summarize_from_hits(user_id, ms_token, selected_files, instructions)
        answer = result.get("answer") if isinstance(result, dict) else str(result)
        return {
            "answer": (answer or "I couldn't extract a useful summary from the selected files.").strip(),
            "files": [{"id": f.get("id"), "name": f.get("name"), "source": f.get("source")} for f in selected_files],
            "used_ms": bool(ms_token),
        }

    # Fallback: filenames-only (should rarely happen now)
    titles = "\n".join(
        f"- {f.get('name','(unnamed)')} [{(f.get('source') or '?').upper()}]" for f in selected_files[:20]
    )
    sys = (
        "You are ECHO, a precise assistant. Summarize based on content provided by the host application. "
        "If you were only given filenames/metadata (no content), say the summary may be incomplete."
    )
    user = f"Files selected:\n{titles}\n\n{instructions}\nIf contents were not provided, mention uncertainty."
    if answer_general_query:
        text = answer_general_query(user_input=user, system_prompt=sys, max_tokens=max_tokens)
    else:
        text = "Summary unavailable: the summarization backend is not configured."

    return {
        "answer": (text or "").strip(),
        "files": [{"id": f.get("id"), "name": f.get("name"), "source": f.get("source")} for f in selected_files],
        "used_ms": bool(ms_token),
    }
