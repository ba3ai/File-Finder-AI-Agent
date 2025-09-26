# backend/services/askgpt_suggester.py
import os
import random
from typing import List, Dict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- Optional: OpenAI (or Azure OpenAI) ----
def _openai_suggestions(history: List[Dict[str, str]], max_suggestions: int = 3) -> List[str]:
    """
    history: [{"role":"user"|"assistant", "content":"..."}]
    Returns a small list of short, clickable suggestions.
    """
    if not OPENAI_API_KEY:
        return []

    try:
        # Works with OpenAI SDK >= 1.0; replace model as you prefer
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        sys_prompt = (
            "You generate at most 3 SHORT suggestion prompts for a chat input box. "
            "Each suggestion must be terse, actionable, and copy-ready. "
            "Avoid punctuation-heavy or long sentences. 3–12 words each.\n"
            "Examples: 'Explain how it works under the hood', 'Give me a quick start guide', "
            "'Show me production-ready code'. Return them as a JSON array of strings ONLY."
        )
        messages = [{"role":"system","content":sys_prompt}] + history[-8:]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=120
        )
        text = resp.choices[0].message.content.strip()
        # Very light JSON parsing guard
        import json
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n",1)[-1]
        suggestions = json.loads(text)
        out = []
        for s in suggestions:
            if isinstance(s, str) and 3 <= len(s.split()) <= 12:
                out.append(s.strip().rstrip("?"))
            if len(out) >= max_suggestions:
                break
        return out
    except Exception:
        # If anything fails, silently fall back
        return []


# ---- Lightweight heuristic fallback (no external API) ----
_CANNED = [
    "Explain how it works under the hood",
    "Give me a quick start guide",
    "Show production-ready example code",
    "List pros and cons briefly",
    "Summarize the key steps",
    "What should I do next?",
    "Break it down step by step",
]

def _heuristic_suggestions(history: List[Dict[str, str]], max_suggestions: int = 3) -> List[str]:
    # Simple intent sniffing on last assistant/user message
    last = (history[-1]["content"] if history else "").lower()
    picks = []
    if any(k in last for k in ["python", "flask", "react", "api", "deploy"]):
        picks += [
            "Show production-ready example code",
            "Give me a quick start guide",
            "Explain how it works under the hood",
        ]
    elif any(k in last for k in ["excel", "sharepoint", "onedrive", "financial"]):
        picks += [
            "Summarize the integration steps",
            "Show a working code snippet",
            "List common pitfalls",
        ]
    else:
        picks += random.sample(_CANNED, k=min(max_suggestions, len(_CANNED)))
    # Deduplicate and trim to max
    seen, out = set(), []
    for s in picks + _CANNED:
        if s not in seen:
            seen.add(s); out.append(s)
        if len(out) >= max_suggestions:
            break
    return out


def get_suggestions(history: List[Dict[str, str]], max_suggestions: int = 3) -> List[str]:
    # Try OpenAI first, then fallback
    out = _openai_suggestions(history, max_suggestions)
    if not out:
        out = _heuristic_suggestions(history, max_suggestions)
    return out
