# uploaded_file_handler.py
from __future__ import annotations
import os, mimetypes
from typing import List, Dict, Any

def attachments_to_selected(
    attachments: List[Dict[str, Any]],
    upload_root: str,
    user_id: int | str,
    chat_id: str,
) -> List[Dict[str, Any]]:
    """
    Convert upload chips from the UI into the 'selected_files' shape expected by the summarizer.
    Each chip should have at least: { id?, name, path, url?, mime? } where `path` is the server token.
    """
    out: List[Dict[str, Any]] = []
    base = os.path.join(upload_root, str(user_id), str(chat_id))
    for a in attachments or []:
        token = (a.get("path") or "").strip()
        if not token:
            continue
        local_path = os.path.join(base, token)
        out.append({
            "id": a.get("id") or token,
            "name": a.get("name") or os.path.basename(token),
            "source": "upload",                     # ★ mark as local upload
            "webUrl": a.get("url") or "",
            "localPath": local_path,               # ★ file the server can read
            "mimeType": a.get("mime") or mimetypes.guess_type(local_path)[0] or "application/octet-stream",
        })
    return out

def summarize_uploads(
    *,
    user_id: int | str,
    chat_id: str,
    attachments: List[Dict[str, Any]],
    prompt: str,
    upload_root: str,
    app_tz: str = "Asia/Dhaka",
    max_tokens: int = 900,
) -> dict:
    """
    Preferred path: use your summarizer.summarize_selected on the uploaded files.
    Fallback: inline-read text and call openai_api.answer_general_query.
    Returns: {"answer": str, "selected": List[dict]}
    """
    selected = attachments_to_selected(attachments, upload_root, user_id, chat_id)
    if not selected:
        return {"answer": "", "selected": []}

    # Try the first-class summarizer if available
    try:
        from summarizer import summarize_selected
        result = summarize_selected(
            user_id=user_id,
            selected_files=selected,
            prompt=prompt,
            ms_token=None,     # uploads don't need a Graph token
            app_tz=app_tz,
            max_tokens=max_tokens,
        )
        return {"answer": (result or {}).get("answer", "") or "", "selected": selected}
    except Exception:
        # Fallback: read raw text (best-effort) and ask the chat model
        parts: list[str] = []
        for f in selected:
            path = f.get("localPath")
            if not path or not os.path.isfile(path):
                continue
            try:
                # best-effort: read up to 1MB and decode as UTF-8
                raw = open(path, "rb").read(1_000_000)
                text = raw.decode("utf-8", "ignore")
                parts.append(f"--- {f.get('name','uploaded_file')} ---\n{text[:6000]}")
            except Exception:
                continue

        from openai_api import answer_general_query
        fused = f"{prompt}\n\nUse only these uploaded files:\n\n" + "\n\n".join(parts) if parts else prompt
        ans = answer_general_query(
            fused,
            history=None,
            system_prompt="You are a file-reading assistant. Use only the provided file excerpts. If unsure, say so.",
            max_tokens=768,
        )
        return {"answer": ans or "", "selected": selected}
