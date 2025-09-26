# google_drive_api.py
# - Token refresh using epoch expiry (matches connections_api.py).
# - Safe search params (no corpora=allDrives 403 issue).
# - Normalizes results and reuses your rule-based ranker.
# - Rich, toggleable debug logging via GD_API_DEBUG=1.

from __future__ import annotations

import os
import json
import time
import datetime as _dt
from typing import Dict, Any, List, Optional

import requests
from models import db, Prefs
from perplexity_ranker import rank_files_rule_based  # your existing ranker
from llm_ranker import rank_files_with_llm  # optional LLM-based ranker

# Pref keys (keep in sync with connections_api.py)
GOOGLE_ACCESS_TOKEN  = "google_access_token"
GOOGLE_REFRESH_TOKEN = "google_refresh_token"
GOOGLE_EXPIRES_AT    = "google_expires_at"
GOOGLE_ACCOUNT_EMAIL = "google_account_email"

# Debugging
_DEBUG = os.getenv("GD_API_DEBUG", "0") == "1"
_DEBUG_MAX_JSON = int(os.getenv("GD_DEBUG_MAX_JSON", "600"))  # max chars printed from JSON bodies


# --------------------------- helpers ---------------------------

def _now_iso():
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _dbg(*msg):
    if _DEBUG:
        print("[GD]", _now_iso(), *msg)

def _j(x: Any) -> str:
    """Compact JSON preview for debug lines."""
    try:
        s = json.dumps(x, ensure_ascii=False, separators=(",", ":"))
        if len(s) > _DEBUG_MAX_JSON:
            s = s[:_DEBUG_MAX_JSON] + "…"
        return s
    except Exception:
        return str(x)

def _get(uid: int, key: str, default=None):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else default

def _set(uid: int, key: str, value: str):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    if row:
        row.value = value
    else:
        row = Prefs(user_id=uid, key=key, value=value)
        db.session.add(row)
    db.session.commit()


# ------------------------ token management --------------------

def _refresh_google_token(uid: int) -> Optional[str]:
    """Refresh the Google access token; returns the new access token or None."""
    refresh = _get(uid, GOOGLE_REFRESH_TOKEN)
    if not refresh:
        _dbg(f"uid={uid} no refresh_token stored")
        return None

    data = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    }
    _dbg(f"uid={uid} refreshing token…")
    try:
        r = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=25)
    except Exception as e:
        _dbg(f"uid={uid} refresh request error: {e}")
        return None

    if not r.ok:
        _dbg(f"uid={uid} refresh failed {r.status_code}: {r.text[:200]}")
        return None

    tok = r.json()
    access = tok.get("access_token")
    expires_in = int(tok.get("expires_in", 0) or 0)

    if access:
        _set(uid, GOOGLE_ACCESS_TOKEN, access)
        _dbg(f"uid={uid} new access token set")

    if expires_in:
        exp_epoch = int(time.time()) + expires_in
        _set(uid, GOOGLE_EXPIRES_AT, str(exp_epoch))
        _dbg(f"uid={uid} new expiry={exp_epoch} (in {expires_in}s)")

    return access

def ensure_google_access_token(uid: int) -> Optional[str]:
    """Return a valid access token for this user; refresh if expiring/expired."""
    access = _get(uid, GOOGLE_ACCESS_TOKEN)
    exp_raw = _get(uid, GOOGLE_EXPIRES_AT) or "0"
    try:
        exp = int(exp_raw)
    except Exception:
        # tolerate legacy ISO strings written by older app.py
        try:
            from dateutil.parser import isoparse  # dateutil is already in your app
            exp = int(isoparse(exp_raw).timestamp())
            # optional: write back as epoch so the next read is fast
            _set(uid, GOOGLE_EXPIRES_AT, str(exp))
        except Exception:
            exp = 0
    now = int(time.time())

    if not access:
        _dbg(f"uid={uid} no access token; trying refresh")
        return _refresh_google_token(uid)

    if exp and now < exp - 60:
        return access

    _dbg(f"uid={uid} token expiring/expired (now={now}, exp={exp}); refreshing")
    return _refresh_google_token(uid)


# ---------------------------- search core --------------------

def _escape_drive_value(s: str) -> str:
    # escape backslash first, then single quotes
    return (s or "").replace("\\", "\\\\").replace("'", "\\'")

def _normalize_drive_file(f: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Google file shape to your MS-style schema used by the ranker."""
    owner = ((f.get("owners") or [{}])[0]).get("emailAddress")
    return {
        "id": f.get("id"),
        "name": f.get("name"),
        "source": "google_drive",
        "createdDateTime": f.get("createdTime"),
        "lastModifiedDateTime": f.get("modifiedTime"),
        "file": {"mimeType": f.get("mimeType")},
        "webUrl": f.get("webViewLink") or (f"https://drive.google.com/file/d/{f.get('id')}/view" if f.get("id") else None),
        "icon": f.get("iconLink"),
        "owner": owner,
        "extracted_text": "",
        "_raw": f,
    }

def search_drive_files(
    uid: int,
    query: str,
    page_token: Optional[str] = None,
    limit: int = 25,
    mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Raw Drive search (one page). Returns:
      { "files": [normalized], "nextPageToken": str|None, "error": str|None, "debug": {...} }
    Debug info carries timings and last-request metadata (safe to ignore in UI).
    """
    t0 = time.time()
    dbg = {"started": _now_iso(), "query": query, "limit": limit, "page_token": page_token}
    print("I am from google drive api")
    _dbg(f"uid={uid} search '{query}' page_token={page_token} limit={limit}")
    access = ensure_google_access_token(uid)
    if not access:
        _dbg(f"uid={uid} no valid token")
        dbg["elapsed_ms"] = int((time.time() - t0) * 1000)
        return {"files": [], "nextPageToken": None, "error": "No Google token", "debug": dbg}

    safe = _escape_drive_value(query)
    q_parts = [f"name contains '{safe}'", f"fullText contains '{safe}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q_str = " or ".join(q_parts)

    params = {
        "q": q_str,
        "pageSize": limit,
        "orderBy": "modifiedTime desc",
        "fields": (
            "files(id,name,mimeType,createdTime,modifiedTime,owners(displayName,emailAddress),"
            "webViewLink,iconLink),nextPageToken"
        ),
        # These are harmless for personal accounts and needed for Shared Drives.
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
        # DO NOT set corpora=allDrives (403 on personal accounts)
    }
    if page_token:
        params["pageToken"] = page_token

    url = "https://www.googleapis.com/drive/v3/files"
    headers = {"Authorization": f"Bearer {access}"}

    _dbg(f"uid={uid} GET {url} params={_j(params)}")
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
    except Exception as e:
        _dbg(f"uid={uid} request error: {e}")
        dbg.update({"elapsed_ms": int((time.time() - t0) * 1000), "exception": str(e)})
        return {"files": [], "nextPageToken": None, "error": str(e), "debug": dbg}

    dbg.update({"status": r.status_code})
    if not r.ok:
        txt = r.text[:400]
        _dbg(f"uid={uid} drive error {r.status_code}: {txt}")
        dbg.update({"elapsed_ms": int((time.time() - t0) * 1000), "body": txt})
        # Common: 403 when corpora=allDrives is used on personal tenants; we don't set it.
        return {"files": [], "nextPageToken": None, "error": txt, "debug": dbg}

    data = r.json()
    files = data.get("files", [])
    normalized = [_normalize_drive_file(f) for f in files]
    token = data.get("nextPageToken")

    dbg.update({
        "elapsed_ms": int((time.time() - t0) * 1000),
        "returned": len(normalized),
        "nextPageToken": token,
    })
    _dbg(f"uid={uid} ok: {len(normalized)} items, next={bool(token)} ({dbg['elapsed_ms']} ms)")
    return {"files": normalized, "nextPageToken": token, "error": None, "debug": dbg}


# ---------------------- ranked facade (use this) ----------------------------

def search_drive_files_ranked(
    uid: int,
    query: str,
    *,
    fetch_pages: int = 2,
    page_size: int = 25,
    mime_type: Optional[str] = None,
    with_debug: bool = False,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    """
    Fetch 1..N pages, normalize, then rank. If with_debug=True, returns:
      { "files": [...ranked...], "trace": [per-page debug dicts] }
    otherwise just the ranked list.
    """
    all_files: List[Dict[str, Any]] = []
    token: Optional[str] = None
    trace: List[Dict[str, Any]] = []
    print(f"UID is {uid} query is {query}")
    for page_idx in range(max(fetch_pages, 1)):
        res = search_drive_files(
            uid=uid,
            query=query,
            page_token=token,
            limit=page_size,
            mime_type=mime_type,
        )
        if with_debug:
            trace.append(res.get("debug", {"page": page_idx, "note": "no-debug"}))

        if res.get("error"):
            _dbg(f"uid={uid} page#{page_idx} error: {res['error']}")
            break

        all_files.extend(res.get("files", []))
        token = res.get("nextPageToken")
        if not token:
            break

    if not all_files:
        return {"files": [], "trace": trace} if with_debug else []

    try:
        ranked = rank_files_with_llm(query, all_files, original_query=query) or []
    except Exception as e:
        _dbg(f"uid={uid} ranker error: {e}")
        # fallback: most recently modified first
        ranked = sorted(all_files, key=lambda x: x.get("lastModifiedDateTime") or "", reverse=True)
    _dbg(f"uid={uid} returning {len(ranked)} ranked items")
    return {"files": ranked, "trace": trace} if with_debug else ranked
