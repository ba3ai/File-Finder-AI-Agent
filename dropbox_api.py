# dropbox_api.py
# Robust Dropbox search + token refresh.
# - Ensures refresh tokens are used when access token expires.
# - Retries once with a forced refresh on 401/invalid token.
# - Clean parsing of /2/files/search_v2 results (files only).
# - Best-effort temporary links (requires files.content.read scope).
# - Safe fallbacks and helpful debug logging.

from __future__ import annotations

import os
import time
import requests
from typing import List, Dict, Any, Optional

from models import Prefs, db
from perplexity_ranker import rank_files_rule_based  # or your preferred ranker

SEARCH_URL   = "https://api.dropboxapi.com/2/files/search_v2"
TMP_LINK_URL = "https://api.dropboxapi.com/2/files/get_temporary_link"
TOKEN_URL    = "https://api.dropboxapi.com/oauth2/token"

# Keep these keys in sync with connections_api.py
DBX_AT   = "dbx_access_token"
DBX_RT   = "dbx_refresh_token"
DBX_EXP  = "dbx_expires_at"   # epoch seconds (string)

DEBUG = os.getenv("DBX_API_DEBUG", "0") == "1"


# -------------------- Prefs helpers --------------------

def _pget(uid: int, key: str, default: Optional[str] = None) -> Optional[str]:
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else default


def _pset(uid: int, key: str, value: str) -> None:
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    if row:
        row.value = value
    else:
        row = Prefs(user_id=uid, key=key, value=value)
        db.session.add(row)
    db.session.commit()


def _dprint(*args):
    if DEBUG:
        print("[dropbox_api]", *args)


# -------------------- Token management --------------------

def ensure_dropbox_access_token(uid: int, force_refresh: bool = False) -> Optional[str]:
    """
    Return a (possibly refreshed) access token for the user.
    If force_refresh=True, always attempt to use the refresh token.
    """
    at  = _pget(uid, DBX_AT)
    rt  = _pget(uid, DBX_RT)
    exp = int(_pget(uid, DBX_EXP, "0") or "0")

    # still valid and not forcing a refresh
    if at and not force_refresh and exp and time.time() < (exp - 60):
        return at

    # No refresh token -> we can't refresh; return whatever we have (may still work)
    if not rt:
        return at

    data = {
        "grant_type": "refresh_token",
        "refresh_token": rt,
        "client_id": os.getenv("DROPBOX_CLIENT_ID"),
        "client_secret": os.getenv("DROPBOX_CLIENT_SECRET"),
    }
    try:
        r = requests.post(TOKEN_URL, data=data, timeout=20)
    except Exception as e:
        _dprint("Token refresh error:", e)
        return at

    if not r.ok:
        _dprint("Token refresh failed:", r.status_code, r.text)
        return at

    t = r.json() or {}
    at = t.get("access_token") or at
    # Dropbox may rotate refresh_token:
    _pset(uid, DBX_AT, at)
    if t.get("refresh_token"):
        _pset(uid, DBX_RT, t["refresh_token"])
    # store new expiry if provided
    try:
        new_exp = int(time.time() + int(t.get("expires_in", 3600)))
        _pset(uid, DBX_EXP, str(new_exp))
    except Exception:
        pass

    return at


# -------------------- Search --------------------

def search_dropbox_files(
    access_token: str,
    query: str,
    limit: int = 50,
    filename_only: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Call /2/files/search_v2.
    - filename_only: if True, restrict to filename matches (useful if account lacks content search).
                     if None, defaults to False (search content too when plan allows).
    """
    if filename_only is None:
        # allow env override if you know your plan does not support content search
        filename_only = os.getenv("DBX_FILENAME_ONLY", "0") == "1"

    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    body = {
        "query": query,
        "options": {
            "filename_only": bool(filename_only),
            "max_results": min(int(limit or 50), 100),
            "file_status": "active",
        },
    }

    r = requests.post(SEARCH_URL, headers=headers, json=body, timeout=30)

    if r.status_code == 401:
        # Dropbox returns 401 for invalid/expired token (often with "invalid_access_token")
        raise PermissionError("dropbox_token_expired")

    if not r.ok:
        _dprint("Dropbox search error:", r.status_code, r.text)
        return []

    data = r.json() or {}
    matches = data.get("matches") or []
    items: List[Dict[str, Any]] = []

    for m in matches:
        md = (m.get("metadata") or {}).get("metadata") or {}
        # keep only files
        if md.get(".tag") != "file":
            continue

        fid = md.get("id")
        name = md.get("name") or ""
        path = md.get("path_display") or ""
        modified = md.get("server_modified") or md.get("client_modified") or ""

        items.append({
            "id": fid or path,
            "name": name or (path.split("/")[-1] if path else ""),
            "webUrl": "",  # filled below if we can mint a temp link
            "lastModifiedDateTime": modified,
            "path_display": path,
            "source": "dropbox",
        })

    # Best-effort temporary link (requires files.content.read). Ignore failures.
    for it in items:
        try:
            resp = requests.post(
                TMP_LINK_URL,
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                json={"path": it.get("path_display")},
                timeout=15,
            )
            if resp.ok:
                it["webUrl"] = (resp.json() or {}).get("link") or ""
        except Exception:
            pass

        # Fallback to a web UI link if no temp link was produced
        if not it.get("webUrl") and it.get("path_display"):
            it["webUrl"] = f"https://www.dropbox.com/home{it['path_display']}"

    return items


def search_dropbox_files_ranked(
    uid: int,
    query: str,
    limit: int = 12,
    candidate_multiplier: int = 3,
) -> List[Dict[str, Any]]:
    """
    High-level search used by app.py:
    - gets/refreshes token
    - retries once with force refresh on 401
    - ranks results
    """
    _dprint(f"Searching Dropbox files for user {uid} with query: {query}")
    token = ensure_dropbox_access_token(uid)
    if not token:
        _dprint("No Dropbox token available")
        return []

    cand_limit = max(limit * candidate_multiplier, 30)

    try:
        candidates = search_dropbox_files(token, query, limit=cand_limit) or []
    except PermissionError:
        # Token invalid → force refresh and retry once
        token = ensure_dropbox_access_token(uid, force_refresh=True)
        if not token:
            _dprint("Token refresh failed; no token to retry")
            return []
        candidates = search_dropbox_files(token, query, limit=cand_limit) or []

    # Normalize expected fields
    for c in candidates:
        c.setdefault("name", "")
        c.setdefault("webUrl", "")
        c.setdefault("lastModifiedDateTime", "")
        c.setdefault("source", "dropbox")

    # Rank (LLM or rule-based)
    try:
        ranked = rank_files_rule_based(query, candidates)
    except Exception:
        ranked = sorted(candidates, key=lambda x: x.get("lastModifiedDateTime", ""), reverse=True)

    return ranked[:limit]
