# google_drive_api.py
import os
import time
import requests
from typing import Dict, Any, List, Optional, Tuple

from models import db, Prefs
from perplexity_ranker import rank_files_rule_based  # <-- your existing ranker

# Keys used in Prefs (same names as in connections_api)
GOOGLE_ACCESS_TOKEN  = "google_access_token"
GOOGLE_REFRESH_TOKEN = "google_refresh_token"
GOOGLE_EXPIRES_AT    = "google_expires_at"
GOOGLE_ACCOUNT_EMAIL = "google_account_email"


# --------------------------- Pref helpers -----------------------------------

def _get(uid, key, default=None):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else default

def _set(uid, key, value):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    if row:
        row.value = value
    else:
        row = Prefs(user_id=uid, key=key, value=value)
        db.session.add(row)
    db.session.commit()


# ------------------------ Token management ----------------------------------

def _refresh_google_token(uid) -> Optional[str]:
    """Refresh access token using the saved refresh_token; write back to Prefs; return the new access token."""
    refresh = _get(uid, GOOGLE_REFRESH_TOKEN)
    if not refresh:
        return None

    data = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    }
    r = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=20)
    if not r.ok:
        return None

    tok = r.json()
    access = tok.get("access_token")
    if access:
        _set(uid, GOOGLE_ACCESS_TOKEN, access)
    if tok.get("expires_in"):
        _set(uid, GOOGLE_EXPIRES_AT, str(int(time.time()) + int(tok["expires_in"])))
    return access

def ensure_google_access_token(uid) -> Optional[str]:
    """Return a valid access token for this user; refresh if needed."""
    access = _get(uid, GOOGLE_ACCESS_TOKEN)
    exp    = int(_get(uid, GOOGLE_EXPIRES_AT) or 0)
    if not access or time.time() > exp - 60:
        access = _refresh_google_token(uid)
    return access


# ---------------------------- Search ----------------------------------------

def _escape_drive_value(s: str) -> str:
    # order matters: escape backslash first, then single quotes
    return (s or "").replace("\\", "\\\\").replace("'", "\\'")

def _normalize_drive_file(f: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map Google fields => the ranker-friendly schema the OneDrive flow already uses:
      - createdDateTime          <- createdTime
      - lastModifiedDateTime     <- modifiedTime
      - file.mimeType            <- mimeType (nest it in a 'file' dict to match MS shape)
      - webUrl                   <- webViewLink
      - owner                    <- owners[0].emailAddress
      - extracted_text           <- '' (no OCR here; ranker handles absence)
    """
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
        # Optional, empty here; ranker is robust to missing text
        "extracted_text": "",
        # Keep raw fields too if you want:
        "_raw": f,
    }

def search_drive_files(
    uid: int,
    query: str,
    page_token: Optional[str] = None,
    limit: int = 25,
    mime_type: Optional[str] = None,
    use_all_drives: bool = True,
) -> Dict[str, Any]:
    """
    Raw Google Drive search (first page). Returns a normalized list for ranking:
      { "files": [ ...normalized... ], "nextPageToken": "<token or None>", "error": str|None }
    """
    access = ensure_google_access_token(uid)
    if not access:
        return {"files": [], "nextPageToken": None, "error": "No Google token"}

    safe = _escape_drive_value(query)
    q_parts = [
        f"name contains '{safe}'",
        f"fullText contains '{safe}'",  # works when fullText is indexed
    ]
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
    }
    if page_token:
        params["pageToken"] = page_token
    if use_all_drives:
        params["supportsAllDrives"]         = "true"
        params["includeItemsFromAllDrives"] = "true"
        params["corpora"] = "allDrives"

    r = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {access}"},
        params=params,
        timeout=25,
    )
    if not r.ok:
        return {"files": [], "nextPageToken": None, "error": r.text}

    data = r.json()
    normalized = [_normalize_drive_file(f) for f in data.get("files", [])]
    return {"files": normalized, "nextPageToken": data.get("nextPageToken"), "error": None}


# ---------------------- Ranked facade (use this) -----------------------------

def search_drive_files_ranked(
    uid: int,
    query: str,
    *,
    # pull extra results so ranking has a decent pool to work with
    fetch_pages: int = 2,
    page_size: int = 25,
    mime_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch 1..N pages from Drive, normalize, then run your rule-based ranker and
    return a **ranked** list (newest-first according to your custom SortKey).
    """
    all_files: List[Dict[str, Any]] = []
    token: Optional[str] = None

    for _ in range(max(fetch_pages, 1)):
        res = search_drive_files(
            uid=uid,
            query=query,
            page_token=token,
            limit=page_size,
            mime_type=mime_type,
        )
        if res.get("error"):
            break
        all_files.extend(res.get("files", []))
        token = res.get("nextPageToken")
        if not token:
            break

    if not all_files:
        return []

    # Reuse your OneDrive/SharePoint ranking logic
    ranked = rank_files_rule_based(query, all_files, original_query=query) or []
    return ranked
