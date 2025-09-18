# box_api.py
import os, time, requests
from typing import List, Dict, Any
from models import Prefs, db
from perplexity_ranker import rank_files_rule_based  # or your llama_ranker if you prefer

BOX_TOKEN_URL = "https://api.box.com/oauth2/token"
BOX_SEARCH    = "https://api.box.com/2.0/search"
BOX_FILE_URL  = "https://app.box.com/file/{id}"

# Prefs keys (match connections_api.py)
BOX_AT   = "box_access_token"
BOX_RT   = "box_refresh_token"
BOX_EXP  = "box_expires_at"

def _pget(uid, key, default=None):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else default

def _pset(uid, key, value):
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    if row: row.value = value
    else:
        row = Prefs(user_id=uid, key=key, value=value)
        db.session.add(row)
    db.session.commit()

def ensure_box_access_token(uid: int) -> str | None:
    at  = _pget(uid, BOX_AT)
    exp = int(_pget(uid, BOX_EXP, "0") or "0")
    if at and exp and time.time() < (exp - 60):  # not expired
        return at
    # refresh
    rt = _pget(uid, BOX_RT)
    if not rt:
        return at  # maybe still good
    data = {
        "grant_type": "refresh_token",
        "refresh_token": rt,
        "client_id": os.getenv("BOX_CLIENT_ID"),
        "client_secret": os.getenv("BOX_CLIENT_SECRET"),
    }
    r = requests.post(BOX_TOKEN_URL, data=data, timeout=20)
    if not r.ok:
        return at
    t = r.json()
    at = t.get("access_token") or at
    rt = t.get("refresh_token") or rt
    exp = int(time.time() + int(t.get("expires_in", 3600)))
    _pset(uid, BOX_AT, at); _pset(uid, BOX_RT, rt); _pset(uid, BOX_EXP, str(exp))
    return at

def search_box_files(access_token: str, query: str, limit: int = 50) -> List[Dict[str,Any]]:
    # Box defaults to 30 items/page. Use limit<=100 per docs; supports offset for pagination.
    print(f"Searching Box for query: {query} with limit: {limit}")
    params = {
        "query": query,
        "type": "file",
        # use name+description by default; content search may not be available on your plan
        "content_types": "name,description",
        "limit": min(limit, 100),
        "fields": "id,name,modified_at,created_at,shared_link,path_collection,size,description,sha1",
        "scope": "user_content",  # be explicit – search the current user's content
    }
    r = requests.get(BOX_SEARCH, headers={"Authorization": f"Bearer {access_token}"}, params=params, timeout=30)
    if r.status_code == 401:
        raise PermissionError("box_token_expired")
    if not r.ok:
        return []

    data = r.json() or {}
    entries = data.get("entries", []) or []

    items = []
    for e in entries:
        fid = e.get("id")
        items.append({
            "id": fid,
            "name": e.get("name") or "",
            "webUrl": BOX_FILE_URL.format(id=fid) if fid else "",
            "lastModifiedDateTime": e.get("modified_at") or e.get("created_at") or "",
            "source": "box",
        })
    return items

def search_box_files_ranked(uid: int, query: str, limit: int = 12, candidate_multiplier: int = 3) -> List[Dict[str,Any]]:
    at = ensure_box_access_token(uid)
    if not at:
        return []
    cand_limit = max(limit * candidate_multiplier, 30)
    try:
        candidates = search_box_files(at, query, limit=cand_limit) or []
    except PermissionError:
        at = ensure_box_access_token(uid)
        if not at:
            return []
        candidates = search_box_files(at, query, limit=cand_limit) or []
    try:
        ranked = rank_files_rule_based(query, candidates)
    except Exception:
        ranked = sorted(candidates, key=lambda x: x.get("lastModifiedDateTime",""), reverse=True)
    return ranked[:limit]
