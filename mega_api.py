# mega_api.py
# MEGAcmd-only integration (no public link system).
# - Account search with a MEGAcmd "session" string saved per user.
# - Stateless: we login in a temp HOME, run the command, logout, and clean up.
# - Optional download helper for selected files.

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from models import db, Prefs

# ------------------------- Pref key -------------------------------------------
MEGA_SESSION_KEY = "mega_session"   # MEGAcmd exported session (string)

# ------------------------- Debug ----------------------------------------------
DEBUG = os.getenv("MEGA_API_DEBUG", "0") == "1"
def _dprint(*args):
    if DEBUG:
        print("[mega_api]", *args, flush=True)

# ------------------------- MEGAcmd binaries -----------------------------------
MEGA_LOGIN_BIN  = os.getenv("MEGA_LOGIN_BIN",  "mega-login")
MEGA_LOGOUT_BIN = os.getenv("MEGA_LOGOUT_BIN", "mega-logout")
MEGA_FIND_BIN   = os.getenv("MEGA_FIND_BIN",   "mega-find")
MEGA_LS_BIN     = os.getenv("MEGA_LS_BIN",     "mega-ls")
MEGA_GET_BIN    = os.getenv("MEGA_GET_BIN",    "mega-get")

def _run(cmd: str, env=None, timeout: int = 40) -> str:
    try:
        if DEBUG: _dprint("RUN:", cmd)
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
        if p.returncode != 0 and DEBUG:
            _dprint("RC:", p.returncode, "STDERR:", (p.stderr or "").strip())
        return p.stdout or ""
    except Exception as e:
        _dprint("Command failed:", e)
        return ""

def _temp_home_env() -> tuple[dict, str]:
    tmp = tempfile.mkdtemp(prefix="mega_home_")
    env = os.environ.copy()
    env["HOME"] = tmp
    if os.name == "nt":
        env["USERPROFILE"] = tmp
    return env, tmp

def _cleanup_home(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)

def _pget(uid: int, key: str, default: Optional[str] = None) -> Optional[str]:
    row = Prefs.query.filter_by(user_id=uid, key=key).first()
    return row.value if row else default

def _to_iso(ts: Any) -> str:
    try:
        ts = int(ts);  return datetime.utcfromtimestamp(ts).isoformat() + "Z"
    except Exception:
        return str(ts) if ts else ""

# ========================= ACCOUNT SEARCH (MEGAcmd) ===========================

def search_mega_account_with_session(uid: int, query: str, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Search the user's own MEGA account by *filename* using MEGAcmd.
    """
    session = _pget(uid, MEGA_SESSION_KEY)
    if not session:
        _dprint("No session saved for user", uid)
        return []

    env, temp_home = _temp_home_env()
    try:
        # 1) login using the session token (non-interactive)
        _run(f'{shlex.quote(MEGA_LOGIN_BIN)} --session {shlex.quote(session)}', env=env, timeout=20)

        # 2) list/search files; prefer mega-find; fallback to mega-ls -R
        out = _run(f'{shlex.quote(MEGA_FIND_BIN)} -R /', env=env, timeout=45)
        if not out.strip():
            out = _run(f'{shlex.quote(MEGA_LS_BIN)} -R --time-format=unix /', env=env, timeout=45)

        q = (query or "").lower().strip()
        words = [w for w in re.split(r"\s+", q) if w]

        items: List[Dict[str, Any]] = []
        for raw in (out or "").splitlines():
            line = raw.strip()
            if not line or line.endswith("/"):
                continue  # skip folders
            name = line.rsplit("/", 1)[-1]
            if words and not all(w in name.lower() for w in words):
                continue
            items.append({
                "id": f"acc::{line}",           # keep full path in the id
                "name": name or "MEGA file",
                "webUrl": "https://mega.nz/fm", # generic web console
                "lastModifiedDateTime": "",
                "source": "mega",
                "mega_path": line,              # needed for download
            })
            if len(items) >= limit:
                break

        _dprint(f"Account search → {len(items)} items for {query!r}")
        return items
    finally:
        try: _run(f'{shlex.quote(MEGA_LOGOUT_BIN)}', env=env, timeout=10)
        except Exception: pass
        _cleanup_home(temp_home)

# ========================= DOWNLOAD (MEGAcmd) =================================

def download_mega_account_file(uid: int, mega_path: str, out_path: str) -> bool:
    """
    Download a file from the user's account to `out_path` using MEGAcmd.
    """
    session = _pget(uid, MEGA_SESSION_KEY)
    if not session:
        _dprint("No session for download.")
        return False

    env, temp_home = _temp_home_env()
    try:
        _run(f'{shlex.quote(MEGA_LOGIN_BIN)} --session {shlex.quote(session)}', env=env, timeout=20)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _run(f'{shlex.quote(MEGA_GET_BIN)} {shlex.quote(mega_path)} -o {shlex.quote(out_path)}',
             env=env, timeout=180)
        ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        _dprint("Download", "OK" if ok else "FAILED", "→", out_path)
        return ok
    finally:
        try: _run(f'{shlex.quote(MEGA_LOGOUT_BIN)}', env=env, timeout=10)
        except Exception: pass
        _cleanup_home(temp_home)

# ========================= Ranked facade (unchanged signature) ================

def _fallback_sort(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(f: Dict[str, Any]):
        dt = f.get("lastModifiedDateTime") or ""
        try: ts = datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp()
        except Exception: ts = 0.0
        return (ts, (f.get("name") or "").lower())
    return sorted(items, key=key, reverse=True)

def search_mega_files_ranked(uid: int, query: str, limit: int = 12) -> List[Dict[str, Any]]:
    """
    Unified search (MEGAcmd only): search the account and (optionally) rank with LLM.
    """
    try:
        from llm_ranker import rank_files_with_llm_simple
    except Exception:
        rank_files_with_llm_simple = None  # type: ignore

    candidates = search_mega_account_with_session(uid, query, limit=64)
    if not candidates:
        return []

    for it in candidates:
        it.setdefault("name", "MEGA file")
        it.setdefault("webUrl", "https://mega.nz/fm")
        it["source"] = "mega"

    if rank_files_with_llm_simple:
        try:
            ranked = rank_files_with_llm_simple(query, candidates, limit=max(limit, 12))
        except Exception as e:
            _dprint("LLM ranker failed:", e)
            ranked = _fallback_sort(candidates)
    else:
        ranked = _fallback_sort(candidates)

    return ranked[:limit]

# ========================= Back-compat NO-OPs (so imports don't break) ========

def search_mega_public_links(uid: int, query: str) -> List[Dict[str, Any]]:
    """Public links removed. Kept for compatibility; always returns empty."""
    return []

def list_user_links(uid: int) -> List[str]:            # legacy no-op
    return []

def add_user_link(uid: int, link: str) -> List[str]:   # legacy no-op
    return []

def remove_user_link(uid: int, link: str) -> List[str]:# legacy no-op
    return []

# ========================= Exports ============================================

__all__ = [
    "search_mega_files_ranked",
    "search_mega_account_with_session",
    "download_mega_account_file",
    # legacy no-ops:
    "search_mega_public_links", "list_user_links", "add_user_link", "remove_user_link",
    "MEGA_SESSION_KEY",
]
