# services_cloud_upload.py
from __future__ import annotations
import os, io, json, logging, mimetypes, requests, re
from typing import Optional, Dict, List, Tuple, Literal
from urllib.parse import quote, urlparse

log = logging.getLogger(__name__)

# ============================
# Token getters (import yours)
# ============================
try:
    from google_drive_api import ensure_google_access_token as _ensure_google
except Exception:
    _ensure_google = lambda uid: None

try:
    from dropbox_api import ensure_dropbox_access_token as _ensure_dropbox
except Exception:
    _ensure_dropbox = lambda uid: None

try:
    from box_api import ensure_box_access_token as _ensure_box
except Exception:
    _ensure_box = lambda uid: None

# For MS Graph (SharePoint/OneDrive)
GRAPH = "https://graph.microsoft.com/v1.0"
SP_SITE_HOST = os.getenv("SP_SITE_HOST")
SP_SITE_PATH = os.getenv("SP_SITE_PATH")
DEFAULT_LIBRARY = os.getenv("SP_LIBRARY", "Shared Documents")

# Provider words (for tolerant parsing)
PROVIDER_WORDS_RX = r"(sharepoint|sp|one\s*drive|onedrive|od|google\s*drive|g\s*drive|drive|dropbox|box|mega)"

# ============================================================
# Provider normalization + connected list + conversation flow
# ============================================================

def normalize_provider(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    k = (name or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if k in {"sharepoint", "sharepoints", "sp"}: return "sharepoint"
    if k in {"onedrive", "msonedrive", "od"}:    return "onedrive"
    if k in {"googledrive", "gdrive", "google", "drive", "google_drive"}: return "google_drive"
    if k in {"dropbox"}:  return "dropbox"
    if k in {"box"}:      return "box"
    if k in {"mega","meganz"}: return "mega"
    return None

def provider_label(p: str) -> str:
    return {
        "sharepoint": "SharePoint",
        "onedrive": "OneDrive",
        "google_drive": "Google Drive",
        "dropbox": "Dropbox",
        "box": "Box",
        "mega": "MEGA",
    }.get(p, p)

def _default_dest_for(provider: str) -> str:
    return {
        "sharepoint": f"{DEFAULT_LIBRARY}/General",
        "onedrive": "Documents",
        "google_drive": "My Drive",
        "dropbox": "/",
        "box": "/",
        "mega": "/",
    }[provider]

def _is_valid_dropbox_token(tok: Optional[str]) -> bool:
    """Lightweight validation to avoid 401s during upload."""
    if not tok:
        return False
    try:
        r = requests.post(
            "https://api.dropboxapi.com/2/users/get_current_account",
            headers={"Authorization": f"Bearer {tok}"},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False

def connected_providers(user_id: int, ms_token: Optional[str] = None) -> List[str]:
    """Best-effort detection of connected providers (tokens are lightly verified)."""
    out: List[str] = []
    if ms_token:
        out.extend(["sharepoint","onedrive"])
    try:
        if _ensure_google(user_id):
            out.append("google_drive")
    except Exception:
        pass
    try:
        tok = _ensure_dropbox(user_id)
        if _is_valid_dropbox_token(tok):
            out.append("dropbox")
    except Exception:
        pass
    try:
        if _ensure_box(user_id):
            out.append("box")
    except Exception:
        pass
    if os.getenv("MEGA_EMAIL") and os.getenv("MEGA_PASSWORD"):
        out.append("mega")
    return out

def _choices_sentence(providers: List[str]) -> str:
    return ", ".join(provider_label(p) for p in providers)

# ---- In-memory upload flow store (per user) ----
_UPLOAD_FLOWS: Dict[int, Dict] = {}

def is_upload_flow_active(user_id: int) -> bool:
    f = _UPLOAD_FLOWS.get(user_id)
    return bool(f and (f.get("awaiting_folder") or f.get("awaiting_confirm")))

_CANCEL_RX = re.compile(r"^\s*(cancel|stop|abort|no|never\s*mind|nevermind|forget\s*it)\s*\.?\s*$", re.I)
_YES_RX    = re.compile(r"^\s*(yes|y|sure|ok|okay|confirm|go\s*ahead)\s*\.?\s*$", re.I)
_DIRECT_RX = re.compile(r"^\s*(direct|upload\s*direct(ly)?|no\s*folder|root)\s*\.?\s*$", re.I)

def _get_connected(user_id: int, ms_token: Optional[str], override_list: Optional[List[str]]) -> List[str]:
    if override_list:
        norm = []
        for p in override_list:
            n = normalize_provider(p)
            if n: norm.append(n)
        seen, final = set(), []
        for p in norm:
            if p not in seen:
                seen.add(p); final.append(p)
        return final
    return connected_providers(user_id, ms_token)

# ===========================
# Local LLM via Ollama (langchain_ollama)
# ===========================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")
NUM_CTX          = int(os.getenv("OLLAMA_NUM_CTX", "20000"))
NUM_PREDICT      = int(os.getenv("OLLAMA_NUM_PREDICT", "800"))
LLM_TEMPERATURE  = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
DEBUG            = bool(int(os.getenv("OPENAI_API_DEBUG", "0")))
APP_TZ_NAME      = os.getenv("APP_TZ", "Asia/Dhaka")

def _llm_available() -> bool:
    return bool(OLLAMA_MODEL)

# Use Pydantic schema with LangChain structured output
try:
    from pydantic import BaseModel, Field
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_ollama import ChatOllama
    _LC_OK = True
except Exception:
    _LC_OK = False
    BaseModel = object  # type: ignore

class IntentSchema(BaseModel):  # type: ignore
    provider: Optional[Literal["sharepoint","onedrive","google_drive","dropbox","box","mega"]] = Field(None)
    folder: Optional[str] = Field(None, description="Folder or path only, no provider prefix.")
    direct: bool = Field(False, description="True if default location / no folder requested.")
    cancel: bool = Field(False, description="True if user wants to cancel.")
    confirm: bool = Field(False, description="True if user says yes/confirm.")

_LLM_SYS = (
    "You are a strict intent parser for file uploads.\n"
    "Return JSON matching the given schema. Do not add extra keys or commentary.\n"
    "provider must be one of: sharepoint, onedrive, google_drive, dropbox, box, mega, or null.\n"
    "folder must contain only the folder/path, without 'SharePoint:' or provider names.\n"
    "Set direct=true if they want default location (no folder). Set cancel/confirm appropriately.\n"
    "Recognize synonyms: SP->sharepoint, OD->onedrive, G Drive/GDrive->google_drive.\n"
    "If unsure, leave fields null/false."
)

def _llm_parse_upload_utterance(text: str) -> dict | None:
    """Ask local Qwen2 via Ollama to parse an utterance. Returns dict or None."""
    if not (_llm_available() and _LC_OK):
        return None
    txt = (text or "").strip()
    if not txt or len(txt) > 800:
        return None
    try:
        chat = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=NUM_PREDICT,
            context_length=NUM_CTX,
            verbose=DEBUG,
        )
        structured = chat.with_structured_output(IntentSchema)
        result: IntentSchema = structured.invoke([
            SystemMessage(content=_LLM_SYS),
            HumanMessage(content=txt),
        ])
        obj = result.dict()
        provider = normalize_provider(obj.get("provider"))
        folder = (obj.get("folder") or "").strip().strip("/") or None
        if folder:
            folder = re.sub(rf'^(?:{PROVIDER_WORDS_RX})\s*[:/\- ,]+', "", folder, flags=re.I).strip().strip("/")
            if re.search(r'\b(upload|this\s*file|these\s*files)\b', folder, re.I):
                folder = None
        return {
            "provider": provider,
            "folder": folder,
            "direct": bool(obj.get("direct")),
            "cancel": bool(obj.get("cancel")),
            "confirm": bool(obj.get("confirm")),
        }
    except Exception as e:
        log.debug("LLM parse failed: %s", e)
        return None

# ======================
# Flow v2: folder-first
# ======================

def start_best_upload_flow(
    *,
    user_id: int,
    attachments: List[Dict],
    ms_token: Optional[str],
    user_text_hint: str | None = None,
    connected_providers: Optional[List[str]] = None,  # accepts override from app.py
) -> Dict:
    """
    Kick off the folder-first upload flow.
    Returns:
      {"status":"ask", "prompt": "..."}
      {"status":"confirm","prompt":"...","targets":[...],"attachments":[...]}
      {"status":"ready","targets":[...],"attachments":[...]}
    """
    text = (user_text_hint or "").strip()

    # Parse provider hint ONLY when explicitly mentioned (prevents hallucinated providers)
    prov_hint = None
    m = re.search(rf"\b(?:to|in|into|onto|on)\s+{PROVIDER_WORDS_RX}\b", text, re.I)
    if m:
        prov_hint = normalize_provider(m.group(1))

    # Parse folder hint (quoted or “on X folder” at the end)
    folder_hint = None
    m2 = re.search(r'\b(?:in|into|on|onto)\s+"([^"]+)"\s*$', text, re.I) \
         or re.search(r"\b(?:in|into|on|onto)\s+'([^']+)'\s*$", text, re.I)
    if m2:
        cand = (m2.group(1) or "").strip().strip("/")
        if not re.search(r'\b(upload|file|this\s+file|these\s+files)\b', cand, re.I):
            folder_hint = cand
    else:
        m3 = re.search(
            rf'\b(?:in|into|on|onto)\s+(?:(?:the|a)\s+)?(?:(?:{PROVIDER_WORDS_RX})\s*[:>\- ]+)?(.+?)\s+(?:folder|directory)\b',
            text, re.I
        )
        if m3:
            cand = (m3.group(m3.lastindex) or "").strip().strip("/")
            cand = re.sub(r'\b(folder|directory)$', '', cand, flags=re.I).strip(" /")
            if cand and not re.search(r'\b(upload|file)\b', cand, re.I):
                folder_hint = cand

    # (Optional) LLM hint — but ignore LLM provider unless user text contains a provider word
    if not folder_hint and text:
        parsed = _llm_parse_upload_utterance(text)
        if parsed:
            if re.search(rf'\b{PROVIDER_WORDS_RX}\b', text, re.I) and not prov_hint and parsed.get("provider"):
                prov_hint = parsed["provider"]
            if parsed.get("cancel"):
                return {"status": "canceled", "prompt": "Ok, I have canceled your upload request."}
            if parsed.get("direct"):
                _UPLOAD_FLOWS.setdefault(user_id, {})["__llm_suggest_direct"] = True
            if parsed.get("folder"):
                folder_hint = parsed["folder"]

    _UPLOAD_FLOWS[user_id] = {
        "awaiting_folder": True,
        "awaiting_confirm": False,
        "attachments": attachments or [],
        "explicit_provider": prov_hint,
        "targets": [],
        "override_connected": connected_providers or None,
    }

    if folder_hint:
        return handle_best_upload_reply(user_id=user_id, text=folder_hint, ms_token=ms_token)

    choices = _get_connected(user_id, ms_token, connected_providers)
    if prov_hint:
        return {
            "status": "ask",
            "prompt": (
                f"Will I upload directly to **{provider_label(prov_hint)}** or into a specific folder there?\n"
                "• Say **direct** to upload to the default location.\n"
                "• Or tell me the folder name (e.g., Website Backup)."
            ),
        }
    if choices:
        return {
            "status": "ask",
            "prompt": (
                "Do you want to upload **directly** or into a **specific folder**?\n"
                "• Say **direct** to upload to the default location of your connected drives.\n"
                "• Or reply with a folder name (e.g., *Website Backup*). You can also say `in SharePoint: Website Backup`.\n"
                f"Connected: {_choices_sentence(choices)}."
            ),
        }
    return {
        "status": "ask",
        "prompt": (
            "Which folder should I upload to? You can say `in SharePoint: Website Backup` (or OneDrive, Google Drive, Dropbox, Box, MEGA). "
            "If no storages are connected yet, connect one first."
        ),
    }

def _confirm_prompt(targets: List[Dict]) -> str:
    lines = []
    for t in targets:
        nice = provider_label(t["provider"])
        drive = t.get("drive_name") or "(library)"
        rel   = t.get("dest_path") or "/"
        if t["provider"] in ("sharepoint","onedrive"):
            lines.append(f"• {nice} → {drive} / {rel}")
        else:
            lines.append(f"• {nice} → {rel}")
    return (
        "I found the destination. Do you want me to upload there?\n"
        + "\n".join(lines)
        + "\nReply **yes** to upload or **no** to cancel."
    )

def handle_best_upload_reply(
    *,
    user_id: int,
    text: str,
    ms_token: Optional[str],
    connected_providers: Optional[List[str]] = None,
) -> Dict:
    """
    Continue the folder-first flow with the user's reply.
    Returns:
      {"status":"canceled","prompt":"..."}
      {"status":"ask","prompt":"..."}
      {"status":"confirm","prompt":"...","targets":[...],"attachments":[...]}
      {"status":"ready","targets":[...],"attachments":[...]}
      {"status":"none"}
    """
    flow = _UPLOAD_FLOWS.get(user_id)
    if not flow:
        return {"status": "none"}

    if connected_providers:
        flow["override_connected"] = connected_providers

    msg = (text or "").strip()

    # Cancel at any time
    if _CANCEL_RX.match(msg):
        cancel_upload_flow(user_id)
        return {"status": "canceled", "prompt": "Ok, I have canceled your upload request."}

    # Confirmation stage
    if flow.get("awaiting_confirm"):
        if _YES_RX.match(msg):
            flow["awaiting_confirm"] = False
            targets = flow.get("targets") or []
            atts = flow.get("attachments") or []
            _UPLOAD_FLOWS.pop(user_id, None)
            return {"status": "ready", "targets": targets, "attachments": atts}
        cancel_upload_flow(user_id)
        return {"status": "canceled", "prompt": "Okay, I won’t upload the file."}

    # Folder/direct stage
    if not flow.get("awaiting_folder"):
        return {"status": "none"}

    override_list = flow.get("override_connected")
    allowed_connected = _get_connected(user_id, ms_token, override_list)

    # 'direct' → default destinations
    if _DIRECT_RX.match(msg) or flow.get("__llm_suggest_direct"):
        prov = flow.get("explicit_provider")
        allowed = [prov] if prov else allowed_connected
        if not allowed:
            return {"status": "ask", "prompt": "I couldn't find any connected cloud storage. Connect SharePoint/OneDrive, Google Drive, Dropbox, Box, or MEGA first."}
        targets = []
        for p in allowed:
            t = {"provider": p, "dest_path": _default_dest_for(p)}
            if p == "sharepoint":
                t["site_host"], t["site_path"] = SP_SITE_HOST, SP_SITE_PATH
            targets.append(t)
        flow["awaiting_folder"] = False
        flow["awaiting_confirm"] = True
        flow["targets"] = targets
        flow.pop("__llm_suggest_direct", None)
        return {"status":"confirm","prompt":_confirm_prompt(targets),"targets":targets,"attachments":flow.get("attachments") or []}

    # --- LLM hint pass (optional) ---
    prov = flow.get("explicit_provider")
    parsed = _llm_parse_upload_utterance(msg)
    if parsed:
        # Ignore LLM provider unless user explicitly wrote a provider word in this reply
        if parsed.get("provider") and not re.search(rf'\b{PROVIDER_WORDS_RX}\b', msg, re.I):
            parsed["provider"] = None

        if parsed.get("provider") and not prov:
            prov = parsed["provider"]
        if parsed.get("cancel"):
            cancel_upload_flow(user_id)
            return {"status": "canceled", "prompt": "Ok, I have canceled your upload request."}
        if parsed.get("direct"):
            msg = "direct"  # reuse the 'direct' branch above
        elif parsed.get("folder"):
            folder = parsed["folder"]
            allowed = [prov] if prov else allowed_connected
            if not allowed:
                return {"status":"ask","prompt":"I couldn't find any connected cloud storage. Connect SharePoint/OneDrive, Google Drive, Dropbox, Box, or MEGA first."}
            found_targets: List[Dict] = []
            for p in allowed:
                try:
                    path_info = _find_existing_folder(provider=p, user_id=user_id, ms_token=ms_token, folder_name=folder)
                    if path_info:
                        t = {"provider": p, "dest_path": path_info["dest_path"]}
                        if "site_host"  in path_info: t["site_host"]  = path_info["site_host"]
                        if "site_path"  in path_info: t["site_path"]  = path_info["site_path"]
                        if "drive_id"   in path_info: t["drive_id"]   = path_info["drive_id"]
                        if "drive_name" in path_info: t["drive_name"] = path_info["drive_name"]
                        found_targets.append(t)
                except Exception as e:
                    log.warning("Folder search failed on %s: %s", p, e)
            if found_targets:
                flow["awaiting_folder"] = False
                flow["awaiting_confirm"] = True
                flow["targets"] = found_targets
                return {"status":"confirm","prompt":_confirm_prompt(found_targets),"targets":found_targets,"attachments":flow.get("attachments") or []}
        # fall through to regex if unresolved

    # --- tolerant provider + folder parsing (regex fallback) ---
    mprov = re.search(rf'\b(?:in|on|to|into|onto)\s*:?\s*{PROVIDER_WORDS_RX}\b', msg, re.I)
    if mprov:
        prov = normalize_provider(mprov.group(1))

    # Extract folder name
    mname = re.search(r'"([^"]+)"', msg) or re.search(r"'([^']+)'", msg)
    if mname:
        folder = (mname.group(1) or "").strip().strip("/")
    else:
        m_on_folder = re.search(
            rf'\b(?:in|into|on|onto)\s+(?:(?:the|a)\s+)?(?:(?:{PROVIDER_WORDS_RX})\s*[:>\- ]+)?(.+?)\s+(?:folder|directory)\b',
            msg, re.I
        )
        if m_on_folder:
            folder = (m_on_folder.group(m_on_folder.lastindex) or "").strip().strip("/")
            folder = re.sub(r'\b(folder|directory)$', '', folder, flags=re.I).strip(" /")
        else:
            folder = re.sub(r'^\s*(?:i\s*want\s*to\s*upload\s*(?:this\s*file|these\s*files)\s*)?(in|on|to|into|onto)\s*:?', "", msg, flags=re.I).strip()
            folder = re.sub(rf'^(?:{PROVIDER_WORDS_RX})\s*[:/\- ,]+', "", folder, flags=re.I).strip()
            folder = re.sub(r'\b(folder|directory)\b', '', folder, flags=re.I).strip(" /")
        if re.search(r'\b(upload|file|these\s*files|this\s*file)\b', folder, re.I):
            folder = ""

    if not folder:
        if prov:
            return {"status":"ask","prompt":f"Will I upload directly to **{provider_label(prov)}** or into a specific folder there? Say **direct** or give the folder name."}
        return {"status":"ask","prompt":"Please say **direct** to upload now, or provide a folder name (e.g., Website Backup). You can also say `in SharePoint: Website Backup`."}

    allowed = [prov] if prov else allowed_connected
    if not allowed:
        return {"status":"ask","prompt":"I couldn't find any connected cloud storage. Connect SharePoint/OneDrive, Google Drive, Dropbox, Box, or MEGA first."}

    found_targets: List[Dict] = []
    for p in allowed:
        try:
            path_info = _find_existing_folder(provider=p, user_id=user_id, ms_token=ms_token, folder_name=folder)
            if path_info:
                t = {"provider": p, "dest_path": path_info["dest_path"]}
                if "site_host"  in path_info: t["site_host"]  = path_info["site_host"]
                if "site_path"  in path_info: t["site_path"]  = path_info["site_path"]
                if "drive_id"   in path_info: t["drive_id"]   = path_info["drive_id"]
                if "drive_name" in path_info: t["drive_name"] = path_info["drive_name"]
                found_targets.append(t)
        except Exception as e:
            log.warning("Folder search failed on %s: %s", p, e)

    if not found_targets:
        if prov:
            return {"status":"ask","prompt":f"I couldn't find a folder named **{folder}** in **{provider_label(prov)}**. Please provide the correct folder name or say **direct**."}
    else:
        flow["awaiting_folder"] = False
        flow["awaiting_confirm"] = True
        flow["targets"] = found_targets
        return {"status":"confirm","prompt":_confirm_prompt(found_targets),"targets":found_targets,"attachments":flow.get("attachments") or []}

    return {"status":"ask","prompt":f"I couldn't find a folder named **{folder}** in any connected storage. Please provide the correct folder name or say **direct**."}

# Backwards-compat wrappers
def start_upload_flow(*, user_id: int, attachments: List[Dict], ms_token: Optional[str], user_text_hint: str | None = None, connected_providers: Optional[List[str]] = None) -> Dict:
    return start_best_upload_flow(
        user_id=user_id,
        attachments=attachments,
        ms_token=ms_token,
        user_text_hint=user_text_hint,
        connected_providers=connected_providers,
    )

def handle_upload_flow_reply(*, user_id: int, text: str, ms_token: Optional[str], connected_providers: Optional[List[str]] = None) -> Dict:
    return handle_best_upload_reply(user_id=user_id, text=text, ms_token=ms_token, connected_providers=connected_providers)

def cancel_upload_flow(user_id: int) -> None:
    _UPLOAD_FLOWS.pop(user_id, None)

# ================= MS Graph utils (SharePoint/OneDrive) =================
def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def ms_pick_any_site(token: str) -> Tuple[str, str]:
    # 1) Followed sites
    try:
        r = requests.get(f"{GRAPH}/me/followedSites", headers=_h(token), timeout=20)
        if r.ok:
            arr = r.json().get("value", [])
            if arr:
                web = arr[0].get("webUrl")
                if web:
                    u = urlparse(web)
                    return u.netloc, u.path
    except Exception as e:
        log.warning("followedSites lookup failed: %s", e)
    # 2) Broad search
    r = requests.get(f"{GRAPH}/sites?search=*", headers=_h(token), timeout=20)
    r.raise_for_status()
    items = r.json().get("value", [])
    if not items:
        raise RuntimeError("No SharePoint sites found for this account.")
    web = items[0].get("webUrl")
    u = urlparse(web)
    return u.netloc, u.path

def ms_resolve_sharepoint_site(token: str, site_host: str, site_path: str) -> dict:
    r = requests.get(f"{GRAPH}/sites/{site_host}:{site_path}", headers=_h(token), timeout=20)
    r.raise_for_status()
    site = r.json()
    site_id = site["id"]
    r2 = requests.get(f"{GRAPH}/sites/{site_id}/drives", headers=_h(token), timeout=20)
    r2.raise_for_status()
    drives = r2.json().get("value", [])
    drive = next((d for d in drives if d.get("name") == DEFAULT_LIBRARY), None) or (drives[0] if drives else None)
    if not drive:
        raise RuntimeError("No document library found on the SharePoint site.")
    return {"siteId": site_id, "driveId": drive["id"], "webUrl": site.get("webUrl")}

def ms_ensure_folder_path(token: str, drive_id: str, folder_path: str) -> dict:
    path = (folder_path or "").strip().strip("/")
    if not path:
        r = requests.get(f"{GRAPH}/drives/{drive_id}/root", headers=_h(token), timeout=20)
        r.raise_for_status()
        return r.json()
    parent = "/"
    item = None
    for seg in [p for p in path.split("/") if p and p != "."]:
        get_url = f"{GRAPH}/drives/{drive_id}/root:{parent}{quote(seg)}"
        r = requests.get(get_url, headers=_h(token), timeout=20)
        if r.status_code == 404:
            create_url = f"{GRAPH}/drives/{drive_id}/root:{parent}:/children"
            payload = {"name": seg, "folder": {}, "@microsoft.graph.conflictBehavior": "replace"}
            r2 = requests.post(create_url, headers={**_h(token), "Content-Type": "application/json"}, data=json.dumps(payload), timeout=20)
            r2.raise_for_status()
            item = r2.json()
        else:
            r.raise_for_status()
            item = r.json()
        parent = f"{parent}{seg}/"
    return item

def onedrive_default_drive(token: str) -> dict:
    r = requests.get(f"{GRAPH}/me/drive", headers=_h(token), timeout=20)
    r.raise_for_status()
    return r.json()

def ms_put_small(token: str, drive_id: str, dest_folder: str, name: str, data: bytes) -> dict:
    if dest_folder:
        url = f"{GRAPH}/drives/{drive_id}/root:/{quote(dest_folder.strip('/'))}/{quote(name)}:/content"
    else:
        url = f"{GRAPH}/drives/{drive_id}/root:/{quote(name)}:/content"
    r = requests.put(url, headers=_h(token), data=data, timeout=180)
    r.raise_for_status()
    return r.json()

def ms_upload_large(token: str, drive_id: str, dest_folder: str, name: str, file_path: str, chunk: int = 5*1024*1024) -> dict:
    if dest_folder:
        url = f"{GRAPH}/drives/{drive_id}/root:/{quote(dest_folder.strip('/'))}/{quote(name)}:/createUploadSession"
    else:
        url = f"{GRAPH}/drives/{drive_id}/root:/{quote(name)}:/createUploadSession"
    r = requests.post(url, headers={**_h(token), "Content-Type": "application/json"}, data="{}", timeout=20)
    r.raise_for_status()
    up = r.json().get("uploadUrl")
    size = os.path.getsize(file_path)
    with open(file_path, "rb") as fh:
        start = 0
        while start < size:
            chunk_bytes = fh.read(chunk)
            end = start + len(chunk_bytes) - 1
            rr = requests.put(
                up,
                headers={"Content-Length": str(len(chunk_bytes)), "Content-Range": f"bytes {start}-{end}/{size}"},
                data=chunk_bytes,
                timeout=300,
            )
            if rr.status_code in (200, 201):
                return rr.json()
            if rr.status_code != 202:
                raise RuntimeError(f"MS large upload failed: {rr.status_code} {rr.text}")
            start = end + 1
    raise RuntimeError("MS upload session ended unexpectedly.")

def upload_sharepoint(token: str, *, site_host: str, site_path: str, library_path: str, local_path: str, display_name: str | None=None) -> dict:
    """Fallback uploader when we DO NOT have a specific drive_id.
    `library_path` is treated as a FOLDER-ONLY path relative to the default library."""
    info = ms_resolve_sharepoint_site(token, site_host, site_path)
    drive_id = info["driveId"]
    parts = [p for p in library_path.strip("/").split("/") if p]
    if parts and parts[0].lower() == DEFAULT_LIBRARY.lower():
        parts = parts[1:]
    # FIX: treat as folder-only (do NOT drop the last segment)
    folder = "/".join(parts) if parts else ""
    if folder:
        ms_ensure_folder_path(token, drive_id, folder)
    name = display_name or os.path.basename(local_path)
    size = os.path.getsize(local_path)
    if size <= 4*1024*1024:
        with open(local_path, "rb") as fh:
            return ms_put_small(token, drive_id, folder, name, fh.read())
    return ms_upload_large(token, drive_id, folder, name, local_path)

# upload directly by a specific drive (library) ID
def upload_sharepoint_by_drive_id(token: str, *, drive_id: str, library_path: str, local_path: str, display_name: str | None=None) -> dict:
    folder = "/".join([p for p in (library_path or "").strip("/").split("/") if p]) if library_path else ""
    if folder:
        ms_ensure_folder_path(token, drive_id, folder)
    name = display_name or os.path.basename(local_path)
    size = os.path.getsize(local_path)
    if size <= 4*1024*1024:
        with open(local_path, "rb") as fh:
            return ms_put_small(token, drive_id, folder, name, fh.read())
    return ms_upload_large(token, drive_id, folder, name, local_path)

def upload_onedrive(token: str, *, dest_path: str, local_path: str, display_name: str | None=None) -> dict:
    drv = onedrive_default_drive(token)
    drive_id = drv["id"]
    folder = "/".join([p for p in dest_path.strip("/").split("/") if p]) if dest_path else ""
    if folder:
        ms_ensure_folder_path(token, drive_id, folder)
    name = display_name or os.path.basename(local_path)
    size = os.path.getsize(local_path)
    if size <= 4*1024*1024:
        with open(local_path, "rb") as fh:
            return ms_put_small(token, drive_id, folder, name, fh.read())
    return ms_upload_large(token, drive_id, folder, name, local_path)

# ---- *Find existing* folder paths (no creation during search) ----
def _ms_search_folder_path(token: str, drive_id: str, name: str) -> Optional[str]:
    """Search a folder path by name, scoring results: exact > startswith > contains."""
    q = (name or "").strip()
    if not q:
        return None
    r = requests.get(f"{GRAPH}/drives/{drive_id}/root/search(q='{quote(q)}')", headers=_h(token), timeout=20)
    if not r.ok:
        return None

    items = r.json().get("value", []) or []
    candidates: List[Tuple[int, str]] = []
    low_q = q.lower()

    for it in items:
        if "folder" not in it:
            continue
        parent_path = (it.get("parentReference", {}) or {}).get("path") or "/drive/root:"
        prefix = "/drive/root:"
        if parent_path.startswith(prefix):
            parent_path = parent_path[len(prefix):]
        parent_path = parent_path.strip("/")
        full = f"{parent_path}/{it.get('name')}".strip("/")
        nm = (it.get("name") or "").strip()
        low_nm = nm.lower()

        if low_nm == low_q: score = 3
        elif low_nm.startswith(low_q): score = 2
        elif low_q in low_nm: score = 1
        else: score = 0

        if score > 0:
            candidates.append((score, full))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (-t[0], len(t[1])))
    return candidates[0][1]

# ========= Site & drive discovery across tenant =========
def _ms_site_from_weburl(token: str, web_url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{GRAPH}/sites/{quote(web_url)}", headers=_h(token), timeout=20)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None

def _ms_candidate_sites(token: str, prefer_host: Optional[str], prefer_path: Optional[str], search_text: Optional[str]) -> List[dict]:
    """Priority-ordered candidate sites:
       configured site > followed sites > sites title-matching search_text > tenant-wide capped list."""
    seen = set()
    out: List[dict] = []

    # 1) Configured site
    if prefer_host and prefer_path:
        try:
            info = ms_resolve_sharepoint_site(token, prefer_host, prefer_path)
            site_id = info.get("siteId")
            if site_id and site_id not in seen:
                out.append({"id": site_id, "webUrl": f"https://{prefer_host}{prefer_path}"})
                seen.add(site_id)
        except Exception:
            pass

    # 2) Followed sites
    try:
        r = requests.get(f"{GRAPH}/me/followedSites", headers=_h(token), timeout=20)
        if r.ok:
            for s in r.json().get("value", []):
                web = s.get("webUrl")
                if not web:
                    continue
                site = _ms_site_from_weburl(token, web)
                if site and site.get("id") not in seen:
                    out.append({"id": site["id"], "webUrl": site.get("webUrl")})
                    seen.add(site["id"])
    except Exception:
        pass

    # 3) Targeted title search
    if search_text:
        try:
            r = requests.get(f"{GRAPH}/sites?search={quote(search_text)}", headers=_h(token), timeout=20)
            if r.ok:
                for s in r.json().get("value", []):
                    sid = s.get("id")
                    if sid and sid not in seen:
                        out.append({"id": sid, "webUrl": s.get("webUrl")})
                        seen.add(sid)
        except Exception:
            pass

    # 4) Broad fallback: tenant-wide (capped)
    if not out:
        try:
            r = requests.get(f"{GRAPH}/sites?search=*", headers=_h(token), timeout=20)
            if r.ok:
                for s in (r.json().get("value") or [])[:25]:
                    sid = s.get("id")
                    if sid and sid not in seen:
                        out.append({"id": sid, "webUrl": s.get("webUrl")})
                        seen.add(sid)
        except Exception:
            pass

    return out

def ms_find_folder_anywhere(token: str, folder_name: str) -> Optional[Dict]:
    """
    Look for a folder OR a document library named `folder_name` across:
      - configured site (SP_SITE_HOST/PATH),
      - followed sites,
      - search-matched sites,
      - tenant-wide capped list.
    Returns: {"site_host","site_path","drive_id","drive_name","dest_path"} or None.
    """
    q = (folder_name or "").strip().strip("/")
    if not q:
        return None

    sites = _ms_candidate_sites(token, SP_SITE_HOST, SP_SITE_PATH, q)
    if not sites:
        try:
            host, path = ms_pick_any_site(token)
            info = ms_resolve_sharepoint_site(token, host, path)
            sites = [{"id": info["siteId"], "webUrl": info.get("webUrl")}]
        except Exception:
            sites = []

    for s in sites:
        sid = s["id"]
        web = s.get("webUrl") or ""
        host = urlparse(web).netloc if web else SP_SITE_HOST
        path = urlparse(web).path if web else SP_SITE_PATH

        try:
            r = requests.get(f"{GRAPH}/sites/{sid}/drives", headers=_h(token), timeout=20)
            if not r.ok:
                continue
            drives = r.json().get("value", []) or []
        except Exception:
            continue

        low_q = q.lower()
        # library (drive) name match → upload to library root
        for d in drives:
            if (d.get("name") or "").lower() == low_q:
                return {"site_host": host, "site_path": path, "drive_id": d["id"], "drive_name": d.get("name"), "dest_path": ""}

        # folder inside each drive
        for d in drives:
            try:
                dest = _ms_search_folder_path(token, d["id"], q)
                if dest:
                    return {"site_host": host, "site_path": path, "drive_id": d["id"], "drive_name": d.get("name"), "dest_path": dest}
            except Exception:
                continue

    return None

def _find_existing_folder(*, provider: str, user_id: int, ms_token: Optional[str], folder_name: str) -> Optional[Dict]:
    p = provider
    name = folder_name.strip().strip("/")
    if not name:
        return None

    if p in ("onedrive", "sharepoint"):
        if not ms_token:
            return None
        if p == "onedrive":
            drv = onedrive_default_drive(ms_token)
            path = _ms_search_folder_path(ms_token, drv["id"], name)
            return {"dest_path": path} if path else None

        # Search any site + any library, and return drive metadata
        found = ms_find_folder_anywhere(ms_token, name)
        if not found:
            return None
        # Keep dest_path RELATIVE to the matched drive root
        return {
            "dest_path": found["dest_path"] or "",
            "site_host": found["site_host"],
            "site_path": found["site_path"],
            "drive_id":  found["drive_id"],
            "drive_name": found.get("drive_name"),
        }

    if p == "google_drive":
        token = _ensure_google(user_id)
        if not token:
            return None
        headers = {"Authorization": f"Bearer {token}"}
        escaped = name.replace("'", "\\'")
        q = (
            f"name = '{escaped}' "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and trashed = false"
        )
        r = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            headers=headers,
            params={
                "q": q,
                "fields": "files(id,name,parents)",
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
                "spaces": "drive",
                "pageSize": 1,
            },
            timeout=20,
        )
        if not r.ok or not r.json().get("files"):
            return None
        return {"dest_path": name}

    if p == "dropbox":
        token = _ensure_dropbox(user_id)
        if not _is_valid_dropbox_token(token):
            return None
        body = {
            "query": name,
            "options": {"filename_only": True, "file_status": "active", "max_results": 20}
        }
        r = requests.post(
            "https://api.dropboxapi.com/2/files/search_v2",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=20,
        )
        if not r.ok:
            return None
        matches = (r.json() or {}).get("matches", [])
        for m in matches:
            md = (((m or {}).get("metadata") or {}).get("metadata") or {})
            if md.get(".tag") == "folder" and md.get("name") == name:
                return {"dest_path": md.get("path_lower") or f"/{name}"}
        return None

    if p == "box":
        token = _ensure_box(user_id)
        if not token:
            return None
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(
            "https://api.box.com/2.0/search",
            headers=headers,
            params={"query": name, "type": "folder", "limit": 5, "content_types": "name"},
            timeout=20,
        )
        if not r.ok:
            return None
        entries = (r.json() or {}).get("entries", [])
        for e in entries:
            if e.get("type") == "folder" and e.get("name") == name:
                fid = e.get("id")
                rr = requests.get(f"https://api.box.com/2.0/folders/{fid}?fields=path_collection,name", headers=headers, timeout=20)
                if not rr.ok:
                    continue
                pc = (rr.json() or {}).get("path_collection", {}).get("entries", [])
                segs = [s["name"] for s in pc if s.get("name") and s.get("name") != "All Files"]
                segs.append(name)
                return {"dest_path": "/".join(segs)}
        return None

    if p == "mega":
        try:
            from mega import Mega
        except Exception:
            return None
        email = os.getenv("MEGA_EMAIL"); password = os.getenv("MEGA_PASSWORD")
        if not email or not password:
            return None
        m = Mega().login(email, password)
        node = m.find(name)
        if node:
            return {"dest_path": f"/{name}"}
        return None

    return None

# =================
# Google Drive (v3)
# =================
def upload_google_drive(user_id: int, *, dest_path: str, local_path: str, display_name: str | None=None) -> dict:
    token = _ensure_google(user_id)
    if not token:
        raise RuntimeError("Google Drive not connected.")
    headers = {"Authorization": f"Bearer {token}"}

    def _find_or_create(parent_id: str, name: str) -> str:
        safe_name = name.replace("'", "\\'")
        q = (
            f"name = '{safe_name}' "
            f"and '{parent_id}' in parents "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )
        r = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            headers=headers,
            params={
                "q": q,
                "fields": "files(id,name)",
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
                "spaces": "drive",
            },
            timeout=20,
        )
        r.raise_for_status()
        arr = r.json().get("files", [])
        if arr:
            return arr[0]["id"]

        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
        r2 = requests.post(
            "https://www.googleapis.com/drive/v3/files",
            headers={**headers, "Content-Type": "application/json"},
            params={"fields": "id", "supportsAllDrives": "true"},
            data=json.dumps(meta),
            timeout=20,
        )
        r2.raise_for_status()
        return r2.json()["id"]

    parent = "root"
    for seg in [p for p in dest_path.strip("/").split("/") if p]:
        parent = _find_or_create(parent, seg)

    name = display_name or os.path.basename(local_path)
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    meta = {"name": name, "parents": [parent]}

    with open(local_path, "rb") as fh:
        files = {
            "metadata": ("metadata.json", json.dumps(meta), "application/json; charset=UTF-8"),
            "file": (name, fh, mime),
        }
        r = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,webViewLink,webContentLink",
            headers=headers,
            files=files,
            timeout=300,
        )
    r.raise_for_status()
    out = r.json()
    out["webUrl"] = f"https://drive.google.com/file/d/{out['id']}/view"
    return out

# =========
# Dropbox
# =========
def _raise_dropbox_http(r, path_for_msg):
    # Try to distinguish scope vs expiry
    try:
        j = r.json()
    except Exception:
        j = {}

    if r.status_code == 401:
        summary = (j or {}).get("error_summary", "").lower()
        if "insufficient_scope" in summary:
            raise RuntimeError("Dropbox is connected with read-only access. Please reconnect and allow file uploads.")
        raise RuntimeError("Dropbox token expired or invalid — please reconnect Dropbox.")

    raise RuntimeError(f"Dropbox upload error ({r.status_code}) for {path_for_msg}: {j or r.text}")


def upload_dropbox(user_id: int, *, dest_path: str, local_path: str, display_name: str | None=None, chunk: int = 40*1024*1024) -> dict:
    token = _ensure_dropbox(user_id)
    if not _is_valid_dropbox_token(token):
        raise RuntimeError("Dropbox not connected or token invalid — please reconnect Dropbox.")
    name = display_name or os.path.basename(local_path)
    if not dest_path.startswith("/"):
        dest_path = "/" + dest_path
    dropbox_path = f"{dest_path.rstrip('/')}/{name}"
    size = os.path.getsize(local_path)

    if size <= chunk:
        with open(local_path, "rb") as fh:
            r = requests.post(
                "https://content.dropboxapi.com/2/files/upload",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Dropbox-API-Arg": json.dumps({"path": dropbox_path, "mode": "add", "autorename": True, "mute": False}),
                    "Content-Type": "application/octet-stream",
                },
                data=fh.read(),
                timeout=300,
            )
        if not r.ok:
            _raise_dropbox_http(r, dropbox_path)
        out = r.json()
        out["webUrl"] = out.get("preview_url") or ""
        return out

    with open(local_path, "rb") as fh:
        r = requests.post(
            "https://content.dropboxapi.com/2/files/upload_session/start",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream", "Dropbox-API-Arg": json.dumps({"close": False})},
            data=fh.read(chunk),
            timeout=300,
        )
        if not r.ok:
            _raise_dropbox_http(r, dropbox_path)
        session_id = r.json()["session_id"]
        cursor = {"session_id": session_id, "offset": chunk}
        while True:
            data = fh.read(chunk)
            if not data:
                break
            cursor["offset"] += len(data)
            rr = requests.post(
                "https://content.dropboxapi.com/2/files/upload_session/append_v2",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream", "Dropbox-API-Arg": json.dumps({"cursor": cursor, "close": False})},
                data=data,
                timeout=300,
            )
            if not rr.ok:
                _raise_dropbox_http(rr, dropbox_path)

        finish = {"cursor": cursor, "commit": {"path": dropbox_path, "mode": "add", "autorename": True, "mute": False}}
        r2 = requests.post(
            "https://content.dropboxapi.com/2/files/upload_session/finish",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream", "Dropbox-API-Arg": json.dumps(finish)},
            data=b"",
            timeout=300,
        )
        if not r2.ok:
            _raise_dropbox_http(r2, dropbox_path)
        out = r2.json()
        out["webUrl"] = out.get("preview_url") or ""
        return out

# =====
# Box
# =====
def upload_box(user_id: int, *, dest_path: str, local_path: str, display_name: str | None=None) -> dict:
    token = _ensure_box(user_id)
    if not token:
        raise RuntimeError("Box not connected.")
    headers = {"Authorization": f"Bearer {token}"}

    def _get_child(parent_id: str, name: str) -> Optional[str]:
        r = requests.get(f"https://api.box.com/2.0/folders/{parent_id}/items?limit=1000", headers=headers, timeout=20)
        r.raise_for_status()
        for it in r.json().get("entries", []):
            if it.get("type") == "folder" and it.get("name") == name:
                return it.get("id")
        return None

    def _ensure_path(path: str) -> str:
        parent = "0"
        for seg in [p for p in path.strip("/").split("/") if p]:
            existing = _get_child(parent, seg)
            if existing:
                parent = existing
                continue
            r = requests.post(
                "https://api.box.com/2.0/folders",
                headers={**headers, "Content-Type": "application/json"},
                data=json.dumps({"name": seg, "parent": {"id": parent}}),
                timeout=20,
            )
            r.raise_for_status()
            parent = r.json()["id"]
        return parent

    parent_id = _ensure_path(dest_path)
    name = display_name or os.path.basename(local_path)
    files = {"file": (name, open(local_path, "rb"))}
    data = {"attributes": json.dumps({"name": name, "parent": {"id": parent_id}})}
    r = requests.post("https://upload.box.com/api/2.0/files/content", headers=headers, files=files, data=data, timeout=300)
    r.raise_for_status()
    out = r.json()["entries"][0]
    out["webUrl"] = f"https://app.box.com/file/{out['id']}"
    return out

# =====
# MEGA
# =====
def upload_mega(*, dest_path: str, local_path: str, display_name: str | None=None) -> dict:
    try:
        from mega import Mega
    except Exception:
        raise RuntimeError("MEGA SDK (mega.py) not installed on server.")
    email = os.getenv("MEGA_EMAIL"); password = os.getenv("MEGA_PASSWORD")
    if not email or not password:
        raise RuntimeError("MEGA credentials not configured (MEGA_EMAIL/MEGA_PASSWORD).")
    m = Mega().login(email, password)
    node = m.find(dest_path) or m.create_folder(dest_path.strip("/"))
    name = display_name or os.path.basename(local_path)
    up = m.upload(local_path, node[0] if isinstance(node, list) else node)
    link = m.get_upload_link(up)
    return {"name": name, "id": str(up), "webUrl": link}

# ==========================
# Unified multi-provider API
# ==========================
def upload_to_provider(
    *,
    provider: str,
    user_id: int,
    ms_token: Optional[str],
    site_host: Optional[str],
    site_path: Optional[str],
    dest_path: str,
    local_path: str,
    display_name: Optional[str] = None,
    drive_id: Optional[str] = None,  # honor SharePoint non-default libraries
) -> dict:
    p = (provider or "").lower().strip()

    if p == "sharepoint":
        if not ms_token:
            raise RuntimeError("Microsoft 365 not connected.")
        # If a specific drive was discovered, honor it
        if drive_id:
            # dest_path must be RELATIVE to the drive root
            dp = (dest_path or "").strip().strip("/")
            # Guard: if someone accidentally included the library name, strip it
            for lib in ("documents", "shared documents"):
                if dp.lower().startswith(lib + "/"):
                    dp = dp[len(lib)+1:]
            return upload_sharepoint_by_drive_id(ms_token, drive_id=drive_id, library_path=dp,
                                                 local_path=local_path, display_name=display_name)
        # else: default site/library route
        site_h = site_host or SP_SITE_HOST
        site_p = site_path or SP_SITE_PATH
        if not site_h or not site_p:
            site_h, site_p = ms_pick_any_site(ms_token)
        return upload_sharepoint(ms_token, site_host=site_h, site_path=site_p,
                                 library_path=dest_path, local_path=local_path, display_name=display_name)

    if p == "onedrive":
        if not ms_token:
            raise RuntimeError("Microsoft 365 not connected.")
        return upload_onedrive(ms_token, dest_path=dest_path, local_path=local_path, display_name=display_name)

    if p == "google_drive":
        return upload_google_drive(user_id, dest_path=dest_path, local_path=local_path, display_name=display_name)

    if p == "dropbox":
        return upload_dropbox(user_id, dest_path=dest_path, local_path=local_path, display_name=display_name)

    if p == "box":
        return upload_box(user_id, dest_path=dest_path, local_path=local_path, display_name=display_name)

    if p == "mega":
        return upload_mega(dest_path=dest_path, local_path=local_path, display_name=display_name)

    raise RuntimeError(f"Unknown provider: {provider}")

# ==========================
# Batch “best flow” executor
# ==========================
def perform_best_upload(
    *,
    targets: List[Dict],
    attachments: List[Dict],
    user_id: int,
    ms_token: Optional[str],
    upload_root: str,
) -> Tuple[Dict[str, List[Dict]], List[str]]:
    """
    Execute uploads to all resolved targets. Returns (uploaded_map, errors).
    Each target: {"provider","dest_path", ["site_host","site_path","drive_id","drive_name"]?}
    attachments: expect .name and .path/.url
    """
    def _resolve_local(att: Dict) -> Optional[str]:
        p = (att.get("path") or "").strip()
        if p and os.path.isabs(p) and os.path.exists(p):
            return p
        if p:
            maybe = os.path.join(upload_root, p.lstrip("/\\"))
            if os.path.exists(maybe):
                return maybe
        u = (att.get("url") or "").strip()
        if u.startswith("/uploads/"):
            maybe = os.path.join(upload_root, u[len("/uploads/"):].lstrip("/\\"))
            if os.path.exists(maybe):
                return maybe
        return None

    uploaded_map: Dict[str, List[Dict]] = {}
    errs: List[str] = []

    for att in attachments:
        local_path = _resolve_local(att)
        if not local_path:
            errs.append(f"Couldn't locate local file for **{att.get('name') or att.get('id')}**.")
            continue

        for t in targets:
            p = t["provider"]
            try:
                # Guard rail: if drive_id is present, enforce path relative to that drive
                if p == "sharepoint" and t.get("drive_id"):
                    dp = (t.get("dest_path") or "").strip().strip("/")
                    dn = (t.get("drive_name") or "").strip().lower()
                    if dp and dp.split("/", 1)[0].strip().lower() in (dn, "documents", "shared documents"):
                        dp = dp.split("/", 1)[1] if "/" in dp else ""
                    t = {**t, "dest_path": dp}

                res = upload_to_provider(
                    provider=p,
                    user_id=user_id,
                    ms_token=ms_token,
                    site_host=t.get("site_host"),
                    site_path=t.get("site_path"),
                    dest_path=t.get("dest_path"),
                    local_path=local_path,
                    display_name=att.get("name") or os.path.basename(local_path),
                    drive_id=t.get("drive_id"),
                )
                uploaded_map.setdefault(p, []).append({
                    "name": res.get("name") or os.path.basename(local_path),
                    "webUrl": res.get("webUrl"),
                    "id": res.get("id"),
                })
            except Exception as e:
                errs.append(f"{provider_label(p)}: **{att.get('name') or os.path.basename(local_path)}** → {e}")
    return uploaded_map, errs
