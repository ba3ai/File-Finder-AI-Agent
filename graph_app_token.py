# graph_app_token.py
import os
import requests

TENANT = os.getenv("MS_TENANT_ID")  # MUST be tenant GUID or domain; not 'common'/'organizations'
CLIENT_ID = os.getenv("MS_CLIENT_ID")
CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")

TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
SCOPE = "https://graph.microsoft.com/.default"  # app-only uses .default

def get_app_access_token() -> str:
    if not (TENANT and CLIENT_ID and CLIENT_SECRET):
        raise RuntimeError("MS_TENANT_ID / MS_CLIENT_ID / MS_CLIENT_SECRET not configured")

    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": SCOPE,
    }
    res = requests.post(TOKEN_URL, data=data, timeout=30)
    try:
        res.raise_for_status()
    except requests.HTTPError:
        # show AAD error body to logs to help debugging
        raise RuntimeError(f"Token request failed: {res.status_code} {res.text}") from None

    return res.json()["access_token"]
