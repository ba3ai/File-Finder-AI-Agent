import requests
import os
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from semantic_search import rank_files_with_llama_prompt
from perplexity_ranker import rank_files_rule_based
from msal_auth import load_token_cache, save_token_cache, build_msal_app
from extractor import extract_text_from_image
from llm_ranker import rank_files_with_llm_simple

logging.basicConfig(level=logging.INFO)

def refresh_token(account_id):
    cache = load_token_cache(account_id)
    app = build_msal_app(cache)
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(os.getenv("SCOPE").split(), account=accounts[0])
        if "access_token" in result:
            save_token_cache(account_id, cache)
            return result["access_token"]
    return None

def retry_request(url, headers, method="get", json=None, max_retries=2, account_id=None):
    backoff = 2
    for i in range(max_retries + 1):
        try:
            res = requests.request(method, url, headers=headers, json=json)
            if res.status_code == 401 and account_id:
                logging.warning("Received 401 Unauthorized. Attempting token refresh...")
                token = refresh_token(account_id)
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                    continue
            elif res.status_code == 429:
                raw = res.headers.get("Retry-After", "5")
                retry_after = 5
                try:
                    # If multiple values are present, take the first chunk
                    raw_first = raw.split(",")[0].strip()
                    retry_after = int(raw_first)
                except Exception:
                    # Retry-After may be an HTTP-date; fall back to a capped backoff
                    retry_after = min(backoff, 60)
                logging.warning(f"Rate limited on {url}. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                backoff = min(backoff * 2, 60)
                continue
            else:
                logging.info(f"Request to {url} returned status {res.status_code}")
                return res
        except Exception as e:
            logging.error(f"Request error on {url}: {e}")
    logging.error(f"Max retries exceeded for {url}")
    return res

def get_file_with_download_url(drive_id, item_id, token):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}"
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.json()
    else:
        logging.warning(f"⚠️ Failed to fetch full metadata for item {item_id}")
        return None

def get_user_email(account_id):
    token = refresh_token(account_id)
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    res = retry_request("https://graph.microsoft.com/v1.0/me", headers)
    if res.status_code == 200:
        return res.json().get("mail") or res.json().get("userPrincipalName")
    return None

def discover_all_sites(token):
    headers = {"Authorization": f"Bearer {token}"}
    sites = []
    url = "https://graph.microsoft.com/v1.0/sites?search=*"
    while url:
        res = retry_request(url, headers)
        if res.status_code == 200:
            data = res.json()
            sites.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        else:
            break
    return sites

def search_all_files(token, query,original_query=None):
    headers = {"Authorization": f"Bearer {token}"}
    all_results = []
    seen_ids = set()

    overall_start = time.time()
    print("🔍 [1] Starting file search...")
    print(f"Search Query: {query}")
    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    year = year_match.group() if year_match else None

    words = query.split()
    if year and year in words:
        words.remove(year)
        
    core = " ".join(words).strip().lower()
    query_batch = [core]

    # Search personal drive
    for q in query_batch:
        me_url = f"https://graph.microsoft.com/v1.0/me/drive/root/search(q='{q}')"
        me_res = retry_request(me_url, headers)
        if me_res.status_code == 200:
            for item in me_res.json().get("value", []):
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    all_results.append(item)

    # Discover and search SharePoint sites in parallel
    sites = discover_all_sites(token)

    def search_drive(drive_id, site_id, q):
        results = []
        search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/search(q='{q}')"
        search_res = retry_request(search_url, headers)
        if search_res.status_code == 200:
            for item in search_res.json().get("value", []):
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    item = tag_site_id([item], site_id)[0]
                    results.append(item)
        return results

    drive_tasks = []
    for site in sites:
        site_id = site.get("id")
        if not site_id:
            continue
        drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
        drives_res = retry_request(drives_url, headers)
        if drives_res.status_code != 200:
            continue
        for drive in drives_res.json().get("value", []):
            for q in query_batch:
                drive_tasks.append((drive["id"], site_id, q))

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(search_drive, drive_id, site_id, q) for drive_id, site_id, q in drive_tasks]
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logging.error(f"❌ Drive search error: {e}")

    if not all_results:
        logging.info("No results from batch search. Using recent files.")
        all_results = fetch_recent_files(token)

    print("⚙️ Enriching metadata in parallel...")
    def enrich(file):
        return get_file_with_download_url(file["parentReference"]["driveId"], file["id"], token)

    with ThreadPoolExecutor(max_workers=30) as executor:
        enriched = list(executor.map(enrich, all_results))

    all_results = [f for f in enriched if f and "folder" not in f]

    print("📄 [2] Processing file content...")
    for f in all_results:
        mime = f.get("file", {}).get("mimeType", "")
        url = f.get("@microsoft.graph.downloadUrl")
        if mime and url:
            if "image" in mime:
                f["extracted_text"] = extract_text_from_image(url)
            else:
                f["extracted_text"] = f"{f['name']} {f.get('webUrl', '')}"
        else:
            f["extracted_text"] = f"{f['name']} {f.get('webUrl', '')}"
    print(f"Total Files Found: {len(all_results)}")
    print("🤖 [3] Ranking files with LLaMA prompt...")
    # ranked_files = rank_files_rule_based(
    #     query=query,
    #     files=all_results,
    #     original_query=original_query
    # )
    ranked_files = rank_files_with_llm_simple(
        query=query,
        files=all_results,
        limit=12,
    )
    print(f"Files after ranking: {ranked_files}")
    total_time = time.time() - overall_start
    print(f"✅ Done. Total pipeline time: {total_time:.2f} seconds.")

    return ranked_files

def fetch_recent_files(token):
    headers = {"Authorization": f"Bearer {token}"}
    res = retry_request("https://graph.microsoft.com/v1.0/me/drive/recent", headers)
    if res.status_code == 200:
        return tag_site_id(res.json().get("value", []), "personal")
    return []

def tag_site_id(items, site_id):
    for item in items:
        if "parentReference" not in item:
            item["parentReference"] = {}
        item["parentReference"]["siteId"] = site_id
    return items

def check_file_access(token, item_id, user_email, site_id=None):
    if os.getenv("PERFORM_ACCESS_CHECK", "false").lower() != "true":
        return True
    headers = {"Authorization": f"Bearer {token}"}
    if site_id and site_id != "personal":
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{item_id}/permissions"
        try:
            res = retry_request(url, headers)
            if res.status_code == 200:
                return True
        except Exception as e:
            logging.warning(f"⚠️ SharePoint access check failed: {e}")
    return False

def send_notification_email(token, to_email, file_name, file_url):
    return send_email(token, to_email, f"Here is the file: {file_name}", f"<p><a href='{file_url}'>{file_name}</a></p>")

def send_multiple_file_email(token, to_email, files):
    links = "".join(f"<p><a href='{f['webUrl']}'>{f['name']}</a></p>" for f in files)
    return send_email(token, to_email, "Your requested files", f"<p>Here are the files you requested:</p>{links}")

def send_email(token, to_email, subject, html_content):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    message = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": html_content
            },
            "toRecipients": [{"emailAddress": {"address": to_email}}]
        },
        "saveToSentItems": True
    }

    try:
        res = retry_request(
            "https://graph.microsoft.com/v1.0/me/sendMail",
            headers,
            method="post",
            json=message
        )
        if res.status_code == 202:
            logging.info(f"✅ Email sent to {to_email}")
            return True
        else:
            logging.error(f"❌ Failed to send email to {to_email}: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        logging.error(f"Email send failed: {e}")
        return False
