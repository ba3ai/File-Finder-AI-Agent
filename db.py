# db.py
import os
import sqlite3
import threading
from datetime import datetime, timedelta

DB_NAME = os.getenv("CHAT_DB_PATH", "chat_history.db")

# Serialize writers to avoid "database is locked"
_WRITE_LOCK = threading.RLock()


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_conn():
    """
    Always create a fresh connection with sane pragmas for concurrency.
    - WAL allows readers while a writer is active
    - busy_timeout gives SQLite time to wait on a lock instead of erroring
    """
    conn = sqlite3.connect(
        DB_NAME,
        timeout=30,                 # wait up to 30s on locks
        check_same_thread=False,    # allow use across threads
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")  # 5s at SQLite level in addition to requests timeout
    return conn


def init_db():
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                user_message TEXT,
                ai_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_email, chat_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_ts ON chat_history(timestamp)")
        conn.commit()
    finally:
        conn.close()


def save_message(user_email, chat_id, user_message=None, ai_response=None):
    # SAFETY: never allow None/empty chat_id into DB
    if not chat_id:
        chat_id = str(int(datetime.now().timestamp()))

    with _WRITE_LOCK:
        conn = get_conn()
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO chat_history (user_email, chat_id, user_message, ai_response, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_email, chat_id, user_message, ai_response, _now_str()),
            )
            conn.commit()
        finally:
            conn.close()


# db.py  ── replace the entire get_user_chats function with this

def get_user_chats(user_email):
    conn = get_conn()
    try:
        c = conn.cursor()
        # Most recent chats first
        c.execute(
            """
            SELECT DISTINCT chat_id
            FROM chat_history
            WHERE user_email = ?
            ORDER BY timestamp DESC
            """,
            (user_email,),
        )
        chat_ids = [row[0] for row in c.fetchall()]

        results = []
        for chat_id in chat_ids:
            # Load all rows for this chat (ASC so first row is earliest)
            c.execute(
                """
                SELECT user_message, ai_response, timestamp, id
                FROM chat_history
                WHERE user_email = ? AND chat_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (user_email, chat_id),
            )
            rows = c.fetchall()

            # Title: prefer explicit [TITLE]..., else build from first timestamp
            title = None
            first_ts = None
            for user_msg, ai_resp, ts, _ in rows:
                if first_ts is None:
                    first_ts = ts
                if user_msg and user_msg.startswith("[TITLE]"):
                    title = user_msg[len("[TITLE]") :].strip()
                    break

            if not title:
                if first_ts:
                    try:
                        dt = datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S")
                        # Friendly UTC-ish label for old rows
                        title = f"Chat - {dt.strftime('%b %d, %Y – %H:%M UTC')}"
                    except Exception:
                        title = f"Chat - {chat_id}"
                else:
                    title = f"Chat - {chat_id}"

            # Preview: last AI message if present; otherwise last user message (not [TITLE])
            preview = None
            for user_msg, ai_resp, ts, _ in reversed(rows):
                if ai_resp:
                    preview = ai_resp.strip()
                    break
                if user_msg and not user_msg.startswith("[TITLE]"):
                    preview = user_msg.strip()
                    break

            if preview:
                # one-line, trimmed
                preview = " ".join(preview.splitlines())
                if len(preview) > 140:
                    preview = preview[:137] + "…"
            else:
                preview = ""  # let UI decide what to show when empty

            # Updated timestamp = last row time
            last_ts = rows[-1][2] if rows else None

            results.append(
                {
                    "id": chat_id,
                    "title": title,
                    "updated": last_ts,
                    "preview": preview,
                }
            )

        return results
    finally:
        conn.close()

def get_chat_messages(chat_id):
    """
    Return a flat list of (sender, message, timestamp)
    by expanding each DB row into up to two messages (user & AI).
    """
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(
            """
            SELECT user_email, user_message, ai_response, timestamp
            FROM chat_history
            WHERE chat_id = ?
            ORDER BY timestamp ASC, id ASC
            """,
            (chat_id,),
        )
        rows = c.fetchall()
        out = []
        for user_email, user_msg, ai_resp, ts in rows:
            if user_msg:
                out.append(("You", user_msg, ts))
            if ai_resp:
                out.append(("AI", ai_resp, ts))
        return out
    finally:
        conn.close()


def delete_old_messages(days=3):
    """
    Delete messages older than N days.
    Use a write-lock so this never races with an insert.
    """
    cutoff = datetime.now() - timedelta(days=days)
    with _WRITE_LOCK:
        conn = get_conn()
        try:
            c = conn.cursor()
            c.execute(
                "DELETE FROM chat_history WHERE timestamp < ?",
                (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            conn.commit()
        finally:
            conn.close()


def delete_old_chats(user_email, limit=None):
    """
    You currently enforce a 3-day retention in delete_old_messages(),
    so we don't need chat-count trimming. Keep this for API compatibility.
    """
    return
