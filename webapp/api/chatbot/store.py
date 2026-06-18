"""
webapp/api/chatbot/store.py
---------------------------
Conversation memory for the assistant. Persisted in the API's OWN auth.db
(AUTH_DB_PATH) — deliberately separate from the engine's person-identity DB, so
chat history never touches identity data (same rule the auth/audit tables follow).

Per session we keep:
  - the running ``active_filters`` (so follow-ups like "only camera 2" refine the
    previous search instead of starting over),
  - the ``last_results`` person-id list (so "open the second one" / "merge those
    two" resolve by reference),
  - a ``pending_action`` (a proposed write awaiting the user's confirmation),
  - the full message transcript (for reloads).
"""

import datetime
import json
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .. import config

_initialized = False


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextmanager
def _conn():
    config.AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(config.AUTH_DB_PATH))
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _ensure() -> None:
    global _initialized
    if _initialized:
        return
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id     TEXT PRIMARY KEY,
                tenant_id      TEXT NOT NULL,
                user_email     TEXT,
                active_filters TEXT DEFAULT '{}',
                last_results   TEXT DEFAULT '[]',
                pending_action TEXT,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_msg ON chat_messages(session_id, id);
            """
        )
    _initialized = True


def _loads(s: Optional[str], default):
    try:
        return json.loads(s) if s else default
    except (TypeError, ValueError):
        return default


# ── Sessions ────────────────────────────────────────────────────────────────
def create_session(tenant_id: str, user_email: str) -> str:
    _ensure()
    sid = uuid.uuid4().hex[:16]
    with _conn() as c:
        c.execute(
            """INSERT INTO chat_sessions
               (session_id, tenant_id, user_email, active_filters, last_results,
                pending_action, created_at, updated_at)
               VALUES (?, ?, ?, '{}', '[]', NULL, ?, ?)""",
            (sid, tenant_id, user_email, _now(), _now()),
        )
    return sid


def get_session(session_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    _ensure()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM chat_sessions WHERE session_id=? AND tenant_id=?",
            (session_id, tenant_id),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["active_filters"] = _loads(d.get("active_filters"), {})
    d["last_results"] = _loads(d.get("last_results"), [])
    d["pending_action"] = _loads(d.get("pending_action"), None)
    return d


def update_session(
    session_id: str,
    active_filters: Optional[Dict[str, Any]] = None,
    last_results: Optional[List[str]] = None,
    pending_action: Optional[Dict[str, Any]] = "__keep__",  # sentinel: leave as-is
) -> None:
    _ensure()
    sets, vals = [], []
    if active_filters is not None:
        sets.append("active_filters=?")
        vals.append(json.dumps(active_filters))
    if last_results is not None:
        sets.append("last_results=?")
        vals.append(json.dumps(last_results))
    if pending_action != "__keep__":
        sets.append("pending_action=?")
        vals.append(json.dumps(pending_action) if pending_action is not None else None)
    if not sets:
        return
    sets.append("updated_at=?")
    vals.append(_now())
    vals.append(session_id)
    with _conn() as c:
        c.execute(f"UPDATE chat_sessions SET {', '.join(sets)} WHERE session_id=?", vals)


# ── Messages ────────────────────────────────────────────────────────────────
def add_message(session_id: str, role: str, content: str) -> None:
    _ensure()
    with _conn() as c:
        c.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )


def get_messages(session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    _ensure()
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, created_at FROM chat_messages "
            "WHERE session_id=? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
