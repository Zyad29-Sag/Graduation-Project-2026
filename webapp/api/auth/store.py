"""
webapp/api/auth/store.py
------------------------
The API's own SQLite store (auth.db) — kept entirely separate from the engine
schema so tenancy/users/audit never touch person identity data.

Tables:
  tenants    — id, name, db_path, snapshots_dir   (the multi-tenant seam:
               each tenant points at its own engine DB + snapshots)
  users      — email, password_hash, role, tenant_id
  audit_log  — append-only record of every correction (who/what/when)
"""

import datetime
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .. import config
from . import security


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


def init_auth_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                db_path       TEXT NOT NULL,
                snapshots_dir TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'viewer',
                tenant_id     TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                user_email  TEXT NOT NULL,
                tenant_id   TEXT NOT NULL,
                action      TEXT NOT NULL,
                target_ids  TEXT,
                detail      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, id DESC);
            """
        )


# ── Tenants ─────────────────────────────────────────────────────────────────
def upsert_tenant(tenant_id: str, name: str, db_path: str, snapshots_dir: str) -> None:
    with _conn() as c:
        c.execute(
            """INSERT INTO tenants (id, name, db_path, snapshots_dir, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, db_path=excluded.db_path,
                 snapshots_dir=excluded.snapshots_dir""",
            (tenant_id, name, db_path, snapshots_dir, _now()),
        )


def get_tenant(tenant_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM tenants WHERE id=?", (tenant_id,)).fetchone()
    return dict(row) if row else None


# ── Users ───────────────────────────────────────────────────────────────────
def create_user(email: str, password: str, role: str, tenant_id: str) -> None:
    with _conn() as c:
        c.execute(
            """INSERT OR IGNORE INTO users
               (email, password_hash, role, tenant_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (email, security.hash_password(password), role, tenant_id, _now()),
        )


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


# ── Audit log ───────────────────────────────────────────────────────────────
def record_audit(
    user_email: str,
    tenant_id: str,
    action: str,
    target_ids: Optional[List[str]] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    with _conn() as c:
        c.execute(
            """INSERT INTO audit_log (ts, user_email, tenant_id, action, target_ids, detail)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                _now(),
                user_email,
                tenant_id,
                action,
                json.dumps(target_ids or []),
                json.dumps(detail or {}),
            ),
        )


def list_audit(tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM audit_log WHERE tenant_id=? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for f in ("target_ids", "detail"):
            try:
                d[f] = json.loads(d.get(f) or "null")
            except (TypeError, ValueError):
                d[f] = None
        out.append(d)
    return out


# ── Demo bootstrap ──────────────────────────────────────────────────────────
def seed_demo_account() -> None:
    """Idempotently create the demo tenant + admin user."""
    init_auth_db()
    upsert_tenant(
        config.DEMO_TENANT_ID,
        config.DEMO_TENANT_NAME,
        str(config.DEMO_DB_PATH),
        str(config.DEMO_SNAPSHOTS_DIR),
    )
    create_user(
        config.DEMO_USER_EMAIL,
        config.DEMO_USER_PASSWORD,
        role="admin",
        tenant_id=config.DEMO_TENANT_ID,
    )
