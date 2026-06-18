"""
webapp/api/seed_demo.py
-----------------------
Seed the API's served demo dataset from the engine's data.

The engine WIPES surveillant/database/surveillant.db before every run, so the
API must serve a separate, persistent copy. This script:
  1. Copies the engine DB into api/data/demo/surveillant.db (consistent copy
     via the sqlite backup API — safe even if the source is in WAL mode).
  2. Copies the snapshot folder for every person referenced in the DB.
  3. Copies the violence alert log if present.
  4. Seeds the demo tenant + demo user into auth.db (if the auth module exists).
  5. Prints a coverage report.

Run from the repo root:
    python -m webapp.api.seed_demo
"""

import json
import shutil
import sqlite3
import sys

from . import config


def _backup_db(src, dst) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dst))
    try:
        with dst_conn:
            src_conn.backup(dst_conn)
    finally:
        src_conn.close()
        dst_conn.close()


def _copy_snapshots(db_path) -> int:
    """Copy snapshot folders only for persons that exist in the DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        person_ids = [r[0] for r in conn.execute("SELECT person_id FROM persons")]
    finally:
        conn.close()

    config.DEMO_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pid in person_ids:
        src = config.ENGINE_SNAPSHOTS_DIR / pid
        if not src.is_dir():
            continue
        dst = config.DEMO_SNAPSHOTS_DIR / pid
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied += 1
    return copied


def _report(db_path) -> None:
    conn = sqlite3.connect(str(db_path))

    def q(sql):
        try:
            return conn.execute(sql).fetchone()[0]
        except Exception as exc:  # noqa: BLE001
            return f"ERR {exc}"

    persons = q("SELECT COUNT(*) FROM persons")
    described = q("SELECT COUNT(*) FROM persons WHERE latest_description_id IS NOT NULL")
    desc_emb = q("SELECT COUNT(*) FROM person_descriptions WHERE embedding IS NOT NULL")
    body_emb = q("SELECT COUNT(*) FROM person_embeddings")
    face_emb = q("SELECT COUNT(*) FROM face_embeddings")
    cam_hist = q("SELECT COUNT(*) FROM camera_history")
    conn.close()

    print("\n-- Seed report ------------------------------------------")
    print(f"  persons               : {persons}")
    print(f"  with description      : {described}")
    print(f"  description embeddings : {desc_emb}")
    print(f"  body embeddings       : {body_emb}")
    print(f"  face embeddings       : {face_emb}")
    print(f"  camera_history rows   : {cam_hist}")
    print("---------------------------------------------------------")
    if described == 0 or desc_emb == 0:
        print("  WARNING: no descriptions/embeddings -> natural-language (chatbot)")
        print("  search will return nothing. Run:  python main.py --phase 4 --describe-all")
        print("  (from surveillant/, with Ollama qwen2.5vl:3b running), then re-seed.")
    print()


def _seed_auth() -> None:
    """Seed the demo tenant + user, if the auth module is available (M2+)."""
    try:
        from .auth import store  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"  (auth not seeded -- auth module not present yet: {exc})")
        return
    store.init_auth_db()
    store.seed_demo_account()
    print(f"  auth: demo tenant '{config.DEMO_TENANT_ID}' + user "
          f"'{config.DEMO_USER_EMAIL}' ready.")


def seed() -> None:
    if not config.ENGINE_DB_PATH.exists():
        print(f"[seed] ERROR: engine DB not found at {config.ENGINE_DB_PATH}")
        print("       Run the engine first (python main.py --phase 2 ...) to populate it.")
        sys.exit(1)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[seed] DB        {config.ENGINE_DB_PATH}  ->  {config.DEMO_DB_PATH}")
    _backup_db(config.ENGINE_DB_PATH, config.DEMO_DB_PATH)

    n = _copy_snapshots(config.DEMO_DB_PATH)
    print(f"[seed] snapshots copied for {n} person(s)  ->  {config.DEMO_SNAPSHOTS_DIR}")

    if config.ENGINE_ALERTS_LOG.exists():
        shutil.copy2(config.ENGINE_ALERTS_LOG, config.DEMO_ALERTS_LOG)
        print(f"[seed] alerts    {config.ENGINE_ALERTS_LOG}  ->  {config.DEMO_ALERTS_LOG}")
    else:
        # Ensure the file exists (empty list) so /alerts never 404s.
        config.DEMO_ALERTS_LOG.write_text("[]", encoding="utf-8")
        print("[seed] alerts    (none found -- wrote empty violence_log.json)")

    _seed_auth()
    _report(config.DEMO_DB_PATH)
    print("[seed] done.")


if __name__ == "__main__":
    seed()
