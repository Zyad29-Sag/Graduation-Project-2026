"""
modules/storage/database.py
----------------------------
SQLite wrapper for SURVEILLANT.

Schema (final version):
  persons          — one row per unique physical person
  person_embeddings — one row per gallery embedding (many per person)
  camera_history   — cross-camera sighting log
  merge_proposals  — reconciliation merge candidates
"""

import sqlite3
import json
import uuid
import datetime
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from config.settings import DB_PATH


class Database:
    """
    SQLite wrapper for the SURVEILLANT system.

    Supports both file-based and in-memory (':memory:') databases.
    In-memory mode is used by tests to avoid file-lock issues on Windows.
    """

    def __init__(self, db_path=DB_PATH) -> None:
        self._in_memory = str(db_path) == ":memory:"

        if self._in_memory:
            self.db_path = ":memory:"
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self.db_path = str(Path(db_path))
            self._conn = None
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Optional hooks for downstream caches (e.g. FAISS index in Part 8).
        # SQLite remains the source of truth; these callbacks keep the
        # in-memory vector index in sync. Both default to None so the
        # Database works standalone without any extra wiring.
        # Signatures:
        #   on_embedding_added(person_id: str, embedding: np.ndarray) -> None
        #   on_merge          (keep_id:   str, remove_id:  str)         -> None
        self.on_embedding_added = None
        self.on_merge           = None

        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Context manager that yields the right connection."""
        if self._in_memory:
            yield self._conn
        else:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                yield conn

    def _init_db(self) -> None:
        """Create / migrate the schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id              TEXT PRIMARY KEY,
                    first_seen_cam         INTEGER,
                    first_seen_time        TEXT,
                    last_seen_cam          INTEGER,
                    last_seen_time         TEXT,
                    status                 TEXT DEFAULT 'unverified',
                    gallery_size           INTEGER DEFAULT 0,
                    known_angles           TEXT DEFAULT '[]',
                    last_gallery_update    TEXT,
                    description            TEXT,
                    gender                 TEXT,
                    age_range              TEXT,
                    snapshot_paths         TEXT DEFAULT '[]',
                    latest_description_id  INTEGER,
                    created_at             TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS person_embeddings (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       TEXT NOT NULL,
                    embedding       BLOB NOT NULL,
                    embedding_type  TEXT NOT NULL,
                    angle_tag       TEXT DEFAULT 'unknown',
                    source_cam      INTEGER,
                    captured_at     TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                );

                CREATE TABLE IF NOT EXISTS camera_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   TEXT NOT NULL,
                    cam_id      INTEGER NOT NULL,
                    track_id    INTEGER NOT NULL,
                    first_seen  TEXT,
                    last_seen   TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                );

                CREATE TABLE IF NOT EXISTS merge_proposals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id_a     TEXT,
                    person_id_b     TEXT,
                    similarity      REAL,
                    proposed_at     TEXT,
                    status          TEXT DEFAULT 'pending',
                    resolved_at     TEXT
                );

                -- ── Part 10 (Phase 4) ─────────────────────────────────────
                -- One row per VLM-generated description. Re-describes
                -- ACCUMULATE — never UPDATE-overwrite. The persons row
                -- carries a denormalised pointer (latest_description_id)
                -- for fast joins, but the full history is preserved here.
                CREATE TABLE IF NOT EXISTS person_descriptions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       TEXT NOT NULL,
                    described_at    TEXT NOT NULL,
                    backend         TEXT NOT NULL,
                    model_id        TEXT NOT NULL,
                    snapshots_used  TEXT NOT NULL,
                    attributes      TEXT NOT NULL,
                    summary         TEXT,
                    confidence      REAL,
                    embedding       BLOB,   -- float32 vector of long_description (Phase 4B semantic search)
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                );

                -- Durable description-job queue. Survives restarts so the
                -- DescriptionWorker can recover in-progress / pending rows
                -- on next startup. Status transitions are append-only:
                --   pending → in_progress → done | failed
                CREATE TABLE IF NOT EXISTS description_queue (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id    TEXT NOT NULL,
                    enqueued_at  TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    last_error   TEXT,
                    attempts     INTEGER DEFAULT 0,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                );

                CREATE INDEX IF NOT EXISTS idx_gallery_pid
                    ON person_embeddings(person_id);
                CREATE INDEX IF NOT EXISTS idx_cam_history_pid
                    ON camera_history(person_id);
                CREATE INDEX IF NOT EXISTS idx_desc_pid
                    ON person_descriptions(person_id);
                CREATE INDEX IF NOT EXISTS idx_descq_status
                    ON description_queue(status);
            """)
            if self._in_memory:
                conn.commit()

        # Migrate legacy schemas (file-based DB that may pre-date this version)
        if not self._in_memory:
            self._migrate()

    def _migrate(self) -> None:
        """Add missing columns to existing databases without losing data."""
        migrations = [
            "ALTER TABLE persons ADD COLUMN status TEXT DEFAULT 'unverified'",
            "ALTER TABLE persons ADD COLUMN gallery_size INTEGER DEFAULT 0",
            "ALTER TABLE persons ADD COLUMN known_angles TEXT DEFAULT '[]'",
            "ALTER TABLE persons ADD COLUMN last_gallery_update TEXT",
            "ALTER TABLE person_embeddings ADD COLUMN source_cam INTEGER",
            # Part 10 — Phase 4 LLM description
            "ALTER TABLE persons ADD COLUMN latest_description_id INTEGER",
            # Phase 4B — semantic search: float32 embedding of long_description
            "ALTER TABLE person_descriptions ADD COLUMN embedding BLOB",
        ]
        with self._get_conn() as conn:
            for sql in migrations:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # column already exists
            # Ensure Part-10 tables exist on legacy DBs
            for ddl in (
                """CREATE TABLE IF NOT EXISTS person_descriptions (
                       id              INTEGER PRIMARY KEY AUTOINCREMENT,
                       person_id       TEXT NOT NULL,
                       described_at    TEXT NOT NULL,
                       backend         TEXT NOT NULL,
                       model_id        TEXT NOT NULL,
                       snapshots_used  TEXT NOT NULL,
                       attributes      TEXT NOT NULL,
                       summary         TEXT,
                       confidence      REAL,
                       FOREIGN KEY (person_id) REFERENCES persons(person_id)
                   )""",
                """CREATE TABLE IF NOT EXISTS description_queue (
                       id           INTEGER PRIMARY KEY AUTOINCREMENT,
                       person_id    TEXT NOT NULL,
                       enqueued_at  TEXT NOT NULL,
                       status       TEXT NOT NULL DEFAULT 'pending',
                       last_error   TEXT,
                       attempts     INTEGER DEFAULT 0,
                       FOREIGN KEY (person_id) REFERENCES persons(person_id)
                   )""",
                "CREATE INDEX IF NOT EXISTS idx_desc_pid ON person_descriptions(person_id)",
                "CREATE INDEX IF NOT EXISTS idx_descq_status ON description_queue(status)",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass

    # ------------------------------------------------------------------
    # Public API — Persons
    # ------------------------------------------------------------------

    def insert_person(self, record: Dict[str, Any]) -> str:
        """
        Insert a newly detected person.
        Accepts an optional 'person_id' key (for tests that need a known UUID).
        Seeds the gallery with the initial embedding if 'embedding' is provided.
        """
        person_id = record.get("person_id") or str(uuid.uuid4())
        snapshot_paths_str = json.dumps(record.get("snapshot_paths", []))
        now = record.get("created_at", datetime.datetime.now().isoformat())

        query = """
        INSERT OR IGNORE INTO persons (
            person_id, first_seen_cam, first_seen_time,
            last_seen_cam, last_seen_time,
            status, gallery_size, known_angles, last_gallery_update,
            description, gender, age_range,
            snapshot_paths, created_at
        ) VALUES (?, ?, ?, ?, ?, 'unverified', 0, '[]', NULL, ?, ?, ?, ?, ?)
        """
        values = (
            person_id,
            record.get("first_seen_cam", record.get("cam_id", 0)),
            record.get("first_seen_time", now),
            record.get("last_seen_cam",  record.get("cam_id", 0)),
            record.get("last_seen_time",  now),
            record.get("description"),
            record.get("gender"),
            record.get("age_range"),
            snapshot_paths_str,
            now,
        )
        with self._get_conn() as conn:
            conn.execute(query, values)
            if self._in_memory:
                conn.commit()

        # Seed gallery with initial embedding (if provided).
        # angle_tag defaults to "initial" but the caller can override with a
        # canonical view ("frontal", "side", "right_moving", "left_moving").
        # This matters: get_view_coverage() only counts canonical views, so
        # using "initial" leaves the person at 0.0 coverage and blocks
        # reconciliation forever.
        emb_bytes = record.get("embedding")
        if emb_bytes:
            self.add_embedding_to_gallery(
                person_id      = person_id,
                embedding_bytes= emb_bytes,
                embedding_type = record.get("embedding_type", "body"),
                angle_tag      = record.get("angle_tag", "initial"),
                source_cam     = record.get("cam_id", 0),
                captured_at    = now,
            )

        return person_id

    def get_all_persons(self) -> List[Dict[str, Any]]:
        """Return all person records as dictionaries."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM persons")
            rows = cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            for field in ("snapshot_paths", "known_angles"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        d[field] = []
            results.append(d)
        return results

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single person by ID."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM persons WHERE person_id = ?", (person_id,)
            )
            row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        for field in ("snapshot_paths", "known_angles"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        return d

    def get_persons_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Return all persons with the given status."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM persons WHERE status = ?", (status,)
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def update_last_seen(self, person_id: str, cam_id: int, timestamp: str) -> None:
        """Update last_seen when a person is re-identified."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE persons SET last_seen_cam=?, last_seen_time=? WHERE person_id=?",
                (cam_id, timestamp, person_id),
            )
            if self._in_memory:
                conn.commit()

    def update_person_status(self, person_id: str, status: str) -> None:
        """
        Update person status (unverified / confirmed / multi_view / flagged / ghost).
        Called automatically when gallery grows or cross-camera match found.
        """
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE persons SET status=? WHERE person_id=?",
                (status, person_id),
            )
            if self._in_memory:
                conn.commit()

    def update_description(
        self, person_id: str, description: str, gender: str, age_range: str
    ) -> None:
        """Update LLM-generated profile attributes."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE persons SET description=?, gender=?, age_range=? WHERE person_id=?",
                (description, gender, age_range, person_id),
            )
            if self._in_memory:
                conn.commit()

    # ------------------------------------------------------------------
    # Public API — Gallery
    # ------------------------------------------------------------------

    def add_embedding_to_gallery(
        self,
        person_id: str,
        embedding_bytes: bytes,
        embedding_type: str,
        angle_tag: str,
        captured_at: str,
        source_cam: int = 0,
    ) -> None:
        """
        Add a new view embedding to a person's gallery.
        Also updates denormalized gallery_size and known_angles on the person row,
        and automatically promotes the person status when gallery grows.
        """
        # Everything in ONE transaction to avoid nested-connection deadlocks on WAL.
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO person_embeddings
                   (person_id, embedding, embedding_type, angle_tag, source_cam, captured_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (person_id, embedding_bytes, embedding_type, angle_tag, source_cam, captured_at),
            )

            cursor = conn.execute(
                "SELECT gallery_size, known_angles FROM persons WHERE person_id=?",
                (person_id,),
            )
            row = cursor.fetchone()
            if row:
                new_size = (row[0] or 0) + 1
                angles_list = json.loads(row[1] or "[]")
                if angle_tag not in angles_list:
                    angles_list.append(angle_tag)
                conn.execute(
                    """UPDATE persons
                       SET gallery_size=?, known_angles=?, last_gallery_update=?
                       WHERE person_id=?""",
                    (new_size, json.dumps(angles_list), captured_at, person_id),
                )
                # Inline status promotion — avoids opening a second connection inside this one
                if new_size == 2:
                    conn.execute(
                        "UPDATE persons SET status='confirmed' WHERE person_id=?",
                        (person_id,),
                    )

            if self._in_memory:
                conn.commit()

        # Notify downstream caches (e.g. FAISS index) AFTER the SQLite
        # transaction has committed. SQLite remains the source of truth;
        # this is a redundant in-memory cache for fast cosine search.
        if self.on_embedding_added is not None:
            try:
                emb_arr = np.frombuffer(embedding_bytes, dtype=np.float32)
                self.on_embedding_added(person_id, emb_arr)
            except Exception as exc:
                # Cache sync failures must NEVER take down the embedding
                # worker — SQLite still has the data.
                print(f"[DB] on_embedding_added hook raised: {exc}")

    def get_gallery(self, person_id: str) -> List[np.ndarray]:
        """Return all gallery embeddings as numpy arrays (vectors only)."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT embedding FROM person_embeddings WHERE person_id=? ORDER BY id ASC",
                (person_id,),
            )
            rows = cursor.fetchall()
        return [np.frombuffer(row[0], dtype=np.float32) for row in rows]

    def get_gallery_typed(self, person_id: str) -> List[Dict[str, Any]]:
        """Return gallery as list of {embedding, type, source_cam} dicts."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT embedding, embedding_type, source_cam FROM person_embeddings "
                "WHERE person_id=? ORDER BY id ASC",
                (person_id,),
            )
            rows = cursor.fetchall()
        return [
            {
                "embedding":  np.frombuffer(row[0], dtype=np.float32),
                "type":       row[1],
                "source_cam": row[2],
            }
            for row in rows
        ]

    def get_gallery_size(self, person_id: str) -> int:
        """Return number of embeddings in a person's gallery (uses denormalized column)."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT gallery_size FROM persons WHERE person_id=?", (person_id,)
            )
            row = cursor.fetchone()
            if row is not None:
                return row[0] or 0
            return 0

    def get_all_galleries_typed(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return {person_id: [{embedding, type, source_cam}, ...]} for all persons."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT person_id, embedding, embedding_type, source_cam "
                "FROM person_embeddings ORDER BY person_id, id ASC"
            )
            rows = cursor.fetchall()
        galleries: Dict[str, List[Dict[str, Any]]] = {}
        for pid, emb_bytes, emb_type, src_cam in rows:
            arr = np.frombuffer(emb_bytes, dtype=np.float32)
            galleries.setdefault(pid, []).append(
                {"embedding": arr, "type": emb_type, "source_cam": src_cam}
            )
        return galleries

    def get_all_galleries(self) -> Dict[str, List[np.ndarray]]:
        """Return {person_id: [ndarray, ...]}. Legacy; prefer get_all_galleries_typed()."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT person_id, embedding FROM person_embeddings ORDER BY person_id, id ASC"
            )
            rows = cursor.fetchall()
        galleries: Dict[str, List[np.ndarray]] = {}
        for pid, emb_bytes in rows:
            galleries.setdefault(pid, []).append(np.frombuffer(emb_bytes, dtype=np.float32))
        return galleries

    # ------------------------------------------------------------------
    # Public API — Camera History
    # ------------------------------------------------------------------

    def upsert_camera_history(
        self, person_id: str, cam_id: int, track_id: int, timestamp: str
    ) -> None:
        """Record or update a camera sighting for a person."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT id FROM camera_history WHERE person_id=? AND cam_id=? AND track_id=?",
                (person_id, cam_id, track_id),
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    "UPDATE camera_history SET last_seen=? WHERE id=?",
                    (timestamp, row[0]),
                )
            else:
                conn.execute(
                    "INSERT INTO camera_history (person_id, cam_id, track_id, first_seen, last_seen) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (person_id, cam_id, track_id, timestamp, timestamp),
                )
            if self._in_memory:
                conn.commit()

    def get_cameras_for_person(self, person_id: str) -> List[int]:
        """Return list of distinct cam_ids where this person has been seen."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT cam_id FROM camera_history WHERE person_id=?",
                (person_id,),
            )
            return [row[0] for row in cursor.fetchall()]

    def get_camera_history(self, person_id: str) -> List[Dict[str, Any]]:
        """
        Return all camera_history rows for a person as a list of dicts:
        [{cam_id, track_id, first_seen, last_seen}, ...].

        Used by the reconciliation worker's co-visibility check (Part 8.5)
        to compute temporal interval overlap across overlap-partner cameras.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT cam_id, track_id, first_seen, last_seen "
                "FROM camera_history WHERE person_id=?",
                (person_id,),
            )
            return [
                {
                    "cam_id":     row[0],
                    "track_id":   row[1],
                    "first_seen": row[2],
                    "last_seen":  row[3],
                }
                for row in cursor.fetchall()
            ]

    # ------------------------------------------------------------------
    # Public API — LLM Descriptions (Part 10 / Phase 4)
    # ------------------------------------------------------------------

    def enqueue_description(self, person_id: str) -> None:
        """
        Upsert a 'pending' row into description_queue for this person.

        Idempotent: if a pending or in-progress row already exists, do nothing
        (avoid double-describing the same person concurrently). If the only
        prior row is 'done' or 'failed', insert a fresh pending row so a
        re-describe can be requested later.
        """
        now = datetime.datetime.now().isoformat()
        with self._get_conn() as conn:
            existing = conn.execute(
                """SELECT id FROM description_queue
                   WHERE person_id=? AND status IN ('pending', 'in_progress')""",
                (person_id,),
            ).fetchone()
            if existing:
                return
            conn.execute(
                "INSERT INTO description_queue (person_id, enqueued_at, status) "
                "VALUES (?, ?, 'pending')",
                (person_id, now),
            )
            if self._in_memory:
                conn.commit()

    def claim_next_description(self) -> Optional[Dict[str, Any]]:
        """
        Atomically claim the oldest 'pending' row, marking it 'in_progress',
        and return it. Returns None if nothing is pending.

        SQLite doesn't support UPDATE ... RETURNING in older versions, so we
        do a SELECT + UPDATE inside one connection — safe under the WAL +
        single-claimer-thread invariant (only one DescriptionWorker thread).
        """
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT id, person_id, enqueued_at, attempts
                   FROM description_queue
                   WHERE status = 'pending'
                   ORDER BY id ASC
                   LIMIT 1"""
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE description_queue SET status='in_progress' WHERE id=?",
                (row["id"],),
            )
            if self._in_memory:
                conn.commit()
            return dict(row)

    def complete_description(self, queue_id: int, description_id: int) -> None:
        """Mark the queue row done and point the person at the new description."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE description_queue SET status='done' WHERE id=?",
                (queue_id,),
            )
            conn.execute(
                "UPDATE persons SET latest_description_id=? "
                "WHERE person_id=(SELECT person_id FROM description_queue WHERE id=?)",
                (description_id, queue_id),
            )
            if self._in_memory:
                conn.commit()

    def fail_description(self, queue_id: int, error_msg: str, max_attempts: int = 3) -> None:
        """
        Record a backend failure for a queue row. If the row has now exceeded
        max_attempts, mark it 'failed'; otherwise return it to 'pending' so the
        next loop iteration / sweep picks it up.
        """
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT attempts FROM description_queue WHERE id=?", (queue_id,)
            ).fetchone()
            if not cur:
                return
            new_attempts = (cur[0] or 0) + 1
            new_status = "failed" if new_attempts >= max_attempts else "pending"
            conn.execute(
                "UPDATE description_queue SET status=?, attempts=?, last_error=? WHERE id=?",
                (new_status, new_attempts, error_msg[:1000], queue_id),
            )
            if self._in_memory:
                conn.commit()

    def insert_description(
        self,
        person_id: str,
        backend: str,
        model_id: str,
        snapshots_used: List[str],
        attributes: Dict[str, Any],
        summary: Optional[str],
        confidence: Optional[float] = None,
        embedding: Optional[bytes] = None,
    ) -> int:
        """
        Insert a new description row and return its id.

        ``embedding`` is the raw float32 bytes of the long_description vector
        (Phase 4B semantic search); may be None if the embedder is unavailable.
        """
        now = datetime.datetime.now().isoformat()
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO person_descriptions
                   (person_id, described_at, backend, model_id,
                    snapshots_used, attributes, summary, confidence, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    person_id, now, backend, model_id,
                    json.dumps(snapshots_used),
                    json.dumps(attributes),
                    summary,
                    confidence,
                    embedding,
                ),
            )
            new_id = cur.lastrowid
            if self._in_memory:
                conn.commit()
            return new_id

    def get_latest_description(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Return the latest description row for a person, or None."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT d.* FROM persons p
                   LEFT JOIN person_descriptions d
                   ON p.latest_description_id = d.id
                   WHERE p.person_id = ?""",
                (person_id,),
            ).fetchone()
            if not row or row["id"] is None:
                return None
            out = dict(row)
            try:
                out["attributes"] = json.loads(out.get("attributes") or "{}")
            except (TypeError, ValueError):
                out["attributes"] = {}
            try:
                out["snapshots_used"] = json.loads(out.get("snapshots_used") or "[]")
            except (TypeError, ValueError):
                out["snapshots_used"] = []
            return out

    def get_persons_without_description(self, limit: int = 100) -> List[str]:
        """Return person_ids with no latest_description_id — sweep targets."""
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT person_id FROM persons "
                "WHERE latest_description_id IS NULL "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            )
            return [row[0] for row in cur.fetchall()]

    def search_persons_by_attributes(
        self, filters: Dict[str, Any], limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Stage 1 of natural-language search (Phase 4B).

        `filters` is a dict matching the description schema (any subset).
        Returns candidate persons with their latest description attributes:
            [{person_id, attributes, summary, last_seen_cam, last_seen_time,
              snapshot_paths}, ...]

        Enum fields → exact match on json_extract(attributes, '$.field').
        Phrase fields (clothing_top, clothing_bottom, headwear) → LIKE.
        Accessories (list) → LIKE on the JSON list serialization.

        Missing filter values are ignored (no filter applied).
        """
        # Sanitise inputs — only known fields are filterable.
        exact_fields  = (
            "gender", "age_range", "body_build", "height_class",
            "hair_color", "hair_length", "beard", "glasses",
            "clothing_top_color", "clothing_bottom_color", "headwear_color",
        )
        phrase_fields = ("clothing_top", "clothing_bottom", "headwear")
        list_fields   = ("accessories",)

        clauses: list = []
        params:  list = []
        for f in exact_fields:
            v = filters.get(f)
            if v is None or v == "" or v == "unknown":
                continue
            clauses.append(f"json_extract(d.attributes, '$.{f}') = ?")
            params.append(str(v))
        for f in phrase_fields:
            v = filters.get(f)
            if v is None or v == "":
                continue
            # Accept a scalar OR a list of synonyms (OR-matched). The search
            # engine passes a synonym group (e.g. hoodie/sweater/jumper) so a
            # query term matches whichever surface form the describer stored.
            items = v if isinstance(v, list) else [v]
            sub = []
            for item in items:
                if item is None or str(item) == "":
                    continue
                sub.append(f"LOWER(json_extract(d.attributes, '$.{f}')) LIKE ?")
                params.append(f"%{str(item).lower()}%")
            if sub:
                clauses.append("(" + " OR ".join(sub) + ")")
        for f in list_fields:
            v = filters.get(f)
            if not v:
                continue
            # `v` is a list of accessory phrases; OR them.
            sub = []
            for item in (v if isinstance(v, list) else [v]):
                sub.append(f"LOWER(d.attributes) LIKE ?")
                params.append(f"%{str(item).lower()}%")
            if sub:
                clauses.append("(" + " OR ".join(sub) + ")")

        where = (" AND " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT p.person_id, p.last_seen_cam, p.last_seen_time, "
            "       p.snapshot_paths, d.attributes, d.summary, d.id AS desc_id "
            "FROM persons p "
            "INNER JOIN person_descriptions d ON p.latest_description_id = d.id "
            "WHERE 1=1" + where + " "
            "ORDER BY p.last_seen_time DESC LIMIT ?"
        )
        params.append(limit)

        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            try:
                attrs = json.loads(row["attributes"] or "{}")
            except (TypeError, ValueError):
                attrs = {}
            try:
                snaps = json.loads(row["snapshot_paths"] or "[]")
            except (TypeError, ValueError):
                snaps = []
            results.append({
                "person_id":      row["person_id"],
                "last_seen_cam":  row["last_seen_cam"],
                "last_seen_time": row["last_seen_time"],
                "snapshot_paths": snaps,
                "attributes":     attrs,
                "summary":        row["summary"],
                "desc_id":        row["desc_id"],
            })
        return results

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """
        Return the latest description per person (with its stored embedding) for
        semantic search. Each item: person_id, last_seen_*, snapshot_paths,
        summary (== long_description), attributes, embedding (raw float32 bytes
        or None), desc_id.
        """
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT p.person_id, p.last_seen_cam, p.last_seen_time, "
                "       p.snapshot_paths, d.summary, d.attributes, d.embedding, "
                "       d.id AS desc_id "
                "FROM persons p "
                "INNER JOIN person_descriptions d ON p.latest_description_id = d.id "
                "WHERE d.summary IS NOT NULL"
            ).fetchall()
        out = []
        for row in rows:
            try:
                snaps = json.loads(row["snapshot_paths"] or "[]")
            except (TypeError, ValueError):
                snaps = []
            try:
                attrs = json.loads(row["attributes"] or "{}")
            except (TypeError, ValueError):
                attrs = {}
            out.append({
                "person_id":      row["person_id"],
                "last_seen_cam":  row["last_seen_cam"],
                "last_seen_time": row["last_seen_time"],
                "snapshot_paths": snaps,
                "summary":        row["summary"],
                "attributes":     attrs,
                "embedding":      row["embedding"],   # raw bytes or None
                "desc_id":        row["desc_id"],
            })
        return out

    def get_all_person_ids(self) -> List[str]:
        """Return every person_id (used by --redescribe-all to re-enqueue all)."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT person_id FROM persons").fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Public API — Merge Proposals
    # ------------------------------------------------------------------

    def propose_merge(self, pid_a: str, pid_b: str, similarity: float) -> None:
        """
        Log a reconciliation merge proposal, deduplicating by person pair.

        If a pending proposal for (pid_a, pid_b) or (pid_b, pid_a) already
        exists, update its similarity score instead of inserting a duplicate.
        Without this, the same wrong pair would accumulate a new row every
        120 seconds, producing dozens of identical false proposals.
        """
        now = datetime.datetime.now().isoformat()
        with self._get_conn() as conn:
            existing = conn.execute(
                """SELECT id FROM merge_proposals
                   WHERE status = 'pending'
                     AND ((person_id_a = ? AND person_id_b = ?)
                          OR (person_id_a = ? AND person_id_b = ?))""",
                (pid_a, pid_b, pid_b, pid_a),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE merge_proposals SET similarity=?, proposed_at=? WHERE id=?",
                    (similarity, now, existing[0]),
                )
            else:
                conn.execute(
                    "INSERT INTO merge_proposals "
                    "(person_id_a, person_id_b, similarity, proposed_at) VALUES (?, ?, ?, ?)",
                    (pid_a, pid_b, similarity, now),
                )
            if self._in_memory:
                conn.commit()

    def get_pending_merges(self) -> List[Dict[str, Any]]:
        """Return all pending merge proposals."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM merge_proposals WHERE status='pending' ORDER BY similarity DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def merge_persons(self, keep_id: str, remove_id: str) -> int:
        """
        Merge remove_id into keep_id.
        Returns number of embeddings transferred.
        """
        with self._get_conn() as conn:
            # Move embeddings
            cursor = conn.execute(
                "UPDATE person_embeddings SET person_id=? WHERE person_id=?",
                (keep_id, remove_id),
            )
            moved = cursor.rowcount

            # Move camera history
            conn.execute(
                "UPDATE camera_history SET person_id=? WHERE person_id=?",
                (keep_id, remove_id),
            )

            # Part 10 — re-point any LLM descriptions from remove_id to keep_id
            # so the merged person inherits the description history of both.
            # latest_description_id on keep_id keeps whatever it had (don't
            # downgrade); the merged-in descriptions are still discoverable
            # via person_descriptions.person_id = keep_id.
            conn.execute(
                "UPDATE person_descriptions SET person_id=? WHERE person_id=?",
                (keep_id, remove_id),
            )
            # If keep_id had no description yet but remove_id had one, inherit it.
            conn.execute("""
                UPDATE persons
                   SET latest_description_id = (
                       SELECT MAX(id) FROM person_descriptions
                       WHERE person_id = ?
                   )
                 WHERE person_id = ? AND latest_description_id IS NULL
            """, (keep_id, keep_id))
            # Cancel any pending description_queue rows for the removed person.
            conn.execute(
                "UPDATE description_queue SET status='done', last_error='merged' "
                "WHERE person_id=? AND status IN ('pending', 'in_progress')",
                (remove_id,),
            )

            # Keep earlier first_seen_time
            conn.execute("""
                UPDATE persons
                SET first_seen_time = (
                    SELECT MIN(first_seen_time)
                    FROM persons WHERE person_id IN (?, ?)
                )
                WHERE person_id = ?
            """, (keep_id, remove_id, keep_id))

            # Delete removed person
            conn.execute("DELETE FROM persons WHERE person_id=?", (remove_id,))

            # Update gallery_size on keep_id
            cursor2 = conn.execute(
                "SELECT COUNT(*) FROM person_embeddings WHERE person_id=?", (keep_id,)
            )
            new_size = cursor2.fetchone()[0]
            conn.execute(
                "UPDATE persons SET gallery_size=? WHERE person_id=?",
                (new_size, keep_id),
            )

            # Mark merge proposals as accepted
            conn.execute(
                "UPDATE merge_proposals SET status='accepted', resolved_at=? "
                "WHERE (person_id_a=? AND person_id_b=?) OR (person_id_a=? AND person_id_b=?)",
                (datetime.datetime.now().isoformat(), keep_id, remove_id, remove_id, keep_id),
            )
            if self._in_memory:
                conn.commit()

        # Notify downstream caches AFTER the SQLite transaction.
        if self.on_merge is not None:
            try:
                self.on_merge(keep_id, remove_id)
            except Exception as exc:
                print(f"[DB] on_merge hook raised: {exc}")

        return moved

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Fetch first embedding per person. Prefer get_all_galleries_typed()."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT person_id, embedding FROM person_embeddings "
                "GROUP BY person_id"
            )
            rows = cursor.fetchall()
        return [(pid, np.frombuffer(emb, dtype=np.float32)) for pid, emb in rows]
