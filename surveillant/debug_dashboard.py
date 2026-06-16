"""
debug_dashboard.py
-------------------
FastAPI server for the SURVEILLANT live debug dashboard.

Run in a separate terminal while Phase 2 is running:
    cd surveillant
    python debug_dashboard.py

Then open: http://localhost:8501
"""

import json
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Bootstrap: make sure we can import from the surveillant package
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    DB_PATH,
    SNAPSHOTS_DIR,
    TRACK_REGISTRY_PATH,
    MAX_GALLERY_SIZE,
    ALERTS_DIR,
)
from debug_activity_tracker import ActivityTracker

# Violence events are logged here by the (optional) Part 11 violence worker.
VIOLENCE_LOG_PATH = Path(TRACK_REGISTRY_PATH).parent / "violence_log.json"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SURVEILLANT Debug Dashboard",
    description="Live database inspector for SURVEILLANT Phase 2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global tracker instance (started on first request)
# ---------------------------------------------------------------------------
_tracker: Optional[ActivityTracker] = None
_start_time = time.time()


def get_tracker() -> ActivityTracker:
    global _tracker
    if _tracker is None:
        _tracker = ActivityTracker(str(DB_PATH), poll_interval=2.0)
        _tracker.start()
    return _tracker


# ---------------------------------------------------------------------------
# DB helpers (read-only)
# ---------------------------------------------------------------------------

def _db_connect() -> sqlite3.Connection:
    """Open a read-only WAL connection to surveillant.db."""
    db_path = str(DB_PATH)
    if not Path(db_path).exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found at {db_path}. Start Phase 2 first.",
        )
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        check_same_thread=False,
        timeout=5.0,
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for field in ("snapshot_paths", "known_angles"):
        if field in d and d[field]:
            try:
                d[field] = json.loads(d[field])
            except Exception:
                d[field] = []
    return d


def _embedding_bytes_to_preview(blob: bytes, dims: int = 8) -> List[float]:
    """Return first `dims` float32 values from raw embedding bytes."""
    if not blob:
        return []
    count = min(dims, len(blob) // 4)
    try:
        return list(struct.unpack_from(f"{count}f", blob))
    except Exception:
        return []


def _embedding_norm(blob: bytes) -> Optional[float]:
    """Compute L2 norm of the embedding vector."""
    if not blob:
        return None
    count = len(blob) // 4
    try:
        vals = struct.unpack_from(f"{count}f", blob)
        sq = sum(v * v for v in vals)
        return round(sq ** 0.5, 6)
    except Exception:
        return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """True if a table exists — used to stay compatible with older DBs that
    predate the Part 11 face_embeddings table."""
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _columns(conn: sqlite3.Connection, table: str) -> set:
    """Return the set of column names on a table (empty on error).

    Keeps the dashboard robust against an un-migrated DB that predates the
    Part 11 persons columns (name / ethnicity / glasses) — querying them
    directly would otherwise raise 'no such column'.
    """
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def _read_violence_log() -> List[Dict[str, Any]]:
    """Read violence_log.json (newest last). Returns [] if absent/unreadable."""
    if not VIOLENCE_LOG_PATH.exists():
        return []
    try:
        data = json.loads(VIOLENCE_LOG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Static file: serve the HTML dashboard at /
# ---------------------------------------------------------------------------
UI_FILE = BASE_DIR / "debug_dashboard_ui.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    if UI_FILE.exists():
        return HTMLResponse(content=UI_FILE.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>UI file not found</h1><p>Place debug_dashboard_ui.html next to this script.</p>",
        status_code=404,
    )


# ---------------------------------------------------------------------------
# API — Stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def get_stats():
    """Aggregate DB counts: persons by status, embeddings, merges, plus the
    Part 10/11 surfaces (descriptions, face embeddings, cameras, violence)."""
    tracker = get_tracker()
    stats = tracker.get_stats()
    stats["uptime_seconds"] = int(time.time() - _start_time)
    stats["max_gallery_size"] = MAX_GALLERY_SIZE
    stats["db_path"] = str(DB_PATH)

    # Enrich with Part 10/11 counts (best-effort, read-only).
    try:
        conn = _db_connect()
        try:
            stats["total_descriptions"] = conn.execute(
                "SELECT COUNT(*) FROM persons WHERE latest_description_id IS NOT NULL"
            ).fetchone()[0]
            stats["total_face_embeddings"] = (
                conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
                if _table_exists(conn, "face_embeddings") else 0
            )
            stats["total_cameras"] = conn.execute(
                "SELECT COUNT(DISTINCT cam_id) FROM camera_history"
            ).fetchone()[0]
            stats["named_persons"] = (
                conn.execute(
                    "SELECT COUNT(*) FROM persons WHERE name IS NOT NULL AND name <> ''"
                ).fetchone()[0]
                if "name" in _columns(conn, "persons") else 0
            )
        finally:
            conn.close()
    except Exception:
        pass

    vlog = _read_violence_log()
    stats["violence_events"] = len(vlog)
    stats["violence_critical"] = sum(1 for e in vlog if e.get("level") == "VIOLENCE")
    return stats


# ---------------------------------------------------------------------------
# API — Persons
# ---------------------------------------------------------------------------

@app.get("/api/persons")
async def get_persons(
    status: Optional[str] = Query(None, description="Filter by status"),
    cam: Optional[int] = Query(None, description="Filter by last_seen_cam"),
    q: Optional[str] = Query(None, description="Keyword: matches person_id / name / description"),
    search: Optional[str] = Query(None, description="(legacy) person_id prefix"),
    gender: Optional[str] = Query(None),
    age_range: Optional[str] = Query(None),
    ethnicity: Optional[str] = Query(None),
    glasses: Optional[str] = Query(None),
    has_description: Optional[str] = Query(None, description="'1' = described, '0' = not"),
    has_face: Optional[str] = Query(None, description="'1' = has face embeddings, '0' = none"),
    sort: str = Query("created_desc", description="created_desc|created_asc|last_seen_desc|gallery_desc"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Return persons with rich filtering, keyword search, the latest
    description summary, and a face-embedding count (Part 11)."""
    conn = _db_connect()
    try:
        has_face_tbl = _table_exists(conn, "face_embeddings")
        pcols = _columns(conn, "persons")
        where: List[str] = ["1=1"]
        params: List[Any] = []

        if status:
            where.append("p.status = ?"); params.append(status)
        if cam is not None:
            where.append("p.last_seen_cam = ?"); params.append(cam)
        if gender and "gender" in pcols:
            where.append("p.gender = ?"); params.append(gender)
        if age_range and "age_range" in pcols:
            where.append("p.age_range = ?"); params.append(age_range)
        if ethnicity and "ethnicity" in pcols:
            where.append("p.ethnicity = ?"); params.append(ethnicity)
        if glasses and "glasses" in pcols:
            where.append("p.glasses = ?"); params.append(glasses)
        if has_description == "1":
            where.append("p.latest_description_id IS NOT NULL")
        elif has_description == "0":
            where.append("p.latest_description_id IS NULL")
        if has_face_tbl and has_face == "1":
            where.append("EXISTS (SELECT 1 FROM face_embeddings fe WHERE fe.person_id = p.person_id)")
        elif has_face_tbl and has_face == "0":
            where.append("NOT EXISTS (SELECT 1 FROM face_embeddings fe WHERE fe.person_id = p.person_id)")

        keyword = q or search
        if keyword:
            like = f"%{keyword.lower()}%"
            name_clause = "OR LOWER(COALESCE(p.name,'')) LIKE ? " if "name" in pcols else ""
            where.append(
                "(LOWER(p.person_id) LIKE ? " + name_clause +
                "OR LOWER(COALESCE(d.summary,'')) LIKE ?)"
            )
            params.append(like)
            if "name" in pcols:
                params.append(like)
            params.append(like)

        order = {
            "created_desc":   "p.created_at DESC",
            "created_asc":    "p.created_at ASC",
            "last_seen_desc": "p.last_seen_time DESC",
            "gallery_desc":   "p.gallery_size DESC",
        }.get(sort, "p.created_at DESC")

        base = ("FROM persons p "
                "LEFT JOIN person_descriptions d ON p.latest_description_id = d.id")
        where_sql = " AND ".join(where)

        total = conn.execute(f"SELECT COUNT(*) {base} WHERE {where_sql}", params).fetchone()[0]

        face_count_expr = (
            "(SELECT COUNT(*) FROM face_embeddings fe WHERE fe.person_id = p.person_id)"
            if has_face_tbl else "0"
        )
        sql = (
            f"SELECT p.*, d.summary AS description_summary, "
            f"{face_count_expr} AS face_count "
            f"{base} WHERE {where_sql} ORDER BY {order} LIMIT ? OFFSET ?"
        )
        rows = [_row_to_dict(r) for r in conn.execute(sql, params + [limit, offset]).fetchall()]

        return {"persons": rows, "total": total, "limit": limit, "offset": offset}
    finally:
        conn.close()


@app.get("/api/persons/{person_id}")
async def get_person(person_id: str):
    """Full detail for one person."""
    conn = _db_connect()
    try:
        cur = conn.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        return _row_to_dict(row)
    finally:
        conn.close()


@app.get("/api/persons/{person_id}/snapshots")
async def get_person_snapshots(person_id: str):
    """List snapshot image paths for a person (relative paths + existence check)."""
    conn = _db_connect()
    try:
        cur = conn.execute(
            "SELECT snapshot_paths FROM persons WHERE person_id = ?", (person_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Person not found")
        try:
            paths = json.loads(row["snapshot_paths"] or "[]")
        except Exception:
            paths = []
        # Also scan the snapshot folder for any crops that weren't stored in DB
        snap_dir = SNAPSHOTS_DIR / person_id
        if snap_dir.exists():
            disk_paths = sorted(snap_dir.glob("*.jpg")) + sorted(snap_dir.glob("*.png"))
            extra = [str(p) for p in disk_paths if str(p) not in paths]
            paths = paths + extra
        return {
            "person_id": person_id,
            "snapshot_paths": paths,
            "count": len(paths),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API — Serve snapshot images
# ---------------------------------------------------------------------------

@app.get("/api/snapshot/{person_id}/{filename}")
async def serve_snapshot(person_id: str, filename: str):
    """Serve a crop JPEG/PNG image for a given person."""
    # Security: prevent path traversal
    safe_filename = Path(filename).name
    image_path = SNAPSHOTS_DIR / person_id / safe_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    media_type = "image/jpeg" if safe_filename.lower().endswith(".jpg") else "image/png"
    return FileResponse(str(image_path), media_type=media_type)


@app.get("/api/snapshot_by_path")
async def serve_snapshot_by_path(path: str = Query(...)):
    """Serve an image by its absolute or relative path."""
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / path
    if not p.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    # Basic safety: must be inside SURVEILLANT directory
    try:
        p.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    media_type = "image/jpeg" if p.suffix.lower() == ".jpg" else "image/png"
    return FileResponse(str(p), media_type=media_type)


# ---------------------------------------------------------------------------
# API — Embeddings
# ---------------------------------------------------------------------------

@app.get("/api/embeddings/{person_id}")
async def get_embeddings(person_id: str):
    """All gallery embeddings for a person with metadata + vector preview."""
    conn = _db_connect()
    try:
        cur = conn.execute(
            "SELECT id, embedding_type, angle_tag, source_cam, captured_at, embedding "
            "FROM person_embeddings WHERE person_id = ? ORDER BY id ASC",
            (person_id,),
        )
        rows = cur.fetchall()
        results = []
        for row in rows:
            blob = row["embedding"]
            results.append({
                "id":             row["id"],
                "embedding_type": row["embedding_type"],
                "angle_tag":      row["angle_tag"],
                "source_cam":     row["source_cam"],
                "captured_at":    row["captured_at"],
                "vector_dims":    len(blob) // 4 if blob else 0,
                "vector_preview": _embedding_bytes_to_preview(blob, dims=8),
                "vector_norm":    _embedding_norm(blob),
            })
        return {"person_id": person_id, "embeddings": results, "count": len(results)}
    finally:
        conn.close()


@app.get("/api/embedding_vectors/{person_id}")
async def get_embedding_vectors(person_id: str):
    """
    Return full embedding vectors as JSON arrays for visualization.
    Applies PCA (2D) if scikit-learn is available, otherwise returns first 2 dims.
    """
    conn = _db_connect()
    try:
        cur = conn.execute(
            "SELECT id, angle_tag, embedding_type, embedding "
            "FROM person_embeddings WHERE person_id = ? ORDER BY id ASC",
            (person_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return {"person_id": person_id, "points": [], "method": "none"}

        labels = []
        full_vectors = []
        for row in rows:
            blob = row["embedding"]
            if blob:
                count = len(blob) // 4
                vec = list(struct.unpack_from(f"{count}f", blob))
                full_vectors.append(vec)
                labels.append({
                    "id":    row["id"],
                    "angle": row["angle_tag"],
                    "type":  row["embedding_type"],
                })

        if len(full_vectors) < 2:
            # Can't do PCA on 1 point — return raw first 2 dims
            points_2d = [[v[0] if len(v) > 0 else 0, v[1] if len(v) > 1 else 0]
                         for v in full_vectors]
            return {
                "person_id": person_id,
                "points": [{"x": p[0], "y": p[1], **labels[i]}
                           for i, p in enumerate(points_2d)],
                "method": "raw_dims",
            }

        try:
            from sklearn.decomposition import PCA
            import numpy as np
            X = np.array(full_vectors, dtype=np.float32)
            n_components = min(2, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(X)
            points = []
            for i, row_r in enumerate(reduced):
                pt = {"x": float(row_r[0]), "y": float(row_r[1]) if len(row_r) > 1 else 0.0}
                pt.update(labels[i])
                points.append(pt)
            return {"person_id": person_id, "points": points, "method": "pca"}
        except ImportError:
            points_2d = [[v[0] if len(v) > 0 else 0, v[1] if len(v) > 1 else 0]
                         for v in full_vectors]
            return {
                "person_id": person_id,
                "points": [{"x": p[0], "y": p[1], **labels[i]}
                           for i, p in enumerate(points_2d)],
                "method": "raw_dims",
            }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API — Camera History
# ---------------------------------------------------------------------------

@app.get("/api/camera_history")
async def get_camera_history(
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """Full camera_history table."""
    conn = _db_connect()
    try:
        cur = conn.execute(
            "SELECT ch.*, p.status as person_status "
            "FROM camera_history ch "
            "LEFT JOIN persons p ON ch.person_id = p.person_id "
            "ORDER BY ch.last_seen DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = [dict(r) for r in cur.fetchall()]
        total = conn.execute("SELECT COUNT(*) FROM camera_history").fetchone()[0]
        return {"history": rows, "total": total}
    finally:
        conn.close()


@app.get("/api/camera_history/{person_id}")
async def get_camera_history_for_person(person_id: str):
    """Camera movement timeline for one person."""
    conn = _db_connect()
    try:
        cur = conn.execute(
            "SELECT * FROM camera_history WHERE person_id = ? ORDER BY first_seen ASC",
            (person_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        return {"person_id": person_id, "history": rows}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API — Merge Proposals
# ---------------------------------------------------------------------------

@app.get("/api/merge_proposals")
async def get_merge_proposals(
    status: Optional[str] = Query(None, description="Filter: pending/accepted/rejected"),
):
    """All merge proposals with person thumbnails."""
    conn = _db_connect()
    try:
        sql = "SELECT * FROM merge_proposals"
        params: List[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY proposed_at DESC"
        cur = conn.execute(sql, params)
        proposals = [dict(r) for r in cur.fetchall()]

        # Attach first snapshot path for each person_id involved
        for prop in proposals:
            for key in ("person_id_a", "person_id_b"):
                pid = prop.get(key)
                if pid:
                    snap_dir = SNAPSHOTS_DIR / pid
                    imgs = sorted(snap_dir.glob("*.jpg")) if snap_dir.exists() else []
                    prop[f"{key}_thumb"] = str(imgs[0]) if imgs else None

        return {"proposals": proposals, "total": len(proposals)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API — Activity Log
# ---------------------------------------------------------------------------

@app.get("/api/activity_log")
async def get_activity_log(since_seconds: float = Query(120.0, ge=1.0, le=3600.0)):
    """Live activity feed — recent DB mutations detected by the tracker."""
    tracker = get_tracker()
    events = tracker.get_recent_activity(since_seconds=since_seconds)
    # Strip internal _ts key from response
    clean = []
    for e in reversed(events):  # newest first
        ev = {k: v for k, v in e.items() if not k.startswith("_")}
        clean.append(ev)
    return {"events": clean, "count": len(clean)}


# ---------------------------------------------------------------------------
# API — Track Registry
# ---------------------------------------------------------------------------

@app.get("/api/track_registry")
async def get_track_registry():
    """Read track_registry_session.json for live cam→person bindings."""
    reg_path = Path(TRACK_REGISTRY_PATH)
    if not reg_path.exists():
        return {"tracks": [], "note": "track_registry_session.json not found — Phase 2 not running?"}
    try:
        data = json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"tracks": [], "error": str(e)}

    # The JSON is a flat dict: {"cam0_track1": "uuid", "cam3_track2": "uuid", ...}
    def _parse_registry_entry(cam_track_key: str, person_id: str) -> dict:
        parts = cam_track_key.split("_track")
        cam_id  = parts[0] if parts else cam_track_key
        track_id = parts[1] if len(parts) > 1 else "?"
        return {"cam_id": cam_id, "track_id": track_id, "person_id": person_id}

    tracks = []
    try:
        conn = _db_connect()
        for cam_track_key, person_id in data.items():
            if not isinstance(person_id, str):
                continue
            entry = _parse_registry_entry(cam_track_key, person_id)
            cur = conn.execute(
                "SELECT status FROM persons WHERE person_id = ?", (person_id,)
            )
            row = cur.fetchone()
            entry["person_status"] = row["status"] if row else "unknown"
            tracks.append(entry)
        conn.close()
    except Exception:
        for cam_track_key, person_id in data.items():
            if not isinstance(person_id, str):
                continue
            entry = _parse_registry_entry(cam_track_key, person_id)
            entry["person_status"] = "unknown"
            tracks.append(entry)

    return {"tracks": tracks, "raw": data}


# ---------------------------------------------------------------------------
# API — Available cameras (distinct cam IDs)
# ---------------------------------------------------------------------------

@app.get("/api/cameras")
async def get_cameras():
    """Return list of distinct cam_ids seen in the DB."""
    conn = _db_connect()
    try:
        cur = conn.execute(
            "SELECT DISTINCT last_seen_cam FROM persons WHERE last_seen_cam IS NOT NULL "
            "UNION SELECT DISTINCT cam_id FROM camera_history ORDER BY 1"
        )
        cams = [row[0] for row in cur.fetchall()]
        return {"cameras": cams}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API — LLM Descriptions (Part 10)
# ---------------------------------------------------------------------------

@app.get("/api/persons/{person_id}/description")
async def get_person_description(person_id: str):
    """Latest + full description history (summary + parsed structured attributes)."""
    conn = _db_connect()
    try:
        prow = conn.execute(
            "SELECT latest_description_id FROM persons WHERE person_id=?", (person_id,)
        ).fetchone()
        if not prow:
            raise HTTPException(status_code=404, detail="Person not found")
        latest_id = prow["latest_description_id"]
        history: List[Dict[str, Any]] = []
        if _table_exists(conn, "person_descriptions"):
            rows = conn.execute(
                "SELECT id, described_at, backend, model_id, summary, attributes, confidence "
                "FROM person_descriptions WHERE person_id=? ORDER BY id DESC",
                (person_id,),
            ).fetchall()
            for r in rows:
                d = dict(r)
                try:
                    d["attributes"] = json.loads(d.get("attributes") or "{}")
                except Exception:
                    d["attributes"] = {}
                d["is_latest"] = (d["id"] == latest_id)
                history.append(d)
        return {"person_id": person_id, "latest_id": latest_id, "descriptions": history}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API — Face embeddings (Part 11 — ISOLATED store)
# ---------------------------------------------------------------------------

@app.get("/api/persons/{person_id}/face_embeddings")
async def get_person_face_embeddings(person_id: str):
    """Face-embedding metadata for a person (from the isolated face store)."""
    conn = _db_connect()
    try:
        if not _table_exists(conn, "face_embeddings"):
            return {"person_id": person_id, "face_embeddings": [], "count": 0}
        rows = conn.execute(
            "SELECT id, source_cam, captured_at, embedding "
            "FROM face_embeddings WHERE person_id=? ORDER BY id ASC",
            (person_id,),
        ).fetchall()
        out = []
        for r in rows:
            blob = r["embedding"]
            out.append({
                "id":          r["id"],
                "source_cam":  r["source_cam"],
                "captured_at": r["captured_at"],
                "vector_dims": len(blob) // 4 if blob else 0,
                "vector_norm": _embedding_norm(blob),
            })
        return {"person_id": person_id, "face_embeddings": out, "count": len(out)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API — Analytics (aggregates for the Overview charts)
# ---------------------------------------------------------------------------

@app.get("/api/analytics")
async def get_analytics():
    """System-wide aggregates for the Overview dashboard charts."""
    conn = _db_connect()
    try:
        pcols = _columns(conn, "persons")

        def grouped(sql: str) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for r in conn.execute(sql).fetchall():
                key = r[0]
                out[str(key) if key not in (None, "") else "unknown"] = r[1]
            return out

        def grouped_col(col: str) -> Dict[str, int]:
            # Skip columns that don't exist yet (un-migrated DB).
            if col not in pcols:
                return {}
            return grouped(f"SELECT {col}, COUNT(*) FROM persons GROUP BY {col}")

        analytics: Dict[str, Any] = {
            "by_status":    grouped("SELECT status, COUNT(*) FROM persons GROUP BY status"),
            "by_camera":    grouped("SELECT last_seen_cam, COUNT(*) FROM persons "
                                    "WHERE last_seen_cam IS NOT NULL GROUP BY last_seen_cam"),
            "by_gender":    grouped_col("gender"),
            "by_age_range": grouped_col("age_range"),
            "by_ethnicity": grouped_col("ethnicity"),
            "by_glasses":   grouped_col("glasses"),
            "gallery_hist": grouped("SELECT gallery_size, COUNT(*) FROM persons GROUP BY gallery_size"),
            "sightings_by_camera": grouped("SELECT cam_id, COUNT(*) FROM camera_history GROUP BY cam_id"),
            "persons_over_time":   grouped(
                "SELECT substr(created_at,1,13), COUNT(*) FROM persons "
                "WHERE created_at IS NOT NULL GROUP BY substr(created_at,1,13) "
                "ORDER BY substr(created_at,1,13)"
            ),
        }
        analytics["embeddings"] = {
            "body": conn.execute("SELECT COUNT(*) FROM person_embeddings").fetchone()[0],
            "face": (conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
                     if _table_exists(conn, "face_embeddings") else 0),
        }
        analytics["descriptions"] = {
            "described": conn.execute(
                "SELECT COUNT(*) FROM persons WHERE latest_description_id IS NOT NULL"
            ).fetchone()[0],
            "undescribed": conn.execute(
                "SELECT COUNT(*) FROM persons WHERE latest_description_id IS NULL"
            ).fetchone()[0],
        }
        return analytics
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API — Violence events (Part 11) + alert media
# ---------------------------------------------------------------------------

@app.get("/api/violence")
async def get_violence_events(limit: int = Query(300, ge=1, le=2000)):
    """Return violence_log.json events, newest first, with a small summary."""
    events = _read_violence_log()
    summary = {
        "total":      len(events),
        "violence":   sum(1 for e in events if e.get("level") == "VIOLENCE"),
        "suspicious": sum(1 for e in events if e.get("level") == "SUSPICIOUS"),
    }
    recent = list(reversed(events))[:limit]
    return {"events": recent, "summary": summary}


@app.get("/api/alert_media")
async def serve_alert_media(path: str = Query(...)):
    """Serve a violence alert snapshot (jpg) or clip (mp4) from the alerts dir."""
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / path
    try:
        p.resolve().relative_to(Path(ALERTS_DIR).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Media not found")
    ext = p.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".mp4": "video/mp4", ".avi": "video/x-msvideo",
    }.get(ext, "application/octet-stream")
    return FileResponse(str(p), media_type=media_type)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 8501))
    print(f"\n{'='*60}")
    print(f"  SURVEILLANT Debug Dashboard")
    print(f"  http://localhost:{port}")
    print(f"  Database: {DB_PATH}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
