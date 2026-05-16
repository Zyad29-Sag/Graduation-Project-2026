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
)
from debug_activity_tracker import ActivityTracker

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
    """Aggregate DB counts: persons by status, embeddings, merges."""
    tracker = get_tracker()
    stats = tracker.get_stats()
    stats["uptime_seconds"] = int(time.time() - _start_time)
    stats["max_gallery_size"] = MAX_GALLERY_SIZE
    stats["db_path"] = str(DB_PATH)
    return stats


# ---------------------------------------------------------------------------
# API — Persons
# ---------------------------------------------------------------------------

@app.get("/api/persons")
async def get_persons(
    status: Optional[str] = Query(None, description="Filter by status"),
    cam: Optional[int] = Query(None, description="Filter by last_seen_cam"),
    search: Optional[str] = Query(None, description="Search by person_id prefix"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Return all persons with optional filtering and pagination."""
    conn = _db_connect()
    try:
        sql = "SELECT * FROM persons WHERE 1=1"
        params: List[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if cam is not None:
            sql += " AND last_seen_cam = ?"
            params.append(cam)
        if search:
            sql += " AND person_id LIKE ?"
            params.append(f"{search}%")
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cur = conn.execute(sql, params)
        rows = [_row_to_dict(r) for r in cur.fetchall()]

        # Count total (for pagination)
        count_sql = sql.replace("SELECT *", "SELECT COUNT(*)", 1)
        count_sql = count_sql[:count_sql.rfind("LIMIT")]
        total = conn.execute(count_sql, params[:-2]).fetchone()[0]

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
