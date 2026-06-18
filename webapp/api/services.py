"""
webapp/api/services.py
----------------------
Pure shaping helpers that turn engine `Database` rows into JSON-friendly dicts
for the API. Kept separate from routers so search (M4) can reuse the same
person/snapshot shaping as the persons explorer.

No heavy ML imports here — only the Database and the filesystem.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# Attribute fields stored directly on the persons row (from Part 11 face layer).
FACE_FIELDS = ("name", "gender", "age_range", "ethnicity", "glasses")

# Image bytes are served by an unauthenticated StaticFiles mount (see main.py).
# Fine for the single-tenant demo; per-tenant signed URLs are the later upgrade.
MEDIA_SNAPSHOTS_PREFIX = "/media/snapshots"


# ── Snapshots ───────────────────────────────────────────────────────────────
def list_snapshot_files(snapshots_dir: Path, person_id: str) -> List[str]:
    """Filenames of the seeded snapshot crops for a person (sorted)."""
    folder = snapshots_dir / person_id
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.glob("*.jpg"))


def snapshot_urls(person_id: str, filenames: List[str]) -> List[str]:
    return [f"{MEDIA_SNAPSHOTS_PREFIX}/{person_id}/{name}" for name in filenames]


# ── Person shaping ──────────────────────────────────────────────────────────
def person_summary(snapshots_dir: Path, p: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight row for the People table / search results."""
    pid = p["person_id"]
    files = list_snapshot_files(snapshots_dir, pid)
    urls = snapshot_urls(pid, files)
    return {
        "person_id": pid,
        "status": p.get("status"),
        "first_seen_cam": p.get("first_seen_cam"),
        "last_seen_cam": p.get("last_seen_cam"),
        "first_seen_time": p.get("first_seen_time"),
        "last_seen_time": p.get("last_seen_time"),
        "gallery_size": p.get("gallery_size"),
        "has_description": p.get("latest_description_id") is not None,
        **{f: p.get(f) for f in FACE_FIELDS},
        "snapshot_count": len(files),
        "thumbnail_url": urls[0] if urls else None,
    }


def search_hit(db, snapshots_dir: Path, person_id: str,
               score: Optional[float] = None,
               summary: Optional[str] = None) -> Dict[str, Any]:
    """Shape a search result: a person summary + an optional score/summary.
    Reused by all three search methods so results render identically."""
    p = db.get_person(person_id)
    if not p:
        return {"person_id": person_id, "score": score, "missing": True}
    s = person_summary(snapshots_dir, p)
    s["score"] = round(float(score), 4) if score is not None else None
    if summary is not None:
        s["summary"] = summary
    return s


def _gallery_meta(db, person_id: str) -> Dict[str, Any]:
    entries = db.get_gallery_typed(person_id)
    by_type: Counter = Counter()
    by_cam: Counter = Counter()
    dim = None
    for e in entries:
        by_type[e.get("type")] += 1
        by_cam[str(e.get("source_cam"))] += 1
        if dim is None and e.get("embedding") is not None:
            dim = int(e["embedding"].shape[0])
    return {
        "count": len(entries),
        "dim": dim,
        "by_type": dict(by_type),
        "by_camera": {k: v for k, v in by_cam.items()},
        # Per-embedding rows WITH ids (no vectors) for split selection.
        "entries": db.get_gallery_entries_meta(person_id),
    }


def shape_journey(db, person_id: str, overlap_groups: List[List[int]]) -> Dict[str, Any]:
    """Camera-history sightings sorted into a cross-camera timeline."""
    rows = db.get_camera_history_rows(person_id)  # includes row id (for split)
    rows.sort(key=lambda r: (r.get("first_seen") or ""))
    cameras = db.get_cameras_for_person(person_id)
    return {
        "person_id": person_id,
        "cameras": sorted(cameras),
        "overlap_groups": overlap_groups,
        "stops": [
            {
                "id": r.get("id"),
                "cam_id": r.get("cam_id"),
                "track_id": r.get("track_id"),
                "first_seen": r.get("first_seen"),
                "last_seen": r.get("last_seen"),
            }
            for r in rows
        ],
    }


def person_detail(db, snapshots_dir: Path, person_id: str,
                  overlap_groups: List[List[int]]) -> Optional[Dict[str, Any]]:
    p = db.get_person(person_id)
    if not p:
        return None
    files = list_snapshot_files(snapshots_dir, person_id)
    desc = db.get_latest_description(person_id)
    if desc:
        # The description row carries a raw float32 embedding BLOB (for semantic
        # search) and absolute host paths — neither is JSON-serialisable/safe to
        # expose. Strip them before returning.
        desc.pop("embedding", None)
        desc.pop("snapshots_used", None)
    return {
        **{k: p.get(k) for k in (
            "person_id", "status", "first_seen_cam", "first_seen_time",
            "last_seen_cam", "last_seen_time", "gallery_size", "known_angles",
            "latest_description_id", "created_at", *FACE_FIELDS,
        )},
        "cameras": sorted(db.get_cameras_for_person(person_id)),
        "gallery": _gallery_meta(db, person_id),
        "journey": shape_journey(db, person_id, overlap_groups),
        "description": desc,  # None when undescribed
        "snapshots": snapshot_urls(person_id, files),
    }


# ── Stats ───────────────────────────────────────────────────────────────────
def build_stats(db, alert_count: int) -> Dict[str, Any]:
    persons = db.get_all_persons()
    total = len(persons)
    by_status: Counter = Counter(p.get("status") for p in persons)
    described = sum(1 for p in persons if p.get("latest_description_id") is not None)
    total_body_emb = sum((p.get("gallery_size") or 0) for p in persons)

    by_gender: Counter = Counter(p.get("gender") for p in persons if p.get("gender"))
    by_ethnicity: Counter = Counter(p.get("ethnicity") for p in persons if p.get("ethnicity"))
    by_glasses: Counter = Counter(p.get("glasses") for p in persons if p.get("glasses"))

    # Per-camera sightings + multi-camera count (demo-scale: iterate persons).
    per_camera: Counter = Counter()
    multi_camera = 0
    for p in persons:
        cams = db.get_cameras_for_person(p["person_id"])
        if len(set(cams)) > 1:
            multi_camera += 1
        for c in cams:
            per_camera[str(c)] += 1

    return {
        "persons": total,
        "by_status": dict(by_status),
        "described": described,
        "undescribed": total - described,
        "multi_camera": multi_camera,
        "total_body_embeddings": total_body_emb,
        "per_camera_sightings": dict(per_camera),
        "distributions": {
            "gender": dict(by_gender),
            "ethnicity": dict(by_ethnicity),
            "glasses": dict(by_glasses),
        },
        "alerts": alert_count,
    }
