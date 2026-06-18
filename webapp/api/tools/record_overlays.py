"""
webapp/api/tools/record_overlays.py
-----------------------------------
Offline "overlay recorder".

The Live-Cams page streams the raw WiseNet videos as MJPEG. To show the *real*
detection boxes + person IDs without running the (CPU-heavy) pipeline on every
request, we pre-compute the overlay metadata ONCE, here, and let the MJPEG
endpoint burn the boxes in at serve time (see routers/cameras.py).

For each demo camera video we:
  1. read every frame sequentially and run the engine's own detector + ByteTrack
     tracker to get per-frame boxes + per-camera track IDs;
  2. keep the largest crop seen for each track, embed it with OSNet, and search
     the seeded demo gallery once per track to bind track_id -> person_id (so the
     overlay IDs match the people shown elsewhere in the webapp);
  3. write a per-camera JSON sidecar keyed by frame index:
         webapp/api/data/demo/overlays/cam{i}.json
     { "total_frames": N,
       "frames": { "0": [ {bbox, track_id, person_id, state, status,
                           gallery_size, name, gender, age_range, ...} ], ... } }

Frame indices are produced by reading frames 0..N-1 in order; the MJPEG endpoint
reads the same file the same way and loops with `idx % total_frames`, so stored
boxes line up exactly with playback.

Run (from the repo root):
    python -m webapp.api.tools.record_overlays
    python -m webapp.api.tools.record_overlays --cam 0 --max-frames 300   # quick test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── Bootstrap: make `webapp.api...` importable even when run as a file ───────
_REPO_ROOT = Path(__file__).resolve().parents[3]  # tools -> api -> webapp -> repo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from webapp.api import config, engine  # engine import puts surveillant/ on sys.path

# Engine modules (importable now that engine.py added surveillant/ to sys.path).
import cv2  # noqa: E402
from config.settings import (  # noqa: E402  (this `config` is the engine's settings package)
    YOLO_MODEL,
    DETECTION_CONF,
    DETECTION_IMGSZ,
    BODY_MATCH_THRESHOLD,
)
from modules.detection.detector import PersonDetector  # noqa: E402
from modules.tracking.tracker import PersonTracker  # noqa: E402

# Minimum crop size before we bother embedding a track for identification.
_MIN_CROP_W = 24
_MIN_CROP_H = 48


def _best_crop_area(bbox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _person_meta(db, person_id: str) -> dict:
    """Pull the display attributes for a matched person from the demo DB."""
    p = db.get_person(person_id) or {}
    return {
        "status":       p.get("status") or "unverified",
        "gallery_size": int(p.get("gallery_size") or 0),
        "name":         p.get("name"),
        "gender":       p.get("gender"),
        "age_range":    p.get("age_range"),
        "ethnicity":    p.get("ethnicity"),
        "glasses":      p.get("glasses"),
    }


def record_camera(cam_id: int, video_path: Path, detector, searcher, db,
                  max_frames: int | None = None) -> dict:
    """Run detection+tracking over one video and return its overlay sidecar dict."""
    print(f"\n[cam{cam_id}] {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[cam{cam_id}] !! could not open video — skipping")
        return {"total_frames": 0, "frames": {}}

    tracker = PersonTracker(cam_id)

    # Pass 1 — per-frame raw boxes + best crop per track.
    raw_by_frame: dict[int, list[dict]] = {}
    best_crop: dict[int, tuple[int, "cv2.Mat"]] = {}  # track_id -> (area, crop)
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        boxes = []
        for tr in tracks:
            tid = tr["track_id"]
            x1, y1, x2, y2 = tr["bbox"]
            boxes.append({"track_id": tid, "bbox": [int(x1), int(y1), int(x2), int(y2)]})

            area = _best_crop_area(tr["bbox"])
            if (x2 - x1) >= _MIN_CROP_W and (y2 - y1) >= _MIN_CROP_H:
                prev = best_crop.get(tid)
                if prev is None or area > prev[0]:
                    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
                    if crop.size > 0:
                        best_crop[tid] = (area, crop)

        if boxes:
            raw_by_frame[frame_idx] = boxes

        frame_idx += 1
        if frame_idx % 100 == 0:
            fps = frame_idx / max(1e-6, time.time() - t0)
            print(f"[cam{cam_id}]   {frame_idx} frames "
                  f"({len(best_crop)} tracks, {fps:.1f} fps)…")
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    total_frames = frame_idx
    print(f"[cam{cam_id}] decode done: {total_frames} frames, {len(best_crop)} tracks. Identifying…")

    # Decide person_id (once) per track + cache its display meta.
    track_person: dict[int, str | None] = {}
    person_meta: dict[str, dict] = {}
    for tid, (_, crop) in best_crop.items():
        vec = searcher.embedder.extract_body_embedding(crop)
        hits = searcher.search_by_embedding(
            vec, query_embedding_type="body", top_k=1, min_threshold=BODY_MATCH_THRESHOLD,
        )
        if hits:
            pid = hits[0]["person_id"]
            track_person[tid] = pid
            if pid not in person_meta:
                person_meta[pid] = _person_meta(db, pid)
        else:
            track_person[tid] = None

    matched = sum(1 for v in track_person.values() if v)
    print(f"[cam{cam_id}] identified {matched}/{len(track_person)} tracks "
          f"-> {len(person_meta)} distinct persons.")

    # Pass 2 — attach identity + state to every recorded box.
    frames_out: dict[str, list[dict]] = {}
    for idx, boxes in raw_by_frame.items():
        out = []
        for b in boxes:
            tid = b["track_id"]
            pid = track_person.get(tid)
            box = {"bbox": b["bbox"], "track_id": tid}
            if pid:
                box["person_id"] = pid
                box["state"] = "new"
                meta = person_meta.get(pid, {})
                box["status"] = meta.get("status", "confirmed")
                box["gallery_size"] = meta.get("gallery_size", 0)
                for k in ("name", "gender", "age_range", "ethnicity", "glasses"):
                    if meta.get(k):
                        box[k] = meta[k]
            else:
                box["state"] = "collecting"
            out.append(box)
        frames_out[str(idx)] = out

    return {"total_frames": total_frames, "frames": frames_out}


def main() -> int:
    ap = argparse.ArgumentParser(description="Record Live-Cams detection overlays.")
    ap.add_argument("--cam", type=int, default=None,
                    help="Record a single camera index (default: all).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Stop each camera after N frames (quick test).")
    args = ap.parse_args()

    out_dir = config.OVERLAYS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    db = engine.get_database(config.DEMO_DB_PATH)
    searcher = engine.get_person_searcher(config.DEMO_DB_PATH)  # OSNet + FAISS over demo gallery
    detector = PersonDetector(YOLO_MODEL, DETECTION_CONF, DETECTION_IMGSZ)

    cam_ids = ([args.cam] if args.cam is not None
               else list(range(len(config.DEMO_CAMERA_VIDEOS))))

    wrote = 0
    for cam_id in cam_ids:
        if cam_id < 0 or cam_id >= len(config.DEMO_CAMERA_VIDEOS):
            print(f"[cam{cam_id}] out of range — skipping")
            continue
        video_path = config.DEMO_CAMERA_VIDEOS[cam_id]
        if not video_path.exists():
            print(f"[cam{cam_id}] video not found: {video_path} — skipping")
            continue
        sidecar = record_camera(cam_id, video_path, detector, searcher, db,
                                max_frames=args.max_frames)
        out_path = config.overlay_sidecar(cam_id)
        out_path.write_text(json.dumps(sidecar), encoding="utf-8")
        kb = out_path.stat().st_size / 1024
        print(f"[cam{cam_id}] wrote {out_path}  ({kb:.0f} KB, "
              f"{len(sidecar['frames'])} annotated frames)")
        wrote += 1

    print(f"\nDone. Wrote {wrote} sidecar(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
