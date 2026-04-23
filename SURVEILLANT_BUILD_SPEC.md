# SURVEILLANT — Build Specification

## IMPORTANT INSTRUCTIONS FOR YOU (THE BUILDER)

- Build **one phase at a time**.
- After completing each phase, **STOP and wait** for the user to test and confirm before proceeding.
- Never merge code from a later phase into an earlier phase's work.
- Every phase must have its own runnable test script.
- When you finish a phase, tell the user exactly what command to run to test it.
- If a phase requires a library install, always provide the exact pip command.
- Write clean, well-commented code. This is a graduation project — it will be read by professors.

---

## Project Overview

**SURVEILLANT** is an AI-powered multi-camera surveillance intelligence system. It ingests video streams from multiple cameras, tracks every person across all cameras, builds a searchable database of embeddings and descriptions, and allows natural-language search queries via an LLM.

**Dataset being used:** WiseNet (Kaggle) — 5 synchronized, overlapping surveillance camera videos. Each video = one camera. All cameras cover the same scene from different angles.

**Final stack (build toward this):**
- Python 3.10+
- YOLOv8 (person detection)
- DeepSORT (per-camera tracking)
- InsightFace (face embeddings)
- SQLite → later PostgreSQL + pgvector
- Ollama + Qwen2.5-VL:2b (LLM descriptions and query parsing)
- OpenCV (video reading and display)
- React + FastAPI (web interface — later phases)
- Docker + Docker Compose (later phases)

---

## Project Folder Structure

Build this structure from scratch. Create all folders and empty `__init__.py` files on start.

```
surveillant/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # all config: paths, thresholds, model names
│
├── modules/
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   └── simulator.py         # reads video files, simulates live streams
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py          # YOLOv8 person detection
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py           # DeepSORT per-camera tracking
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py          # InsightFace face + body embeddings
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── database.py          # SQLite operations
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   └── searcher.py          # cosine similarity search
│   │
│   └── llm/
│       ├── __init__.py
│       └── describer.py         # Ollama + Qwen integration
│
├── display/
│   ├── __init__.py
│   └── visualizer.py            # OpenCV multi-cam grid display
│
├── data/
│   ├── videos/                  # user places WiseNet .mp4 files here
│   │   └── .gitkeep
│   └── snapshots/               # auto-created, saved person crop images
│       └── .gitkeep
│
├── database/
│   └── surveillant.db           # auto-created SQLite file
│
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── test_phase4.py
│
├── main.py                      # entry point — orchestrates all phases
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## .gitignore (create this immediately)

```
__pycache__/
*.pyc
*.pyo
.env
data/snapshots/*
!data/snapshots/.gitkeep
database/surveillant.db
*.mp4
*.avi
*.mov
.DS_Store
```

---

## requirements.txt (full project — install as needed per phase)

```
# Core
opencv-python==4.9.0.80
numpy==1.26.4
Pillow==10.3.0

# Detection
ultralytics==8.2.0

# Tracking
deep-sort-realtime==1.3.2

# Embedding
insightface==0.7.3
onnxruntime==1.18.0

# Storage
SQLAlchemy==2.0.30

# LLM
ollama==0.2.1

# Search
scikit-learn==1.5.0

# Utils
python-dotenv==1.0.1
tqdm==4.66.4
```

---

## config/settings.py

```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Paths
VIDEOS_DIR     = BASE_DIR / "data" / "videos"
SNAPSHOTS_DIR  = BASE_DIR / "data" / "snapshots"
DB_PATH        = BASE_DIR / "database" / "surveillant.db"

# Camera
FPS_TARGET     = 5          # process every Nth frame for performance
DISPLAY_WIDTH  = 640        # width of each camera window in grid
DISPLAY_HEIGHT = 360        # height of each camera window in grid
GRID_COLS      = 3          # cameras per row in display grid

# Detection
YOLO_MODEL     = "yolov8n.pt"   # nano = fastest, use yolov8s.pt for better accuracy
DETECTION_CONF = 0.5            # minimum confidence for person detection
DETECTION_CLASS = 0             # class 0 = person in COCO

# Tracking
MAX_AGE        = 30         # frames before a lost track is deleted
MIN_HITS       = 3          # frames before a track is confirmed
IOU_THRESHOLD  = 0.3

# Embedding
FACE_DET_SIZE  = (640, 640)
MIN_FACE_CONF  = 0.5
NUM_FRAMES_FOR_EMBEDDING = 5    # collect this many frames before computing final embedding

# Search
SIMILARITY_THRESHOLD = 0.6      # cosine similarity threshold for a match

# LLM
OLLAMA_HOST    = "http://localhost:11434"
LLM_MODEL      = "qwen2.5vl:2b"

# Storage
SNAPSHOT_QUALITY = 90       # JPEG quality for saved crops
MAX_SNAPSHOTS_PER_PERSON = 5
```

---

# PHASE 1 — Camera Simulator + Detection + Tracking + Display

## Goal
Build a system that:
1. Accepts N video files (simulating N live cameras)
2. Reads all videos simultaneously in sync
3. Runs YOLOv8 person detection on each frame
4. Runs DeepSORT tracking to assign persistent Track IDs per camera
5. Displays all cameras in a real-time grid with bounding boxes and Track IDs drawn
6. Prints a live log: which camera, which track ID, bounding box coordinates

## What to build

### modules/camera/simulator.py
- Class `CameraSimulator`
- `__init__(self, video_paths: list[str], fps_target: int)`
- `start()` — opens all VideoCapture objects
- `read_frames()` → yields `dict[cam_id: int, frame: np.ndarray]` for each synchronized frame set
- `release()` — releases all captures
- Loops videos when they end (simulate infinite live stream)
- Skips frames to match `fps_target` (don't process every frame)

### modules/detection/detector.py
- Class `PersonDetector`
- `__init__(self, model_name: str, conf: float)`
- `detect(self, frame: np.ndarray)` → returns `list[dict]`
  - Each dict: `{bbox: [x1,y1,x2,y2], confidence: float}`
- Only detects class 0 (person)

### modules/tracking/tracker.py
- Class `PersonTracker`
- One tracker instance per camera
- `__init__(self, cam_id: int)`
- `update(self, detections: list[dict], frame: np.ndarray)` → returns `list[dict]`
  - Each dict: `{track_id: int, bbox: [x1,y1,x2,y2], cam_id: int}`
- Uses `deep_sort_realtime.DeepSort`

### display/visualizer.py
- Class `GridDisplay`
- `__init__(self, num_cams: int, cols: int, cell_w: int, cell_h: int)`
- `update(self, cam_id: int, frame: np.ndarray, tracks: list[dict])` — draws boxes and labels on frame
- `render()` — assembles all frames into one grid image and shows with `cv2.imshow`
- `should_quit()` → bool — returns True if user pressed 'q'
- Each bounding box should be drawn in a unique color per Track ID
- Label format: `CAM{cam_id} | T{track_id}`

### main.py (Phase 1 version)
```python
# Entry point for Phase 1
# Usage: python main.py --phase 1 --videos data/videos/cam1.mp4 data/videos/cam2.mp4 ...

import argparse
from modules.camera.simulator import CameraSimulator
from modules.detection.detector import PersonDetector
from modules.tracking.tracker import PersonTracker
from display.visualizer import GridDisplay
from config.settings import *

def run_phase1(video_paths):
    simulator = CameraSimulator(video_paths, FPS_TARGET)
    detector  = PersonDetector(YOLO_MODEL, DETECTION_CONF)
    trackers  = {i: PersonTracker(i) for i in range(len(video_paths))}
    display   = GridDisplay(len(video_paths), GRID_COLS, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    simulator.start()

    while not display.should_quit():
        frames = simulator.read_frames()
        if not frames:
            break

        for cam_id, frame in frames.items():
            detections = detector.detect(frame)
            tracks     = trackers[cam_id].update(detections, frame)
            display.update(cam_id, frame, tracks)

            for t in tracks:
                print(f"[CAM {cam_id}] Track {t['track_id']} @ {t['bbox']}")

        display.render()

    simulator.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    args = parser.parse_args()

    if args.phase == 1:
        run_phase1(args.videos)
```

## Install commands for Phase 1
```bash
pip install opencv-python ultralytics deep-sort-realtime numpy
```

## Acceptance criteria (test before moving on)
- [ ] All 5 camera feeds display simultaneously in one window grid
- [ ] Bounding boxes appear around every person detected
- [ ] Track IDs are stable — the same person keeps the same ID across frames
- [ ] When a person exits and re-enters the same camera, they may get a new ID (this is expected in Phase 1)
- [ ] Pressing 'q' closes the display cleanly
- [ ] No crash when a camera video loops back to start

## Test script: tests/test_phase1.py
```python
"""
Run: python tests/test_phase1.py
Tests Phase 1 components in isolation without needing video files.
"""
import numpy as np
import cv2
from modules.detection.detector import PersonDetector
from modules.tracking.tracker import PersonTracker
from display.visualizer import GridDisplay

def test_detector_runs():
    detector = PersonDetector("yolov8n.pt", 0.5)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(fake_frame)
    assert isinstance(result, list), "Detector must return a list"
    print("✓ Detector runs on blank frame without crash")

def test_tracker_runs():
    tracker = PersonTracker(cam_id=0)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_detections = [{"bbox": [100, 100, 200, 300], "confidence": 0.9}]
    result = tracker.update(fake_detections, fake_frame)
    assert isinstance(result, list), "Tracker must return a list"
    print("✓ Tracker runs with fake detections without crash")

def test_display_grid():
    display = GridDisplay(num_cams=2, cols=2, cell_w=320, cell_h=240)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_tracks = [{"track_id": 1, "bbox": [50, 50, 150, 200], "cam_id": 0}]
    display.update(0, fake_frame, fake_tracks)
    display.update(1, fake_frame, [])
    display.render()
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print("✓ Grid display renders two cameras without crash")

if __name__ == "__main__":
    test_detector_runs()
    test_tracker_runs()
    test_display_grid()
    print("\nAll Phase 1 tests passed.")
```

## ⛔ STOP HERE
Run `python tests/test_phase1.py` then run the full system with your 5 WiseNet videos.
**Do not proceed to Phase 2 until the user confirms Phase 1 is working correctly.**

---

# PHASE 2 — Person Capture + Embedding + Simple Storage

## Goal
For every newly confirmed track (a Track ID that has been stable for `MIN_HITS` frames):
1. Collect `NUM_FRAMES_FOR_EMBEDDING` (5) crops of the person at different moments
2. Try to extract a face embedding using InsightFace
3. If no face detected, extract a body embedding using the full bounding box crop resized to 112x112
4. Average all embeddings into one final representative vector
5. Save 1-3 best crop images to `data/snapshots/person_{global_id}/`
6. Write a record to SQLite database
7. Continue displaying everything from Phase 1 with an additional indicator when a person is "processed" (green box = processed, yellow = collecting, red = just detected)

## What to build

### modules/storage/database.py
- Creates `database/surveillant.db` on first run
- Table `persons`:
  ```sql
  CREATE TABLE IF NOT EXISTS persons (
      person_id     TEXT PRIMARY KEY,   -- UUID
      track_id      INTEGER NOT NULL,
      cam_id        INTEGER NOT NULL,
      embedding     BLOB NOT NULL,      -- numpy array serialized as bytes
      embedding_type TEXT NOT NULL,     -- 'face' or 'body'
      description   TEXT,              -- NULL until Phase 4
      gender        TEXT,
      age_range     TEXT,
      first_seen_cam  INTEGER,
      first_seen_time TEXT,
      last_seen_cam   INTEGER,
      last_seen_time  TEXT,
      snapshot_paths  TEXT,             -- JSON list of file paths
      created_at    TEXT NOT NULL
  );
  ```
- Functions:
  - `insert_person(record: dict) → str` (returns person_id)
  - `get_all_persons() → list[dict]`
  - `get_person(person_id: str) → dict`
  - `update_last_seen(person_id, cam_id, timestamp)`
  - `update_description(person_id, description, gender, age_range)`
  - `get_all_embeddings() → list[tuple[person_id, np.ndarray]]`

### modules/embedding/embedder.py
- Class `PersonEmbedder`
- `__init__(self)`  — loads InsightFace model `buffalo_sc` (small, fast)
- `extract_face_embedding(self, crop: np.ndarray) → np.ndarray | None`
  - Returns 512-dim embedding or None if no face detected
- `extract_body_embedding(self, crop: np.ndarray) → np.ndarray`
  - Resizes crop to 112x112, normalizes pixel values, flattens → 512-dim vector (fallback)
- `aggregate_embeddings(self, embeddings: list[np.ndarray]) → np.ndarray`
  - Returns `np.mean(embeddings, axis=0)` normalized to unit length
- `serialize(self, embedding: np.ndarray) → bytes`
- `deserialize(self, data: bytes) → np.ndarray`

### Update main.py for Phase 2
- Add a `FrameBuffer` dict: `{(cam_id, track_id): list[np.ndarray crops]}`
- When a track has been seen for `MIN_HITS` frames but buffer has < `NUM_FRAMES_FOR_EMBEDDING` crops → add crop to buffer, draw YELLOW box
- When buffer reaches `NUM_FRAMES_FOR_EMBEDDING` → process: extract embedding, save snapshots, write to DB, draw GREEN box
- Print: `[DB] New person stored: person_id={...} cam={...} type=face/body`

## Acceptance criteria
- [ ] After running for ~30 seconds, `database/surveillant.db` contains records
- [ ] `data/snapshots/` folder contains subfolders with person crop images
- [ ] Bounding box colors change: yellow (collecting) → green (stored)
- [ ] No duplicate records for the same track in the same camera
- [ ] Can open `surveillant.db` with any SQLite viewer and see proper records
- [ ] Embeddings stored as BLOB can be deserialized back to numpy arrays

## Test script: tests/test_phase2.py
```python
"""Run: python tests/test_phase2.py"""
import numpy as np
import os, json
from modules.embedding.embedder import PersonEmbedder
from modules.storage.database import Database

def test_embedder():
    emb = PersonEmbedder()
    fake_crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    body_vec = emb.extract_body_embedding(fake_crop)
    assert body_vec.shape == (512,), f"Expected (512,) got {body_vec.shape}"
    print(f"✓ Body embedding shape: {body_vec.shape}")

def test_embedding_aggregation():
    emb = PersonEmbedder()
    vecs = [np.random.randn(512).astype(np.float32) for _ in range(5)]
    avg = emb.aggregate_embeddings(vecs)
    assert avg.shape == (512,)
    norm = np.linalg.norm(avg)
    assert abs(norm - 1.0) < 0.01, f"Aggregated embedding should be unit norm, got {norm}"
    print(f"✓ Aggregation produces unit-norm vector: norm={norm:.4f}")

def test_serialize_deserialize():
    emb = PersonEmbedder()
    original = np.random.randn(512).astype(np.float32)
    serialized = emb.serialize(original)
    restored = emb.deserialize(serialized)
    assert np.allclose(original, restored), "Serialization roundtrip failed"
    print("✓ Embedding serialization/deserialization roundtrip OK")

def test_database_insert_and_read():
    db = Database(":memory:")  # use in-memory DB for testing
    record = {
        "track_id": 1,
        "cam_id": 0,
        "embedding": np.random.randn(512).astype(np.float32).tobytes(),
        "embedding_type": "body",
        "first_seen_cam": 0,
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_cam": 0,
        "last_seen_time": "2024-01-01T10:00:05",
        "snapshot_paths": json.dumps([]),
    }
    pid = db.insert_person(record)
    assert pid is not None
    persons = db.get_all_persons()
    assert len(persons) == 1
    print(f"✓ Database insert/read OK, person_id={pid}")

if __name__ == "__main__":
    test_embedder()
    test_embedding_aggregation()
    test_serialize_deserialize()
    test_database_insert_and_read()
    print("\nAll Phase 2 tests passed.")
```

## ⛔ STOP HERE
Run tests, then run the full system and let it process the 5 videos for at least 2 minutes.
Check the database and snapshot folder.
**Do not proceed to Phase 3 until the user confirms.**

---

# PHASE 3 — Photo Search

## Goal
Given a query photo of a person, find all matching records in the database and return:
- The matched person's ID
- Which camera they were last seen on
- Their timestamp
- Their saved snapshot
- Similarity score

## What to build

### modules/search/searcher.py
- Class `PersonSearcher`
- `__init__(self, db: Database, embedder: PersonEmbedder)`
- `search_by_photo(self, query_image_path: str, top_k: int = 5) → list[dict]`
  - Load query image
  - Try face embedding first, fall back to body embedding
  - Load all embeddings from database
  - Compute cosine similarity against all stored embeddings
  - Return top_k matches sorted by score descending
  - Each result: `{person_id, cam_id, last_seen_time, similarity_score, snapshot_path}`
- `search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5) → list[dict]`
  - Same as above but accepts pre-computed embedding

### Add to main.py — interactive search mode
```
python main.py --phase 3 --query path/to/photo.jpg
```
- Prints top 5 results to terminal in a readable format
- Opens the matched snapshot images side by side with the query image using OpenCV for visual confirmation

## Acceptance criteria
- [ ] Given a cropped image of a person from `data/snapshots/`, the top result is the correct person
- [ ] Similarity scores are between 0.0 and 1.0
- [ ] Query on a completely different person returns low scores (< 0.4)
- [ ] Results include camera ID and timestamp
- [ ] Visual comparison window opens showing query vs match

## Test script: tests/test_phase3.py
```python
"""Run: python tests/test_phase3.py"""
import numpy as np
from modules.search.searcher import PersonSearcher
from modules.embedding.embedder import PersonEmbedder
from modules.storage.database import Database
import json

def test_cosine_similarity_identical():
    from sklearn.metrics.pairwise import cosine_similarity
    v = np.random.randn(512).astype(np.float32)
    v = v / np.linalg.norm(v)
    score = cosine_similarity([v], [v])[0][0]
    assert abs(score - 1.0) < 0.001, "Identical vectors must have similarity 1.0"
    print(f"✓ Cosine similarity of identical vectors = {score:.4f}")

def test_cosine_similarity_different():
    from sklearn.metrics.pairwise import cosine_similarity
    v1 = np.random.randn(512).astype(np.float32)
    v2 = np.random.randn(512).astype(np.float32)
    score = cosine_similarity([v1], [v2])[0][0]
    assert score < 0.5, f"Random vectors should have low similarity, got {score}"
    print(f"✓ Cosine similarity of random vectors = {score:.4f} (low as expected)")

def test_search_returns_correct_format():
    db = Database(":memory:")
    emb = PersonEmbedder()
    embedding = np.random.randn(512).astype(np.float32)
    db.insert_person({
        "track_id": 1,
        "cam_id": 2,
        "embedding": emb.serialize(embedding),
        "embedding_type": "body",
        "first_seen_cam": 2,
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_cam": 2,
        "last_seen_time": "2024-01-01T10:00:10",
        "snapshot_paths": json.dumps([]),
    })
    searcher = PersonSearcher(db, emb)
    results = searcher.search_by_embedding(embedding, top_k=3)
    assert len(results) >= 1
    assert "person_id" in results[0]
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] > 0.99
    print(f"✓ Search returns correct format, top score = {results[0]['similarity_score']:.4f}")

if __name__ == "__main__":
    test_cosine_similarity_identical()
    test_cosine_similarity_different()
    test_search_returns_correct_format()
    print("\nAll Phase 3 tests passed.")
```

## ⛔ STOP HERE
Take one of the saved snapshots from Phase 2 and use it as a query photo.
The system should find the same person.
**Do not proceed to Phase 4 until the user confirms.**

---

# PHASE 4 — LLM Description Generation

## Goal
For every person in the database that has no description yet, use Qwen2.5-VL:2b via Ollama to:
1. Analyze their best snapshot image
2. Generate a structured description: age range, gender, clothing top, clothing bottom, hair color, beard, glasses, distinctive features
3. Store the description in the database

Also expose a text-query search: "find a young man with a red jacket" → parse with LLM → search DB.

## Prerequisites
```bash
# Install Ollama first: https://ollama.ai/install
ollama pull qwen2.5vl:2b
```

## What to build

### modules/llm/describer.py
- Class `PersonDescriber`
- `__init__(self, model: str, host: str)`
- `describe_from_image(self, image_path: str) → dict`
  - Sends image to Qwen2.5-VL
  - Returns structured dict:
    ```python
    {
      "gender": "male|female|unknown",
      "age_range": "0-10|10-20|20-30|30-45|45-60|60+",
      "hair_color": str or None,
      "beard": bool or None,
      "glasses": bool or None,
      "clothing_top": str or None,
      "clothing_bottom": str or None,
      "clothing_colors": list[str],
      "distinctive_features": str or None,
      "summary": str
    }
    ```
  - Prompt must instruct model to return ONLY JSON, no markdown
  - Include JSON cleaning logic (strip ```json fences if present)

- `parse_search_query(self, natural_language: str) → dict`
  - Parses a security guard's text description into the same structured format
  - Used for text-based search in Phase 4

### Run mode: batch describe
```
python main.py --phase 4 --describe-all
```
- Loops through all persons in DB with no description
- Sends their best snapshot to Qwen
- Updates DB with description, gender, age_range
- Prints progress: `[LLM] person_id=... → male, 30-45, red jacket, black pants`

### Run mode: text search
```
python main.py --phase 4 --search-text "young man with red jacket near entrance"
```
- Parses query with LLM
- Filters DB by matching gender, age_range, description keywords
- Returns matches

## Acceptance criteria
- [ ] `ollama run qwen2.5vl:2b` works before running this phase
- [ ] All persons in database get a text description after `--describe-all`
- [ ] Descriptions are plausible (not random/hallucinated) for the person in the image
- [ ] Text search returns relevant results for a description that matches a stored person
- [ ] LLM errors are handled gracefully (timeout, model not found → log error, skip person, continue)

## Test script: tests/test_phase4.py
```python
"""Run: python tests/test_phase4.py"""
import json
from modules.llm.describer import PersonDescriber

def test_query_parser_output_format():
    desc = PersonDescriber(model="qwen2.5vl:2b", host="http://localhost:11434")
    result = desc.parse_search_query(
        "A man around 30, wearing a blue shirt and black pants, short hair, no glasses"
    )
    assert isinstance(result, dict), "Must return a dict"
    required_keys = ["gender", "age_range", "clothing_top"]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"
    print(f"✓ Query parser returns dict with required keys")
    print(f"  Result: {json.dumps(result, indent=2)}")

def test_json_cleaning():
    raw = '```json\n{"gender": "male", "age_range": "20-30"}\n```'
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    parsed = json.loads(cleaned.strip())
    assert parsed["gender"] == "male"
    print("✓ JSON fence cleaning logic works correctly")

if __name__ == "__main__":
    test_json_cleaning()
    try:
        test_query_parser_output_format()
    except Exception as e:
        print(f"⚠ LLM test skipped (Ollama not running): {e}")
    print("\nPhase 4 tests complete.")
```

## ⛔ STOP HERE
Run `--describe-all` on your populated database, then run a text search.
**Do not proceed to Phase 5 until the user confirms.**

---

# PHASE 5 — Cross-Camera Person Linking

## Goal
Detect when the same physical person appears on two different cameras and link their records in the database. A person who walks from Camera 1 to Camera 3 should end up as a single database entry, not two separate ones.

## Approach
Use embedding cosine similarity across cameras. When a new person is stored on Camera B with a similarity score > `SIMILARITY_THRESHOLD` against an existing record from Camera A, merge the records.

## What to build

### Add to modules/storage/database.py
- Table `cross_camera_links`:
  ```sql
  CREATE TABLE IF NOT EXISTS cross_camera_links (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      person_id_a   TEXT,
      person_id_b   TEXT,
      similarity    REAL,
      linked_at     TEXT,
      FOREIGN KEY (person_id_a) REFERENCES persons(person_id),
      FOREIGN KEY (person_id_b) REFERENCES persons(person_id)
  );
  ```
- `link_persons(person_id_a, person_id_b, similarity)` — insert a link
- `get_person_links(person_id) → list[dict]` — get all linked persons
- `get_canonical_id(person_id) → str` — follow link chain to master record

### Add cross-camera matching in main pipeline
- After storing a new person in Phase 2, immediately run a search
- If similarity > `SIMILARITY_THRESHOLD` with an existing record from a **different camera** → create a link
- In the display: persons linked across cameras share the same color bounding box

## Acceptance criteria
- [ ] The same person appearing on 2 cameras creates a `cross_camera_links` record
- [ ] `get_canonical_id` returns the same master ID for both camera records
- [ ] Bounding boxes for the same person are the same color across all cameras in the display

## ⛔ STOP HERE. Confirm with user before continuing.

---

# PHASE 6 — FastAPI Web Interface

## Goal
Expose the system's capabilities as a REST API so a frontend can interact with it.

## Endpoints to build

```
GET  /health                          → {"status": "ok", "persons_in_db": N}

GET  /persons                         → list all stored persons (id, cam, time, snapshot_url)
GET  /persons/{person_id}             → full record for one person
GET  /persons/{person_id}/snapshot    → serve the snapshot image file

POST /search/by-photo                 → upload image, returns top 5 matches
     body: multipart/form-data { image: file }
     response: [{person_id, cam_id, similarity, snapshot_url, description}]

POST /search/by-text                  → text query search
     body: { "query": "young man with red jacket" }
     response: [{person_id, cam_id, similarity, description}]

GET  /cameras/live                    → WebSocket endpoint for live frame stream
     streams: MJPEG or base64 frames with bounding boxes for each camera

GET  /analytics/demographics          → age/gender breakdown across all stored persons
GET  /analytics/timeline/{person_id}  → movement timeline for one person across cameras
```

## Tech
- FastAPI with uvicorn
- Serve snapshot images as static files
- WebSocket for live camera feeds

## ⛔ STOP HERE. Confirm with user before continuing.

---

# PHASE 7 — React Frontend Dashboard

## Goal
A browser-based dashboard with:
- Live multi-camera grid (WebSocket stream from Phase 6)
- Search panel: upload photo OR type text description
- Results panel: shows matched persons with snapshots and location info
- Demographics sidebar: real-time age/gender bar charts
- Person timeline view: click a person → see their movement across cameras

## Tech
- React 18
- Vite (dev server)
- Axios (API calls)
- Chart.js (demographics charts)
- Served by Nginx in Docker

## ⛔ STOP HERE. Confirm with user before continuing.

---

# PHASE 8 — Dockerize Everything

## Goal
All services run from a single `docker compose up` command.

## Services in docker-compose.yml
- `db` — PostgreSQL 16 + pgvector (migrate from SQLite)
- `redis` — message queue between services
- `ollama` — Qwen2.5-VL:2b local model server
- `vision` — YOLOv8 + InsightFace service
- `api` — FastAPI service
- `frontend` — React + Nginx
- `ingestion` — camera simulator service

## Migration task
- Write a migration script: `migrate_sqlite_to_postgres.py`
- Move all SQLite records to PostgreSQL
- Replace embedding BLOB with pgvector column type
- Update all DB queries to use asyncpg

## Acceptance criteria
- [ ] `docker compose up` starts all services
- [ ] `docker compose down` stops all cleanly
- [ ] All Phase 1-7 functionality works inside Docker
- [ ] `.env` controls all secrets and paths

## ⛔ STOP HERE. Confirm with user before continuing.

---

# General Notes for the Builder

## Error handling rules
- Never let an error in one camera crash the others
- Always wrap InsightFace calls in try/except (some faces will fail)
- Always wrap Ollama calls in try/except with a timeout of 30 seconds
- Log all errors to console with `[ERROR]` prefix but continue processing

## Performance rules
- Never process every single frame — use `FPS_TARGET` to skip frames
- Never call the LLM for every tracked person on every frame — only call it once per `person_id` when their buffer is full
- Cache the YOLOv8 model — only load it once at startup

## Code style
- All functions must have docstrings
- All classes must have `__repr__` methods
- Use type hints everywhere
- No magic numbers — all thresholds and sizes must come from `config/settings.py`

## Git workflow
```bash
git init
git add .
git commit -m "feat: project skeleton"
git checkout -b feature/phase-1-camera-simulator
# work on phase 1
git add .
git commit -m "feat(phase1): camera simulator with YOLO detection and DeepSORT tracking"
git checkout main
git merge feature/phase-1-camera-simulator
```

One branch per phase. Merge to main only after user confirms the phase works.
