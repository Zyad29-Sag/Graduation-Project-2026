"""
SURVEILLANT — Central Configuration
All thresholds, paths, and model names live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIDEOS_DIR    = BASE_DIR / "data" / "videos"
SNAPSHOTS_DIR = BASE_DIR / "data" / "snapshots"
DB_PATH       = BASE_DIR / "database" / "surveillant.db"
TRACK_REGISTRY_PATH = BASE_DIR / "database" / "track_registry_session.json"

# ---------------------------------------------------------------------------
# Camera / simulator
# ---------------------------------------------------------------------------
FPS_TARGET     = 30
DISPLAY_WIDTH  = 480
DISPLAY_HEIGHT = 270
GRID_COLS      = 3

# ---------------------------------------------------------------------------
# Detection (YOLOv8)
# ---------------------------------------------------------------------------
YOLO_MODEL      = "yolov8n.pt"
DETECTION_CONF  = 0.3
DETECTION_CLASS = 0
DETECTION_IMGSZ = 320

# ---------------------------------------------------------------------------
# Tracking (DeepSORT)
# ---------------------------------------------------------------------------
MAX_AGE       = 2
MIN_HITS      = 2
IOU_THRESHOLD = 0.3

# Display staleness — tracks older than this (seconds) are cleared from screen
STALE_TRACK_TIMEOUT = 1.5

# ---------------------------------------------------------------------------
# Embedding (MobileNetV3 body)
# ---------------------------------------------------------------------------
FACE_DET_SIZE            = (640, 640)
MIN_FACE_CONF            = 0.5
NUM_FRAMES_FOR_EMBEDDING = 4    # reduced from 8 to assign colors/IDs much faster

# ---------------------------------------------------------------------------
# Cross-camera identity matching thresholds
# ---------------------------------------------------------------------------
FACE_MATCH_THRESHOLD  = 0.55   # face embeddings: lower threshold OK (faces are reliable)
BODY_MATCH_THRESHOLD  = 0.72   # body embeddings: reduced from 0.82 for cross-cam realism
CROSS_TYPE_MULTIPLIER = 0.85   # penalty when query type != stored type

# Legacy aliases kept for old import sites
FACE_SIMILARITY_THRESHOLD = FACE_MATCH_THRESHOLD
BODY_SIMILARITY_THRESHOLD = BODY_MATCH_THRESHOLD
SIMILARITY_THRESHOLD      = BODY_MATCH_THRESHOLD

# Minimum gallery size before a person is eligible as a cross-cam match target
MIN_GALLERY_FOR_MATCHING = 2

# ---------------------------------------------------------------------------
# Gallery update thresholds (is this angle different enough to store?)
# ---------------------------------------------------------------------------
FACE_GALLERY_ADD_DISTANCE  = 0.25   # cosine DISTANCE threshold for face views
BODY_GALLERY_ADD_DISTANCE  = 0.35   # cosine DISTANCE threshold for body views
GALLERY_MAX_DISTANCE       = 0.50   # reject if further than this (garbage/occlusion)
MAX_GALLERY_SIZE           = 10     # increased from 8

# Sampling rate for gallery updates
MIN_FRAMES_BETWEEN_SAMPLES = 15    # at ~5 FPS = every ~3 seconds per track

# Legacy aliases
GALLERY_NEW_VIEW_THRESHOLD = BODY_GALLERY_ADD_DISTANCE
NEW_VIEW_DISTANCE_THRESHOLD = BODY_GALLERY_ADD_DISTANCE
MAX_VIEW_DISTANCE_TO_ACCEPT = GALLERY_MAX_DISTANCE
MIN_FRAMES_BEFORE_SAMPLE    = MIN_FRAMES_BETWEEN_SAMPLES

# ---------------------------------------------------------------------------
# Background Reconciliation
# ---------------------------------------------------------------------------
RECONCILIATION_INTERVAL_SEC = 120   # run every 2 minutes
MERGE_CANDIDATE_THRESHOLD   = 0.88  # two person_ids this similar => merge candidate
AUTO_MERGE_THRESHOLD        = 0.95  # auto-merge (no operator confirmation needed)
GHOST_TTL_SEC               = 180   # person not seen for 3 min => mark inactive

# ---------------------------------------------------------------------------
# LLM (Ollama + Qwen2.5-VL)
# ---------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL   = "qwen2.5vl:2b"

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
SNAPSHOT_QUALITY         = 90
MAX_SNAPSHOTS_PER_PERSON = 5
