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
# Segmentation variant — provides per-person pixel masks used by the
# preprocessing stage to suppress background before embedding (Part 4).
YOLO_MODEL      = "yolov8n-seg.pt"
# Low threshold so ByteTrack receives both high- and low-confidence detections
# for its two-stage association (Part 7). Crops for embedding are gated
# separately by BYTETRACK_TRACK_THRESH so embedding quality is preserved.
DETECTION_CONF  = 0.10
DETECTION_CLASS = 0
DETECTION_IMGSZ = 256   # 256 instead of original 320 — reduces inference time ~30%

# ---------------------------------------------------------------------------
# Frame preprocessing (Part 3 — low-light / adverse lighting enhancement)
# ---------------------------------------------------------------------------
ENABLE_FRAME_ENHANCEMENT = True
CLAHE_CLIP_LIMIT         = 2.0
CLAHE_TILE_GRID_SIZE     = (8, 8)
AUTO_GAMMA_THRESHOLD     = 60
AUTO_GAMMA_VALUE         = 0.5

# ---------------------------------------------------------------------------
# Crop quality gate (Part 2 — reject garbage crops before gallery storage)
# ---------------------------------------------------------------------------
CROP_BLUR_THRESHOLD     = 50.0   # Laplacian variance minimum (50 accepts mild motion blur)
CROP_MIN_WIDTH          = 48
CROP_MIN_HEIGHT         = 96
CROP_DARKNESS_THRESHOLD = 30     # HSV V mean minimum

# ---------------------------------------------------------------------------
# Background isolation via YOLO segmentation (Part 4)
# ---------------------------------------------------------------------------
USE_SEGMENTATION             = True
SEGMENTATION_MASK_THRESHOLD  = 0.5
SEGMENTATION_TRACK_IOU       = 0.4
BACKGROUND_REPLACEMENT_COLOR = 128   # neutral gray

# ---------------------------------------------------------------------------
# Tracking (ByteTrack — Part 7, replaces DeepSORT)
# ---------------------------------------------------------------------------
# ByteTrack two-stage association thresholds:
#   First stage  : high-confidence detections (>= BYTETRACK_TRACK_THRESH)
#   Second stage : low-confidence detections  (>= BYTETRACK_LOW_THRESH, < BYTETRACK_TRACK_THRESH)
#                  used to keep coasting tracks alive through brief occlusions
BYTETRACK_TRACK_THRESH = 0.45   # high-conf gate (first-stage association + embedding crops)
BYTETRACK_LOW_THRESH   = 0.10   # low-conf gate  (second-stage, keeps tracks alive)
BYTETRACK_MATCH_THRESH = 0.80   # IoU matching threshold
BYTETRACK_TRACK_BUFFER = 30     # frames to hold a lost track before dropping (~1 s @ 30 fps)

# Display filter: hide Kalman-predicted box after this many consecutive missed
# detection cycles. ByteTrack handles internal coasting up to BYTETRACK_TRACK_BUFFER;
# this is a stricter display-side filter to reduce ghost-box drift.
TRACKING_COAST_FRAMES = 4

# Display staleness — tracks older than this (seconds) are cleared from screen
STALE_TRACK_TIMEOUT = 3.0

# ---------------------------------------------------------------------------
# Embedding (OSNet x1.0 via torchreid — Part 5, replaces ResNet-50)
# ---------------------------------------------------------------------------
EMBEDDING_DIM            = 512  # OSNet x1.0 global feature dimension
FACE_DET_SIZE            = (640, 640)
MIN_FACE_CONF            = 0.5
NUM_FRAMES_FOR_EMBEDDING = 4

# ---------------------------------------------------------------------------
# Cross-camera identity matching thresholds (recalibrated for OSNet — Part 5)
# ---------------------------------------------------------------------------
# OSNet's same-/different-person distributions barely overlap, making 0.72
# a reliable decision boundary. ResNet-50 needed 0.63 because its distributions
# were much wider (same person front→back scored only 0.45–0.65).
FACE_MATCH_THRESHOLD  = 0.55

# Dual-threshold matching (introduced after side-view sponge + cross-cam split regression):
#
# A single threshold cannot separate two overlapping ranges in the WiseNet dataset:
#   • Different people, same camera, side view  : observed ~0.70–0.72  → must NOT match
#   • Same person, different cameras / lighting : observed ~0.68–0.72  → must match
#
# Context-aware logic in main.py selects the right threshold per candidate:
#   SAME camera  → stricter gate  (a truly returning person scores 0.85+; 0.75 safely
#                                  clears the ~0.72 false-positive ceiling for side views)
#   CROSS camera → looser gate    (angle/lighting change can drop same-person scores to 0.68)
BODY_MATCH_THRESHOLD_SAME_CAM  = 0.75
BODY_MATCH_THRESHOLD_CROSS_CAM = 0.68

# ---------------------------------------------------------------------------
# Overlap-aware matching (Part 8.5 — prelude to Part 9 spatio-temporal)
# ---------------------------------------------------------------------------
# WiseNet (and most real surveillance deployments) has cameras whose fields
# of view overlap — two or more cameras in the same room, possibly pointed
# at the same spot from different angles. The same physical person can then
# appear on cam_A and cam_B *simultaneously*, with two views taken from
# sharply different angles. OSNet's same-person score across such a sharp
# angle change in the same instant is empirically 0.55–0.70 — BELOW the
# 0.68 cross-cam threshold, which would split one person into two IDs.
#
# CAMERA_OVERLAP_GROUPS declares which cam_ids share physical space.
# Each entry is a set of cam_ids that overlap. Example:
#     CAMERA_OVERLAP_GROUPS = [{1, 2}, {4, 5}]
# meaning cam1↔cam2 share one room and cam4↔cam5 share another.
#
# Empty list (default) = no overlap declared → system behaves exactly as
# before (the overlap branch in main.py never fires). Drop-in safe.
#
# Constraint: groups must be DISJOINT (a camera belongs to at most one
# room). validate_overlap_topology() enforces this at startup.
CAMERA_OVERLAP_GROUPS: list[set[int]] = []

# Threshold used when the current cam and the candidate's last_seen_cam
# are in the same overlap group. Looser than cross-cam (0.68) because
# same-person-different-angle-same-instant scores from a different
# distribution than same-person-different-time-cross-cam (sequential
# transitions land in 0.68–0.78; simultaneous overlap views land lower).
#
# Invariant #12: BODY_MATCH_THRESHOLD_OVERLAP must satisfy
#   BODY_MATCH_THRESHOLD_OVERLAP <= BODY_MATCH_THRESHOLD_CROSS_CAM
#                                <= BODY_MATCH_THRESHOLD_SAME_CAM
# Otherwise the overlap threshold defeats the purpose of cross-cam
# strictness, or worse, allows force-accept to absorb wrong embeddings
# in declared overlap pairs (1 - 0.62 = 0.38 > FORCE_ACCEPT_MAX_DISTANCE
# 0.35 — still safe at the default 0.62; do not lower further).
BODY_MATCH_THRESHOLD_OVERLAP   = 0.62

# Searcher default floor / legacy alias — used by callers that do NOT pass
# a `min_threshold=` override (notably Phase-3 offline photo search, which
# has no camera context and should match at cross-cam strictness).
#
# Phase-2 live identification has camera context and explicitly passes
# `min_threshold=BODY_MATCH_THRESHOLD_OVERLAP` to widen the candidate pool
# before main.py's triple-threshold decision. So this constant being equal
# to CROSS_CAM does NOT pre-filter live-identification overlap candidates.
#
# Invariant #10 (refined): BODY_MATCH_THRESHOLD must equal
# BODY_MATCH_THRESHOLD_CROSS_CAM. Phase-3 callers and any context-less
# caller depend on this strictness.
BODY_MATCH_THRESHOLD = BODY_MATCH_THRESHOLD_CROSS_CAM

CROSS_TYPE_MULTIPLIER = 0.85

# Legacy aliases
FACE_SIMILARITY_THRESHOLD = FACE_MATCH_THRESHOLD
BODY_SIMILARITY_THRESHOLD = BODY_MATCH_THRESHOLD
SIMILARITY_THRESHOLD      = BODY_MATCH_THRESHOLD

MIN_GALLERY_FOR_MATCHING = 1

# ---------------------------------------------------------------------------
# Gallery update thresholds (recalibrated for OSNet — Part 5)
# ---------------------------------------------------------------------------
FACE_GALLERY_ADD_DISTANCE  = 0.25
BODY_GALLERY_ADD_DISTANCE  = 0.20   # was 0.40; OSNet embeddings cluster tighter
GALLERY_MAX_DISTANCE       = 0.55   # accepts challenging same-person poses (distance
                                   # up to 0.55 = similarity down to 0.45) while
                                   # blocking obviously-different-person embeddings;
                                   # 0.65 was too permissive — gallery got polluted

# Maximum cosine distance allowed for a FORCE-ACCEPT (canonical-slot fill).
# Without this, any crop with distance <= GALLERY_MAX_DISTANCE (sim >= 0.45)
# could be force-accepted into an empty canonical slot — far too loose.
# The force-accept path bypasses the novelty/diversity gate, so it needs its
# own tighter guard to prevent wrong-person embeddings from being absorbed when
# a track is (rarely) wrongly bound to another person's gallery.
# 0.35 = sim >= 0.65 — matches the pre-fix BODY_MATCH_THRESHOLD, so force-accept
# requires the same minimum similarity as identification itself.
FORCE_ACCEPT_MAX_DISTANCE  = 0.35

MAX_GALLERY_SIZE           = 10
MIN_FRAMES_BETWEEN_SAMPLES = 15

# Legacy aliases
GALLERY_NEW_VIEW_THRESHOLD  = BODY_GALLERY_ADD_DISTANCE
NEW_VIEW_DISTANCE_THRESHOLD = BODY_GALLERY_ADD_DISTANCE
MAX_VIEW_DISTANCE_TO_ACCEPT = GALLERY_MAX_DISTANCE
MIN_FRAMES_BEFORE_SAMPLE    = MIN_FRAMES_BETWEEN_SAMPLES

# ---------------------------------------------------------------------------
# Pose-aware gallery (Part 6)
# ---------------------------------------------------------------------------
# Four canonical viewpoints the gallery tries to cover for each person.
# estimate_view() in gallery.py maps a bounding box to one of these tags.
# Uncovered canonical slots get force-accepted regardless of cosine distance.
CANONICAL_VIEWS = ("frontal", "right_moving", "left_moving", "side")

# Minimum view coverage fraction (covered slots / 4) for a person to be
# considered a reliable cross-camera match target. Below this threshold the
# searcher skips the person to avoid matching on a single-angle prototype.
MIN_VIEW_COVERAGE_FOR_MATCHING = 0.5   # at least 2 distinct canonical views

# ---------------------------------------------------------------------------
# Background Reconciliation
# ---------------------------------------------------------------------------
RECONCILIATION_INTERVAL_SEC = 120

# Thresholds are for MEAN-POOL similarity (average across all compatible pairs),
# not max-pool. Mean-pool is far more reliable for merge decisions — a single
# accidentally-similar pair can no longer trigger a false proposal.
#
# Expected mean-pool scores (OSNet):
#   Same person, multi-angle gallery  : 0.60–0.82
#   Different people, similar clothes : 0.15–0.45
MERGE_CANDIDATE_THRESHOLD   = 0.58   # mean-pool: propose pairs for human review
AUTO_MERGE_THRESHOLD        = 0.82   # mean-pool: auto-merge only when very confident;
                                     # 0.75 was causing false auto-merges of different people
GHOST_TTL_SEC               = 180

# Minimum gallery size a person must have before being a reconciliation target.
# 2 is sufficient — 3 was preventing fresh duplicates from ever being caught.
MIN_GALLERY_FOR_RECONCILIATION = 2

# ---------------------------------------------------------------------------
# Co-visibility boost (Part 8.5, Layer 2 — backstop for missed live merges)
# ---------------------------------------------------------------------------
# When two person_ids show up *together* on declared overlap-partner cameras
# in lock-step across multiple independent moments, they are almost certainly
# the same physical person split by the live matcher. Two unrelated people
# would not coincidentally share the same time window on the same overlap
# pair more than once or twice.
#
# The reconciliation worker uses these constants to relax MERGE_CANDIDATE_THRESHOLD
# down to MERGE_CANDIDATE_THRESHOLD_OVERLAP_BOOSTED for any pair (pid_a, pid_b)
# whose camera_history intervals overlap on at least CO_VISIBILITY_MIN_MOMENTS
# independent moments — each at least CO_VISIBILITY_MIN_OVERLAP_SEC seconds long
# — on cameras that are in the same CAMERA_OVERLAP_GROUPS entry.
CO_VISIBILITY_MIN_MOMENTS           = 3
CO_VISIBILITY_MIN_OVERLAP_SEC       = 0.5
MERGE_CANDIDATE_THRESHOLD_OVERLAP_BOOSTED = 0.45

# ---------------------------------------------------------------------------
# Search backend (Part 8 — FAISS in-memory vector index)
# ---------------------------------------------------------------------------
# Kill switch. When False, the PersonSearcher is constructed with
# faiss_index=None and falls back to the SQLite linear scan. Used for
# debugging the FAISS regression reported on 2026-05-28 — lets us A/B
# the system with and without FAISS to determine whether FAISS itself
# is the cause of the "gallery sponge" / wrong-color binding bug.
ENABLE_FAISS = True    # H3 confirmed — FAISS is not the problem; re-enabled.

# Diagnostic mode. When True, every identification query runs BOTH the
# FAISS path AND the SQLite linear scan, and any disagreement above
# 0.01 in score or any difference in top-1 person_id is logged as
# [FAISS_DRIFT] ... — this lets us confirm whether FAISS is producing
# inflated scores compared to the linear cosine scan.
# Costs ~10 ms per identify call (rare event) — never leave on in prod.
FAISS_AUDIT_MODE = False

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


# ---------------------------------------------------------------------------
# Camera-topology helpers (Part 8.5)
# ---------------------------------------------------------------------------

def are_overlapping_cams(cam_a: int, cam_b: int) -> bool:
    """
    Return True iff cam_a and cam_b are declared as sharing physical space
    in CAMERA_OVERLAP_GROUPS. Returns False when cam_a == cam_b — same-camera
    is handled by the BODY_MATCH_THRESHOLD_SAME_CAM branch, not the overlap
    branch.
    """
    if cam_a == cam_b:
        return False
    for group in CAMERA_OVERLAP_GROUPS:
        if cam_a in group and cam_b in group:
            return True
    return False


def validate_overlap_topology(num_cams: int) -> list[str]:
    """
    Validate CAMERA_OVERLAP_GROUPS and threshold ordering at startup.

    Returns a list of warning messages (empty if the config is clean).
    Per WORKING_INSTRUCTIONS §5, a bad topology must not crash a camera
    thread — this function only WARNS; it does not raise.

    Checks:
      1. Threshold ordering: OVERLAP <= CROSS_CAM <= SAME_CAM (invariant #12).
      2. Force-accept safety: 1 - OVERLAP > FORCE_ACCEPT_MAX_DISTANCE
         (otherwise force-accept becomes a sponge on overlap pairs).
      3. Disjoint groups: a cam_id belongs to at most one overlap group.
      4. Known cam_ids: every cam_id in a group is in range [0, num_cams).
      5. Singleton groups are pointless — warn but ignore.
    """
    warnings: list[str] = []

    if not (
        BODY_MATCH_THRESHOLD_OVERLAP
        <= BODY_MATCH_THRESHOLD_CROSS_CAM
        <= BODY_MATCH_THRESHOLD_SAME_CAM
    ):
        warnings.append(
            f"[CONFIG] Invariant #12 violated: threshold ordering — "
            f"OVERLAP={BODY_MATCH_THRESHOLD_OVERLAP} "
            f"CROSS_CAM={BODY_MATCH_THRESHOLD_CROSS_CAM} "
            f"SAME_CAM={BODY_MATCH_THRESHOLD_SAME_CAM}"
        )

    if (1.0 - BODY_MATCH_THRESHOLD_OVERLAP) <= FORCE_ACCEPT_MAX_DISTANCE:
        warnings.append(
            f"[CONFIG] Overlap threshold {BODY_MATCH_THRESHOLD_OVERLAP} too loose: "
            f"force-accept guard ({FORCE_ACCEPT_MAX_DISTANCE}) may absorb "
            f"wrong-person embeddings on overlap pairs."
        )

    seen_cams: dict[int, int] = {}
    for idx, group in enumerate(CAMERA_OVERLAP_GROUPS):
        if len(group) < 2:
            warnings.append(
                f"[CONFIG] Overlap group #{idx} has fewer than 2 cameras: {group} — ignored."
            )
            continue
        for cam_id in group:
            if not isinstance(cam_id, int) or cam_id < 0 or cam_id >= num_cams:
                warnings.append(
                    f"[CONFIG] Overlap group #{idx} contains unknown cam_id {cam_id} "
                    f"(valid range 0..{num_cams - 1}) — pair will be ignored at runtime."
                )
            if cam_id in seen_cams:
                warnings.append(
                    f"[CONFIG] cam_id {cam_id} appears in multiple overlap groups "
                    f"(#{seen_cams[cam_id]} and #{idx}). Groups must be disjoint."
                )
            else:
                seen_cams[cam_id] = idx

    return warnings
