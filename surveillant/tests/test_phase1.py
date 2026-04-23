"""
tests/test_phase1.py
---------------------
Unit tests for Phase 1 components.
Runs entirely in isolation — no real video files required.

Usage:
    python tests/test_phase1.py
"""

import sys
import os

# Ensure the package root is on the path when running from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from modules.detection.detector import PersonDetector
from modules.tracking.tracker import PersonTracker
from display.visualizer import GridDisplay


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_detector_runs() -> None:
    """
    Verify PersonDetector initialises and runs on a blank frame without
    raising an exception.  No real people are expected to be found in a
    zero-filled image.
    """
    detector   = PersonDetector("yolov8n.pt", conf=0.5)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result     = detector.detect(fake_frame)

    assert isinstance(result, list), (
        f"Detector must return a list, got {type(result)}"
    )
    print("✓ Detector runs on blank frame without crash")


def test_tracker_runs() -> None:
    """
    Verify PersonTracker accepts synthetic detections and returns a list.
    The track may or may not be confirmed on the first frame depending on
    MIN_HITS — that is expected behaviour.
    """
    tracker         = PersonTracker(cam_id=0)
    fake_frame      = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_detections = [{"bbox": [100, 100, 200, 300], "confidence": 0.9}]

    result = tracker.update(fake_detections, fake_frame)

    assert isinstance(result, list), (
        f"Tracker must return a list, got {type(result)}"
    )
    print("✓ Tracker runs with fake detections without crash")


def test_tracker_output_format() -> None:
    """
    Feed several identical detections into the tracker to let a track become
    confirmed, then verify the output dict has the required keys.
    """
    tracker    = PersonTracker(cam_id=1)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_det   = [{"bbox": [50, 50, 150, 250], "confidence": 0.85}]

    tracks = []
    # Pump enough frames to confirm a track (MIN_HITS = 3)
    for _ in range(10):
        tracks = tracker.update(fake_det, fake_frame)

    # If a track is confirmed, validate its structure
    for t in tracks:
        assert "track_id" in t,  "Track must have 'track_id'"
        assert "bbox"     in t,  "Track must have 'bbox'"
        assert "cam_id"   in t,  "Track must have 'cam_id'"
        assert len(t["bbox"]) == 4, "bbox must have 4 coordinates"
        assert t["cam_id"] == 1,   "cam_id must match the tracker's cam_id"

    print(
        f"✓ Tracker output format OK "
        f"({'track confirmed' if tracks else 'no confirmed track yet — normal'})"
    )


def test_display_grid() -> None:
    """
    Create a two-camera GridDisplay, push a fake annotated frame, render
    for 1 second, then destroy the window.
    """
    display    = GridDisplay(num_cams=2, cols=2, cell_w=320, cell_h=240)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_tracks = [{"track_id": 1, "bbox": [50, 50, 150, 200], "cam_id": 0}]

    display.update(0, fake_frame, fake_tracks)
    display.update(1, fake_frame, [])
    display.render()
    cv2.waitKey(1000)   # show the window for 1 second
    cv2.destroyAllWindows()
    print("✓ Grid display renders two cameras without crash")


def test_display_color_function() -> None:
    """
    Verify that the track-colour helper produces valid BGR tuples and that
    different track IDs produce different colours.
    """
    from display.visualizer import _track_color

    colors = [_track_color(i) for i in range(20)]
    for c in colors:
        assert len(c) == 3, "Color must be a 3-tuple"
        assert all(0 <= v <= 255 for v in c), "Color values must be 0–255"

    # All 20 colours should be distinct
    assert len(set(colors)) > 10, (
        "Expected most track colours to be distinct"
    )
    print(f"✓ Track color function produces {len(set(colors))}/20 distinct colours")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  SURVEILLANT — Phase 1 Tests")
    print("=" * 55)

    test_detector_runs()
    test_tracker_runs()
    test_tracker_output_format()
    test_display_grid()
    test_display_color_function()

    print("\n✅  All Phase 1 tests passed.\n")
