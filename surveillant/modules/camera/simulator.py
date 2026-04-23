"""
modules/camera/simulator.py
---------------------------
CameraSimulator reads N video files and yields synchronized frame sets,
simulating N live camera streams.

Each video loops infinitely so the simulation never ends.
Frame pacing is enforced to match the native video FPS.
"""

import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List


class CameraSimulator:
    """
    Simulates multiple live camera streams by reading from video files.
    Reads one frame per camera per call, paced to the video's native FPS
    for smooth playback.
    """

    def __init__(self, video_paths: List[str], fps_target: int) -> None:
        self.video_paths: List[str] = video_paths
        self.fps_target:  int       = fps_target
        self.captures: Dict[int, cv2.VideoCapture] = {}
        self.native_fps: float = 30.0
        self._last_read_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"CameraSimulator(cameras={len(self.video_paths)}, "
            f"native_fps={self.native_fps})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open all VideoCapture objects."""
        for cam_id, path in enumerate(self.video_paths):
            if not Path(path).exists():
                raise FileNotFoundError(f"[CameraSimulator] Video not found: {path}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"[CameraSimulator] Cannot open: {path}")

            self.captures[cam_id] = cap
            print(f"[CameraSimulator] Camera {cam_id} opened -> {path}")

            # Determine native FPS from the first valid capture
            if cam_id == 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    self.native_fps = fps
                    print(f"[CameraSimulator] Native FPS detected: {fps:.1f}")

        self._last_read_time = time.time()

    def read_frames(self) -> Dict[int, np.ndarray]:
        """
        Read exactly one frame from every camera, enforcing native FPS pacing.
        This produces smooth video playback.
        """
        # Pace to native video FPS for smooth playback
        frame_duration = 1.0 / self.native_fps
        now = time.time()
        elapsed = now - self._last_read_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)
        self._last_read_time = time.time()

        frames: Dict[int, np.ndarray] = {}
        for cam_id, cap in self.captures.items():
            ret, frame = cap.read()
            if not ret:
                # End of video — loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    print(f"[ERROR] Camera {cam_id} failed to loop.")
                    continue
            frames[cam_id] = frame

        return frames

    def release(self) -> None:
        """Release all VideoCapture objects."""
        for cam_id, cap in self.captures.items():
            cap.release()
            print(f"[CameraSimulator] Camera {cam_id} released.")
        self.captures.clear()
