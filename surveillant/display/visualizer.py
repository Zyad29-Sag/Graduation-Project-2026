"""
display/visualizer.py
---------------------
GridDisplay assembles all camera frames into a single tiled grid window
using OpenCV.

ColorRegistry
-------------
Assigns a unique, stable BGR color to each person_id (UUID).
Color is deterministic (same UUID → same color every run, every camera)
and is derived via MD5 + HSV colour wheel so adjacent UUIDs get very
different hues.

Pipeline state is shown via BORDER STYLE, not box color:
  - White thin border   → collecting (< NUM_FRAMES_FOR_EMBEDDING frames buffered)
  - Thick unique color  → confirmed new person
  - Thick unique color (dashed effect via double rect) → re-identified / returning
"""

import colorsys
import hashlib
import cv2
import math
import numpy as np
from typing import Dict, List, Any, Optional

from display.overlay_draw import draw_tracks

# ------------------------------------------------------------------
# ColorRegistry — Bug Category 1 fix
# ------------------------------------------------------------------

class ColorRegistry:
    """
    Assigns a unique, stable BGR color to each person_id (UUID).

    Color is derived deterministically from the UUID so it is:
    - The same across all cameras for the same person
    - The same every time the system runs
    - Visually distinct from nearby assignments (golden-angle hue spread)
    """

    def __init__(self) -> None:
        self._registry: Dict[str, tuple] = {}          # person_id → (B,G,R)
        self._aliases:  Dict[str, str]   = {}          # "cam0_track7" → person_id

    # ------ Color lookup ---------------------------------------------------

    def get_color(self, person_id: str) -> tuple:
        """
        Returns a BGR color tuple for this person_id.
        If not yet registered, generates one from the UUID hash.
        """
        if person_id not in self._registry:
            self._registry[person_id] = self._generate_color(person_id)
        return self._registry[person_id]

    def _generate_color(self, person_id: str) -> tuple:
        """
        Hash person_id → deterministic HSV → BGR for OpenCV.
        Full hue range (0.0–1.0) × golden angle spread, fixed high
        saturation and value so colors are always vivid.
        """
        hash_bytes = hashlib.md5(person_id.encode()).digest()
        # Use first byte for hue, second byte to add a golden-angle offset
        # so nearby sequential UUIDs get very different hues.
        raw_hue = hash_bytes[0] / 255.0
        offset  = (hash_bytes[1] / 255.0) * 0.618033988  # golden ratio fraction
        hue = (raw_hue + offset) % 1.0

        saturation = 0.90
        value      = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = (int(b * 255), int(g * 255), int(r * 255))   # BGR for OpenCV

        hex_color = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
        print(f"[COLOR] Assigned {hex_color} to person_id={person_id[:8]}...")
        return bgr

    # ------ Alias registry (track → person_id) ----------------------------

    def register_alias(self, cam_id_or_key, track_id_or_pid=None, person_id: str = None) -> None:
        """
        Link a (cam_id, track_id) pair to a known person_id.
        Accepts both call styles for backward compatibility:
          register_alias(cam_id: int, track_id: int, person_id: str)   # new
          register_alias(key: str, person_id: str)                      # old (tests)
        """
        if isinstance(cam_id_or_key, str):
            # Old style: register_alias("cam0_track3", pid)
            key       = cam_id_or_key
            person_id = track_id_or_pid
        else:
            key = f"cam{cam_id_or_key}_track{track_id_or_pid}"

        old = self._aliases.get(key)
        if old != person_id:
            color = self.get_color(person_id)
            hex_c = "#{:02X}{:02X}{:02X}".format(color[2], color[1], color[0])
            print(f"[COLOR] Cross-camera link: {key} -> existing person {person_id[:8]}... ({hex_c})")
        self._aliases[key] = person_id

    def resolve_person_id(self, cam_id: int, track_id: int) -> Optional[str]:
        """Returns the person_id for a given (cam_id, track_id) or None."""
        return self._aliases.get(f"cam{cam_id}_track{track_id}")


# ------------------------------------------------------------------
# GridDisplay
# ------------------------------------------------------------------

class GridDisplay:
    """
    Multi-camera grid display.

    Maintains an internal buffer of the latest annotated frame for each
    camera.  When ``render()`` is called, all frames are assembled into a
    single image and shown in one OpenCV window.

    Args:
        num_cams (int):       Total number of camera feeds.
        cols     (int):       Number of camera columns in the grid.
        cell_w   (int):       Width  of each camera cell in pixels.
        cell_h   (int):       Height of each camera cell in pixels.
        color_registry (ColorRegistry): Shared registry for person colors.
    """

    WINDOW_NAME = "SURVEILLANT - Live Feed"

    # White used for un-matched tracks (collecting state)
    COLOR_COLLECTING = (255, 255, 255)

    def __init__(
        self,
        num_cams: int,
        cols: int,
        cell_w: int,
        cell_h: int,
        color_registry: Optional[ColorRegistry] = None,
    ) -> None:
        self.num_cams:       int           = num_cams
        self.cols:           int           = cols
        self.cell_w:         int           = cell_w
        self.cell_h:         int           = cell_h
        self.color_registry: ColorRegistry = color_registry or ColorRegistry()

        self.rows: int = math.ceil(num_cams / cols)

        blank = np.full((cell_h, cell_w, 3), 30, dtype=np.uint8)
        self._frames: Dict[int, np.ndarray] = {
            i: blank.copy() for i in range(num_cams)
        }

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.WINDOW_NAME,
            min(cols * cell_w, 1920),
            min(self.rows * cell_h, 1080),
        )

    def __repr__(self) -> str:
        return (
            f"GridDisplay(num_cams={self.num_cams}, cols={self.cols}, "
            f"cell={self.cell_w}x{self.cell_h})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        cam_id: int,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        banner: Optional[str] = None,
    ) -> None:
        """
        Annotate a frame with bounding boxes and track labels, then store
        it in the internal buffer.

        Each track dict may contain:
            track_id     : int
            bbox         : (x1, y1, x2, y2)
            cam_id       : int
            state        : 'collecting' | 'new' | 'returning' | 'flash_green'
            person_id    : str (UUID) — present once matched
            gallery_size : int — number of gallery embeddings so far
            label        : str — optional override label (legacy support)
            color        : tuple — optional override BGR color (legacy support)
            # Part 11 — additive face attributes (display only):
            name, gender, age_range, ethnicity, glasses, returning_face

        ``banner`` (Part 11) is an optional per-camera violence status string
        (e.g. "VIOLENCE (0.91)"). When present and not "OK", a colored banner +
        frame border is drawn. None → nothing drawn (unchanged behavior).
        """
        # Box + label drawing is shared with the webapp overlay recorder.
        annotated = draw_tracks(frame, tracks, cam_id, self.color_registry, banner)

        # Camera watermark
        resized = cv2.resize(annotated, (self.cell_w, self.cell_h))
        cv2.putText(
            resized,
            f"CAMERA {cam_id}",
            (8, self.cell_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        self._frames[cam_id] = resized

    def render(self) -> None:
        """Assemble all buffered frames into a grid and display."""
        blank = np.full((self.cell_h, self.cell_w, 3), 20, dtype=np.uint8)
        rows_images = []

        for row in range(self.rows):
            row_frames = []
            for col in range(self.cols):
                cid = row * self.cols + col
                row_frames.append(self._frames[cid] if cid < self.num_cams else blank)
            rows_images.append(np.hstack(row_frames))

        cv2.imshow(self.WINDOW_NAME, np.vstack(rows_images))

    def should_quit(self) -> bool:
        """Return True if user pressed 'q' or closed the window."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return True
        try:
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            return True
        return False
