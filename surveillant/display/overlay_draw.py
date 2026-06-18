"""
display/overlay_draw.py
-----------------------
Standalone bounding-box / label renderer.

This is the single source of truth for how a person box is drawn on a frame:
state-based border styling, the ``P:xxxxxx`` identity label, the status badge
and (Part 11) the additive face attributes. It was factored out of
``GridDisplay.update`` so that BOTH the desktop live-feed window and the webapp
overlay recorder draw boxes identically.

``draw_tracks`` takes a frame + the same per-track dicts ``GridDisplay`` uses and
returns a NEW annotated copy (it never mutates the input). It does NOT resize or
watermark — that stays a concern of whoever owns the surface (the grid display
tiles into cells, the MJPEG stream resizes to a fixed width).
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# White used for un-matched tracks (collecting state). Kept here so the helper
# is self-contained; GridDisplay re-exports it for backward compatibility.
COLOR_COLLECTING = (255, 255, 255)

# Status badge (ASCII-safe for Windows cp1252).
STATUS_BADGE = {
    "unverified": "[*]",
    "confirmed":  "[+]",
    "multi_view": "[M]",
    "flagged":    "[F]",
    "ghost":      "[G]",
}


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    cam_id: int,
    color_registry,
    banner: Optional[str] = None,
) -> np.ndarray:
    """
    Annotate ``frame`` with bounding boxes + labels for every track and return
    a new annotated image.

    Each track dict may contain:
        track_id     : int
        bbox         : (x1, y1, x2, y2)
        state        : 'collecting' | 'new' | 'returning' | 'flash_green'
        person_id    : str (UUID) — present once matched
        status       : 'unverified' | 'confirmed' | 'multi_view' | 'flagged' | 'ghost'
        gallery_size : int — number of gallery embeddings so far
        buffer_len / buffer_total : int — collecting progress (optional)
        # Part 11 — additive face attributes (display only):
        name, gender, age_range, ethnicity, glasses, returning_face

    ``banner`` is an optional per-camera violence status string (e.g.
    "VIOLENCE (0.91)"). When present and not "OK", a colored banner + frame
    border is drawn. ``color_registry`` is a ``ColorRegistry`` (deterministic
    person_id -> BGR color).
    """
    annotated = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        tid    = track.get("track_id")
        state  = track.get("state", "collecting")
        pid    = track.get("person_id")
        g_sz   = track.get("gallery_size", 0)
        status = track.get("status", "unverified")

        badge = STATUS_BADGE.get(status, "[?]")

        if state == "flash_green":
            color     = (0, 255, 0)
            thickness = 3
            label     = f"PROCESSED! {badge}[G:{g_sz}]" if pid else "PROCESSED!"

        elif state == "returning" and pid:
            color     = color_registry.get_color(pid)
            thickness = 3
            label     = f"CAM{cam_id} | P:{pid[:6]} {badge}[G:{g_sz}] <"

        elif state == "new" and pid:
            color     = color_registry.get_color(pid)
            thickness = 3
            label     = f"CAM{cam_id} | P:{pid[:6]} {badge}[G:{g_sz}]"

        else:
            # Collecting — white thin border
            color     = COLOR_COLLECTING
            thickness = 1
            buf_len   = track.get("buffer_len", 0)
            total     = track.get("buffer_total", 8)
            label     = f"CAM{cam_id} | T{tid} | ({buf_len}/{total})"

        # Part 11 — append additive face attributes (display only). Absent fields
        # are simply skipped, so the label is unchanged when the face layer is off.
        if pid:
            attr_bits = []
            if track.get("name"):           attr_bits.append(str(track["name"]))
            if track.get("gender"):         attr_bits.append(str(track["gender"]))
            if track.get("age_range"):      attr_bits.append(str(track["age_range"]))
            if track.get("ethnicity"):      attr_bits.append(str(track["ethnicity"]))
            if track.get("glasses"):        attr_bits.append(str(track["glasses"]))
            if track.get("returning_face"): attr_bits.append("RET")
            if attr_bits:
                label = label + " | " + " | ".join(attr_bits)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # For returning persons, an inner rectangle simulates a "dashed" look,
        # making them visually distinct from new persons.
        if state == "returning" and pid:
            inner_margin = 3
            cv2.rectangle(
                annotated,
                (x1 + inner_margin, y1 + inner_margin),
                (x2 - inner_margin, y2 - inner_margin),
                color,
                1,
            )

        # Label background + text
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

        lbl_x1 = x1
        lbl_y1 = max(0, y1 - th - 8)
        lbl_x2 = x1 + tw + 6
        lbl_y2 = y1

        cv2.rectangle(annotated, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (lbl_x1 + 3, lbl_y2 - 4),
            font,
            font_scale,
            (0, 0, 0) if state == "flash_green" else (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    # Part 11 — violence banner + frame border (drawn once per frame). Only
    # VIOLENCE / SUSPICIOUS are shown prominently; "OK" stays quiet.
    if banner and "OK" not in banner.upper():
        up = banner.upper()
        bcolor = (0, 0, 255) if "VIOLENCE" in up else (0, 165, 255)  # red / orange
        h_a, w_a = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w_a, 28), bcolor, -1)
        cv2.putText(annotated, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(annotated, (0, 0), (w_a - 1, h_a - 1), bcolor, 4)

    return annotated
