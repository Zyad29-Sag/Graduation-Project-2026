"""
webapp/api/overlays.py
----------------------
Serve-time helper for the Live-Cams overlay.

Loads the per-camera overlay sidecars produced offline by
``webapp.api.tools.record_overlays`` and draws the recorded boxes/IDs onto a
frame using the SAME renderer the desktop live feed uses
(``display.overlay_draw.draw_tracks``). Box styling therefore stays identical
across the desktop app and the webapp.

The sidecar is memo-cached (it never changes at runtime). A single process-wide
``ColorRegistry`` keeps person colors stable across cameras and requests.
"""

from functools import lru_cache
from typing import Any, Dict

import numpy as np

from . import config, engine  # engine import puts surveillant/ on sys.path

_color_registry = None  # lazy singleton (display import needs surveillant on path)


def get_color_registry():
    global _color_registry
    if _color_registry is None:
        from display.visualizer import ColorRegistry  # noqa: PLC0415

        _color_registry = ColorRegistry()
    return _color_registry


@lru_cache(maxsize=32)
def load_overlay(cam_id: int) -> Dict[str, Any]:
    """Return the camera's overlay sidecar ({total_frames, frames}) or empty."""
    path = config.overlay_sidecar(cam_id)
    if not path.exists():
        return {"total_frames": 0, "frames": {}}
    import json  # noqa: PLC0415

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"total_frames": 0, "frames": {}}


def has_overlay(cam_id: int) -> bool:
    return bool(load_overlay(cam_id).get("frames"))


def annotate(frame: np.ndarray, cam_id: int, frame_idx: int) -> np.ndarray:
    """
    Draw the recorded boxes for ``frame_idx`` onto ``frame``. Returns the frame
    unchanged when no sidecar / no boxes exist for that index, so the stream
    degrades gracefully to raw playback.
    """
    data = load_overlay(cam_id)
    frames = data.get("frames") or {}
    if not frames:
        return frame
    total = int(data.get("total_frames") or 0)
    key = str(frame_idx % total) if total else str(frame_idx)
    boxes = frames.get(key)
    if not boxes:
        return frame

    from display.overlay_draw import draw_tracks  # noqa: PLC0415

    return draw_tracks(frame, boxes, cam_id, get_color_registry())
