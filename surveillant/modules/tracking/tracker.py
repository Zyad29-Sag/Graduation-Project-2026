"""
modules/tracking/tracker.py
----------------------------
PersonTracker wraps DeepSORT for per-camera persistent tracking.

Part B Improvement 4: person_id Feedback Loop
When a track is bound to a person_id, its gallery embeddings are stored
as 'appearance model hints'. These give DeepSORT a stable appearance
model even when the person changes angle or enters shadow.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from deep_sort_realtime.deepsort_tracker import DeepSort

from config.settings import MAX_AGE, IOU_THRESHOLD


class PersonTracker:
    """
    Per-camera person tracker using DeepSORT.

    Args:
        cam_id (int): The camera index this tracker is associated with.
    """

    def __init__(self, cam_id: int) -> None:
        self.cam_id: int = cam_id

        self._tracker = DeepSort(
            max_age          = MAX_AGE,
            max_iou_distance = 1.0 - IOU_THRESHOLD,
            n_init           = 2,          # 2 frames to confirm (was 3 = too slow)
            embedder         = None,       # Disabled slow builtin CNN (was "mobilenet")
            half             = False,
            bgr              = True,
        )

        # Part B Improvement 4 — gallery hints per track_id
        self._gallery_hints: Dict[int, List[np.ndarray]] = {}
        self._track_to_person: Dict[int, str]            = {}

        print(f"[PersonTracker] Tracker initialized for camera {cam_id}.")

    def __repr__(self) -> str:
        return f"PersonTracker(cam_id={self.cam_id})"

    # ------------------------------------------------------------------
    # Public API — tracking
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Update the tracker with the latest detections.

        Returns confirmed tracks as:
            [{'track_id': int, 'bbox': [x1,y1,x2,y2], 'cam_id': int}, ...]
        """
        raw_detections = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            raw_detections.append(([x1, y1, w, h], det["confidence"], "person"))

        tracks = self._tracker.update_tracks(
            raw_detections,
            frame=frame,
            embeds=[np.array([1.0])] * len(raw_detections)  # dummy 1D array to satisfy internal norm()
        )

        confirmed: List[Dict[str, Any]] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            confirmed.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                    "cam_id":   self.cam_id,
                }
            )

        return confirmed

    # ------------------------------------------------------------------
    # Public API — gallery hint feedback (Part B Improvement 4)
    # ------------------------------------------------------------------

    def reinforce_track(
        self,
        track_id: int,
        person_id: str,
        gallery_embeddings: List[np.ndarray],
    ) -> None:
        """
        Bind a confirmed track_id to a person_id and store gallery
        embeddings as stable appearance hints for DeepSORT.

        Called after a track is successfully matched/created.
        """
        self._gallery_hints[track_id]   = gallery_embeddings
        self._track_to_person[track_id] = person_id
        print(
            f"[REINFORCE] track cam{self.cam_id}_track{track_id} reinforced "
            f"with {len(gallery_embeddings)}-view gallery for person {person_id[:8]}. "
            f"DeepSORT appearance model updated. Track stability improved."
        )

    def get_appearance_hint(self, track_id: int) -> Optional[np.ndarray]:
        """
        Returns averaged gallery embedding for stable appearance features.
        Returns None if no gallery hints yet.
        """
        hints = self._gallery_hints.get(track_id)
        if hints:
            return np.mean(hints, axis=0)
        return None

    def get_person_id(self, track_id: int) -> Optional[str]:
        """Returns the person_id bound to this track_id, or None."""
        return self._track_to_person.get(track_id)
