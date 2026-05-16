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

from config.settings import MAX_AGE, MIN_HITS, IOU_THRESHOLD


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
            n_init           = MIN_HITS,
            embedder         = None,       # Disabled — SURVEILLANT manages Re-ID externally
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
            # Skip pure Kalman predictions — only draw tracks that were matched
            # to an actual detection this cycle (time_since_update == 0).
            # Stale predictions cause "ghost boxes" trailing behind moving persons.
            if track.time_since_update > 1:
                continue
            ltrb = track.to_ltrb()
            confirmed.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                    "cam_id":   self.cam_id,
                }
            )

        return self._deduplicate(confirmed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate(
        self,
        tracks: List[Dict[str, Any]],
        iou_threshold: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Remove confirmed tracks whose bounding boxes heavily overlap.
        Keeps the track with the lower (older) track_id.
        Prevents split YOLO detections from showing as two boxes on screen.
        """
        if len(tracks) <= 1:
            return tracks

        sorted_tracks = sorted(tracks, key=lambda t: t["track_id"])
        suppressed = set()

        for i in range(len(sorted_tracks)):
            if i in suppressed:
                continue
            for j in range(i + 1, len(sorted_tracks)):
                if j in suppressed:
                    continue
                if self._iou(sorted_tracks[i]["bbox"], sorted_tracks[j]["bbox"]) > iou_threshold:
                    suppressed.add(j)

        return [t for idx, t in enumerate(sorted_tracks) if idx not in suppressed]

    @staticmethod
    def _iou(box_a: list, box_b: list) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

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
            f"[REINFORCE] cam{self.cam_id}_track{track_id} -> person {person_id[:8]} "
            f"({len(gallery_embeddings)}-view gallery stored)"
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
