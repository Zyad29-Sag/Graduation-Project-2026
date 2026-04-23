"""
modules/embedding/gallery.py
-----------------------------
GalleryManager — decides when a new embedding is novel enough to store
and provides the angle-heuristic tag.

Part B Improvement 1: Multi-Angle Gallery Learning
Rules:
  1. Always accept the first embedding.
  2. Hard cap at MAX_GALLERY_SIZE.
  3. Reject if cosine distance > GALLERY_MAX_DISTANCE (garbage/occlusion).
  4. Prefer face embeddings — reject a new body if gallery has >= 2 faces
     and is >= 75% full.
  5. Require minimum distance to avoid near-duplicates:
       face: FACE_GALLERY_ADD_DISTANCE
       body: BODY_GALLERY_ADD_DISTANCE
"""

import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

from config.settings import (
    MAX_GALLERY_SIZE,
    FACE_GALLERY_ADD_DISTANCE,
    BODY_GALLERY_ADD_DISTANCE,
    GALLERY_MAX_DISTANCE,
    MIN_FRAMES_BETWEEN_SAMPLES,
)


class GalleryManager:
    """
    Stateless gallery decision engine.
    All state lives in the SQLite person_embeddings table.
    """

    # ----------------------------------------------------------------
    # Core should_add decision
    # ----------------------------------------------------------------

    def should_add_to_gallery(
        self,
        new_embedding: np.ndarray,
        new_type: str,                        # 'face' or 'body'
        existing_gallery: List[Dict[str, Any]],  # [{'embedding', 'type', 'source_cam'}, ...]
    ) -> bool:
        """
        Returns True if new_embedding is novel enough to store.

        Args:
            new_embedding:    Normalized 1-D float32 array.
            new_type:         'face' or 'body'.
            existing_gallery: List of dicts with 'embedding', 'type', 'source_cam'.
        """
        if not existing_gallery:
            return True

        if len(existing_gallery) >= MAX_GALLERY_SIZE:
            return False

        existing_vecs  = [e["embedding"] for e in existing_gallery]
        existing_types = [e["type"]      for e in existing_gallery]

        face_count = sum(1 for t in existing_types if t == "face")

        # Protect face embeddings from being crowded out by body data
        if (new_type == "body"
                and face_count >= 2
                and len(existing_gallery) >= int(MAX_GALLERY_SIZE * 0.75)):
            return False

        query_2d      = new_embedding.reshape(1, -1)
        gallery_array = np.array(existing_vecs)
        similarities  = cosine_similarity(query_2d, gallery_array)[0]
        max_sim       = float(np.max(similarities))
        distance      = 1.0 - max_sim

        # Reject garbage crops (too dissimilar from everything we know)
        if distance > GALLERY_MAX_DISTANCE:
            return False

        novelty_thresh = (
            FACE_GALLERY_ADD_DISTANCE if new_type == "face"
            else BODY_GALLERY_ADD_DISTANCE
        )
        return distance > novelty_thresh

    # ----------------------------------------------------------------
    # Angle heuristic
    # ----------------------------------------------------------------

    def get_angle_tag(
        self,
        new_embedding: np.ndarray,
        existing_gallery: List[Dict[str, Any]],
        source_cam: Optional[int] = None,
        person_first_cam: Optional[int] = None,
    ) -> str:
        """
        Heuristic angle classification.

        Priority:
          'initial'          — first embedding ever
          'cross_cam_view'   — source_cam differs from person_first_cam
          'very_different'   — cosine distance > 0.5 (likely back view)
          'same_cam_new_angle' — different angle, same camera
          'partial_view'     — small difference
        """
        if not existing_gallery:
            return "initial"

        if (source_cam is not None
                and person_first_cam is not None
                and source_cam != person_first_cam):
            return "cross_cam_view"

        existing_vecs = [e["embedding"] for e in existing_gallery]
        query_2d      = new_embedding.reshape(1, -1)
        gallery_array = np.array(existing_vecs)
        similarities  = cosine_similarity(query_2d, gallery_array)[0]
        distance      = 1.0 - float(np.max(similarities))

        if distance > 0.5:
            return "very_different"
        elif distance > BODY_GALLERY_ADD_DISTANCE:
            return "same_cam_new_angle"
        else:
            return "partial_view"

    # ----------------------------------------------------------------
    # Gallery update orchestration (Part B Improvement 1)
    # ----------------------------------------------------------------

    def maybe_update_gallery(
        self,
        person_id: str,
        crop: np.ndarray,
        embedder,        # PersonEmbedder
        db,              # Database
        frame_count: int,
        cam_id: int = 0,
    ) -> bool:
        """
        Called on every frame for a bound track.
        Rate-limited to every MIN_FRAMES_BETWEEN_SAMPLES frames.

        Returns True if a new embedding was added to the gallery.
        """
        if frame_count % MIN_FRAMES_BETWEEN_SAMPLES != 0:
            return False

        if crop is None or crop.size == 0:
            return False

        gallery_size = db.get_gallery_size(person_id)
        if gallery_size >= MAX_GALLERY_SIZE:
            return False

        # Extract embedding
        new_emb  = embedder.extract_body_embedding(crop)
        new_type = "body"

        existing = db.get_gallery_typed(person_id)

        if not self.should_add_to_gallery(new_emb, new_type, existing):
            if existing:
                existing_vecs = [e["embedding"] for e in existing]
                query_2d      = new_emb.reshape(1, -1)
                sims          = cosine_similarity(query_2d, np.array(existing_vecs))[0]
                max_sim       = float(np.max(sims))
                dist          = 1.0 - max_sim
                print(
                    f"[GALLERY] person {person_id[:8]} | {new_type} view REJECTED "
                    f"dist={dist:.2f} (too similar)"
                )
            return False

        # Determine angle tag
        person_data = db.get_person(person_id)
        first_cam   = person_data.get("first_seen_cam") if person_data else None
        angle_tag   = self.get_angle_tag(new_emb, existing, cam_id, first_cam)

        now_str = datetime.datetime.now().isoformat()
        db.add_embedding_to_gallery(
            person_id      = person_id,
            embedding_bytes= embedder.serialize(new_emb),
            embedding_type = new_type,
            angle_tag      = angle_tag,
            source_cam     = cam_id,
            captured_at    = now_str,
        )
        print(
            f"[GALLERY] person {person_id[:8]} | {new_type} view added "
            f"angle={angle_tag} (gallery: {gallery_size}->{gallery_size+1})"
        )
        return True
