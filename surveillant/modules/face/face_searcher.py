"""
modules/face/face_searcher.py
-----------------------------
Face-image search over the ISOLATED face_embeddings table (Part 11).

"Search for a person by their face image": extract the query's face embedding
(InsightFace) and rank stored persons by max-pool cosine similarity over their
face embeddings.

This path is completely separate from the body PersonSearcher / FAISS index:
it reads ONLY face_embeddings (never person_embeddings), so face search and
body identity never interfere. Linear scan is fine here — face galleries are
small and queries are interactive/offline.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config.settings import FACE_SEARCH_THRESHOLD


class FaceSearcher:
    """Cosine search over the isolated face_embeddings store."""

    def __init__(self, db, face_analyzer) -> None:
        self.db   = db
        self.face = face_analyzer

    def search_by_face_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        threshold = FACE_SEARCH_THRESHOLD if min_threshold is None else float(min_threshold)
        galleries = self.db.get_all_face_embeddings()
        if not galleries or query_embedding is None:
            return []

        scores: Dict[str, float] = {}
        for pid, embs in galleries.items():
            best = 0.0
            for e in embs:
                if e.shape[0] != query_embedding.shape[0]:
                    continue
                sim = float(np.dot(query_embedding, e))   # both L2-normalized → cosine
                if sim > best:
                    best = sim
            scores[pid] = best

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for pid, sc in ranked:
            if sc < threshold:
                continue
            person = self.db.get_person(pid)
            if person:
                person["similarity_score"] = sc
                results.append(person)
                if len(results) >= top_k:
                    break
        return results

    def search_by_face_photo(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Load an image, extract its face embedding, and search the face store."""
        if not self.face.ready:
            print("[FACE] FaceAnalyzer not ready — cannot run face-image search.")
            return []
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load query image at {image_path}")
        query = self.face.extract_face_embedding(img)
        if query is None:
            print("[FACE] No face detected in the query image.")
            return []
        return self.search_by_face_embedding(query, top_k=top_k)
