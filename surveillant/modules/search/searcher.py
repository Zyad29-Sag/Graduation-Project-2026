"""
modules/search/searcher.py
--------------------------
Max-pooling cross-camera person search.

Two backends:
  1. FAISS in-memory index (Part 8 — `IndexFlatIP`, ~1000x faster than linear scan)
  2. SQLite linear-scan fallback (legacy path — used when FAISS is disabled or
     the index is empty)

SQLite remains the source of truth in both cases. The Searcher always fetches
person metadata from SQLite for the top-k matches — FAISS only accelerates
the cosine-similarity computation.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from modules.storage.database import Database
from modules.embedding.embedder import PersonEmbedder
from modules.search.faiss_index import FAISSIndex
from config.settings import (
    FACE_MATCH_THRESHOLD,
    BODY_MATCH_THRESHOLD,
    MIN_GALLERY_FOR_MATCHING,
    FAISS_AUDIT_MODE,
)


class PersonSearcher:
    """
    Compares incoming embeddings against all known persons in the DB.

    Max-pooling: score(person) = max similarity over all of that person's
    stored gallery embeddings.
    """

    def __init__(
        self,
        db: Database,
        embedder: PersonEmbedder,
        faiss_index: Optional[FAISSIndex] = None,
    ) -> None:
        self.db          = db
        self.embedder    = embedder
        self.faiss_index = faiss_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        query_embedding_type: str = "body",
        top_k: int = 5,
        min_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return up to ``top_k`` person dicts whose galleries best match the query.

        Each dict has all the fields from ``Database.get_person()`` plus a
        ``similarity_score`` key.

        Args:
            min_threshold: Override the config threshold for this call. Used by
                main.py Phase-2 identification to pass BODY_MATCH_THRESHOLD_CROSS_CAM
                (0.68) so that cross-camera candidates are not filtered out before
                the context-aware same-cam / cross-cam decision is applied.
                If None, the config value for the embedding type is used.
        """
        if min_threshold is not None:
            threshold = min_threshold
        else:
            threshold = (
                FACE_MATCH_THRESHOLD if query_embedding_type == "face"
                else BODY_MATCH_THRESHOLD
            )

        faiss_available = (
            self.faiss_index is not None
            and self.faiss_index.enabled
            and self.faiss_index.size > 0
        )

        # Audit mode (off by default): run BOTH paths on every query and log
        # any disagreement. Returns the SQLite result for safety while the
        # audit is in progress. Used to diagnose the FAISS gallery-sponge
        # regression — see DECISION_LOG.md Part 8.
        if FAISS_AUDIT_MODE and faiss_available:
            faiss_scored  = self.faiss_index.search(query_embedding, top_k=top_k)
            sqlite_result = self._search_via_sqlite_raw(query_embedding, top_k)
            self._log_audit_drift(faiss_scored, sqlite_result)
            return self._hydrate(sqlite_result, threshold, top_k)

        # Path 1 — FAISS (preferred): the index already aggregates per-person
        # max-pool scores and returns sorted (pid, sim) pairs.
        if faiss_available:
            scored = self.faiss_index.search(query_embedding, top_k=top_k)
            return self._hydrate(scored, threshold, top_k)

        # Path 2 — SQLite linear scan (fallback): the original behaviour.
        return self._search_via_sqlite(query_embedding, threshold, top_k)

    def search_by_photo(
        self, query_image_path: str, top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Load an image from disk, extract body embedding, and search."""
        img = cv2.imread(query_image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {query_image_path}")
        vector = self.embedder.extract_body_embedding(img)
        return self.search_by_embedding(vector, query_embedding_type="body", top_k=top_k)

    # ------------------------------------------------------------------
    # Internal — SQLite fallback (kept for safety net + tests)
    # ------------------------------------------------------------------

    def _search_via_sqlite(
        self,
        query_embedding: np.ndarray,
        threshold: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        scored = self._search_via_sqlite_raw(query_embedding, top_k)
        return self._hydrate(scored, threshold, top_k)

    def _search_via_sqlite_raw(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[tuple]:
        """
        SQLite linear scan that returns the raw sorted ``(pid, score)`` list
        without applying the threshold or hydrating to person dicts. Shared
        between the standard SQLite path and the FAISS audit comparison.
        """
        all_galleries = self.db.get_all_galleries_typed()
        if not all_galleries:
            return []

        query_2d = query_embedding.reshape(1, -1)
        person_scores: Dict[str, float] = {}

        for pid, gallery_entries in all_galleries.items():
            if len(gallery_entries) < MIN_GALLERY_FOR_MATCHING:
                continue

            best_score = 0.0
            for entry in gallery_entries:
                stored_vec = entry["embedding"]
                if stored_vec.shape[0] != query_embedding.shape[0]:
                    continue  # stale embedding from a different backbone
                sim = float(cosine_similarity(query_2d, stored_vec.reshape(1, -1))[0][0])
                if sim > best_score:
                    best_score = sim
            person_scores[pid] = best_score

        return sorted(person_scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Internal — FAISS_AUDIT_MODE drift logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_audit_drift(
        faiss_scored: List[tuple],
        sqlite_scored: List[tuple],
    ) -> None:
        """
        Print a [FAISS_DRIFT] line when the FAISS and SQLite paths
        disagree on either the top-1 person_id or the top-1 score by more
        than 0.01.

        Silence is the goal — every line that fires is a clue.
        """
        faiss_top  = faiss_scored[0]  if faiss_scored  else (None, 0.0)
        sqlite_top = sqlite_scored[0] if sqlite_scored else (None, 0.0)

        pid_diff   = faiss_top[0] != sqlite_top[0]
        score_diff = abs(float(faiss_top[1]) - float(sqlite_top[1])) > 0.01
        if not pid_diff and not score_diff:
            return

        def _fmt(pid, score):
            short = pid[:8] if pid else "None"
            return f"({short}, {float(score):.4f})"

        print(
            f"[FAISS_DRIFT] faiss={_fmt(*faiss_top)} sqlite={_fmt(*sqlite_top)} "
            f"Δscore={abs(float(faiss_top[1]) - float(sqlite_top[1])):.4f}"
            f"{' [PID DIFFER]' if pid_diff else ''}"
        )

    # ------------------------------------------------------------------
    # Internal — shared post-processing
    # ------------------------------------------------------------------

    def _hydrate(
        self,
        scored: List[tuple],
        threshold: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Convert (pid, score) pairs into full person dicts above threshold."""
        results: List[Dict[str, Any]] = []
        for pid, score in scored:
            if score < threshold:
                continue
            person_data = self.db.get_person(pid)
            if person_data:
                person_data["similarity_score"] = float(score)
                results.append(person_data)
                if len(results) >= top_k:
                    break
        return results
