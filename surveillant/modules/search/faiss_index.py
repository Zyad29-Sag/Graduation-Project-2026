"""
modules/search/faiss_index.py
------------------------------
In-memory FAISS vector index for fast cosine-similarity person search.

Part 8 of the Enhancement Proposal.

SQLite remains the source of truth — this index is a redundant in-memory
copy that lives alongside it for O(1) nearest-neighbour search. The
existing SQLite linear-scan path in `searcher.py` is preserved as a
fallback, so the system still functions even if `faiss-cpu` is missing
or the index gets out of sync.

Design:
  - `IndexFlatIP(EMBEDDING_DIM)` — exact inner-product search. On L2-normalized
    vectors this equals cosine similarity. No training step required.
  - Each FAISS vector has a parallel `idx → person_id` mapping. A single
    person can own multiple vectors (one per gallery view); search aggregates
    them via max-pool.
  - All writes go through a lock so concurrent insertions from the embedding
    thread and merges from the reconciliation worker are safe.

Synchronisation flow:
  - On startup, `rebuild_from_db()` reads every embedding from SQLite and
    bulk-loads the index.
  - The Database's `on_embedding_added` callback fires on every new gallery
    entry → `FAISSIndex.add()`.
  - The Database's `on_merge` callback fires on every reconciliation merge →
    `FAISSIndex.reassign_person()` (cheap relabel — no rebuild needed).
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from config.settings import EMBEDDING_DIM


class FAISSIndex:
    """
    Thread-safe in-memory FAISS index over the global person embedding set.

    If faiss-cpu is not installed, the object stays in a disabled state and
    every method becomes a no-op. The Searcher detects this via `.enabled`
    and falls back to the SQLite linear scan automatically.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim     = int(dim)
        self.enabled = _FAISS_AVAILABLE
        self._lock   = threading.Lock()

        if not self.enabled:
            print("[FAISS] faiss-cpu not installed — falling back to SQLite linear scan.")
            return

        self._index = faiss.IndexFlatIP(self.dim)
        self._idx_to_pid: List[Optional[str]] = []           # parallel array — index i → person_id (None = tombstone)
        self._pid_to_idxs: Dict[str, List[int]] = {}         # reverse map for re-labelling on merge

        print(f"[FAISS] In-memory index ready (dim={self.dim}, IndexFlatIP).")

    # ------------------------------------------------------------------
    # Mutation API (called from Database callbacks)
    # ------------------------------------------------------------------

    def add(self, person_id: str, embedding: np.ndarray) -> None:
        """Append a single embedding for ``person_id`` to the index."""
        if not self.enabled or embedding is None:
            return
        # Skip stale/incompatible-backbone embeddings without crashing.
        if embedding.shape[0] != self.dim:
            return

        vec = embedding.astype(np.float32, copy=False).reshape(1, -1)
        with self._lock:
            self._index.add(vec)
            new_idx = len(self._idx_to_pid)
            self._idx_to_pid.append(person_id)
            self._pid_to_idxs.setdefault(person_id, []).append(new_idx)

    def reassign_person(self, keep_id: str, remove_id: str) -> None:
        """
        After a reconciliation merge, relabel all vectors belonging to
        ``remove_id`` so they now belong to ``keep_id``.

        Signature matches ``Database.merge_persons(keep_id, remove_id)`` so
        it can be wired directly as the ``Database.on_merge`` callback.

        Cheaper than rebuilding the index — we only touch the id-map, not
        the FAISS vectors themselves.
        """
        if not self.enabled or keep_id == remove_id:
            return
        with self._lock:
            idxs = self._pid_to_idxs.pop(remove_id, [])
            if not idxs:
                return
            for idx in idxs:
                if 0 <= idx < len(self._idx_to_pid):
                    self._idx_to_pid[idx] = keep_id
            self._pid_to_idxs.setdefault(keep_id, []).extend(idxs)

    def rebuild_from_db(self, db) -> int:
        """
        Reset the index and bulk-load every embedding from SQLite.

        Called once at startup. Also safe to call any time the index is
        suspected of being out of sync.

        Returns the number of vectors added.
        """
        if not self.enabled:
            return 0

        all_galleries = db.get_all_galleries_typed()

        with self._lock:
            self._index       = faiss.IndexFlatIP(self.dim)
            self._idx_to_pid  = []
            self._pid_to_idxs = {}

            count = 0
            for pid, entries in all_galleries.items():
                for entry in entries:
                    emb = entry.get("embedding")
                    if emb is None or emb.shape[0] != self.dim:
                        continue
                    vec = emb.astype(np.float32, copy=False).reshape(1, -1)
                    self._index.add(vec)
                    new_idx = len(self._idx_to_pid)
                    self._idx_to_pid.append(pid)
                    self._pid_to_idxs.setdefault(pid, []).append(new_idx)
                    count += 1

        return count

    # ------------------------------------------------------------------
    # Query API (called from Searcher)
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Return up to ``top_k`` ``(person_id, similarity)`` pairs sorted by
        descending similarity.

        Aggregation is max-pool per person (mirrors the SQLite searcher's
        behaviour): if one person has multiple stored embeddings and several
        score highly, only the best score for that person is returned.
        """
        if not self.enabled or query is None or query.shape[0] != self.dim:
            return []

        vec = query.astype(np.float32, copy=False).reshape(1, -1)

        with self._lock:
            total = self._index.ntotal
            if total == 0:
                return []
            # Over-fetch: one person owns multiple vectors, so to get top_k
            # distinct people we need more raw hits. 10× is plenty in practice.
            k = min(total, max(top_k * 10, 32))
            scores, idxs = self._index.search(vec, k)

            best_per_pid: Dict[str, float] = {}
            for idx, score in zip(idxs[0], scores[0]):
                if idx < 0 or idx >= len(self._idx_to_pid):
                    continue
                pid = self._idx_to_pid[idx]
                if pid is None:    # tombstone
                    continue
                if pid not in best_per_pid or score > best_per_pid[pid]:
                    best_per_pid[pid] = float(score)

        return sorted(best_per_pid.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently in the index (including tombstones)."""
        if not self.enabled:
            return 0
        with self._lock:
            return int(self._index.ntotal)

    @property
    def num_persons(self) -> int:
        """Number of distinct persons indexed."""
        if not self.enabled:
            return 0
        with self._lock:
            return len(self._pid_to_idxs)
