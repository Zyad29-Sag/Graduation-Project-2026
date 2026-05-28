"""
modules/llm/description_worker.py
----------------------------------
Background daemon that consumes description tasks and writes structured
results into ``person_descriptions``. Mirrors the ReconciliationWorker
pattern so the lifecycle is consistent (run_forever / stop / per-task
try/except → never crash the daemon).

Producer side: the embedding worker calls ``db.enqueue_description(pid)``
right after binding a track to a person_id, AND pushes a hint onto an
in-memory queue so this worker wakes up quickly. If the in-memory queue
is full the daemon still picks the row up on its next sweep, so we
never block the embedding worker.

Lifecycle:

    startup_recovery()   — re-pending any 'in_progress' rows left over
                            from a previous run (durability)
    run_forever()        — pop hint from in-memory queue OR fall through
                            to claim_next_description() OR sweep for
                            persons without descriptions, dispatch, write
    stop()               — signal the daemon to exit cleanly
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from config.settings import (
    DESCRIPTION_SWEEP_INTERVAL_SEC,
    MAX_DESCRIPTION_ATTEMPTS,
)
from modules.llm.describer import Describer
from modules.preprocessing.quality_gate import CropQualityGate


class DescriptionWorker:
    """
    Single-threaded consumer for the description pipeline.

    Invariants preserved (project_invariants.md):
      • Never blocks the embedding worker (uses a bounded in-memory hint
        queue with non-blocking gets and the durable description_queue
        SQLite table as the source of truth).
      • person_descriptions rows ACCUMULATE — every successful describe
        inserts a new row; latest_description_id moves forward.
      • description_queue rows move append-only:
        pending → in_progress → done | failed.
    """

    def __init__(
        self,
        describer: Describer,
        db,
        llm_queue: queue.Queue,
        max_attempts: int = MAX_DESCRIPTION_ATTEMPTS,
        sweep_interval_sec: int = DESCRIPTION_SWEEP_INTERVAL_SEC,
        quality_gate: Optional[CropQualityGate] = None,
    ) -> None:
        self._stop_event       = threading.Event()
        self._describer        = describer
        self._db               = db
        self._q                = llm_queue
        self._max_attempts     = int(max_attempts)
        self._sweep_interval   = float(sweep_interval_sec)
        self._gate             = quality_gate or CropQualityGate()
        self._last_sweep_ts    = 0.0
        self._n_described      = 0
        self._n_failed         = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()

    def stats(self) -> Dict[str, int]:
        return {
            "described": self._n_described,
            "failed":    self._n_failed,
        }

    def startup_recovery(self) -> int:
        """
        Re-pending any 'in_progress' rows left over from a previous run.
        Returns the number of rows reset.
        """
        # We don't expose a dedicated DB method; the simplest correct
        # implementation is one UPDATE. The daemon is the only writer
        # for in_progress, so this is race-free at startup.
        with self._db._get_conn() as conn:  # noqa: SLF001
            cur = conn.execute(
                "UPDATE description_queue SET status='pending' WHERE status='in_progress'"
            )
            n = cur.rowcount
            if self._db._in_memory:  # noqa: SLF001
                conn.commit()
        if n:
            print(f"[DESCRIBE] recovered {n} stuck 'in_progress' row(s) → pending")
        return n

    def run_forever(self) -> None:
        """Daemon loop. Catches everything; never propagates."""
        self.startup_recovery()
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                import traceback
                print(f"[DESCRIBE] daemon tick error: {exc}\n{traceback.format_exc()}")
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # One iteration
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        # 1. Drain in-memory hint queue (non-blocking) — these are the
        #    freshest persons that the embedding worker just bound.
        try:
            _ = self._q.get_nowait()
        except queue.Empty:
            pass

        # 2. Periodic sweep: any person with NULL latest_description_id
        #    that isn't already enqueued gets a pending row.
        now_ts = time.time()
        if now_ts - self._last_sweep_ts > self._sweep_interval:
            self._last_sweep_ts = now_ts
            self._sweep_for_missing_descriptions()

        # 3. Claim the next pending row and process it. Most of the wall
        #    time goes into the VLM call below.
        claim = self._db.claim_next_description()
        if claim is None:
            # Nothing to do; sleep briefly so we don't busy-loop.
            time.sleep(1.0)
            return

        self._handle(claim)

    def _sweep_for_missing_descriptions(self) -> None:
        missing = self._db.get_persons_without_description(limit=50)
        for pid in missing:
            self._db.enqueue_description(pid)
        if missing:
            print(f"[DESCRIBE] sweep enqueued {len(missing)} missing descriptions")

    def _handle(self, claim: Dict[str, Any]) -> None:
        queue_id  = claim["id"]
        person_id = claim["person_id"]
        t0 = time.time()

        # 1. Load and select best snapshot.
        snapshot_paths = self._load_snapshot_paths(person_id)
        if not snapshot_paths:
            self._db.fail_description(
                queue_id, "no snapshots on disk", self._max_attempts,
            )
            self._n_failed += 1
            return

        best = self._pick_best_snapshot(snapshot_paths)
        if best is None:
            self._db.fail_description(
                queue_id, "no usable snapshot (all rejected by quality gate)",
                self._max_attempts,
            )
            self._n_failed += 1
            return

        # 2. Call backend.
        attributes = self._describer.describe([best])
        if attributes is None:
            self._db.fail_description(
                queue_id, "backend returned None", self._max_attempts,
            )
            self._n_failed += 1
            return

        # 3. Persist.
        summary = attributes.get("summary")
        desc_id = self._db.insert_description(
            person_id      = person_id,
            backend        = self._describer.backend_name,
            model_id       = self._describer.model_id,
            snapshots_used = [best],
            attributes     = attributes,
            summary        = summary,
            confidence     = attributes.get("_confidence"),  # backends may add this
        )
        self._db.complete_description(queue_id, desc_id)
        self._n_described += 1

        dt = time.time() - t0
        short_summary = (summary or "(no summary)")[:80]
        print(
            f"[DESCRIBE] person {person_id[:8]} <- {short_summary} "
            f"(model={self._describer.model_id}, {dt:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Snapshot selection
    # ------------------------------------------------------------------

    def _load_snapshot_paths(self, person_id: str) -> List[str]:
        person = self._db.get_person(person_id)
        if not person:
            return []
        paths = person.get("snapshot_paths") or []
        if isinstance(paths, str):
            try:
                paths = json.loads(paths)
            except (TypeError, ValueError):
                paths = []
        return [p for p in paths if p and Path(p).is_file()]

    def _pick_best_snapshot(self, paths: List[str]) -> Optional[str]:
        """
        Pick the sharpest snapshot that passes the existing quality gate.

        Uses Laplacian variance — the same blur measure CropQualityGate
        already computes — so no new image-quality code is introduced.
        Returns None if every snapshot fails the gate.
        """
        best_path: Optional[str] = None
        best_blur: float = -1.0
        for p in paths:
            img = cv2.imread(p)
            if img is None or img.size == 0:
                continue
            report = self._gate.assess(img)
            if not report.passes:
                continue
            if report.blur_score > best_blur:
                best_blur = report.blur_score
                best_path = p
        return best_path
