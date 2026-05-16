"""
debug_activity_tracker.py
--------------------------
Lightweight DB change-detection for the SURVEILLANT debug dashboard.

Strategy: periodic snapshotting + diffing — no triggers, no WAL parsing.
The ActivityTracker polls the SQLite database every N seconds and compares
the current state against the previous snapshot to emit structured change events.

Safe for concurrent use: reads in WAL mode (read-only intent).
"""

import sqlite3
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------
EVENT_NEW_PERSON    = "NEW_PERSON"
EVENT_STATUS_CHANGE = "STATUS_CHANGE"
EVENT_GALLERY_UPDATE= "GALLERY_UPDATE"
EVENT_MERGE         = "MERGE"
EVENT_LAST_SEEN     = "LAST_SEEN_UPDATE"
EVENT_GHOST         = "GHOST_MARKED"


class ActivityTracker:
    """
    Polls the database and produces an activity log of DB mutations.

    Usage:
        tracker = ActivityTracker(db_path)
        tracker.start()           # starts background polling thread
        events = tracker.get_recent_activity(since_seconds=60)
        tracker.stop()
    """

    MAX_EVENTS = 500  # keep last N events in memory

    def __init__(self, db_path: str, poll_interval: float = 2.0):
        self.db_path = str(db_path)
        self.poll_interval = poll_interval

        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []   # newest at end
        self._last_snapshot: Optional[Dict[str, Any]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        # Take first snapshot immediately
        self._last_snapshot = self._snapshot()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_recent_activity(self, since_seconds: float = 120) -> List[Dict[str, Any]]:
        """Return events that occurred in the last `since_seconds` seconds."""
        cutoff = time.time() - since_seconds
        with self._lock:
            return [e for e in self._events if e["_ts"] >= cutoff]

    def get_all_activity(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate DB counts."""
        return self._query_stats()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)
            try:
                new_snap = self._snapshot()
                if self._last_snapshot is not None:
                    new_events = self._diff(self._last_snapshot, new_snap)
                    if new_events:
                        with self._lock:
                            self._events.extend(new_events)
                            # Trim to MAX_EVENTS
                            if len(self._events) > self.MAX_EVENTS:
                                self._events = self._events[-self.MAX_EVENTS:]
                self._last_snapshot = new_snap
            except Exception:
                pass  # DB not ready yet — silently retry

    def _connect(self) -> sqlite3.Connection:
        """Open a read-only WAL connection."""
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=5.0,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _snapshot(self) -> Dict[str, Any]:
        """
        Capture the current DB state as a lightweight dict.
        Keys:
          persons:  {person_id: {status, gallery_size, last_seen_cam, last_seen_time}}
          merge_count: int (total proposals ever)
          merge_statuses: {id: status}
          embedding_counts: {person_id: count}
        """
        snap: Dict[str, Any] = {
            "persons": {},
            "merge_count": 0,
            "merge_statuses": {},
            "embedding_counts": {},
            "ts": time.time(),
        }
        try:
            conn = self._connect()
            # Persons
            cur = conn.execute(
                "SELECT person_id, status, gallery_size, last_seen_cam, last_seen_time, "
                "first_seen_cam, first_seen_time, known_angles FROM persons"
            )
            for row in cur.fetchall():
                snap["persons"][row["person_id"]] = {
                    "status":          row["status"],
                    "gallery_size":    row["gallery_size"] or 0,
                    "last_seen_cam":   row["last_seen_cam"],
                    "last_seen_time":  row["last_seen_time"],
                    "first_seen_cam":  row["first_seen_cam"],
                    "first_seen_time": row["first_seen_time"],
                    "known_angles":    row["known_angles"] or "[]",
                }
            # Merge proposals
            cur2 = conn.execute("SELECT id, status, person_id_a, person_id_b FROM merge_proposals")
            for row in cur2.fetchall():
                snap["merge_statuses"][row["id"]] = {
                    "status": row["status"],
                    "pid_a":  row["person_id_a"],
                    "pid_b":  row["person_id_b"],
                }
            snap["merge_count"] = len(snap["merge_statuses"])
            conn.close()
        except Exception:
            pass
        return snap

    def _diff(
        self, old: Dict[str, Any], new: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare two snapshots and return list of change events."""
        events: List[Dict[str, Any]] = []
        now_ts = time.time()
        now_iso = datetime.now().isoformat(timespec="seconds")

        old_persons = old.get("persons", {})
        new_persons = new.get("persons", {})

        # ---- Detect new persons ----
        for pid in new_persons:
            if pid not in old_persons:
                p = new_persons[pid]
                events.append({
                    "_ts":       now_ts,
                    "type":      EVENT_NEW_PERSON,
                    "person_id": pid,
                    "cam_id":    p["first_seen_cam"],
                    "time":      now_iso,
                    "message":   f"New person detected on cam {p['first_seen_cam']}",
                })

        # ---- Detect changes in existing persons ----
        for pid in new_persons:
            if pid not in old_persons:
                continue
            o = old_persons[pid]
            n = new_persons[pid]

            # Status change
            if o["status"] != n["status"]:
                is_ghost = n["status"] == "ghost"
                events.append({
                    "_ts":       now_ts,
                    "type":      EVENT_GHOST if is_ghost else EVENT_STATUS_CHANGE,
                    "person_id": pid,
                    "old_status": o["status"],
                    "new_status": n["status"],
                    "time":      now_iso,
                    "message":   (
                        f"Person marked ghost"
                        if is_ghost
                        else f"Status: {o['status']} → {n['status']}"
                    ),
                })

            # Gallery update
            if (n["gallery_size"] or 0) > (o["gallery_size"] or 0):
                new_angles = _parse_angles(n["known_angles"])
                old_angles = _parse_angles(o["known_angles"])
                added = [a for a in new_angles if a not in old_angles]
                events.append({
                    "_ts":        now_ts,
                    "type":       EVENT_GALLERY_UPDATE,
                    "person_id":  pid,
                    "gallery_size": n["gallery_size"],
                    "new_angles": added,
                    "time":       now_iso,
                    "message":    (
                        f"Gallery → {n['gallery_size']} embeddings"
                        + (f" (+{', '.join(added)})" if added else "")
                    ),
                })

            # Last-seen update (cross-camera movement)
            if (
                o["last_seen_cam"] != n["last_seen_cam"]
                and n["last_seen_cam"] is not None
            ):
                events.append({
                    "_ts":       now_ts,
                    "type":      EVENT_LAST_SEEN,
                    "person_id": pid,
                    "from_cam":  o["last_seen_cam"],
                    "to_cam":    n["last_seen_cam"],
                    "time":      now_iso,
                    "message":   f"Moved cam {o['last_seen_cam']} → cam {n['last_seen_cam']}",
                })

        # ---- Detect merge executions (pending → accepted) ----
        old_merges = old.get("merge_statuses", {})
        new_merges = new.get("merge_statuses", {})
        for mid, nm in new_merges.items():
            if mid in old_merges:
                om = old_merges[mid]
                if om["status"] == "pending" and nm["status"] == "accepted":
                    events.append({
                        "_ts":      now_ts,
                        "type":     EVENT_MERGE,
                        "merge_id": mid,
                        "person_id_a": nm["pid_a"],
                        "person_id_b": nm["pid_b"],
                        "time":     now_iso,
                        "message":  f"{_short(nm['pid_a'])} merged → {_short(nm['pid_b'])}",
                    })

        return events

    def _query_stats(self) -> Dict[str, Any]:
        """Return live aggregate stats from the database."""
        stats: Dict[str, Any] = {
            "total_persons":    0,
            "by_status":        {},
            "total_embeddings": 0,
            "pending_merges":   0,
            "total_merges":     0,
            "db_ok":            False,
        }
        try:
            conn = self._connect()
            # Person counts
            cur = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM persons GROUP BY status"
            )
            total = 0
            for row in cur.fetchall():
                stats["by_status"][row["status"]] = row["cnt"]
                total += row["cnt"]
            stats["total_persons"] = total

            # Embedding count
            cur2 = conn.execute("SELECT COUNT(*) FROM person_embeddings")
            stats["total_embeddings"] = cur2.fetchone()[0]

            # Merge counts
            cur3 = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM merge_proposals GROUP BY status"
            )
            for row in cur3.fetchall():
                if row["status"] == "pending":
                    stats["pending_merges"] = row["cnt"]
                stats["total_merges"] = stats.get("total_merges", 0) + row["cnt"]

            stats["db_ok"] = True
            conn.close()
        except Exception:
            pass
        return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_angles(json_str: str) -> List[str]:
    try:
        return json.loads(json_str or "[]")
    except Exception:
        return []


def _short(person_id: str) -> str:
    """Return first 8 chars of a UUID for readable log messages."""
    return person_id[:8] if person_id else "???"
