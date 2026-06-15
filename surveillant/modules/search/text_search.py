"""
modules/search/text_search.py
------------------------------
Natural-language search over the body-description database (Phase 4B).

**Semantic, embedding-based.** Each person's ``long_description`` is embedded at
describe time (``all-MiniLM-L6-v2``) and stored. A query is embedded the same way
and ranked against every stored vector by cosine similarity, so results are the
NEAREST IN MEANING — not a brittle SQL ``LIKE`` keyword match. This handles
synonyms, paraphrases and partial descriptions naturally ("guy in a dark tee"
matches "a man wearing a black t-shirt") with no hand-curated synonym table.

The engine NEVER raises into the caller: a missing embedding model or an empty
database returns an empty list (with a printed note).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class TextSearchEngine:
    """Semantic natural-language search facade. One construction per process."""

    def __init__(self, db, parser: Optional[Any] = None) -> None:
        # `parser` is accepted for backward-compatible construction but is no
        # longer used — search is fully semantic now.
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Embed ``query`` and rank every person's stored description embedding by
        cosine similarity. Returns a list of dicts sorted by descending score:
            {person_id, summary, attributes, score,
             last_seen_cam, last_seen_time, snapshot_paths}
        Empty list = no descriptions in the DB, or the embedding model is
        unavailable.
        """
        if not query or not query.strip():
            return []

        rows = self._db.get_all_summaries()
        if not rows:
            print("[SEARCH] No descriptions in the database yet — run "
                  "`--describe-all` first.")
            return []

        try:
            from modules.search.text_embedder import encode
            q_vec = encode([query])[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[SEARCH] semantic model unavailable ({exc}). "
                  "Install with: pip install sentence-transformers")
            return []

        scored: List[Dict[str, Any]] = []
        missing = 0
        for row in rows:
            emb = row.get("embedding")
            if not emb:
                missing += 1
                continue
            vec = np.frombuffer(emb, dtype="float32")
            if vec.shape[0] != q_vec.shape[0]:
                continue  # stale/incompatible vector — skip
            r = dict(row)
            r["score"] = float(np.dot(vec, q_vec))  # both L2-normalised → cosine
            scored.append(r)

        if missing:
            print(f"[SEARCH] {missing} person(s) have a description but no "
                  "embedding yet — re-run `--redescribe-all` to index them.")

        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# CLI helper (used by main.py --search-text)
# ---------------------------------------------------------------------------

def format_results(results: List[Dict[str, Any]]) -> str:
    """Pretty-print search results for the CLI."""
    if not results:
        return "(no matches)"
    lines = [f"  Found {len(results)} match(es):", ""]
    for i, r in enumerate(results, 1):
        pid_short = r["person_id"][:8]
        score = r.get("score", 0.0)
        cam = r.get("last_seen_cam")
        when = r.get("last_seen_time", "")
        summary = r.get("summary") or "(no summary)"
        snaps = r.get("snapshot_paths") or []
        first_snap = snaps[0] if snaps else "(no snapshot)"
        lines.append(
            f"  {i}. person {pid_short}  score={score:.2f}  "
            f"last seen cam{cam} @ {when}"
        )
        lines.append(f"     {summary}")
        lines.append(f"     snapshot: {first_snap}")
        lines.append("")
    return "\n".join(lines)
