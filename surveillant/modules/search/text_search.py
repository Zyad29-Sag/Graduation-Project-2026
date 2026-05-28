"""
modules/search/text_search.py
------------------------------
Natural-language search over the body-description database (Phase 4B).

Pipeline:
    1. Parse the free-text query with QueryParser → partial schema dict.
    2. Stage 1 — structured SQL filter against persons + latest description.
    3. Stage 2 — soft re-rank using a hand-curated synonym table.
    4. Stage 3 — fallback to sentence-transformer cosine similarity over
                 every stored summary, but only if Stage 1 returned no hits
                 AND ``ENABLE_TEXT_FALLBACK_RERANK`` is on.

The engine NEVER raises into the caller. A failed parse or empty result
returns an empty list (with a printed warning).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from config.settings import ENABLE_TEXT_FALLBACK_RERANK
from modules.llm.describer import (
    QueryParser,
    REQUIRED_DESCRIPTION_FIELDS,
    COLOR_PALETTE,
)


# ---------------------------------------------------------------------------
# Small synonym table for soft-matching (Stage 2)
# ---------------------------------------------------------------------------
# Curated by hand; expand as new failure cases show up in real queries.
# Each key is what the operator might say; the value is the canonical
# token used in our schema.
_SYNONYMS: Dict[str, str] = {
    # colors
    "crimson": "red", "scarlet": "red", "burgundy": "red", "maroon": "red",
    "navy": "blue", "azure": "blue", "sky": "blue", "turquoise": "blue",
    "lime": "green", "olive": "green", "emerald": "green",
    "golden": "yellow", "amber": "yellow",
    "dark": "black", "noir": "black",
    "ivory": "white", "cream": "white", "beige": "white",
    "silver": "gray", "ash": "gray", "grey": "gray",
    "tan": "brown", "khaki": "brown", "chocolate": "brown",
    # garments
    "tshirt": "t-shirt", "tee": "t-shirt",
    "hoodie": "sweater", "jumper": "sweater", "pullover": "sweater",
    "trousers": "pants", "slacks": "pants",
    "kicks": "shoes", "sneakers": "shoes",
    # headwear
    "cap": "hat",
    # body
    "chubby": "heavy", "fat": "heavy", "overweight": "heavy", "stocky": "heavy",
    "thin": "slim", "skinny": "slim", "lean": "slim",
}


def _canonicalise(token: str) -> str:
    """Map a word/phrase to its canonical schema token if known."""
    if not token:
        return token
    t = token.strip().lower()
    return _SYNONYMS.get(t, t)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TextSearchEngine:
    """
    Natural-language search facade. One construction per process.
    """

    def __init__(self, db, parser: Optional[QueryParser] = None) -> None:
        self._db = db
        self._parser = parser or QueryParser()
        # Stage-3 reranker is loaded lazily — the import is heavy and
        # most queries won't need it.
        self._st_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Run the full pipeline. Returns a list of dicts:
            {person_id, summary, attributes, score,
             last_seen_cam, last_seen_time, snapshot_paths}
        sorted by descending score.

        An empty list means no candidate matched (Stage 1) and the Stage-3
        fallback either returned nothing or is disabled.
        """
        if not query or not query.strip():
            return []

        # 1. Parse free text → filter dict
        raw_filters = self._parser.parse(query)
        if not raw_filters:
            print("[SEARCH] parser returned no filters; falling back to summary rerank.")
            filters = {}
        else:
            filters = self._canonicalise_filters(raw_filters)
        print(f"[SEARCH] parsed filters: {filters}")

        # 2. Stage 1 — SQL filter
        candidates = self._db.search_persons_by_attributes(filters, limit=200)
        if candidates:
            ranked = self._stage2_rerank(candidates, filters)
            return ranked[:top_k]

        print("[SEARCH] Stage 1 returned 0 candidates.")
        # 3. Stage 3 — semantic fallback over all summaries
        if ENABLE_TEXT_FALLBACK_RERANK:
            return self._stage3_text_fallback(query, top_k=top_k)
        return []

    # ------------------------------------------------------------------
    # Stage 2 — soft re-rank
    # ------------------------------------------------------------------

    def _stage2_rerank(
        self,
        candidates: List[Dict[str, Any]],
        filters:    Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Score each candidate by how many filter fields match its
        attributes. Exact match (after canonicalisation) = +1.0; the
        SQL filter has already enforced these so most hits score
        identically — we keep the framework in place for future fuzzy
        scoring (Stage 4D).
        """
        n_fields = max(len(filters), 1)
        scored: List[Dict[str, Any]] = []
        for cand in candidates:
            attrs = cand.get("attributes") or {}
            hits = 0
            for f, want in filters.items():
                got = attrs.get(f)
                if f == "accessories":
                    got_list = got if isinstance(got, list) else []
                    want_list = want if isinstance(want, list) else [want]
                    if any(_canonicalise(str(w)) in [_canonicalise(str(g)) for g in got_list]
                           for w in want_list):
                        hits += 1
                else:
                    if _canonicalise(str(got or "")) == _canonicalise(str(want or "")):
                        hits += 1
            cand = dict(cand)
            cand["score"] = hits / n_fields
            scored.append(cand)
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Stage 3 — sentence-transformer fallback
    # ------------------------------------------------------------------

    def _stage3_text_fallback(
        self, query: str, top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Rank ALL stored summaries by cosine similarity to the raw query.
        Used only when Stage 1 returned nothing. Requires the optional
        sentence-transformers dependency; if it's missing or fails we
        return [].
        """
        all_rows = self._db.get_all_summaries()
        if not all_rows:
            return []

        try:
            model = self._load_st_model()
        except Exception as exc:
            print(f"[SEARCH] Stage 3 unavailable: {exc}")
            return []

        try:
            import numpy as np  # already a project dep
            q_vec  = model.encode([query], normalize_embeddings=True)[0]
            summaries = [r.get("summary") or "" for r in all_rows]
            s_vecs = model.encode(summaries, normalize_embeddings=True)
            sims = (s_vecs @ q_vec).tolist()
        except Exception as exc:
            print(f"[SEARCH] Stage 3 inference failed: {exc}")
            return []

        ranked = []
        for row, sim in zip(all_rows, sims):
            r = dict(row)
            r["score"] = float(sim)
            ranked.append(r)
        ranked.sort(key=lambda c: c["score"], reverse=True)
        return ranked[:top_k]

    def _load_st_model(self):
        if self._st_model is not None:
            return self._st_model
        from sentence_transformers import SentenceTransformer
        self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonicalise_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f, v in filters.items():
            if f == "accessories":
                if isinstance(v, list):
                    out[f] = [_canonicalise(str(x)) for x in v if x]
                elif isinstance(v, str) and v:
                    out[f] = [_canonicalise(v)]
            elif isinstance(v, str):
                out[f] = _canonicalise(v)
            else:
                out[f] = v
        return out


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
