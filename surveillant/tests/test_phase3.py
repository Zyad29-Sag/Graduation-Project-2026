"""
Run: python tests/test_phase3.py
Verifies: Cosine similarity math, max-pooling search (typed API),
          cross-type penalty, body threshold guard, gallery-learning
          cross-angle query, reconciliation duplicate detection.
"""
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.search.searcher import PersonSearcher, CROSS_TYPE_PENALTY
from modules.embedding.embedder import PersonEmbedder
from modules.storage.database import Database


def make_unit_vec(size: int = 576) -> np.ndarray:
    v = np.random.randn(size).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Cosine similarity math
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical() -> None:
    from sklearn.metrics.pairwise import cosine_similarity
    v     = make_unit_vec()
    score = cosine_similarity([v], [v])[0][0]
    assert abs(score - 1.0) < 0.001, "Identical vectors must have similarity 1.0"
    print(f"[OK] Cosine similarity of identical vectors = {score:.4f}")


def test_cosine_similarity_different() -> None:
    from sklearn.metrics.pairwise import cosine_similarity
    v1    = make_unit_vec()
    v2    = make_unit_vec()
    score = cosine_similarity([v1], [v2])[0][0]
    assert score < 0.5, f"Random vectors should have low similarity, got {score}"
    print(f"[OK] Cosine similarity of random vectors = {score:.4f} (low as expected)")


# ---------------------------------------------------------------------------
# Search format
# ---------------------------------------------------------------------------

def test_search_returns_correct_format() -> None:
    db  = Database(":memory:")
    emb = PersonEmbedder()
    vec = make_unit_vec()

    pid = db.insert_person({
        "cam_id"         : 2,
        "embedding"      : emb.serialize(vec),
        "embedding_type" : "body",
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_time" : "2024-01-01T10:00:10",
        "snapshot_paths" : [],
        "created_at"     : "2024-01-01T10:00:10",
    })
    # Add a 2nd embedding so it passes MIN_GALLERY_FOR_MATCHING=2
    db.add_embedding_to_gallery(pid, emb.serialize(make_unit_vec()), "body", "side", "2024-01-01T10:00:11")

    searcher = PersonSearcher(db, emb)
    results  = searcher.search_by_embedding(vec, query_embedding_type="body", top_k=3)
    assert len(results) >= 1
    assert "person_id"       in results[0]
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] > 0.99, (
        f"Expected >0.99 (exact match), got {results[0]['similarity_score']}"
    )
    print(f"[OK] Search returns correct format, top score = {results[0]['similarity_score']:.4f}")


# ---------------------------------------------------------------------------
# Max-pooling
# ---------------------------------------------------------------------------

def test_search_max_pooling() -> None:
    """Person with 3 gallery embeddings; query matches only the 3rd -> still top result."""
    db  = Database(":memory:")
    emb = PersonEmbedder()
    v1  = make_unit_vec()

    pid = db.insert_person({
        "cam_id"         : 0,
        "embedding"      : emb.serialize(v1),
        "embedding_type" : "body",
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_time" : "2024-01-01T10:00:00",
        "snapshot_paths" : [],
        "created_at"     : "2024-01-01T10:00:00",
    })

    v2 = make_unit_vec()
    v3 = make_unit_vec()
    db.add_embedding_to_gallery(pid, emb.serialize(v2), "body", "side", "2024-01-01T10:00:05")
    db.add_embedding_to_gallery(pid, emb.serialize(v3), "body", "back", "2024-01-01T10:00:10")

    # Query with v3 — the person has this exact view in their gallery
    searcher = PersonSearcher(db, emb)
    results  = searcher.search_by_embedding(v3, query_embedding_type="body", top_k=1)

    assert len(results) >= 1,         "Expected at least 1 result"
    assert results[0]["person_id"] == pid, "Max-pooling should return the correct person"
    assert results[0]["similarity_score"] > 0.99, (
        f"v3 is in gallery, score should be ~1.0, got {results[0]['similarity_score']:.4f}"
    )
    print(f"[OK] Max-pooling search: correct person found. Score={results[0]['similarity_score']:.4f}")


# ---------------------------------------------------------------------------
# Body threshold
# ---------------------------------------------------------------------------

def test_body_search_requires_high_score() -> None:
    """Body embedding query should NOT match a random person."""
    from config.settings import BODY_MATCH_THRESHOLD
    db  = Database(":memory:")
    emb = PersonEmbedder()

    stored_vec = make_unit_vec()
    pid = db.insert_person({
        "cam_id"         : 0,
        "embedding"      : emb.serialize(stored_vec),
        "embedding_type" : "body",
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_time" : "2024-01-01T10:00:00",
        "snapshot_paths" : [],
        "created_at"     : "2024-01-01T10:00:00",
    })
    db.add_embedding_to_gallery(pid, emb.serialize(make_unit_vec()), "body", "side", "2024")

    query_vec = make_unit_vec()
    searcher  = PersonSearcher(db, emb)
    results   = searcher.search_by_embedding(query_vec, query_embedding_type="body", top_k=1)
    assert len(results) == 0, (
        f"Expected no match for random body query, but got {len(results)} results"
    )
    print(f"[OK] Body search with random vector correctly returns no match (threshold={BODY_MATCH_THRESHOLD})")


# ---------------------------------------------------------------------------
# Cross-type penalty
# ---------------------------------------------------------------------------

def test_cross_type_penalty_applied() -> None:
    from sklearn.metrics.pairwise import cosine_similarity
    v          = make_unit_vec()
    raw_score  = cosine_similarity([v], [v])[0][0]
    penalized  = raw_score * CROSS_TYPE_PENALTY
    assert penalized < raw_score, "Cross-type penalty must reduce score"
    print(f"[OK] Cross-type penalty applied: {raw_score:.4f} -> {penalized:.4f}")


# ---------------------------------------------------------------------------
# NEW: Gallery learning across angles (Part B Improvement 1)
# ---------------------------------------------------------------------------

def test_gallery_learning_across_angles() -> None:
    """Person with front+back embeddings should match a side-view query."""
    db  = Database(":memory:")
    emb = PersonEmbedder()

    pid = "test-person-multi-angle"
    db.insert_person({
        "person_id"      : pid,
        "cam_id"         : 0,
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_time" : "2024-01-01T10:00:10",
        "snapshot_paths" : [],
        "created_at"     : "2024-01-01T10:00:00",
    })

    front_emb  = np.random.randn(576).astype(np.float32)
    front_emb /= np.linalg.norm(front_emb)
    back_emb   = -front_emb + np.random.randn(576).astype(np.float32) * 0.1
    back_emb  /= np.linalg.norm(back_emb)

    db.add_embedding_to_gallery(pid, emb.serialize(front_emb), "body", "front", "2024-01-01T10:00:00")
    db.add_embedding_to_gallery(pid, emb.serialize(back_emb),  "body", "back",  "2024-01-01T10:00:05")

    # Construct a side-view guaranteed to score 0.98 similarity to front_emb
    ortho      = np.random.randn(576).astype(np.float32)
    ortho      = ortho - np.dot(ortho, front_emb) * front_emb
    ortho     /= np.linalg.norm(ortho)
    sim_target = 0.98
    side_query = sim_target * front_emb + np.sqrt(1 - sim_target**2) * ortho


    searcher = PersonSearcher(db, emb)
    results  = searcher.search_by_embedding(side_query, "body", top_k=3)
    assert len(results) > 0 and results[0]["person_id"] == pid, (
        "Side-view query should find person via max-pooling gallery"
    )
    print(f"[OK] Multi-angle gallery: side-view query found person (sim={results[0]['similarity_score']:.3f})")


# ---------------------------------------------------------------------------
# NEW: Reconciliation detects duplicate (Part B Improvement 3)
# ---------------------------------------------------------------------------

def test_reconciliation_detects_duplicate() -> None:
    """Reconciliation worker should flag two very similar persons as merge candidates."""
    from modules.reconciliation.worker import ReconciliationWorker
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    db  = Database(":memory:")
    emb = PersonEmbedder()
    import datetime
    now = datetime.datetime.now().isoformat()

    base = np.random.randn(576).astype(np.float32)
    base /= np.linalg.norm(base)

    # Build nearly identical gallery for two persons
    for pid in ["person-A", "person-B"]:
        # tiny noise — similarity should be ~0.9998
        v  = base + np.random.randn(576).astype(np.float32) * 0.001
        v /= np.linalg.norm(v)
        v2 = base + np.random.randn(576).astype(np.float32) * 0.001
        v2 /= np.linalg.norm(v2)
        db.insert_person({
            "person_id"      : pid,
            "cam_id"         : 0,
            "first_seen_time": now,
            "last_seen_time" : now,
            "snapshot_paths" : [],
            "created_at"     : now,
        })
        db.add_embedding_to_gallery(pid, emb.serialize(v),  "body", "front", now)
        db.add_embedding_to_gallery(pid, emb.serialize(v2), "body", "side",  now)

    worker  = ReconciliationWorker()
    summary = worker.run_cycle(db, track_registry={})
    total   = summary["merge_proposals"] + summary["auto_merges"]
    assert total >= 1, (
        f"Should have found >= 1 merge candidate/auto-merge, got: {summary}"
    )
    print(f"[OK] Reconciliation detected duplicate persons (proposals={summary['merge_proposals']}, auto={summary['auto_merges']})")




if __name__ == "__main__":
    test_cosine_similarity_identical()
    test_cosine_similarity_different()
    test_search_returns_correct_format()
    test_search_max_pooling()
    test_body_search_requires_high_score()
    test_cross_type_penalty_applied()
    test_gallery_learning_across_angles()
    test_reconciliation_detects_duplicate()
    print("\nAll Phase 3 tests passed.")
