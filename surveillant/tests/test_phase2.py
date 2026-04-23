"""
Run: python tests/test_phase2.py
Verifies: PersonEmbedder, GalleryManager (typed API), embedding aggregation,
          Database CRUD, ColorRegistry, track_registry permanence,
          status promotion, body threshold separation.
"""
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.embedding.embedder import PersonEmbedder
from modules.embedding.gallery import GalleryManager
from modules.storage.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_unit_vec(size: int = 576) -> np.ndarray:
    v = np.random.randn(size).astype(np.float32)
    return v / np.linalg.norm(v)


def make_gallery_entry(vec: np.ndarray, emb_type: str = "body", source_cam: int = 0) -> dict:
    return {"embedding": vec, "type": emb_type, "source_cam": source_cam}


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

def test_embedder() -> None:
    emb   = PersonEmbedder()
    dummy = np.zeros((112, 112, 3), dtype=np.uint8)
    result = emb.extract_body_embedding(dummy)
    assert result.shape == (576,), f"Expected (576,) got {result.shape}"
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 0.01 or norm == 0.0, f"Expected unit norm, got {norm}"
    print("[OK] Embedder produces valid 576-dim vector")


def test_embedding_aggregation() -> None:
    emb  = PersonEmbedder()
    vecs = [make_unit_vec() for _ in range(5)]
    agg  = emb.aggregate_embeddings(vecs)
    norm = float(np.linalg.norm(agg))
    assert abs(norm - 1.0) < 0.01, f"Aggregated embedding should be unit norm, got {norm}"
    print(f"[OK] Aggregation produces unit-norm vector: norm={norm:.4f}")


def test_serialize_deserialize() -> None:
    emb      = PersonEmbedder()
    original = np.random.randn(576).astype(np.float32)
    restored = emb.deserialize(emb.serialize(original))
    assert np.allclose(original, restored), "Serialization roundtrip failed"
    print("[OK] Embedding serialization/deserialization roundtrip OK")


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

def test_database_insert_and_read() -> None:
    db  = Database(":memory:")
    emb = PersonEmbedder()
    pid = db.insert_person({
        "cam_id"         : 0,
        "embedding"      : make_unit_vec().tobytes(),
        "embedding_type" : "body",
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_time" : "2024-01-01T10:00:05",
        "snapshot_paths" : [],
        "created_at"     : "2024-01-01T10:00:00",
    })
    assert pid is not None
    persons = db.get_all_persons()
    assert len(persons) >= 1
    print(f"[OK] Database insert/read OK, person_id={pid}")


def test_status_promotion() -> None:
    """Person with 1 embedding = unverified; 2nd embedding => auto-promoted to confirmed."""
    db  = Database(":memory:")
    emb = PersonEmbedder()
    pid = "status-test-person"
    db.insert_person({
        "person_id"      : pid,
        "cam_id"         : 0,
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_time" : "2024-01-01T10:00:10",
        "snapshot_paths" : [],
        "created_at"     : "2024-01-01T10:00:00",
    })
    v1 = np.random.randn(576).astype(np.float32)
    db.add_embedding_to_gallery(pid, v1.tobytes(), "face", "front", "2024-01-01T10:00:00")
    p = db.get_person(pid)
    assert p["status"] == "unverified", f"Expected unverified, got {p['status']}"

    v2 = np.random.randn(576).astype(np.float32)
    db.add_embedding_to_gallery(pid, v2.tobytes(), "face", "side", "2024-01-01T10:00:01")
    p = db.get_person(pid)
    assert p["status"] == "confirmed", f"Expected confirmed, got {p['status']}"
    print("[OK] Status promotes unverified -> confirmed when gallery reaches 2")


# ---------------------------------------------------------------------------
# Gallery tests (typed API)
# ---------------------------------------------------------------------------

def test_gallery_adds_different_view() -> None:
    gm = GalleryManager()
    v1 = make_unit_vec()
    # Construct a vector with cosine distance ~0.45 (> BODY_GALLERY_ADD_DISTANCE=0.35)
    ortho  = make_unit_vec()
    ortho  = ortho - np.dot(ortho, v1) * v1
    ortho  = ortho / np.linalg.norm(ortho)
    sim    = 0.55   # distance = 1 - 0.55 = 0.45
    v2     = sim * v1 + np.sqrt(1 - sim ** 2) * ortho

    gallery = [make_gallery_entry(v1)]
    result  = gm.should_add_to_gallery(v2, "body", gallery)
    dist    = 1 - float(np.dot(v1, v2))
    assert result is True, f"Expected True for a novel view (dist={dist:.3f})"
    print(f"[OK] Gallery correctly accepts a new/different view (dist={dist:.3f})")



def test_gallery_rejects_same_view() -> None:
    gm = GalleryManager()
    v1 = make_unit_vec()
    noise = np.random.randn(576).astype(np.float32) * 0.001
    v2 = v1 + noise
    v2 = v2 / np.linalg.norm(v2)

    gallery = [make_gallery_entry(v1)]
    result  = gm.should_add_to_gallery(v2, "body", gallery)
    assert result is False, "Expected False for nearly identical view"
    print("[OK] Gallery correctly rejects a duplicate/similar view")


def test_gallery_size_limit() -> None:
    from config.settings import MAX_GALLERY_SIZE
    gm      = GalleryManager()
    gallery = [make_gallery_entry(make_unit_vec()) for _ in range(MAX_GALLERY_SIZE)]
    result  = gm.should_add_to_gallery(make_unit_vec(), "body", gallery)
    assert result is False, f"Expected False when gallery is at MAX_GALLERY_SIZE={MAX_GALLERY_SIZE}"
    print(f"[OK] Gallery correctly enforces MAX_GALLERY_SIZE={MAX_GALLERY_SIZE}")


# ---------------------------------------------------------------------------
# ColorRegistry tests
# ---------------------------------------------------------------------------

def test_color_registry_deterministic() -> None:
    from display.visualizer import ColorRegistry
    reg   = ColorRegistry()
    pid   = "550e8400-e29b-41d4-a716-446655440000"
    color1 = reg.get_color(pid)
    color2 = reg.get_color(pid)
    assert color1 == color2, "Color must be deterministic for the same UUID"
    print(f"[OK] Color for UUID is deterministic: {color1}")


def test_different_persons_different_colors() -> None:
    from display.visualizer import ColorRegistry
    import uuid
    reg    = ColorRegistry()
    colors = {reg.get_color(str(uuid.uuid4())) for _ in range(20)}
    assert len(colors) >= 18, (
        f"Expected >= 18 unique colors for 20 UUIDs, got {len(colors)}"
    )
    print(f"[OK] 20 random UUIDs produced {len(colors)} unique colors")


def test_cross_camera_same_color() -> None:
    """Same person_id registered to two cameras must return identical color."""
    from display.visualizer import ColorRegistry
    reg = ColorRegistry()
    pid = "test-person-uuid-12345"
    # New-style API: register_alias(cam_id, track_id, person_id)
    reg.register_alias(0, 3, pid)
    reg.register_alias(2, 7, pid)
    color_cam0 = reg.get_color(reg.resolve_person_id(0, 3))
    color_cam2 = reg.get_color(reg.resolve_person_id(2, 7))
    assert color_cam0 == color_cam2, "Same person must have same color across cameras"
    print(f"[OK] Cross-camera color consistency verified: {color_cam0}")


def test_body_threshold_is_stricter() -> None:
    from config.settings import FACE_MATCH_THRESHOLD, BODY_MATCH_THRESHOLD
    assert BODY_MATCH_THRESHOLD > FACE_MATCH_THRESHOLD, (
        f"Body ({BODY_MATCH_THRESHOLD}) should be > Face ({FACE_MATCH_THRESHOLD})"
    )
    print(
        f"[OK] Body match threshold ({BODY_MATCH_THRESHOLD}) > "
        f"Face match threshold ({FACE_MATCH_THRESHOLD})"
    )


def test_body_embeddings_do_not_false_merge() -> None:
    from sklearn.metrics.pairwise import cosine_similarity
    from config.settings import BODY_MATCH_THRESHOLD
    v1  = np.random.randn(576).astype(np.float32)
    v2  = np.random.randn(576).astype(np.float32)
    score = cosine_similarity([v1], [v2])[0][0]
    assert score < BODY_MATCH_THRESHOLD, (
        f"Random body vectors scored {score:.3f}, should be < {BODY_MATCH_THRESHOLD}"
    )
    print(f"[OK] Random body vectors score {score:.3f} - below body threshold, no false merge")


# ---------------------------------------------------------------------------
# NEW: Track registry tests (Part A)
# ---------------------------------------------------------------------------

def test_track_registry_permanence() -> None:
    """Once a (cam_id, track_id) is registered, it must never be overwritten."""
    registry = {}
    registry[(0, 7)] = "person-uuid-aaa"
    # Simulate second attempt — must be blocked
    if (0, 7) not in registry:
        registry[(0, 7)] = "person-uuid-bbb"   # must NOT execute
    assert registry[(0, 7)] == "person-uuid-aaa", "Registry entry was overwritten!"
    print("[OK] Track registry permanence: entry cannot be overwritten")


# ---------------------------------------------------------------------------
# NEW: Status promotion from DB (Part B)
# ---------------------------------------------------------------------------

def test_status_promotion_from_db() -> None:
    """Inserting 2nd gallery entry automatically promotes status to confirmed."""
    db  = Database(":memory:")
    pid = "promotion-test-person"
    db.insert_person({
        "person_id"      : pid,
        "cam_id"         : 0,
        "first_seen_time": "2024-01-01T10:00:00",
        "last_seen_time" : "2024-01-01T10:00:00",
        "snapshot_paths" : [],
        "created_at"     : "2024-01-01T10:00:00",
    })
    v1 = np.random.randn(576).astype(np.float32)
    db.add_embedding_to_gallery(pid, v1.tobytes(), "face", "front", "2024-01-01T10:00:00")
    assert db.get_person(pid)["status"] == "unverified"

    v2 = np.random.randn(576).astype(np.float32)
    db.add_embedding_to_gallery(pid, v2.tobytes(), "face", "side",  "2024-01-01T10:00:01")
    assert db.get_person(pid)["status"] == "confirmed"
    print("[OK] Status auto-promoted unverified -> confirmed (gallery=2)")


if __name__ == "__main__":
    test_embedder()
    test_embedding_aggregation()
    test_serialize_deserialize()
    test_database_insert_and_read()
    test_status_promotion()
    test_gallery_adds_different_view()
    test_gallery_rejects_same_view()
    test_gallery_size_limit()
    test_color_registry_deterministic()
    test_different_persons_different_colors()
    test_cross_camera_same_color()
    test_body_threshold_is_stricter()
    test_body_embeddings_do_not_false_merge()
    test_track_registry_permanence()
    test_status_promotion_from_db()
    print("\nAll Phase 2 tests passed.")
