"""
modules/face/face_analyzer.py
-----------------------------
InsightFace-based face analysis (Part 11), merged & consolidated from the team
branch (their PersonEmbedder face methods).

Responsibilities (ALL additive — none affect body identity):
  - detect the most prominent face in a person crop;
  - 512-d face embedding (for storage in the ISOLATED face_embeddings table and
    for Phase-3 face-image search);
  - age range + gender (from InsightFace);
  - named watchlist match (KNOWN_FACES_DIR → person name);
  - "returning face" badge (FACESFROMVID_DIR chip gallery) — label only;
  - composes the optional GlassesDetector + EthnicityClassifier so main.py makes
    a single analyze() call.

Graceful-disable: if `insightface` is not installed or the model fails to load,
`ready` is False and analyze() returns empty attributes + no embedding. The
caller treats that as "no face info this track" and continues normally.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from config.settings import (
    INSIGHTFACE_MODEL,
    FACE_DET_SIZE,
    KNOWN_FACES_DIR,
    FACESFROMVID_DIR,
    KNOWN_FACE_TOLERANCE,
    RETURNING_FACE_TOLERANCE,
)

_SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp"}


def _age_to_range(age: float) -> str:
    """Map a numeric age estimate to a human-readable range string."""
    age = int(age)
    if age <= 3:  return "0-3"
    if age <= 6:  return "4-6"
    if age <= 10: return "7-10"
    if age <= 19: return "11-19"
    if age <= 30: return "20-30"
    if age <= 45: return "30-45"
    if age <= 55: return "45-55"
    if age <= 70: return "55-70"
    return "70+"


def _largest(faces):
    """Return the face with the largest bbox area, or None."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _norm(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return (vec / n) if n > 0 else vec


class FaceAnalyzer:
    """InsightFace wrapper + named/returning galleries + glasses/ethnicity."""

    def __init__(self, glasses_detector=None, ethnicity_classifier=None) -> None:
        self.face_app = None
        self.glasses  = glasses_detector
        self.ethnicity = ethnicity_classifier
        self.known_tol     = float(KNOWN_FACE_TOLERANCE)
        self.returning_tol = float(RETURNING_FACE_TOLERANCE)

        # Named watchlist (KNOWN_FACES_DIR) and returning chips (FACESFROMVID_DIR)
        self.known_embeddings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.returning_embeddings: List[np.ndarray] = []

        self._init_insightface()
        if self.face_app is not None:
            self._load_named_faces()
            self._load_returning_faces()

        print(f"[FACE] {self!r}")

    def __repr__(self) -> str:
        status = "ready" if self.ready else "disabled"
        g = "on" if (self.glasses and self.glasses.is_ready) else "off"
        e = "on" if (self.ethnicity and self.ethnicity.is_ready) else "off"
        return (f"FaceAnalyzer({status}, named={len(self.known_names)}, "
                f"returning={len(self.returning_embeddings)}, glasses={g}, ethnicity={e})")

    @property
    def ready(self) -> bool:
        return self.face_app is not None

    # ── Initialization ─────────────────────────────────────────────────────

    def _init_insightface(self) -> None:
        try:
            import torch
            from insightface.app import FaceAnalysis
            self.face_app = FaceAnalysis(name=INSIGHTFACE_MODEL)
            ctx = 0 if torch.cuda.is_available() else -1
            self.face_app.prepare(ctx_id=ctx, det_size=tuple(FACE_DET_SIZE))
            print(f"[FACE] InsightFace ({INSIGHTFACE_MODEL}) loaded on "
                  f"{'GPU' if ctx == 0 else 'CPU'}.")
        except Exception as exc:
            print(f"[FACE] InsightFace not available ({exc}). "
                  "Face analysis DISABLED (system runs normally).")
            self.face_app = None

    def _load_named_faces(self) -> None:
        d = Path(KNOWN_FACES_DIR)
        d.mkdir(parents=True, exist_ok=True)
        loaded = 0
        for img_path in sorted(d.iterdir()):
            if img_path.suffix.lower() not in _SUPPORTED_IMG:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            face = _largest(self._safe_get(img))
            if face is None or getattr(face, "embedding", None) is None:
                continue
            self.known_embeddings.append(_norm(np.array(face.embedding, dtype=np.float32)))
            self.known_names.append(img_path.stem.replace("_", " ").title())
            loaded += 1
        print(f"[FACE] {loaded} named watchlist face(s) loaded from {d}.")

    def _load_returning_faces(self) -> None:
        d = Path(FACESFROMVID_DIR)
        if not d.exists():
            return
        loaded = 0
        for img_path in sorted(d.iterdir()):
            if img_path.suffix.lower() not in _SUPPORTED_IMG:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            face = _largest(self._safe_get(img))
            if face is None or getattr(face, "embedding", None) is None:
                continue
            emb = _norm(np.array(face.embedding, dtype=np.float32))
            self.returning_embeddings.append(emb)
            loaded += 1
        print(f"[FACE] {loaded} returning-face chip(s) loaded from {d}.")

    # ── Core helpers ────────────────────────────────────────────────────────

    def _safe_get(self, img):
        try:
            return self.face_app.get(img)
        except Exception:
            return []

    def _match(self, emb: np.ndarray) -> Tuple[Optional[str], bool]:
        """Return (name_or_None, returning_bool) for a normalized face embedding."""
        name = None
        returning = False
        if self.known_embeddings:
            sims = [float(np.dot(emb, k)) for k in self.known_embeddings]
            bi = int(np.argmax(sims))
            if sims[bi] >= self.known_tol:
                name = self.known_names[bi]
        if self.returning_embeddings:
            sims = [float(np.dot(emb, k)) for k in self.returning_embeddings]
            if sims and max(sims) >= self.returning_tol:
                returning = True
        if name is not None:
            returning = True   # a named (watchlisted) person is by definition returning
        return name, returning

    # ── Public API ────────────────────────────────────────────────────────

    def extract_face_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Return a normalized 512-d face embedding from a crop, or None.

        Used both for live storage and for Phase-3 face-image search queries.
        """
        if not self.ready or crop is None or crop.size == 0:
            return None
        face = _largest(self._safe_get(crop))
        if face is None or getattr(face, "embedding", None) is None:
            return None
        return _norm(np.array(face.embedding, dtype=np.float32))

    def analyze(self, buf: List[np.ndarray]) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Analyze a buffer of person crops for the most prominent face.

        Returns (attributes, face_embedding):
          attributes = {age_range, gender, name, returning, ethnicity, glasses}
          face_embedding = normalized 512-d vector for the best face, or None.

        Never raises; on any failure returns empty attributes + None so the
        embedding worker is never disrupted.
        """
        attrs: Dict[str, Any] = {
            "age_range": None, "gender": None, "name": None,
            "returning": False, "ethnicity": None, "glasses": None,
        }
        if not self.ready or not buf:
            return attrs, None

        best = None  # (area, src_crop, face, (x1,y1,x2,y2)_in_orig)
        glasses_votes: List[str] = []

        for crop in buf:
            if crop is None or crop.size == 0:
                continue
            # Upsample 2× before detection — faces are small in body crops.
            try:
                crop_up = cv2.resize(crop, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            except Exception:
                continue
            faces = self._safe_get(crop_up)
            for f in faces:
                x1 = max(0, int(f.bbox[0] / 2)); y1 = max(0, int(f.bbox[1] / 2))
                x2 = min(crop.shape[1], int(f.bbox[2] / 2)); y2 = min(crop.shape[0], int(f.bbox[3] / 2))
                area = (x2 - x1) * (y2 - y1)
                if area <= 0:
                    continue
                if best is None or area > best[0]:
                    best = (area, crop, f, (x1, y1, x2, y2))
                # Glasses vote on the face chip (majority across the buffer)
                if self.glasses is not None and self.glasses.is_ready:
                    label = self.glasses.predict(crop[y1:y2, x1:x2])
                    if label is not None:
                        glasses_votes.append(label)

        if glasses_votes:
            attrs["glasses"] = max(set(glasses_votes), key=glasses_votes.count)

        if best is None:
            return attrs, None

        _, src_crop, face, (fx1, fy1, fx2, fy2) = best

        if getattr(face, "age", None) is not None:
            try:
                attrs["age_range"] = _age_to_range(float(face.age))
            except (TypeError, ValueError):
                pass
        g = getattr(face, "gender", None)
        if g is not None:
            attrs["gender"] = "Male" if g == 1 else "Female"

        emb = None
        if getattr(face, "embedding", None) is not None:
            emb = _norm(np.array(face.embedding, dtype=np.float32))
            attrs["name"], attrs["returning"] = self._match(emb)

        if self.ethnicity is not None and self.ethnicity.is_ready:
            face_region = src_crop[fy1:fy2, fx1:fx2]
            if face_region.size > 0:
                attrs["ethnicity"] = self.ethnicity.predict(face_region)

        return attrs, emb
