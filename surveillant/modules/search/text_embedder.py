"""
modules/search/text_embedder.py
--------------------------------
Shared sentence-transformer text encoder (Phase 4B — semantic search).

One lazily-loaded `all-MiniLM-L6-v2` model (22M params, CPU-fast) is reused by
BOTH sides of the pipeline:

  * the DescriptionWorker, to embed each person's ``long_description`` at
    describe time and store the vector alongside the row;
  * the TextSearchEngine, to embed the operator's free-text query and rank
    stored vectors by cosine similarity (== nearest meaning).

Replacing the old SQL ``LIKE`` keyword filter with embeddings means a search
for "a man in a black t-shirt" matches descriptions by MEANING, not exact
tokens — far more accurate and robust to wording.

Embeddings are L2-normalised float32, so cosine similarity is a plain dot
product. The dependency is optional and lazily imported; callers handle a
missing model gracefully (return ``None`` / empty results).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# all-MiniLM-L6-v2 output dimension.
EMBED_DIM = 384
_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None  # lazily loaded singleton


def _get_model():
    """Load (once) and return the SentenceTransformer model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # heavy, optional
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def encode(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised float32 vectors,
    shape ``(len(texts), EMBED_DIM)``. Raises if the dependency is missing —
    callers that must not fail should wrap this in try/except.
    """
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def embed_text(text: str) -> Optional[bytes]:
    """
    Embed a single string and return its raw float32 bytes (for DB storage),
    or ``None`` if the text is empty or the model is unavailable. Never raises.
    """
    if not text or not text.strip():
        return None
    try:
        return encode([text])[0].tobytes()
    except Exception as exc:  # noqa: BLE001 — embedding must never crash describe
        print(f"[EMBED] text embedding failed: {exc}")
        return None
