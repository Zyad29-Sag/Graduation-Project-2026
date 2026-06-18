"""
webapp/api/schemas.py
---------------------
Pydantic request models for endpoints that take a JSON body.
Responses are plain dicts shaped by services.py (kept flexible for the demo).
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TextSearchRequest(BaseModel):
    """Chatbot / natural-language semantic search."""
    query: str = Field(..., examples=["a man wearing glasses"])
    top_k: int = Field(10, ge=1, le=100)


class FilterSearchRequest(BaseModel):
    """Structured DB filters. Face-attribute fields (gender/age/ethnicity/
    glasses/status/camera/name) match the persons row and work today.
    Appearance fields match the LLM description attributes and need
    --describe-all to have been run."""
    # persons-row (face layer) filters
    gender: Optional[str] = None
    age_range: Optional[str] = None
    ethnicity: Optional[str] = None
    glasses: Optional[str] = None
    status: Optional[str] = None
    camera: Optional[int] = None
    name: Optional[str] = None
    # description-attribute (appearance) filters
    clothing_top: Optional[str] = None
    clothing_bottom: Optional[str] = None
    clothing_top_color: Optional[str] = None
    clothing_bottom_color: Optional[str] = None
    hair_color: Optional[str] = None
    headwear: Optional[str] = None
    body_build: Optional[str] = None
    accessories: Optional[List[str]] = None
    top_k: int = Field(50, ge=1, le=500)


# ── Corrections ─────────────────────────────────────────────────────────────
class MergeRequest(BaseModel):
    keep_id: str
    remove_id: str


class SplitRequest(BaseModel):
    """Peel the selected gallery embeddings (and optional camera-history rows)
    into a new person. ids come from GET /persons/{id} (gallery.entries[].id and
    journey.stops[].id)."""
    embedding_ids: List[int] = Field(default_factory=list)
    history_ids: List[int] = Field(default_factory=list)


class AttributesUpdate(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None
    ethnicity: Optional[str] = None
    glasses: Optional[str] = None
