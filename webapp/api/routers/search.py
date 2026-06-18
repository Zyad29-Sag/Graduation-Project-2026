"""
webapp/api/routers/search.py
----------------------------
The three search methods the client asked for:
  POST /search/text     — chatbot / natural-language (semantic, MiniLM)
  POST /search/filters  — structured DB filters (Male / age / glasses / ...)
  POST /search/image    — search by an uploaded image (body Re-ID or face)

Heavy models (OSNet, InsightFace, FAISS) are lazy-loaded the first time the
relevant endpoint is hit (see engine.py).
"""

import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from .. import engine, services
from ..auth.deps import TenantCtx, get_tenant_ctx
from ..schemas import FilterSearchRequest, TextSearchRequest

router = APIRouter(prefix="/search", tags=["search"])

# Appearance fields routed to the description-attribute matcher.
_APPEARANCE_FIELDS = (
    "clothing_top", "clothing_bottom", "clothing_top_color",
    "clothing_bottom_color", "hair_color", "headwear", "body_build", "accessories",
)


@router.post("/text")
def search_text(req: TextSearchRequest, ctx: TenantCtx = Depends(get_tenant_ctx)):
    """Chatbot / natural-language search over LLM body descriptions."""
    tse = engine.get_text_search_engine(ctx.db)
    results = tse.search(req.query, top_k=req.top_k)
    hits = [
        services.search_hit(ctx.db, ctx.snapshots_dir, r["person_id"],
                            score=r.get("score"), summary=r.get("summary"))
        for r in results
    ]
    return {
        "query": req.query,
        "count": len(hits),
        "results": hits,
        "note": None if results else (
            "No descriptions indexed yet — run `python main.py --phase 4 "
            "--describe-all` (Ollama qwen2.5vl:3b), then re-seed."
        ),
    }


@router.post("/filters")
def search_filters(req: FilterSearchRequest, ctx: TenantCtx = Depends(get_tenant_ctx)):
    """Structured filters: face attributes (persons row) AND/OR appearance
    (description attributes), intersected."""
    persons = ctx.db.get_all_persons()

    def keep(p) -> bool:
        if req.status and p.get("status") != req.status:
            return False
        if req.gender and (p.get("gender") or "").lower() != req.gender.lower():
            return False
        if req.age_range and p.get("age_range") != req.age_range:
            return False
        if req.ethnicity and (p.get("ethnicity") or "").lower() != req.ethnicity.lower():
            return False
        if req.glasses and (p.get("glasses") or "").lower() != req.glasses.lower():
            return False
        if req.name and req.name.lower() not in (p.get("name") or "").lower():
            return False
        if req.camera is not None and req.camera not in ctx.db.get_cameras_for_person(p["person_id"]):
            return False
        return True

    base = [p for p in persons if keep(p)]
    result_ids = {p["person_id"] for p in base}

    appearance = {f: getattr(req, f) for f in _APPEARANCE_FIELDS if getattr(req, f)}
    if appearance:
        attr_rows = ctx.db.search_persons_by_attributes(appearance, limit=10_000)
        result_ids &= {r["person_id"] for r in attr_rows}

    hits = [
        services.search_hit(ctx.db, ctx.snapshots_dir, p["person_id"])
        for p in base if p["person_id"] in result_ids
    ][: req.top_k]
    return {"count": len(hits), "results": hits, "appearance_filters_used": bool(appearance)}


@router.post("/image")
async def search_image(
    file: UploadFile = File(...),
    mode: str = Form("body", description="body (OSNet Re-ID) or face (InsightFace)"),
    top_k: int = Form(5),
    ctx: TenantCtx = Depends(get_tenant_ctx),
):
    """Search by an uploaded image — body Re-ID or face."""
    if mode not in ("body", "face"):
        raise HTTPException(status_code=400, detail="mode must be 'body' or 'face'")

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
        try:
            if mode == "body":
                matches = engine.get_person_searcher(ctx.db_path).search_by_photo(
                    tmp_path, top_k=top_k
                )
            else:
                matches = engine.get_face_searcher(ctx.db_path).search_by_face_photo(
                    tmp_path, top_k=top_k
                )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Could not read the uploaded image")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    hits = [
        services.search_hit(ctx.db, ctx.snapshots_dir, m["person_id"],
                            score=m.get("similarity_score"))
        for m in matches
    ]
    note = None
    if mode == "face" and not matches:
        note = "No face match (no face detected in the query, or below threshold)."
    return {"mode": mode, "count": len(hits), "results": hits, "note": note}
