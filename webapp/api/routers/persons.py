"""
webapp/api/routers/persons.py
-----------------------------
The person database explorer: list (with filters + pagination), detail
(gallery meta + cross-camera journey + LLM description + face attributes),
and journey/snapshot views.

All reads go through the engine's `Database` via the tenant context.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import config, services
from ..auth.deps import TenantCtx, get_tenant_ctx

router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("")
def list_persons(
    ctx: TenantCtx = Depends(get_tenant_ctx),
    status: Optional[str] = Query(None),
    camera: Optional[int] = Query(None, description="cam_id seen on (any sighting)"),
    gender: Optional[str] = Query(None),
    age_range: Optional[str] = Query(None),
    ethnicity: Optional[str] = Query(None),
    glasses: Optional[str] = Query(None),
    has_description: Optional[bool] = Query(None),
    q: Optional[str] = Query(None, description="substring match on person_id or name"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    persons = ctx.db.get_all_persons()

    def keep(p) -> bool:
        if status and p.get("status") != status:
            return False
        if gender and (p.get("gender") or "").lower() != gender.lower():
            return False
        if age_range and p.get("age_range") != age_range:
            return False
        if ethnicity and (p.get("ethnicity") or "").lower() != ethnicity.lower():
            return False
        if glasses and (p.get("glasses") or "").lower() != glasses.lower():
            return False
        if has_description is not None and (p.get("latest_description_id") is not None) != has_description:
            return False
        if camera is not None:
            cams = ctx.db.get_cameras_for_person(p["person_id"])
            if camera not in cams:
                return False
        if q:
            ql = q.lower()
            if ql not in p["person_id"].lower() and ql not in (p.get("name") or "").lower():
                return False
        return True

    filtered = [p for p in persons if keep(p)]
    filtered.sort(key=lambda p: (p.get("last_seen_time") or ""), reverse=True)
    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [services.person_summary(ctx.snapshots_dir, p) for p in page],
    }


@router.get("/{person_id}")
def get_person(person_id: str, ctx: TenantCtx = Depends(get_tenant_ctx)):
    detail = services.person_detail(
        ctx.db, ctx.snapshots_dir, person_id, config.overlap_groups()
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return detail


@router.get("/{person_id}/journey")
def get_journey(person_id: str, ctx: TenantCtx = Depends(get_tenant_ctx)):
    if ctx.db.get_person(person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return services.shape_journey(ctx.db, person_id, config.overlap_groups())


@router.get("/{person_id}/snapshots")
def list_person_snapshots(person_id: str, ctx: TenantCtx = Depends(get_tenant_ctx)):
    files = services.list_snapshot_files(ctx.snapshots_dir, person_id)
    return {"person_id": person_id, "snapshots": services.snapshot_urls(person_id, files)}
