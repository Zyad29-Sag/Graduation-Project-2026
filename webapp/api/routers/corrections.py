"""
webapp/api/routers/corrections.py
---------------------------------
Human-in-the-loop correction tools. The actual work lives in
``webapp/api/corrections_service.py`` so the chatbot assistant can perform the
same operations through one audited, cache-invalidating code path. These routes
are thin: role-gate, then delegate.
"""

from fastapi import APIRouter, Depends

from .. import corrections_service as svc
from ..auth import store
from ..auth.deps import TenantCtx, get_tenant_ctx, require_role
from ..schemas import AttributesUpdate, MergeRequest, SplitRequest

router = APIRouter(tags=["corrections"])
WRITE_ROLES = ("admin", "operator")


# ── Edit classification ─────────────────────────────────────────────────────
@router.patch("/persons/{person_id}/attributes")
def edit_attributes(
    person_id: str,
    body: AttributesUpdate,
    ctx: TenantCtx = Depends(get_tenant_ctx),
    _: dict = Depends(require_role(*WRITE_ROLES)),
):
    return svc.edit_attributes(ctx, person_id, body.model_dump(exclude_none=True))


# ── Re-describe (re-classify) ───────────────────────────────────────────────
@router.post("/persons/{person_id}/redescribe")
def redescribe(
    person_id: str,
    ctx: TenantCtx = Depends(get_tenant_ctx),
    _: dict = Depends(require_role(*WRITE_ROLES)),
):
    return svc.redescribe(ctx, person_id)


# ── Merge ───────────────────────────────────────────────────────────────────
@router.post("/corrections/merge")
def merge(
    body: MergeRequest,
    ctx: TenantCtx = Depends(get_tenant_ctx),
    _: dict = Depends(require_role(*WRITE_ROLES)),
):
    return svc.merge(ctx, body.keep_id, body.remove_id)


# ── Split ───────────────────────────────────────────────────────────────────
@router.post("/persons/{person_id}/split")
def split(
    person_id: str,
    body: SplitRequest,
    ctx: TenantCtx = Depends(get_tenant_ctx),
    _: dict = Depends(require_role(*WRITE_ROLES)),
):
    return svc.split(ctx, person_id, body.embedding_ids, body.history_ids)


# ── Delete ──────────────────────────────────────────────────────────────────
@router.delete("/persons/{person_id}")
def delete_person(
    person_id: str,
    ctx: TenantCtx = Depends(get_tenant_ctx),
    _: dict = Depends(require_role(*WRITE_ROLES)),
):
    return svc.delete_person(ctx, person_id)


# ── Optional: process the description queue inline (needs Ollama) ────────────
@router.post("/corrections/run-descriptions")
def run_descriptions(
    limit: int = 50,
    ctx: TenantCtx = Depends(get_tenant_ctx),
    _: dict = Depends(require_role(*WRITE_ROLES)),
):
    return svc.run_descriptions(ctx, limit)


# ── Audit log ───────────────────────────────────────────────────────────────
@router.get("/corrections/audit")
def audit_log(limit: int = 100, ctx: TenantCtx = Depends(get_tenant_ctx)):
    return {"items": store.list_audit(ctx.user["tenant_id"], limit=limit)}
