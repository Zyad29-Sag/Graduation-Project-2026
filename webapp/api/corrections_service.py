"""
webapp/api/corrections_service.py
---------------------------------
Identity-mutating operations, factored out of routers/corrections.py so that
BOTH the REST routes and the chatbot assistant share one code path.

Every function here:
  - goes through the engine's invariant-safe Database methods (never raw SQL),
  - records the write in the audit_log,
  - and, when it mutates identity (merge/delete/split), invalidates the cached
    FAISS index so the next photo search rebuilds from SQLite (the source of truth).

These take a ``TenantCtx`` and plain args (not pydantic models) so non-HTTP
callers (the chatbot) can use them directly. They raise ``HTTPException`` on
bad input / not-found; FastAPI turns that into a response for REST callers, and
the chatbot router catches it to produce a friendly chat message.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from . import config, engine, services
from .auth import store
from .auth.deps import TenantCtx


def audit(ctx: TenantCtx, action: str, target_ids, detail=None) -> None:
    store.record_audit(ctx.user["email"], ctx.user["tenant_id"], action, target_ids, detail)


def _merge_snapshot_folders(snaps: Path, keep_id: str, remove_id: str) -> None:
    """Move remove_id's crops into keep_id's folder, then drop remove_id's."""
    src, dst = snaps / remove_id, snaps / keep_id
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.jpg"):
            target = dst / f"merged_{remove_id[:8]}_{f.name}"
            try:
                shutil.copy2(f, target)
            except OSError:
                pass
        shutil.rmtree(src, ignore_errors=True)


# ── Edit classification ─────────────────────────────────────────────────────
def edit_attributes(ctx: TenantCtx, person_id: str, fields: Dict[str, str]) -> dict:
    if ctx.db.get_person(person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    fields = {k: v for k, v in (fields or {}).items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    ctx.db.update_person_attributes(person_id, **fields)
    audit(ctx, "edit_attributes", [person_id], fields)
    return services.person_detail(ctx.db, ctx.snapshots_dir, person_id, config.overlap_groups())


# ── Re-describe (re-classify) ───────────────────────────────────────────────
def redescribe(ctx: TenantCtx, person_id: str) -> dict:
    if ctx.db.get_person(person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    ctx.db.enqueue_description(person_id)
    audit(ctx, "redescribe_enqueue", [person_id])
    return {
        "ok": True,
        "queued": person_id,
        "note": "Process with POST /corrections/run-descriptions (Ollama required).",
    }


# ── Merge ───────────────────────────────────────────────────────────────────
def merge(ctx: TenantCtx, keep_id: str, remove_id: str) -> dict:
    if keep_id == remove_id:
        raise HTTPException(status_code=400, detail="keep_id and remove_id must differ")
    if ctx.db.get_person(keep_id) is None or ctx.db.get_person(remove_id) is None:
        raise HTTPException(status_code=404, detail="keep_id or remove_id not found")
    moved = ctx.db.merge_persons(keep_id, remove_id)
    _merge_snapshot_folders(ctx.snapshots_dir, keep_id, remove_id)
    engine.invalidate_search_caches(ctx.db_path)
    audit(ctx, "merge", [keep_id, remove_id], {"moved_embeddings": moved})
    return {"ok": True, "keep_id": keep_id, "removed": remove_id, "moved_embeddings": moved}


# ── Split ───────────────────────────────────────────────────────────────────
def split(ctx: TenantCtx, person_id: str, embedding_ids: List[int],
          history_ids: Optional[List[int]] = None) -> dict:
    if ctx.db.get_person(person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    new_id = ctx.db.split_person(person_id, embedding_ids, history_ids or [])
    if new_id is None:
        raise HTTPException(
            status_code=400,
            detail="Nothing selected to split (need embedding_ids and/or history_ids)",
        )
    src, dst = ctx.snapshots_dir / person_id, ctx.snapshots_dir / new_id
    if src.is_dir() and not dst.exists():
        try:
            shutil.copytree(src, dst)
        except OSError:
            pass
    engine.invalidate_search_caches(ctx.db_path)
    audit(ctx, "split", [person_id, new_id],
          {"embedding_ids": embedding_ids, "history_ids": history_ids})
    return {"ok": True, "source": person_id, "new_person_id": new_id}


# ── Delete ──────────────────────────────────────────────────────────────────
def delete_person(ctx: TenantCtx, person_id: str) -> dict:
    if not ctx.db.delete_person(person_id):
        raise HTTPException(status_code=404, detail="Person not found")
    folder = ctx.snapshots_dir / person_id
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    engine.invalidate_search_caches(ctx.db_path)
    audit(ctx, "delete", [person_id])
    return {"ok": True, "deleted": person_id}


# ── Optional: process the description queue inline (needs Ollama) ────────────
def run_descriptions(ctx: TenantCtx, limit: int = 50) -> dict:
    """Drain pending description jobs using the configured VLM backend and write
    results straight into the served DB. Requires Ollama (qwen2.5vl:3b)."""
    import queue as _q

    try:
        from modules.llm.describer import build_describer
        from modules.llm.description_worker import DescriptionWorker
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"LLM modules unavailable: {exc}")

    try:
        describer = build_describer()
        worker = DescriptionWorker(describer, ctx.db, _q.Queue())
        worker.startup_recovery()
        processed = 0
        while processed < limit:
            claim = ctx.db.claim_next_description()
            if claim is None:
                break
            worker._handle(claim)
            processed += 1
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Describer failed (Ollama up?): {exc}")

    audit(ctx, "run_descriptions", [], {"processed": processed})
    return {"ok": True, "processed": processed}
