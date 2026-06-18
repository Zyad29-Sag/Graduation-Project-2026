"""
webapp/api/routers/stats.py
---------------------------
Dashboard overview stats + the violence alerts feed + pending merge proposals.
"""

import json
from typing import List

from fastapi import APIRouter, Depends

from .. import config, services
from ..auth.deps import TenantCtx, get_tenant_ctx

router = APIRouter(tags=["dashboard"])


def _read_alerts() -> List[dict]:
    try:
        data = json.loads(config.DEMO_ALERTS_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


@router.get("/stats")
def stats(ctx: TenantCtx = Depends(get_tenant_ctx)):
    alerts = _read_alerts()
    return services.build_stats(ctx.db, alert_count=len(alerts))


@router.get("/alerts")
def alerts(limit: int = 100, ctx: TenantCtx = Depends(get_tenant_ctx)):
    """Violence/anomaly alerts (most recent first) from the engine's log."""
    items = _read_alerts()
    items = list(reversed(items))[:limit]
    return {"total": len(items), "items": items}


@router.get("/merges/pending")
def pending_merges(ctx: TenantCtx = Depends(get_tenant_ctx)):
    """Reconciliation merge proposals awaiting human review."""
    return {"items": ctx.db.get_pending_merges()}
