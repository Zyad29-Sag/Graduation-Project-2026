"""
webapp/api/chatbot/router.py
----------------------------
FastAPI router for the conversational assistant.

  POST /chat
    body: { session_id?: str, message: str }
    → { session_id, reply, results?, open_person_id?, proposed_action? }

  GET  /chat/sessions/{session_id}/messages
    → { messages: [...] }

Write actions (merge / delete / edit_attributes / redescribe) are PROPOSED by
the brain and returned as ``proposed_action``. They execute ONLY when the user
sends a follow-up that `brain.is_affirmative()` recognises AND a ``pending_action``
is stored in the session. Any logged-in user can read; writes require role
admin or operator (matching the corrections REST gate).
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth.deps import TenantCtx, get_tenant_ctx
from . import brain, store
from .. import corrections_service

router = APIRouter(prefix="/chat", tags=["chat"])

_WRITE_ROLES = ("admin", "operator")


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    results: Optional[Any] = None
    open_person_id: Optional[str] = None
    proposed_action: Optional[Dict[str, Any]] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _exec_action(ctx: TenantCtx, action: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a confirmed write action. Raises HTTPException on any problem."""
    t = action.get("type")
    args = action.get("args", {})
    if t == "merge":
        return corrections_service.merge(ctx, args["keep_id"], args["remove_id"])
    if t == "delete":
        return corrections_service.delete_person(ctx, args["person_id"])
    if t == "edit_attributes":
        return corrections_service.edit_attributes(ctx, args["person_id"], args.get("fields", {}))
    if t == "redescribe":
        return corrections_service.redescribe(ctx, args["person_id"])
    raise HTTPException(status_code=400, detail=f"Unknown action type: {t!r}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
def post_chat(req: ChatRequest, ctx: TenantCtx = Depends(get_tenant_ctx)):
    """Send a message to the assistant and get a reply."""
    # 1. Resolve / create session
    session_id = req.session_id
    if session_id:
        session = store.get_session(session_id, ctx.user["tenant_id"])
        if session is None:
            # stale / foreign — start fresh
            session_id = None
    if not session_id:
        session_id = store.create_session(ctx.user["tenant_id"], ctx.user["email"])
        session = store.get_session(session_id, ctx.user["tenant_id"])

    # Attach recent message history for Tier-2 context
    session["_recent_msgs"] = store.get_messages(session_id, limit=20)

    text = (req.message or "").strip()
    pending = session.get("pending_action")  # previously proposed write

    # 2. Confirmation of a pending write?
    if pending and brain.is_affirmative(text):
        role = ctx.user.get("role", "")
        if role not in _WRITE_ROLES:
            reply_text = (
                f"Sorry — write actions require the admin or operator role "
                f"(you are '{role}')."
            )
            result_payload = {
                "reply": reply_text,
                "results": None,
                "open_person_id": None,
                "proposed_action": None,
            }
        else:
            try:
                _exec_action(ctx, pending)
                summary = (pending.get("summary") or pending.get("type", "action")).title()
                reply_text = f"Done — {summary} completed successfully."
                result_payload = {
                    "reply": reply_text,
                    "results": None,
                    "open_person_id": None,
                    "proposed_action": None,
                }
            except HTTPException as exc:
                reply_text = f"Couldn't complete the action: {exc.detail}"
                result_payload = {
                    "reply": reply_text,
                    "results": None,
                    "open_person_id": None,
                    "proposed_action": None,
                }
        # Clear pending regardless
        store.update_session(session_id, pending_action=None)
        store.add_message(session_id, "user", text)
        store.add_message(session_id, "assistant", reply_text)
        return ChatResponse(session_id=session_id, **result_payload)

    # 3. Negation of a pending write
    if pending and brain.is_negative(text):
        store.update_session(session_id, pending_action=None)
        reply_text = "No problem — action cancelled."
        store.add_message(session_id, "user", text)
        store.add_message(session_id, "assistant", reply_text)
        return ChatResponse(session_id=session_id, reply=reply_text)

    # 4. Normal turn — route through the brain
    output = brain.respond(ctx, session, text)

    # 5. Persist state
    new_filters = output.get("active_filters")
    new_results = output.get("last_results")
    new_pending = output.get("pending_action")  # may be "__keep__" sentinel

    update_kwargs: Dict[str, Any] = {}
    if new_filters is not None:
        update_kwargs["active_filters"] = new_filters
    if new_results is not None:
        update_kwargs["last_results"] = new_results
    if new_pending != "__keep__":
        update_kwargs["pending_action"] = new_pending if new_pending else None

    if update_kwargs:
        store.update_session(session_id, **update_kwargs)

    reply_text = output.get("reply", "")
    store.add_message(session_id, "user", text)
    store.add_message(session_id, "assistant", reply_text)

    return ChatResponse(
        session_id=session_id,
        reply=reply_text,
        results=output.get("results"),
        open_person_id=output.get("open_person_id"),
        proposed_action=output.get("proposed_action"),
    )


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    ctx: TenantCtx = Depends(get_tenant_ctx),
):
    """Return the message history for a session (for page reloads)."""
    session = store.get_session(session_id, ctx.user["tenant_id"])
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": store.get_messages(session_id, limit=200)}
