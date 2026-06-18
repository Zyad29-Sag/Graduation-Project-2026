"""
webapp/api/auth/deps.py
-----------------------
FastAPI dependencies: current user, role gate, and the tenant context.

`get_tenant_ctx` is the multi-tenant seam — it resolves the logged-in user's
tenant to a concrete engine `Database` + snapshots dir. Today there is one
tenant (the demo). Later, each customer is a tenant row with its own db_path,
and nothing else in the API changes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from ..engine import get_database
from . import security, store

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


_CREDS_EXC = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def user_from_token(token: str) -> Dict[str, Any]:
    """Validate a raw JWT and return the user. Used by streaming endpoints that
    receive the token via a query param (an <img> tag can't set headers)."""
    try:
        payload = security.decode_access_token(token)
    except jwt.PyJWTError:
        raise _CREDS_EXC
    email = payload.get("sub")
    if not email:
        raise _CREDS_EXC
    user = store.get_user_by_email(email)
    if not user:
        raise _CREDS_EXC
    return user


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    return user_from_token(token)


def require_role(*roles: str):
    """Dependency factory: allow only the listed roles (empty = any logged-in)."""

    def checker(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if roles and user["role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(roles)}",
            )
        return user

    return checker


@dataclass
class TenantCtx:
    user: Dict[str, Any]
    tenant: Dict[str, Any]
    db: Any                 # engine Database for this tenant
    db_path: str            # cache key for engine.invalidate_search_caches
    snapshots_dir: Path


def get_tenant_ctx(user: Dict[str, Any] = Depends(get_current_user)) -> TenantCtx:
    tenant = store.get_tenant(user["tenant_id"])
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tenant '{user['tenant_id']}' is not configured.",
        )
    return TenantCtx(
        user=user,
        tenant=tenant,
        db=get_database(tenant["db_path"]),
        db_path=tenant["db_path"],
        snapshots_dir=Path(tenant["snapshots_dir"]),
    )
