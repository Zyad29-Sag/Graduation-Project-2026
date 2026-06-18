"""
webapp/api/auth/router.py
-------------------------
Auth endpoints: login (OAuth2 password form so Swagger's Authorize button works
out of the box) and a /me echo. `username` carries the email.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from . import security, store
from .deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = store.get_user_by_email(form.username)
    if not user or not security.verify_password(form.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    token = security.create_access_token(user["email"], user["role"], user["tenant_id"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "email": user["email"],
            "role": user["role"],
            "tenant_id": user["tenant_id"],
        },
    }


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return {
        "email": user["email"],
        "role": user["role"],
        "tenant_id": user["tenant_id"],
    }
