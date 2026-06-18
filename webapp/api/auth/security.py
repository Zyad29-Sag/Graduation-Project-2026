"""
webapp/api/auth/security.py
---------------------------
Password hashing (bcrypt) + JWT creation/verification (PyJWT).

Kept dependency-light on purpose: bcrypt directly (no passlib) and PyJWT.
"""

import datetime

import bcrypt
import jwt

from .. import config


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(email: str, role: str, tenant_id: str) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": email,
        "role": role,
        "tenant_id": tenant_id,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=config.JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALG)


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError on invalid/expired tokens."""
    return jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALG])
