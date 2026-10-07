from __future__ import annotations

import re

import bcrypt

from rag_cache.config import Settings
from rag_cache.db import Database

_USERNAME = re.compile(r"^[A-Za-z0-9._-]{3,40}$")


class AuthError(ValueError):
    pass


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def public_user(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "first_name": row["first_name"],
        "last_name": row["last_name"],
        "username": row["username"],
        "role": row["role"],
    }


def register_user(
    db: Database,
    *,
    first_name: str,
    last_name: str,
    username: str,
    password: str,
    confirm_password: str,
) -> dict:
    first_name = first_name.strip()
    last_name = last_name.strip()
    username = username.strip()
    if not first_name or not last_name:
        raise AuthError("First name and last name are required.")
    if len(first_name) > 80 or len(last_name) > 80:
        raise AuthError("Names must be 80 characters or fewer.")
    if not _USERNAME.match(username):
        raise AuthError("Username must be 3–40 characters and use letters, numbers, dots, underscores, or hyphens.")
    if password != confirm_password:
        raise AuthError("Passwords do not match.")
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    if len(password.encode("utf-8")) > 72:
        raise AuthError("Password must be 72 bytes or fewer.")
    if db.get_user_by_username(username):
        raise AuthError("That username is already registered.")
    return db.create_user(
        first_name=first_name,
        last_name=last_name,
        username=username,
        password_hash=hash_password(password),
        role="user",
    )


def authenticate(db: Database, username: str, password: str) -> dict | None:
    row = db.get_user_by_username(username.strip())
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return public_user(row)


def ensure_admin(db: Database, settings: Settings) -> None:
    if db.get_user_by_username(settings.admin_username):
        return
    password = settings.admin_password
    if not password or password == "change-me":
        return
    if len(password.encode("utf-8")) > 72:
        raise AuthError("ADMIN_PASSWORD must be 72 bytes or fewer.")
    db.create_user(
        first_name=settings.admin_first_name.strip() or "Admin",
        last_name=settings.admin_last_name.strip() or "User",
        username=settings.admin_username.strip(),
        password_hash=hash_password(password),
        role="admin",
    )
