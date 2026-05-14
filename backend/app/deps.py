from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, WebSocket, status
from sqlalchemy.orm import Session

from app.auth import decode_access_token
from app.config import settings
from app.database import get_db
from app.models import Organization, User
from app.permissions import list_permissions


def get_token_from_cookie(request: Request) -> str | None:
    return request.cookies.get(settings.cookie_name)


def get_token_from_websocket(websocket: WebSocket) -> str | None:
    return websocket.cookies.get(settings.cookie_name)


@dataclass(frozen=True)
class AuthContext:
    user: User
    organization: Organization
    permissions: tuple[str, ...]

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


def resolve_current_user_context(db: Session, token: str | None) -> AuthContext:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    organization = db.get(Organization, user.organization_id)
    if organization is None or not organization.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Organization not found")
    permissions = tuple(list_permissions(user.role))
    return AuthContext(user=user, organization=organization, permissions=permissions)


def resolve_current_user(db: Session, token: str | None) -> User:
    return resolve_current_user_context(db, token).user


def get_current_user_context(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> AuthContext:
    return resolve_current_user_context(db, get_token_from_cookie(request))


def get_current_user(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    return resolve_current_user(db, get_token_from_cookie(request))


def require_permission(permission: str):
    def dependency(context: Annotated[AuthContext, Depends(get_current_user_context)]) -> AuthContext:
        if not context.has_permission(permission):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied")
        return context

    return dependency
