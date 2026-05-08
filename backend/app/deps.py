from typing import Annotated

from fastapi import Depends, HTTPException, Request, WebSocket, status
from sqlalchemy.orm import Session

from app.auth import decode_access_token
from app.config import settings
from app.database import get_db
from app.models import User


def get_token_from_cookie(request: Request) -> str | None:
    return request.cookies.get(settings.cookie_name)


def get_token_from_websocket(websocket: WebSocket) -> str | None:
    return websocket.cookies.get(settings.cookie_name)


def resolve_current_user(db: Session, token: str | None) -> User:
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
    return user


def get_current_user(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    return resolve_current_user(db, get_token_from_cookie(request))
