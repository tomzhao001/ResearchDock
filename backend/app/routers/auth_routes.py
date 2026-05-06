from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import create_access_token, verify_password
from app.config import settings
from app.database import get_db
from app.deps import get_current_user
from app.models import User
from app.schemas import LoginRequest, MessageResponse, UserPublic

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login")
def login(
    body: LoginRequest,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    user = db.scalar(select(User).where(User.username == body.username))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    token = create_access_token(user_id=user.id, username=user.username)
    max_age = 3600 * settings.jwt_expire_hours
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        httponly=True,
        max_age=max_age,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )
    return UserPublic(id=user.id, username=user.username)


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key=settings.cookie_name, path="/")
    return MessageResponse(message="logged out")


@router.get("/me", response_model=UserPublic)
def me(user: Annotated[User, Depends(get_current_user)]):
    return UserPublic(id=user.id, username=user.username)
