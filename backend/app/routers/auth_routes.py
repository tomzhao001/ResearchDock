from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import create_access_token, verify_password
from app.config import settings
from app.database import get_db
from app.deps import AuthContext, get_current_user_context
from app.models import Organization, User
from app.permissions import list_permissions, require_known_role
from app.schemas import LoginRequest, MessageResponse, OrganizationPublic, UserSessionPublic

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _build_user_session_response(user: User, organization: Organization, *, permissions: list[str]) -> UserSessionPublic:
    return UserSessionPublic(
        id=user.id,
        username=user.username,
        role=require_known_role(user.role),
        permissions=permissions,
        organization=OrganizationPublic.model_validate(organization),
    )


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
    organization = db.get(Organization, user.organization_id)
    if organization is None or not organization.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Organization not found")
    token = create_access_token(
        user_id=user.id,
        username=user.username,
        organization_id=user.organization_id,
        role=require_known_role(user.role),
    )
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
    return _build_user_session_response(user, organization, permissions=list_permissions(user.role))


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key=settings.cookie_name, path="/")
    return MessageResponse(message="logged out")


@router.get("/me", response_model=UserSessionPublic)
def me(context: Annotated[AuthContext, Depends(get_current_user_context)]):
    return _build_user_session_response(context.user, context.organization, permissions=list(context.permissions))
