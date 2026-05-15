from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import AuthContext, require_permission
from app.schemas import OrganizationQuestionItem, OrganizationQuestionSetResponse, OrganizationQuestionSetUpdateRequest
from app.services.org_settings import get_or_create_organization_settings, get_organization_question_items, update_organization_question_items

router = APIRouter(prefix="/api/org-settings", tags=["org-settings"])


@router.get("/questions", response_model=OrganizationQuestionSetResponse)
def get_question_set(
    db: Annotated[Session, Depends(get_db)],
    context: Annotated[AuthContext, Depends(require_permission("org_settings:read"))],
):
    settings = get_or_create_organization_settings(db, organization_id=context.organization.id)
    questions = [OrganizationQuestionItem.model_validate(item) for item in get_organization_question_items(db, organization_id=context.organization.id)]
    return OrganizationQuestionSetResponse(
        organization_id=context.organization.id,
        questions=questions,
        updated_at=settings.updated_at,
    )


@router.put("/questions", response_model=OrganizationQuestionSetResponse)
def put_question_set(
    payload: OrganizationQuestionSetUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    context: Annotated[AuthContext, Depends(require_permission("org_settings:write"))],
):
    try:
        settings = update_organization_question_items(
            db,
            organization_id=context.organization.id,
            questions=payload.questions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    questions = [OrganizationQuestionItem.model_validate(item) for item in settings.auto_extraction_questions_json or []]
    return OrganizationQuestionSetResponse(
        organization_id=context.organization.id,
        questions=questions,
        updated_at=settings.updated_at,
    )
