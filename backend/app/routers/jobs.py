from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import AuthContext, require_permission
from app.models import Job, Paper
from app.schemas import JobListResponse, JobPublic, MessageResponse
from app.services.papers import delete_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=JobListResponse)
def list_jobs(
    db: Annotated[Session, Depends(get_db)],
    context: Annotated[AuthContext, Depends(require_permission("jobs:read"))],
):
    items = db.scalars(
        select(Job)
        .join(Paper, Paper.id == Job.paper_id)
        .where(Paper.organization_id == context.organization.id, Paper.deleted_at.is_(None))
        .order_by(Job.id.desc())
        .limit(20)
    ).all()
    return JobListResponse(items=[JobPublic.model_validate(item) for item in items])


@router.get("/{job_id}", response_model=JobPublic)
def get_job(
    job_id: int,
    db: Annotated[Session, Depends(get_db)],
    context: Annotated[AuthContext, Depends(require_permission("jobs:read"))],
):
    job = db.scalar(
        select(Job)
        .join(Paper, Paper.id == Job.paper_id)
        .where(Job.id == job_id, Paper.organization_id == context.organization.id, Paper.deleted_at.is_(None))
    )
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobPublic.model_validate(job)


@router.delete("/{job_id}", response_model=MessageResponse)
def delete_job_item(
    job_id: int,
    db: Annotated[Session, Depends(get_db)],
    context: Annotated[AuthContext, Depends(require_permission("jobs:manage"))],
):
    try:
        deleted = delete_job(db, job_id, organization_id=context.organization.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return MessageResponse(message="Job deleted")
