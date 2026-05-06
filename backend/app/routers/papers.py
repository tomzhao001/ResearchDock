from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import User
from app.schemas import (
    JobPublic,
    PaperDetailResponse,
    PaperListItem,
    PaperListResponse,
    UploadAcceptedResponse,
)
from app.services.papers import create_upload_artifacts, get_paper_detail, list_papers
from app.tasks.paper_ingest import process_uploaded_pdf

router = APIRouter(prefix="/api/papers", tags=["papers"])

ALLOWED_PDF_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
}


@router.get("", response_model=PaperListResponse)
def get_papers(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    items = [PaperListItem.model_validate(item) for item in list_papers(db)]
    return PaperListResponse(items=items)


@router.get("/{paper_id}", response_model=PaperDetailResponse)
def get_paper(
    paper_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    detail = get_paper_detail(db, paper_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found")

    metadata = detail.asset.metadata_json if detail.asset else None
    original_filename = None
    extraction_metadata = None
    structured_summary = None
    if isinstance(metadata, dict):
        original_filename = metadata.get("original_filename")
        extraction_metadata = metadata.get("extraction")
        structured_summary = metadata.get("structured_summary")

    latest_job = JobPublic.model_validate(detail.latest_job) if detail.latest_job else None
    return PaperDetailResponse(
        id=detail.paper.id,
        title=detail.paper.title,
        authors=detail.paper.authors,
        abstract_raw=detail.paper.abstract_raw,
        source_url=detail.paper.source_url,
        pdf_url=detail.paper.pdf_url,
        doi=detail.paper.doi,
        published_at=detail.paper.published_at,
        status=detail.paper.status,
        created_at=detail.paper.created_at,
        updated_at=detail.paper.updated_at,
        original_filename=original_filename if isinstance(original_filename, str) else None,
        preview_text=detail.asset.raw_text if detail.asset else None,
        extraction_metadata=extraction_metadata if isinstance(extraction_metadata, dict) else None,
        structured_summary=structured_summary if isinstance(structured_summary, dict) else None,
        latest_job=latest_job,
    )


@router.post("/upload", response_model=UploadAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_paper(
    file: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    filename = file.filename or "upload.pdf"
    mime_type = file.content_type or ""
    if mime_type not in ALLOWED_PDF_MIME_TYPES and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported")

    payload = await file.read()
    await file.close()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    artifacts = create_upload_artifacts(
        db,
        filename=filename,
        content_type=mime_type or "application/pdf",
        payload=payload,
    )
    process_uploaded_pdf.delay(artifacts.job_id)
    return UploadAcceptedResponse(
        paper_id=artifacts.paper_id,
        job_id=artifacts.job_id,
        filename=artifacts.filename,
        status="queued",
    )
