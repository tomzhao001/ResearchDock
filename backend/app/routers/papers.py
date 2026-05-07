from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import User
from app.schemas import (
    JobAcceptedResponse,
    JobPublic,
    MessageResponse,
    PaperDetailResponse,
    PaperListItem,
    PaperListResponse,
    PaperUpdateRequest,
    UploadAcceptedResponse,
)
from app.services.papers import (
    DuplicateFilenameError,
    create_upload_artifacts,
    delete_paper,
    enqueue_paper_ocr_rerun,
    enqueue_paper_summary_regeneration,
    get_original_filename,
    get_paper_detail,
    list_papers,
    update_paper_metadata,
)
from app.tasks.paper_ingest import process_paper_summary, process_uploaded_pdf

router = APIRouter(prefix="/api/papers", tags=["papers"])

ALLOWED_PDF_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
}


def build_paper_detail_response(detail) -> PaperDetailResponse:
    metadata = detail.asset.metadata_json if detail.asset else None
    extraction_metadata = None
    structured_summary = None
    if isinstance(metadata, dict):
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
        original_filename=get_original_filename(detail.asset),
        preview_text=detail.asset.raw_text if detail.asset else None,
        extraction_metadata=extraction_metadata if isinstance(extraction_metadata, dict) else None,
        structured_summary=structured_summary if isinstance(structured_summary, dict) else None,
        latest_job=latest_job,
    )


@router.get("", response_model=PaperListResponse)
def get_papers(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    items = []
    for item in list_papers(db):
        detail = get_paper_detail(db, item.id)
        original_filename = get_original_filename(detail.asset) if detail else None
        items.append(
            PaperListItem(
                id=item.id,
                title=item.title,
                original_filename=original_filename,
                abstract_raw=item.abstract_raw,
                status=item.status,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
        )
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
    return build_paper_detail_response(detail)


@router.patch("/{paper_id}", response_model=PaperDetailResponse)
def patch_paper(
    paper_id: int,
    payload: PaperUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    try:
        detail = update_paper_metadata(db, paper_id, payload.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found")
    return build_paper_detail_response(detail)


@router.delete("/{paper_id}", response_model=MessageResponse)
def delete_paper_item(
    paper_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    try:
        deleted = delete_paper(db, paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found")
    return MessageResponse(message="Paper deleted")


@router.post("/{paper_id}/rerun-ocr", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def rerun_paper_ocr(
    paper_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    try:
        job = enqueue_paper_ocr_rerun(db, paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found")

    process_uploaded_pdf.delay(job.id)
    return JobAcceptedResponse(
        paper_id=paper_id,
        job_id=job.id,
        job_type=job.job_type or "pdf_ingest",
        status=job.status or "queued",
    )


@router.post("/{paper_id}/regenerate-summary", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def regenerate_paper_summary(
    paper_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    try:
        job = enqueue_paper_summary_regeneration(db, paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found")

    process_paper_summary.delay(job.id)
    return JobAcceptedResponse(
        paper_id=paper_id,
        job_id=job.id,
        job_type=job.job_type or "paper_summary",
        status=job.status or "queued",
    )


@router.post("/upload", response_model=UploadAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_paper(
    file: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
    overwrite: Annotated[bool, Form()] = False,
):
    filename = file.filename or "upload.pdf"
    mime_type = file.content_type or ""
    if mime_type not in ALLOWED_PDF_MIME_TYPES and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported")

    payload = await file.read()
    await file.close()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    try:
        artifacts = create_upload_artifacts(
            db,
            filename=filename,
            content_type=mime_type or "application/pdf",
            payload=payload,
            overwrite=overwrite,
        )
    except DuplicateFilenameError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "已有相同文件名的文档，是否需要覆盖上传？",
                "existing_paper_id": exc.existing_paper_id,
                "filename": exc.filename,
            },
        ) from exc

    process_uploaded_pdf.delay(artifacts.job_id)
    return UploadAcceptedResponse(
        paper_id=artifacts.paper_id,
        job_id=artifacts.job_id,
        filename=artifacts.filename,
        status="queued",
    )
