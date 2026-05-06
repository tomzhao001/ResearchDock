from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import User
from app.schemas import UploadAcceptedResponse
from app.services.papers import create_upload_artifacts
from app.tasks.paper_ingest import process_uploaded_pdf

router = APIRouter(prefix="/api/papers", tags=["papers"])

ALLOWED_PDF_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
}


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
