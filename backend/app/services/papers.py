from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.models import Job, Paper, PaperAsset, PaperChunk, PaperDocumentBlock, PaperDocumentPage, PaperDocumentPicture, PaperDocumentTable
from app.services.document_extraction import DocumentExtractor, ExtractedDocument, ExtractedPicture
from app.services.llm import answer_question_set_questions, is_chat_llm_configured, summarize_paper_text
from app.services.org_settings import get_organization_question_items
from app.services.task_events import publish_task_status_event
from app.services.vision import PictureDescriptionAdapter, PictureDescriptionRequest


@dataclass
class UploadArtifacts:
    paper_id: int
    job_id: int
    filename: str


@dataclass
class PaperDetailData:
    paper: Paper
    asset: PaperAsset | None
    latest_job: Job | None
    latest_ocr_job: Job | None
    latest_summary_job: Job | None
    latest_question_set_job: Job | None


class DuplicateFilenameError(RuntimeError):
    def __init__(self, filename: str, existing_paper_id: int):
        super().__init__(f"Duplicate filename: {filename}")
        self.filename = filename
        self.existing_paper_id = existing_paper_id


class JobCancellationRequested(RuntimeError):
    """Signal cooperative task cancellation without marking the job as failed."""


ACTIVE_JOB_STATUSES = {"queued", "processing", "cancel_requested"}
DELETABLE_JOB_STATUSES = {"completed", "failed", "cancelled"}
DOI_PATTERN = re.compile(r"^10\.\S+/\S+$", re.IGNORECASE)


def normalize_filename(filename: str) -> str:
    return Path(filename or "upload.pdf").name.strip().casefold()


def _active_paper_asset_rows(db: Session, *, organization_id: int | None = None) -> list[tuple[Paper, PaperAsset]]:
    statement = (
        select(Paper, PaperAsset)
        .join(PaperAsset, PaperAsset.paper_id == Paper.id)
        .where(
            Paper.deleted_at.is_(None),
            PaperAsset.asset_type == "original_pdf",
        )
    )
    if organization_id is not None:
        statement = statement.where(Paper.organization_id == organization_id)
    rows = db.execute(statement).all()
    return [(paper, asset) for paper, asset in rows]


def _get_scoped_paper(db: Session, paper_id: int, *, organization_id: int | None = None) -> Paper | None:
    statement = select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None))
    if organization_id is not None:
        statement = statement.where(Paper.organization_id == organization_id)
    return db.scalar(statement)


def find_active_paper_by_original_filename(db: Session, filename: str, *, organization_id: int) -> Paper | None:
    normalized = normalize_filename(filename)
    for paper, asset in _active_paper_asset_rows(db, organization_id=organization_id):
        metadata = asset.metadata_json or {}
        original_filename = metadata.get("original_filename") if isinstance(metadata, dict) else None
        if isinstance(original_filename, str) and normalize_filename(original_filename) == normalized:
            return paper
    return None


def _get_original_pdf_asset(db: Session, paper_id: int) -> PaperAsset | None:
    return db.scalar(
        select(PaperAsset).where(
            PaperAsset.paper_id == paper_id,
            PaperAsset.asset_type == "original_pdf",
        )
    )


def _structure_sort_key(*, reading_order: int | None, page_number: int | None, fallback_group: int, fallback_index: int) -> tuple[int, int, int, int]:
    if reading_order is not None:
        return (0, int(reading_order), fallback_group, fallback_index)
    return (1, int(page_number or 0), fallback_group, fallback_index)


def _serialize_table_rows(data: object, *, max_rows: int = 6) -> str:
    if not isinstance(data, list):
        return ""
    lines: list[str] = []
    for row in data[:max_rows]:
        if not isinstance(row, dict):
            continue
        cells = [f"{str(key).strip()}: {str(value).strip()}" for key, value in row.items() if str(value).strip()]
        if cells:
            lines.append("; ".join(cells))
    return "\n".join(lines)


def _normalize_rendered_lines(lines: list[str]) -> str:
    compacted: list[str] = []
    for line in lines:
        normalized = str(line or "").strip()
        if not normalized:
            if compacted and compacted[-1] != "":
                compacted.append("")
            continue
        compacted.append(normalized)
    while compacted and compacted[-1] == "":
        compacted.pop()
    return "\n".join(compacted).strip()


def render_document_text(document: ExtractedDocument) -> str:
    from app.services.paper_pipeline.rendering import render_document_text as render_document_text_impl

    return render_document_text_impl(document)


def render_paper_text_from_structure(db: Session, paper_id: int) -> str:
    from app.services.paper_pipeline.rendering import render_paper_text_from_structure as render_paper_text_from_structure_impl

    return render_paper_text_from_structure_impl(db, paper_id)


def _get_latest_job_for_paper(db: Session, paper_id: int, job_type: str) -> Job | None:
    return db.scalar(
        select(Job)
        .where(
            Job.paper_id == paper_id,
            Job.job_type == job_type,
            Job.deleted_at.is_(None),
        )
        .order_by(Job.id.desc())
        .limit(1)
    )


def _get_active_job_for_paper(db: Session, paper_id: int) -> Job | None:
    return db.scalar(
        select(Job)
        .where(
            Job.paper_id == paper_id,
            Job.deleted_at.is_(None),
            Job.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(Job.id.desc())
        .limit(1)
    )


def _get_latest_visible_job(db: Session, paper_id: int) -> Job | None:
    return db.scalar(
        select(Job)
        .where(Job.paper_id == paper_id, Job.deleted_at.is_(None))
        .order_by(Job.id.desc())
        .limit(1)
    )


def _get_scoped_job(db: Session, job_id: int, *, organization_id: int | None = None) -> Job | None:
    if organization_id is None:
        job = db.get(Job, job_id)
        if job is None or job.deleted_at is not None:
            return None
        return job
    return db.scalar(
        select(Job)
        .join(Paper, Paper.id == Job.paper_id)
        .where(
            Job.id == job_id,
            Job.deleted_at.is_(None),
            Paper.organization_id == organization_id,
            Paper.deleted_at.is_(None),
        )
    )


def _set_paper_status_from_latest_job(db: Session, paper: Paper) -> None:
    latest_job = _get_latest_visible_job(db, paper.id)
    if latest_job is None:
        return
    if latest_job.status in ACTIVE_JOB_STATUSES:
        paper.status = latest_job.status
    elif latest_job.status == "failed":
        paper.status = "failed"
    elif latest_job.status == "cancelled":
        paper.status = "cancelled"
    elif latest_job.status == "completed":
        paper.status = "completed"


def set_job_celery_task_id(
    job_id: int,
    celery_task_id: str | None,
    *,
    db: Session | None = None,
    session_factory: Callable[[], Session] = SessionLocal,
) -> None:
    if not celery_task_id:
        return
    owns_session = db is None
    if db is None:
        db = session_factory()
    try:
        job = db.get(Job, job_id)
        if job is None or job.deleted_at is not None:
            return
        job.celery_task_id = celery_task_id
        db.commit()
    finally:
        if owns_session:
            db.close()


def _mark_job_cancelled(db: Session, job: Job, paper: Paper | None) -> None:
    now = datetime.now(timezone.utc)
    job.status = "cancelled"
    job.error_message = None
    job.finished_at = job.finished_at or now
    if paper is not None:
        paper.updated_at = now
        _set_paper_status_from_latest_job(db, paper)
    db.commit()
    if paper is not None:
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)


def _raise_if_cancel_requested(db: Session, job: Job, paper: Paper | None) -> None:
    from app.services.paper_pipeline.helpers import raise_if_cancel_requested

    return raise_if_cancel_requested(db, job, paper)


def _queue_summary_job_if_needed(db: Session, paper: Paper, asset: PaperAsset) -> int | None:
    from app.services.paper_pipeline.helpers import queue_summary_job_if_needed

    return queue_summary_job_if_needed(db, paper, asset)


def _extract_structured_summary_from_asset(asset: PaperAsset | None) -> dict | None:
    from app.services.paper_pipeline.helpers import extract_structured_summary_from_asset

    return extract_structured_summary_from_asset(asset)


def _queue_question_set_job_if_needed(db: Session, paper: Paper, asset: PaperAsset) -> int | None:
    from app.services.paper_pipeline.helpers import queue_question_set_job_if_needed

    return queue_question_set_job_if_needed(db, paper, asset)


def get_job_phase_status(job: Job | None) -> str | None:
    return job.status if job else None


def _normalize_summary_text(value: object) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    return normalized


def _normalize_summary_doi(value: object) -> str:
    normalized = _normalize_summary_text(value)
    if not normalized:
        return ""
    lowered = normalized.lower()
    if lowered.startswith("doi:"):
        normalized = normalized[4:].strip()
    elif lowered.startswith("https://doi.org/") or lowered.startswith("http://doi.org/"):
        parsed = urlparse(normalized)
        normalized = parsed.path.lstrip("/").strip()
    return normalized if DOI_PATTERN.match(normalized) else ""


def _normalize_summary_url(value: object) -> str:
    normalized = _normalize_summary_text(value)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return normalized


def _parse_summary_published_at(value: object) -> datetime | None:
    normalized = _normalize_summary_text(value)
    if not normalized:
        return None
    iso_value = f"{normalized[:-1]}+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _serialize_summary_published_at(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _paper_text_field_is_empty(value: str | None) -> bool:
    return value is None or not str(value).strip()


def create_upload_artifacts(
    db: Session,
    *,
    organization_id: int,
    filename: str,
    content_type: str,
    payload: bytes,
    overwrite: bool = False,
) -> UploadArtifacts:
    safe_name = Path(filename or "upload.pdf").name
    existing_paper = find_active_paper_by_original_filename(db, safe_name, organization_id=organization_id)
    if existing_paper is not None and not overwrite:
        raise DuplicateFilenameError(safe_name, existing_paper.id)

    if existing_paper is not None and overwrite:
        now = datetime.now(timezone.utc)
        existing_paper.deleted_at = now
        existing_paper.updated_at = now
        existing_paper.status = "deleted"

    content_hash = hashlib.sha256(payload).hexdigest()
    storage_dir = settings.file_storage_path / content_hash[:2]
    storage_dir.mkdir(parents=True, exist_ok=True)
    stored_path = storage_dir / f"{content_hash}.pdf"
    stored_path.write_bytes(payload)

    paper = Paper(
        organization_id=organization_id,
        title=Path(safe_name).stem or "Untitled PDF",
        content_hash=content_hash,
        ingest_type="upload",
        status="uploaded",
    )
    db.add(paper)
    db.flush()

    asset = PaperAsset(
        paper_id=paper.id,
        asset_type="original_pdf",
        storage_path=str(stored_path),
        mime_type=content_type,
        metadata_json={
            "original_filename": safe_name,
            "content_hash": content_hash,
            "size_bytes": len(payload),
        },
    )
    job = Job(
        job_type="pdf_ingest",
        paper_id=paper.id,
        status="queued",
    )
    db.add(asset)
    db.add(job)
    db.commit()
    db.refresh(job)
    publish_task_status_event(db, paper_id=paper.id, job_id=job.id)
    return UploadArtifacts(paper_id=paper.id, job_id=job.id, filename=safe_name)


def list_papers(db: Session, *, organization_id: int, limit: int = 50) -> list[Paper]:
    statement = (
        select(Paper)
        .where(Paper.deleted_at.is_(None), Paper.organization_id == organization_id)
        .order_by(Paper.updated_at.desc(), Paper.id.desc())
        .limit(limit)
    )
    return db.scalars(statement).all()


def get_paper_detail(db: Session, paper_id: int, *, organization_id: int | None = None) -> PaperDetailData | None:
    paper = _get_scoped_paper(db, paper_id, organization_id=organization_id)
    if paper is None:
        return None

    asset = _get_original_pdf_asset(db, paper_id)
    latest_ocr_job = _get_latest_job_for_paper(db, paper_id, "pdf_ingest")
    latest_summary_job = _get_latest_job_for_paper(db, paper_id, "paper_summary")
    latest_question_set_job = _get_latest_job_for_paper(db, paper_id, "paper_question_set")
    latest_job = _get_latest_visible_job(db, paper_id)
    return PaperDetailData(
        paper=paper,
        asset=asset,
        latest_job=latest_job,
        latest_ocr_job=latest_ocr_job,
        latest_summary_job=latest_summary_job,
        latest_question_set_job=latest_question_set_job,
    )


def get_original_filename(asset: PaperAsset | None) -> str | None:
    metadata = asset.metadata_json if asset else None
    if not isinstance(metadata, dict):
        return None
    original_filename = metadata.get("original_filename")
    return original_filename if isinstance(original_filename, str) else None


def update_paper_metadata(db: Session, paper_id: int, updates: dict, *, organization_id: int) -> PaperDetailData | None:
    paper = _get_scoped_paper(db, paper_id, organization_id=organization_id)
    if paper is None:
        return None

    payload = {key: value for key, value in updates.items() if key in {"title", "authors", "doi", "source_url", "published_at"}}
    if not payload:
        raise ValueError("No fields to update")

    if "title" in payload:
        title = payload["title"]
        if title is None:
            raise ValueError("Title is required")
        normalized_title = str(title).strip()
        if not normalized_title:
            raise ValueError("Title is required")
        paper.title = normalized_title

    for field_name in ("authors", "doi", "source_url"):
        if field_name not in payload:
            continue
        value = payload[field_name]
        if value is None:
            setattr(paper, field_name, None)
            continue
        normalized_value = str(value).strip()
        setattr(paper, field_name, normalized_value or None)

    if "published_at" in payload:
        paper.published_at = payload["published_at"]

    paper.updated_at = datetime.now(timezone.utc)
    db.commit()
    return get_paper_detail(db, paper_id, organization_id=organization_id)


def update_paper_title(db: Session, paper_id: int, title: str, *, organization_id: int) -> PaperDetailData | None:
    return update_paper_metadata(db, paper_id, {"title": title}, organization_id=organization_id)


def delete_job(db: Session, job_id: int, *, organization_id: int | None = None) -> bool:
    job = _get_scoped_job(db, job_id, organization_id=organization_id)
    if job is None:
        return False
    if job.status not in DELETABLE_JOB_STATUSES:
        raise ValueError("Only completed, failed, or cancelled jobs can be deleted")

    now = datetime.now(timezone.utc)
    job.deleted_at = now
    paper = db.get(Paper, job.paper_id) if job.paper_id is not None else None
    if paper is not None and paper.deleted_at is None:
        paper.updated_at = now
        _set_paper_status_from_latest_job(db, paper)
    db.commit()
    return True


def cancel_job(db: Session, job_id: int, *, organization_id: int) -> Job | None:
    job = _get_scoped_job(db, job_id, organization_id=organization_id)
    if job is None:
        return None
    if job.status == "cancel_requested":
        return job
    if job.status not in {"queued", "processing"}:
        raise ValueError("Only queued or processing jobs can be cancelled")

    paper = db.get(Paper, job.paper_id) if job.paper_id is not None else None
    now = datetime.now(timezone.utc)
    job.cancel_requested_at = job.cancel_requested_at or now
    job.error_message = None
    if job.status == "queued":
        job.status = "cancelled"
        job.finished_at = now
        if paper is not None:
            paper.updated_at = now
            _set_paper_status_from_latest_job(db, paper)
        db.commit()
        if paper is not None:
            publish_task_status_event(db, paper_id=paper.id, job_id=job.id)
    else:
        job.status = "cancel_requested"
        if paper is not None and paper.deleted_at is None:
            paper.status = "cancel_requested"
            paper.updated_at = now
        db.commit()
        if paper is not None:
            publish_task_status_event(db, paper_id=paper.id, job_id=job.id)

    if job.celery_task_id:
        celery_app.control.revoke(job.celery_task_id)
    db.refresh(job)
    return job


def delete_paper(db: Session, paper_id: int, *, organization_id: int) -> bool:
    paper = _get_scoped_paper(db, paper_id, organization_id=organization_id)
    if paper is None:
        return False

    if _get_active_job_for_paper(db, paper_id) is not None:
        raise ValueError("Papers with active jobs cannot be deleted")

    now = datetime.now(timezone.utc)
    paper.deleted_at = now
    paper.updated_at = now
    paper.status = "deleted"
    for job in db.scalars(select(Job).where(Job.paper_id == paper_id, Job.deleted_at.is_(None))).all():
        job.deleted_at = now
    db.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
    db.commit()
    return True


def enqueue_paper_reparse(db: Session, paper_id: int, *, organization_id: int) -> Job | None:
    paper = _get_scoped_paper(db, paper_id, organization_id=organization_id)
    if paper is None:
        return None
    if _get_original_pdf_asset(db, paper_id) is None:
        raise ValueError("Original PDF not found")
    if _get_active_job_for_paper(db, paper_id) is not None:
        raise ValueError("Paper already has an active job")

    now = datetime.now(timezone.utc)
    paper.status = "queued"
    paper.updated_at = now
    job = Job(job_type="pdf_ingest", paper_id=paper_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    publish_task_status_event(db, paper_id=paper_id, job_id=job.id)
    return job


def _clear_document_structure(db: Session, paper_id: int) -> None:
    from app.services.paper_pipeline.helpers import clear_document_structure

    return clear_document_structure(db, paper_id)


def _picture_context(document: ExtractedDocument, picture: ExtractedPicture) -> str | None:
    if picture.page_number is None:
        return None
    nearby = [
        block.text.strip()
        for block in document.blocks
        if block.page_number == picture.page_number and block.text.strip()
    ][:5]
    return "\n".join(nearby) or None


def _describe_pictures(document: ExtractedDocument, adapter: PictureDescriptionAdapter) -> None:
    from app.services.paper_pipeline.helpers import describe_pictures

    return describe_pictures(document, adapter)


def _persist_document_structure(db: Session, *, paper_id: int, asset_id: int, document: ExtractedDocument) -> None:
    from app.services.paper_pipeline.helpers import persist_document_structure

    return persist_document_structure(db, paper_id=paper_id, asset_id=asset_id, document=document)


def enqueue_paper_summary_regeneration(db: Session, paper_id: int, *, organization_id: int) -> Job | None:
    paper = _get_scoped_paper(db, paper_id, organization_id=organization_id)
    if paper is None:
        return None
    asset = _get_original_pdf_asset(db, paper_id)
    if asset is None:
        raise ValueError("Original PDF not found")
    if not render_paper_text_from_structure(db, paper_id):
        raise ValueError("No parsed text available")
    if not is_chat_llm_configured():
        raise ValueError("LLM is not configured for summarization")
    if _get_active_job_for_paper(db, paper_id) is not None:
        raise ValueError("Paper already has an active job")

    now = datetime.now(timezone.utc)
    paper.status = "queued"
    paper.updated_at = now
    job = Job(job_type="paper_summary", paper_id=paper_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    publish_task_status_event(db, paper_id=paper_id, job_id=job.id)
    return job


def enqueue_paper_question_set_regeneration(db: Session, paper_id: int, *, organization_id: int) -> Job | None:
    paper = _get_scoped_paper(db, paper_id, organization_id=organization_id)
    if paper is None:
        return None
    asset = _get_original_pdf_asset(db, paper_id)
    if asset is None:
        raise ValueError("Original PDF not found")
    if not render_paper_text_from_structure(db, paper_id):
        raise ValueError("No parsed text available")
    if _extract_structured_summary_from_asset(asset) is None:
        raise ValueError("No structured summary available")
    if not is_chat_llm_configured():
        raise ValueError("LLM is not configured for question set extraction")
    if not get_organization_question_items(db, organization_id=organization_id):
        raise ValueError("No organization question set configured")
    if _get_active_job_for_paper(db, paper_id) is not None:
        raise ValueError("Paper already has an active job")

    now = datetime.now(timezone.utc)
    paper.status = "queued"
    paper.updated_at = now
    job = Job(job_type="paper_question_set", paper_id=paper_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    publish_task_status_event(db, paper_id=paper_id, job_id=job.id)
    return job


def run_pdf_ingest_job(
    job_id: int,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
    extractor: DocumentExtractor | None = None,
    picture_adapter: PictureDescriptionAdapter | None = None,
) -> int | None:
    from app.services.paper_pipeline.workflow import get_paper_workflow_service

    return get_paper_workflow_service().run_pdf_ingest_job(
        job_id,
        session_factory=session_factory,
        extractor=extractor,
        picture_adapter=picture_adapter,
    )


def run_paper_summary_job(
    job_id: int,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
) -> int | None:
    from app.services.paper_pipeline.workflow import get_paper_workflow_service

    return get_paper_workflow_service().run_paper_summary_job(
        job_id,
        session_factory=session_factory,
    )


def run_paper_question_set_job(
    job_id: int,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
) -> None:
    from app.services.paper_pipeline.workflow import get_paper_workflow_service

    return get_paper_workflow_service().run_paper_question_set_job(
        job_id,
        session_factory=session_factory,
    )
