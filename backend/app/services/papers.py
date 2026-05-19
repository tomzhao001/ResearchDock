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
from app.services.docling_extraction import build_document_extractor
from app.services.document_extraction import DocumentExtractor, ExtractedDocument, ExtractedPicture
from app.services.llm import answer_question_set_questions, is_chat_llm_configured, summarize_paper_text
from app.services.org_settings import get_organization_question_items
from app.services.rag import rebuild_paper_index_from_document_structure
from app.services.task_events import publish_task_status_event
from app.services.vision import PictureDescriptionAdapter, PictureDescriptionRequest, build_picture_description_adapter


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
    rows: list[tuple[tuple[int, int, int, int], str]] = []
    for fallback_index, block in enumerate(document.blocks):
        text = str(block.text or "").strip()
        if not text:
            continue
        if block.block_type == "heading":
            heading_level = max(int(block.heading_level or 1), 1)
            text = f"{'#' * min(heading_level, 6)} {text}"
        rows.append(
            (
                _structure_sort_key(
                    reading_order=block.reading_order,
                    page_number=block.page_number,
                    fallback_group=0,
                    fallback_index=fallback_index,
                ),
                text,
            )
        )
    for fallback_index, table in enumerate(document.tables):
        parts = [str(table.caption or "").strip()]
        table_rows = _serialize_table_rows(table.data)
        if table_rows:
            parts.append(table_rows)
        elif str(table.markdown or "").strip():
            parts.append(str(table.markdown or "").strip())
        rendered = "\n".join(part for part in parts if part)
        if not rendered:
            continue
        rows.append(
            (
                _structure_sort_key(
                    reading_order=table.reading_order,
                    page_number=table.page_from,
                    fallback_group=1,
                    fallback_index=fallback_index,
                ),
                rendered,
            )
        )
    for fallback_index, picture in enumerate(document.pictures):
        rendered = "\n".join(
            part
            for part in (str(picture.caption or "").strip(), str(picture.description or "").strip())
            if part
        )
        if not rendered:
            continue
        rows.append(
            (
                _structure_sort_key(
                    reading_order=picture.reading_order,
                    page_number=picture.page_number,
                    fallback_group=2,
                    fallback_index=fallback_index,
                ),
                rendered,
            )
        )
    return _normalize_rendered_lines([text for _sort_key, text in sorted(rows, key=lambda item: item[0])])


def render_paper_text_from_structure(db: Session, paper_id: int) -> str:
    rows: list[tuple[tuple[int, int, int, int], str]] = []
    blocks = db.scalars(
        select(PaperDocumentBlock)
        .where(PaperDocumentBlock.paper_id == paper_id)
        .order_by(PaperDocumentBlock.block_index.asc(), PaperDocumentBlock.id.asc())
    ).all()
    for fallback_index, block in enumerate(blocks):
        text = str(block.text or "").strip()
        if not text:
            continue
        if (block.block_type or "") == "heading":
            heading_level = max(int(block.heading_level or 1), 1)
            text = f"{'#' * min(heading_level, 6)} {text}"
        page_number = None
        page = getattr(block, "page_id", None)
        if page is not None:
            linked_page = db.get(PaperDocumentPage, block.page_id)
            page_number = linked_page.page_number if linked_page is not None else None
        rows.append(
            (
                _structure_sort_key(
                    reading_order=block.reading_order,
                    page_number=page_number,
                    fallback_group=0,
                    fallback_index=fallback_index,
                ),
                text,
            )
        )
    tables = db.scalars(
        select(PaperDocumentTable)
        .where(PaperDocumentTable.paper_id == paper_id)
        .order_by(PaperDocumentTable.table_index.asc(), PaperDocumentTable.id.asc())
    ).all()
    for fallback_index, table in enumerate(tables):
        parts = [str(table.caption or "").strip()]
        table_rows = _serialize_table_rows(table.data_json)
        if table_rows:
            parts.append(table_rows)
        elif str(table.markdown or "").strip():
            parts.append(str(table.markdown or "").strip())
        rendered = "\n".join(part for part in parts if part)
        if not rendered:
            continue
        rows.append(
            (
                _structure_sort_key(
                    reading_order=table.reading_order,
                    page_number=table.page_from,
                    fallback_group=1,
                    fallback_index=fallback_index,
                ),
                rendered,
            )
        )
    pictures = db.scalars(
        select(PaperDocumentPicture)
        .where(PaperDocumentPicture.paper_id == paper_id)
        .order_by(PaperDocumentPicture.picture_index.asc(), PaperDocumentPicture.id.asc())
    ).all()
    for fallback_index, picture in enumerate(pictures):
        rendered = "\n".join(
            part
            for part in (str(picture.caption or "").strip(), str(picture.description or "").strip())
            if part
        )
        if not rendered:
            continue
        rows.append(
            (
                _structure_sort_key(
                    reading_order=picture.reading_order,
                    page_number=picture.page_number,
                    fallback_group=2,
                    fallback_index=fallback_index,
                ),
                rendered,
            )
        )
    return _normalize_rendered_lines([text for _sort_key, text in sorted(rows, key=lambda item: item[0])])


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
    db.refresh(job)
    if job.deleted_at is not None or job.status == "cancelled":
        raise JobCancellationRequested()
    if job.status == "cancel_requested":
        _mark_job_cancelled(db, job, paper)
        raise JobCancellationRequested()


def _queue_summary_job_if_needed(db: Session, paper: Paper, asset: PaperAsset) -> int | None:
    if not is_chat_llm_configured():
        return None
    if not render_paper_text_from_structure(db, paper.id):
        return None

    latest_summary_job = _get_latest_job_for_paper(db, paper.id, "paper_summary")
    if latest_summary_job is not None and latest_summary_job.status in ACTIVE_JOB_STATUSES:
        return latest_summary_job.id

    paper.status = "queued"
    paper.updated_at = datetime.now(timezone.utc)
    summary_job = Job(job_type="paper_summary", paper_id=paper.id, status="queued")
    db.add(summary_job)
    db.commit()
    db.refresh(summary_job)
    publish_task_status_event(db, paper_id=paper.id, job_id=summary_job.id)
    return summary_job.id


def _extract_structured_summary_from_asset(asset: PaperAsset | None) -> dict | None:
    metadata = asset.metadata_json if asset else None
    if not isinstance(metadata, dict):
        return None
    structured_summary = metadata.get("structured_summary")
    return structured_summary if isinstance(structured_summary, dict) else None


def _queue_question_set_job_if_needed(db: Session, paper: Paper, asset: PaperAsset) -> int | None:
    if not is_chat_llm_configured():
        return None
    if not render_paper_text_from_structure(db, paper.id):
        return None
    if _extract_structured_summary_from_asset(asset) is None:
        return None
    if not get_organization_question_items(db, organization_id=paper.organization_id):
        return None

    latest_question_set_job = _get_latest_job_for_paper(db, paper.id, "paper_question_set")
    if latest_question_set_job is not None and latest_question_set_job.status in ACTIVE_JOB_STATUSES:
        return latest_question_set_job.id

    paper.status = "queued"
    paper.updated_at = datetime.now(timezone.utc)
    question_set_job = Job(job_type="paper_question_set", paper_id=paper.id, status="queued")
    db.add(question_set_job)
    db.commit()
    db.refresh(question_set_job)
    publish_task_status_event(db, paper_id=paper.id, job_id=question_set_job.id)
    return question_set_job.id


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
    db.execute(delete(PaperDocumentPicture).where(PaperDocumentPicture.paper_id == paper_id))
    db.execute(delete(PaperDocumentTable).where(PaperDocumentTable.paper_id == paper_id))
    db.execute(delete(PaperDocumentBlock).where(PaperDocumentBlock.paper_id == paper_id))
    db.execute(delete(PaperDocumentPage).where(PaperDocumentPage.paper_id == paper_id))


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
    for picture in document.pictures:
        result = adapter.describe(
            PictureDescriptionRequest(
                image_bytes=picture.image_bytes,
                caption=picture.caption,
                page_number=picture.page_number,
                bbox=picture.bbox,
                context=_picture_context(document, picture),
            )
        )
        if result.description:
            picture.description = result.description
        picture.description_model = result.model_name
        picture.description_prompt_version = result.prompt_version
        metadata = dict(picture.metadata or {})
        metadata["description"] = {
            "usage": result.usage,
            "error": result.error,
        }
        if result.raw_response is not None:
            metadata["description"]["raw_response"] = result.raw_response
        picture.metadata = metadata


def _persist_document_structure(db: Session, *, paper_id: int, asset_id: int, document: ExtractedDocument) -> None:
    page_id_by_number: dict[int, int] = {}
    for page in document.pages:
        record = PaperDocumentPage(
            paper_id=paper_id,
            asset_id=asset_id,
            page_number=page.page_number,
            text=page.text or None,
            width=int(page.width) if page.width is not None else None,
            height=int(page.height) if page.height is not None else None,
            metadata_json=page.metadata,
        )
        db.add(record)
        db.flush()
        page_id_by_number[page.page_number] = record.id

    for block in document.blocks:
        db.add(
            PaperDocumentBlock(
                paper_id=paper_id,
                page_id=page_id_by_number.get(block.page_number or 0),
                block_index=block.block_index,
                reading_order=block.reading_order,
                block_type=block.block_type or "paragraph",
                docling_label=block.docling_label,
                heading_level=block.heading_level,
                section_path=block.section_path,
                text=block.text,
                bbox_json=block.bbox,
                provenance_json=block.provenance,
                metadata_json=block.metadata,
            )
        )

    for table in document.tables:
        db.add(
            PaperDocumentTable(
                paper_id=paper_id,
                page_from=table.page_from,
                page_to=table.page_to,
                table_index=table.table_index,
                reading_order=table.reading_order,
                heading_level=table.heading_level,
                section_path=table.section_path,
                caption=table.caption,
                markdown=table.markdown,
                data_json=table.data,
                bbox_json=table.bbox,
                provenance_json=table.provenance,
                metadata_json=table.metadata,
            )
        )

    for picture in document.pictures:
        db.add(
            PaperDocumentPicture(
                paper_id=paper_id,
                page_number=picture.page_number,
                picture_index=picture.picture_index,
                reading_order=picture.reading_order,
                heading_level=picture.heading_level,
                section_path=picture.section_path,
                caption=picture.caption,
                description=picture.description,
                description_model=picture.description_model,
                description_prompt_version=picture.description_prompt_version,
                bbox_json=picture.bbox,
                provenance_json=picture.provenance,
                image_asset_path=picture.image_asset_path,
                metadata_json=picture.metadata,
            )
        )


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
    db = session_factory()
    try:
        job = db.get(Job, job_id)
        if not job or job.deleted_at is not None or job.status == "cancelled":
            return None

        paper = db.get(Paper, job.paper_id) if job.paper_id is not None else None
        asset = db.scalar(
            select(PaperAsset).where(
                PaperAsset.paper_id == job.paper_id,
                PaperAsset.asset_type == "original_pdf",
            )
        )
        if paper is None or paper.deleted_at is not None or asset is None or not asset.storage_path:
            raise RuntimeError("Missing upload asset for job")
        _raise_if_cancel_requested(db, job, paper)

        now = datetime.now(timezone.utc)
        job.status = "processing"
        job.started_at = now
        paper.status = "processing"
        paper.updated_at = now
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)

        _raise_if_cancel_requested(db, job, paper)
        document = (extractor or build_document_extractor()).extract(Path(asset.storage_path))
        _raise_if_cancel_requested(db, job, paper)
        _describe_pictures(document, picture_adapter or build_picture_description_adapter())
        _raise_if_cancel_requested(db, job, paper)
        _clear_document_structure(db, paper.id)
        db.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper.id))
        _persist_document_structure(db, paper_id=paper.id, asset_id=asset.id, document=document)
        db.flush()
        metadata = dict(asset.metadata_json or {})
        metadata["extraction"] = {
            **document.extraction_metadata(),
            "picture_vlm": {
                "provider": settings.picture_vlm_provider,
                "model": settings.picture_vlm_model,
                "prompt_version": settings.picture_vlm_prompt_version,
            },
        }
        asset.metadata_json = metadata
        asset.raw_text = None
        rebuild_paper_index_from_document_structure(
            db,
            paper_id=paper.id,
            paper_title=paper.title,
        )
        _raise_if_cancel_requested(db, job, paper)
        paper.status = "completed"
        paper.updated_at = datetime.now(timezone.utc)
        job.status = "completed"
        job.error_message = None
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)
        return _queue_summary_job_if_needed(db, paper, asset)
    except JobCancellationRequested:
        return None
    except Exception as exc:
        if "job" in locals() and job is not None:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
        if "paper" in locals() and paper is not None:
            paper.status = "failed"
            paper.updated_at = datetime.now(timezone.utc)
        db.commit()
        if "paper" in locals() and paper is not None:
            publish_task_status_event(db, paper_id=paper.id, job_id=job.id if "job" in locals() and job is not None else None)
        raise
    finally:
        db.close()


def run_paper_summary_job(
    job_id: int,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
) -> int | None:
    db = session_factory()
    try:
        job = db.get(Job, job_id)
        if not job or job.deleted_at is not None or job.status == "cancelled":
            return None

        paper = db.get(Paper, job.paper_id) if job.paper_id is not None else None
        asset = _get_original_pdf_asset(db, job.paper_id) if job.paper_id is not None else None
        if paper is None or paper.deleted_at is not None or asset is None:
            raise RuntimeError("Missing paper or upload asset for summary job")
        if not is_chat_llm_configured():
            raise RuntimeError("LLM is not configured for summarization")
        paper_text = render_paper_text_from_structure(db, paper.id)
        if not paper_text:
            raise RuntimeError("No parsed text available")
        _raise_if_cancel_requested(db, job, paper)

        now = datetime.now(timezone.utc)
        job.status = "processing"
        job.started_at = now
        paper.status = "processing"
        paper.updated_at = now
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)

        _raise_if_cancel_requested(db, job, paper)
        summary = summarize_paper_text(paper_text)
        _raise_if_cancel_requested(db, job, paper)
        summary["authors"] = _normalize_summary_text(summary.get("authors"))
        summary["doi"] = _normalize_summary_doi(summary.get("doi"))
        summary["source_url"] = _normalize_summary_url(summary.get("source_url"))
        published_at = _parse_summary_published_at(summary.get("published_at"))
        summary["published_at"] = _serialize_summary_published_at(published_at)
        metadata = dict(asset.metadata_json or {})
        metadata["structured_summary"] = summary
        asset.metadata_json = metadata
        paper.abstract_raw = summary.get("abstract_cn") or paper.abstract_raw
        if _paper_text_field_is_empty(paper.authors) and summary["authors"]:
            paper.authors = summary["authors"]
        if _paper_text_field_is_empty(paper.doi) and summary["doi"]:
            paper.doi = summary["doi"]
        if _paper_text_field_is_empty(paper.source_url) and summary["source_url"]:
            paper.source_url = summary["source_url"]
        if paper.published_at is None and published_at is not None:
            paper.published_at = published_at
        paper.status = "completed"
        paper.updated_at = datetime.now(timezone.utc)
        job.status = "completed"
        job.error_message = None
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)
        return _queue_question_set_job_if_needed(db, paper, asset)
    except JobCancellationRequested:
        return None
    except Exception as exc:
        if "job" in locals() and job is not None:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
        if "paper" in locals() and paper is not None:
            paper.status = "failed"
            paper.updated_at = datetime.now(timezone.utc)
        db.commit()
        if "paper" in locals() and paper is not None:
            publish_task_status_event(db, paper_id=paper.id, job_id=job.id if "job" in locals() and job is not None else None)
        raise
    finally:
        db.close()


def run_paper_question_set_job(
    job_id: int,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
) -> None:
    db = session_factory()
    try:
        job = db.get(Job, job_id)
        if not job or job.deleted_at is not None or job.status == "cancelled":
            return

        paper = db.get(Paper, job.paper_id) if job.paper_id is not None else None
        asset = _get_original_pdf_asset(db, job.paper_id) if job.paper_id is not None else None
        if paper is None or paper.deleted_at is not None or asset is None:
            raise RuntimeError("Missing paper or upload asset for question set job")
        if not is_chat_llm_configured():
            raise RuntimeError("LLM is not configured for question set extraction")
        paper_text = render_paper_text_from_structure(db, paper.id)
        if not paper_text:
            raise RuntimeError("No parsed text available")

        structured_summary = _extract_structured_summary_from_asset(asset)
        if structured_summary is None:
            raise RuntimeError("No structured summary available")

        questions = get_organization_question_items(db, organization_id=paper.organization_id)
        if not questions:
            raise RuntimeError("No organization question set configured")
        _raise_if_cancel_requested(db, job, paper)

        now = datetime.now(timezone.utc)
        job.status = "processing"
        job.started_at = now
        paper.status = "processing"
        paper.updated_at = now
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)

        _raise_if_cancel_requested(db, job, paper)
        result = answer_question_set_questions(
            paper_text,
            structured_summary=structured_summary,
            questions=questions,
        )
        _raise_if_cancel_requested(db, job, paper)
        metadata = dict(asset.metadata_json or {})
        metadata["question_set_extraction"] = result
        asset.metadata_json = metadata
        paper.status = "completed"
        paper.updated_at = datetime.now(timezone.utc)
        job.status = "completed"
        job.error_message = None
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)
    except JobCancellationRequested:
        return
    except Exception as exc:
        if "job" in locals() and job is not None:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
        if "paper" in locals() and paper is not None:
            paper.status = "failed"
            paper.updated_at = datetime.now(timezone.utc)
        db.commit()
        if "paper" in locals() and paper is not None:
            publish_task_status_event(db, paper_id=paper.id, job_id=job.id if "job" in locals() and job is not None else None)
        raise
    finally:
        db.close()
