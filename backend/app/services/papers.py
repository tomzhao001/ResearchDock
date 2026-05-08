from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import Job, Paper, PaperAsset, PaperChunk
from app.services.llm import summarize_paper_text
from app.services.pdf_extraction import PDFTextExtractor
from app.services.rag import rebuild_paper_index
from app.services.task_events import publish_task_status_event


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


class DuplicateFilenameError(RuntimeError):
    def __init__(self, filename: str, existing_paper_id: int):
        super().__init__(f"Duplicate filename: {filename}")
        self.filename = filename
        self.existing_paper_id = existing_paper_id


ACTIVE_JOB_STATUSES = {"queued", "processing"}
NON_DELETABLE_JOB_STATUSES = {"processing"}


def normalize_filename(filename: str) -> str:
    return Path(filename or "upload.pdf").name.strip().casefold()


def _active_paper_asset_rows(db: Session) -> list[tuple[Paper, PaperAsset]]:
    rows = db.execute(
        select(Paper, PaperAsset)
        .join(PaperAsset, PaperAsset.paper_id == Paper.id)
        .where(
            Paper.deleted_at.is_(None),
            PaperAsset.asset_type == "original_pdf",
        )
    ).all()
    return [(paper, asset) for paper, asset in rows]


def find_active_paper_by_original_filename(db: Session, filename: str) -> Paper | None:
    normalized = normalize_filename(filename)
    for paper, asset in _active_paper_asset_rows(db):
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


def _get_latest_job_for_paper(db: Session, paper_id: int, job_type: str) -> Job | None:
    return db.scalar(
        select(Job)
        .where(
            Job.paper_id == paper_id,
            Job.job_type == job_type,
        )
        .order_by(Job.id.desc())
        .limit(1)
    )


def _get_active_job_for_paper(db: Session, paper_id: int) -> Job | None:
    return db.scalar(
        select(Job)
        .where(
            Job.paper_id == paper_id,
            Job.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(Job.id.desc())
        .limit(1)
    )


def _queue_summary_job_if_needed(db: Session, paper: Paper, asset: PaperAsset) -> int | None:
    if not settings.openai_api_key.strip():
        return None
    if not (asset.raw_text or "").strip():
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


def get_job_phase_status(job: Job | None) -> str | None:
    return job.status if job else None


def create_upload_artifacts(
    db: Session,
    *,
    filename: str,
    content_type: str,
    payload: bytes,
    overwrite: bool = False,
) -> UploadArtifacts:
    safe_name = Path(filename or "upload.pdf").name
    existing_paper = find_active_paper_by_original_filename(db, safe_name)
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


def list_papers(db: Session, *, limit: int = 50) -> list[Paper]:
    statement = (
        select(Paper)
        .where(Paper.deleted_at.is_(None))
        .order_by(Paper.updated_at.desc(), Paper.id.desc())
        .limit(limit)
    )
    return db.scalars(statement).all()


def get_paper_detail(db: Session, paper_id: int) -> PaperDetailData | None:
    paper = db.scalar(select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None)))
    if paper is None:
        return None

    asset = _get_original_pdf_asset(db, paper_id)
    latest_ocr_job = _get_latest_job_for_paper(db, paper_id, "pdf_ingest")
    latest_summary_job = _get_latest_job_for_paper(db, paper_id, "paper_summary")
    latest_job = db.scalar(
        select(Job).where(Job.paper_id == paper_id).order_by(Job.id.desc()).limit(1)
    )
    return PaperDetailData(
        paper=paper,
        asset=asset,
        latest_job=latest_job,
        latest_ocr_job=latest_ocr_job,
        latest_summary_job=latest_summary_job,
    )


def get_original_filename(asset: PaperAsset | None) -> str | None:
    metadata = asset.metadata_json if asset else None
    if not isinstance(metadata, dict):
        return None
    original_filename = metadata.get("original_filename")
    return original_filename if isinstance(original_filename, str) else None


def update_paper_metadata(db: Session, paper_id: int, updates: dict) -> PaperDetailData | None:
    paper = db.scalar(select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None)))
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
    return get_paper_detail(db, paper_id)


def update_paper_title(db: Session, paper_id: int, title: str) -> PaperDetailData | None:
    return update_paper_metadata(db, paper_id, {"title": title})


def delete_job(db: Session, job_id: int) -> bool:
    job = db.get(Job, job_id)
    if job is None:
        return False
    if job.status in NON_DELETABLE_JOB_STATUSES:
        raise ValueError("Processing jobs cannot be deleted")

    db.delete(job)
    db.commit()
    return True


def delete_paper(db: Session, paper_id: int) -> bool:
    paper = db.scalar(select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None)))
    if paper is None:
        return False

    if _get_active_job_for_paper(db, paper_id) is not None:
        raise ValueError("Papers with active jobs cannot be deleted")

    now = datetime.now(timezone.utc)
    paper.deleted_at = now
    paper.updated_at = now
    paper.status = "deleted"
    db.execute(delete(Job).where(Job.paper_id == paper_id))
    db.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
    db.commit()
    return True


def enqueue_paper_ocr_rerun(db: Session, paper_id: int) -> Job | None:
    paper = db.scalar(select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None)))
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


def enqueue_paper_summary_regeneration(db: Session, paper_id: int) -> Job | None:
    paper = db.scalar(select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None)))
    if paper is None:
        return None
    asset = _get_original_pdf_asset(db, paper_id)
    if asset is None:
        raise ValueError("Original PDF not found")
    if not (asset.raw_text or "").strip():
        raise ValueError("No OCR text available")
    if not settings.openai_api_key.strip():
        raise ValueError("Summary model is not configured")
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


def run_pdf_ingest_job(
    job_id: int,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
    extractor: PDFTextExtractor | None = None,
) -> int | None:
    db = session_factory()
    try:
        job = db.get(Job, job_id)
        if not job:
            return None

        paper = db.get(Paper, job.paper_id) if job.paper_id is not None else None
        asset = db.scalar(
            select(PaperAsset).where(
                PaperAsset.paper_id == job.paper_id,
                PaperAsset.asset_type == "original_pdf",
            )
        )
        if paper is None or asset is None or not asset.storage_path:
            raise RuntimeError("Missing upload asset for job")

        now = datetime.now(timezone.utc)
        job.status = "processing"
        job.started_at = now
        paper.status = "processing"
        paper.updated_at = now
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)

        document = (extractor or PDFTextExtractor()).extract(Path(asset.storage_path))
        metadata = dict(asset.metadata_json or {})
        metadata["extraction"] = document.metadata
        asset.metadata_json = metadata
        asset.raw_text = document.raw_text
        rebuild_paper_index(db, paper_id=paper.id, raw_text=document.raw_text)
        paper.status = "completed"
        paper.updated_at = datetime.now(timezone.utc)
        job.status = "completed"
        job.error_message = None
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)
        return _queue_summary_job_if_needed(db, paper, asset)
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
) -> None:
    db = session_factory()
    try:
        job = db.get(Job, job_id)
        if not job:
            return

        paper = db.get(Paper, job.paper_id) if job.paper_id is not None else None
        asset = _get_original_pdf_asset(db, job.paper_id) if job.paper_id is not None else None
        if paper is None or paper.deleted_at is not None or asset is None:
            raise RuntimeError("Missing paper or upload asset for summary job")
        if not settings.openai_api_key.strip():
            raise RuntimeError("Summary model is not configured")
        if not (asset.raw_text or "").strip():
            raise RuntimeError("No OCR text available")

        now = datetime.now(timezone.utc)
        job.status = "processing"
        job.started_at = now
        paper.status = "processing"
        paper.updated_at = now
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)

        summary = summarize_paper_text(asset.raw_text)
        metadata = dict(asset.metadata_json or {})
        metadata["structured_summary"] = summary
        asset.metadata_json = metadata
        paper.abstract_raw = summary.get("abstract_cn") or paper.abstract_raw
        paper.status = "completed"
        paper.updated_at = datetime.now(timezone.utc)
        job.status = "completed"
        job.error_message = None
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)
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
