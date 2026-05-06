from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import Job, Paper, PaperAsset
from app.services.llm import summarize_paper_text
from app.services.pdf_extraction import PDFTextExtractor


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


class DuplicateFilenameError(RuntimeError):
    def __init__(self, filename: str, existing_paper_id: int):
        super().__init__(f"Duplicate filename: {filename}")
        self.filename = filename
        self.existing_paper_id = existing_paper_id


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

    asset = db.scalar(
        select(PaperAsset).where(
            PaperAsset.paper_id == paper_id,
            PaperAsset.asset_type == "original_pdf",
        )
    )
    latest_job = db.scalar(
        select(Job).where(Job.paper_id == paper_id).order_by(Job.id.desc()).limit(1)
    )
    return PaperDetailData(paper=paper, asset=asset, latest_job=latest_job)


def get_original_filename(asset: PaperAsset | None) -> str | None:
    metadata = asset.metadata_json if asset else None
    if not isinstance(metadata, dict):
        return None
    original_filename = metadata.get("original_filename")
    return original_filename if isinstance(original_filename, str) else None


def update_paper_title(db: Session, paper_id: int, title: str) -> PaperDetailData | None:
    paper = db.scalar(select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None)))
    if paper is None:
        return None

    normalized_title = title.strip()
    if not normalized_title:
        raise ValueError("Title is required")

    paper.title = normalized_title
    paper.updated_at = datetime.now(timezone.utc)
    db.commit()
    return get_paper_detail(db, paper_id)


def run_pdf_ingest_job(
    job_id: int,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
    extractor: PDFTextExtractor | None = None,
) -> None:
    db = session_factory()
    try:
        job = db.get(Job, job_id)
        if not job:
            return

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

        document = (extractor or PDFTextExtractor()).extract(Path(asset.storage_path))
        metadata = dict(asset.metadata_json or {})
        metadata["extraction"] = document.metadata
        asset.metadata_json = metadata
        asset.raw_text = document.raw_text
        if settings.openai_api_key.strip():
            summary = summarize_paper_text(document.raw_text)
            paper.abstract_raw = summary.get("abstract_cn") or paper.abstract_raw
            metadata["structured_summary"] = summary
            asset.metadata_json = metadata
        paper.status = "completed"
        paper.updated_at = datetime.now(timezone.utc)
        job.status = "completed"
        job.error_message = None
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        if "job" in locals() and job is not None:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)
        if "paper" in locals() and paper is not None:
            paper.status = "failed"
            paper.updated_at = datetime.now(timezone.utc)
        db.commit()
        raise
    finally:
        db.close()
