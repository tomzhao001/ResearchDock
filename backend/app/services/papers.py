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
from app.services.pdf_extraction import PDFTextExtractor


@dataclass
class UploadArtifacts:
    paper_id: int
    job_id: int
    filename: str


def create_upload_artifacts(
    db: Session,
    *,
    filename: str,
    content_type: str,
    payload: bytes,
) -> UploadArtifacts:
    safe_name = Path(filename or "upload.pdf").name
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
        db.commit()

        document = (extractor or PDFTextExtractor()).extract(Path(asset.storage_path))
        metadata = dict(asset.metadata_json or {})
        metadata["extraction"] = document.metadata
        asset.metadata_json = metadata
        asset.raw_text = document.raw_text
        paper.status = "completed"
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
        db.commit()
        raise
    finally:
        db.close()
