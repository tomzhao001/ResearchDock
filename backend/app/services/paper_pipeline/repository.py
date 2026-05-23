from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import Job, Paper, PaperAsset
from app.services import papers as legacy_papers
from app.services.task_events import publish_task_status_event


class PaperJobRepository:
    def get_job(self, db: Session, job_id: int) -> Job | None:
        job = db.get(Job, job_id)
        if not job or job.deleted_at is not None or job.status == "cancelled":
            return None
        return job

    def get_paper(self, db: Session, paper_id: int | None) -> Paper | None:
        if paper_id is None:
            return None
        paper = db.get(Paper, paper_id)
        if paper is None or paper.deleted_at is not None:
            return None
        return paper

    def get_original_pdf_asset(self, db: Session, paper_id: int | None) -> PaperAsset | None:
        if paper_id is None:
            return None
        return legacy_papers._get_original_pdf_asset(db, paper_id)

    def mark_processing(self, db: Session, *, job: Job, paper: Paper) -> None:
        now = datetime.now(timezone.utc)
        job.status = "processing"
        job.started_at = now
        paper.status = "processing"
        paper.updated_at = now
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)

    def mark_completed(self, db: Session, *, job: Job, paper: Paper) -> None:
        now = datetime.now(timezone.utc)
        paper.status = "completed"
        paper.updated_at = now
        job.status = "completed"
        job.error_message = None
        job.finished_at = now
        db.commit()
        publish_task_status_event(db, paper_id=paper.id, job_id=job.id)

    def mark_failed(self, db: Session, *, job: Job | None, paper: Paper | None, error: Exception) -> None:
        if job is not None:
            job.status = "failed"
            job.error_message = str(error)
            job.finished_at = datetime.now(timezone.utc)
        if paper is not None:
            paper.status = "failed"
            paper.updated_at = datetime.now(timezone.utc)
        db.commit()
        if paper is not None:
            publish_task_status_event(db, paper_id=paper.id, job_id=job.id if job is not None else None)

    def ensure_not_cancelled(self, db: Session, *, job: Job, paper: Paper) -> None:
        legacy_papers._raise_if_cancel_requested(db, job, paper)
