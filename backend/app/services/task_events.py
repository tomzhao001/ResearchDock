from __future__ import annotations

import logging

from redis import Redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Job
from app.schemas import JobPublic, PaperDetailResponse, PaperListItem, QuestionSetExtractionPublic, TaskStatusEvent

TASK_STATUS_CHANNEL = "researchdock:task-status"

logger = logging.getLogger(__name__)
_publisher: Redis | None = None


def _get_publisher() -> Redis:
    global _publisher
    if _publisher is None:
        _publisher = Redis.from_url(settings.redis_url, decode_responses=True)
    return _publisher


def get_task_status_channel(organization_id: int) -> str:
    return f"{TASK_STATUS_CHANNEL}:{organization_id}"


def _build_paper_detail_response(detail) -> PaperDetailResponse:
    from app.services.papers import get_job_phase_status, get_original_filename

    metadata = detail.asset.metadata_json if detail.asset else None
    extraction_metadata = None
    structured_summary = None
    question_set_extraction = None
    if isinstance(metadata, dict):
        extraction_metadata = metadata.get("extraction")
        structured_summary = metadata.get("structured_summary")
        raw_question_set_extraction = metadata.get("question_set_extraction")
        if isinstance(raw_question_set_extraction, dict):
            question_set_extraction = QuestionSetExtractionPublic.model_validate(raw_question_set_extraction)

    latest_job = JobPublic.model_validate(detail.latest_job) if detail.latest_job else None
    latest_ocr_job = JobPublic.model_validate(detail.latest_ocr_job) if detail.latest_ocr_job else None
    latest_summary_job = JobPublic.model_validate(detail.latest_summary_job) if detail.latest_summary_job else None
    latest_question_set_job = JobPublic.model_validate(detail.latest_question_set_job) if detail.latest_question_set_job else None
    return PaperDetailResponse(
        id=detail.paper.id,
        organization_id=detail.paper.organization_id,
        title=detail.paper.title,
        authors=detail.paper.authors,
        abstract_raw=detail.paper.abstract_raw,
        source_url=detail.paper.source_url,
        pdf_url=detail.paper.pdf_url,
        doi=detail.paper.doi,
        published_at=detail.paper.published_at,
        status=detail.paper.status,
        ocr_status=get_job_phase_status(detail.latest_ocr_job),
        summary_status=get_job_phase_status(detail.latest_summary_job),
        question_set_status=get_job_phase_status(detail.latest_question_set_job),
        created_at=detail.paper.created_at,
        updated_at=detail.paper.updated_at,
        original_filename=get_original_filename(detail.asset),
        preview_text=detail.asset.raw_text if detail.asset else None,
        extraction_metadata=extraction_metadata if isinstance(extraction_metadata, dict) else None,
        structured_summary=structured_summary if isinstance(structured_summary, dict) else None,
        question_set_extraction=question_set_extraction,
        latest_job=latest_job,
        latest_ocr_job=latest_ocr_job,
        latest_summary_job=latest_summary_job,
        latest_question_set_job=latest_question_set_job,
    )


def _build_paper_list_item(detail) -> PaperListItem:
    from app.services.papers import get_job_phase_status, get_original_filename

    return PaperListItem(
        id=detail.paper.id,
        organization_id=detail.paper.organization_id,
        title=detail.paper.title,
        original_filename=get_original_filename(detail.asset),
        abstract_raw=detail.paper.abstract_raw,
        status=detail.paper.status,
        ocr_status=get_job_phase_status(detail.latest_ocr_job),
        summary_status=get_job_phase_status(detail.latest_summary_job),
        question_set_status=get_job_phase_status(detail.latest_question_set_job),
        created_at=detail.paper.created_at,
        updated_at=detail.paper.updated_at,
    )


def build_task_status_event(db: Session, *, paper_id: int, job_id: int | None = None) -> TaskStatusEvent | None:
    from app.services.papers import get_paper_detail

    detail = get_paper_detail(db, paper_id, organization_id=None)
    if detail is None:
        return None

    job = db.get(Job, job_id) if job_id is not None else detail.latest_job
    if job is None and detail.latest_job is not None:
        job = detail.latest_job
    if job is None:
        job = db.scalar(select(Job).where(Job.paper_id == paper_id).order_by(Job.id.desc()).limit(1))

    detail_response = _build_paper_detail_response(detail)
    return TaskStatusEvent(
        paper_id=paper_id,
        job_id=job.id if job else None,
        job_type=job.job_type if job else None,
        job_status=job.status if job else None,
        paper_status=detail.paper.status,
        ocr_status=detail_response.ocr_status,
        summary_status=detail_response.summary_status,
        question_set_status=detail_response.question_set_status,
        error_message=job.error_message if job else None,
        updated_at=detail.paper.updated_at,
        job=JobPublic.model_validate(job) if job else None,
        paper_list_item=_build_paper_list_item(detail),
        paper_detail=detail_response,
    )


def publish_task_status_event(db: Session, *, paper_id: int, job_id: int | None = None) -> None:
    event = build_task_status_event(db, paper_id=paper_id, job_id=job_id)
    if event is None:
        return
    try:
        _get_publisher().publish(get_task_status_channel(event.paper_detail.organization_id), event.model_dump_json())
    except Exception:
        logger.exception("Failed to publish task status event for paper %s", paper_id)
