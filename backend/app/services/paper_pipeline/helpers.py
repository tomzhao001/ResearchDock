from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import Job, Paper, PaperAsset, PaperDocumentBlock, PaperDocumentPage, PaperDocumentPicture, PaperDocumentTable
from app.services import papers as legacy_papers
from app.services.document_extraction import ExtractedDocument, ExtractedPicture
from app.services.paper_pipeline.rendering import render_paper_text_from_structure
from app.services.task_events import publish_task_status_event
from app.services.vision import PictureDescriptionAdapter, PictureDescriptionRequest


def raise_if_cancel_requested(db: Session, job: Job, paper: Paper | None) -> None:
    db.refresh(job)
    if job.deleted_at is not None or job.status == "cancelled":
        raise legacy_papers.JobCancellationRequested()
    if job.status == "cancel_requested":
        legacy_papers._mark_job_cancelled(db, job, paper)
        raise legacy_papers.JobCancellationRequested()


def queue_summary_job_if_needed(db: Session, paper: Paper, asset: PaperAsset) -> int | None:
    if not legacy_papers.is_chat_llm_configured():
        return None
    if not render_paper_text_from_structure(db, paper.id):
        return None

    latest_summary_job = legacy_papers._get_latest_job_for_paper(db, paper.id, "paper_summary")
    if latest_summary_job is not None and latest_summary_job.status in legacy_papers.ACTIVE_JOB_STATUSES:
        return latest_summary_job.id

    paper.status = "queued"
    paper.updated_at = datetime.now(timezone.utc)
    summary_job = Job(job_type="paper_summary", paper_id=paper.id, status="queued")
    db.add(summary_job)
    db.commit()
    db.refresh(summary_job)
    publish_task_status_event(db, paper_id=paper.id, job_id=summary_job.id)
    return summary_job.id


def extract_structured_summary_from_asset(asset: PaperAsset | None) -> dict | None:
    metadata = asset.metadata_json if asset else None
    if not isinstance(metadata, dict):
        return None
    structured_summary = metadata.get("structured_summary")
    return structured_summary if isinstance(structured_summary, dict) else None


def queue_question_set_job_if_needed(db: Session, paper: Paper, asset: PaperAsset) -> int | None:
    if not legacy_papers.is_chat_llm_configured():
        return None
    if not render_paper_text_from_structure(db, paper.id):
        return None
    if extract_structured_summary_from_asset(asset) is None:
        return None
    if not legacy_papers.get_organization_question_items(db, organization_id=paper.organization_id):
        return None

    latest_question_set_job = legacy_papers._get_latest_job_for_paper(db, paper.id, "paper_question_set")
    if latest_question_set_job is not None and latest_question_set_job.status in legacy_papers.ACTIVE_JOB_STATUSES:
        return latest_question_set_job.id

    paper.status = "queued"
    paper.updated_at = datetime.now(timezone.utc)
    question_set_job = Job(job_type="paper_question_set", paper_id=paper.id, status="queued")
    db.add(question_set_job)
    db.commit()
    db.refresh(question_set_job)
    publish_task_status_event(db, paper_id=paper.id, job_id=question_set_job.id)
    return question_set_job.id


def clear_document_structure(db: Session, paper_id: int) -> None:
    db.execute(delete(PaperDocumentPicture).where(PaperDocumentPicture.paper_id == paper_id))
    db.execute(delete(PaperDocumentTable).where(PaperDocumentTable.paper_id == paper_id))
    db.execute(delete(PaperDocumentBlock).where(PaperDocumentBlock.paper_id == paper_id))
    db.execute(delete(PaperDocumentPage).where(PaperDocumentPage.paper_id == paper_id))


def picture_context(document: ExtractedDocument, picture: ExtractedPicture) -> str | None:
    if picture.page_number is None:
        return None
    nearby = [
        block.text.strip()
        for block in document.blocks
        if block.page_number == picture.page_number and block.text.strip()
    ][:5]
    return "\n".join(nearby) or None


def describe_pictures(document: ExtractedDocument, adapter: PictureDescriptionAdapter) -> None:
    for picture in document.pictures:
        result = adapter.describe(
            PictureDescriptionRequest(
                image_bytes=picture.image_bytes,
                caption=picture.caption,
                page_number=picture.page_number,
                bbox=picture.bbox,
                context=picture_context(document, picture),
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


def persist_document_structure(db: Session, *, paper_id: int, asset_id: int, document: ExtractedDocument) -> None:
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
