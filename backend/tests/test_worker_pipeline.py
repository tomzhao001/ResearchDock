import fitz
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job, PaperAsset, PaperChunk
from app.services.ocr.base import OcrRecognitionResult
from app.services.papers import create_upload_artifacts, run_pdf_ingest_job
from app.services.pdf_extraction import PDFTextExtractor


class MockOcrAdapter:
    def recognize_page(self, *, image_bytes: bytes, page_number: int, hints: dict | None = None) -> OcrRecognitionResult:
        return OcrRecognitionResult(
            text="ocr recovered text",
            metadata={
                "provider": "glm_ocr",
                "page_number": page_number,
                "hints": hints or {},
            },
        )


class FailingOcrAdapter:
    def recognize_page(self, *, image_bytes: bytes, page_number: int, hints: dict | None = None) -> OcrRecognitionResult:
        raise RuntimeError("GLM-OCR request failed")


def make_blank_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page()
    payload = doc.tobytes()
    doc.close()
    return payload


def test_worker_persists_ocr_fallback_result(
    db_session: Session,
    session_factory: sessionmaker,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        filename="scan.pdf",
        content_type="application/pdf",
        payload=make_blank_pdf(),
    )

    run_pdf_ingest_job(
        artifacts.job_id,
        session_factory=session_factory,
        extractor=PDFTextExtractor(ocr_backend=MockOcrAdapter()),
    )

    job = db_session.get(Job, artifacts.job_id)
    asset = db_session.scalar(select(PaperAsset).where(PaperAsset.paper_id == artifacts.paper_id))
    chunks = db_session.scalars(select(PaperChunk).where(PaperChunk.paper_id == artifacts.paper_id)).all()

    assert job is not None
    assert job.status == "completed"
    assert asset is not None
    assert asset.raw_text == "ocr recovered text"
    assert asset.metadata_json is not None
    assert asset.metadata_json["extraction"]["used_ocr_pages"] == [1]
    assert asset.metadata_json["extraction"]["pages"][0]["ocr_metadata"]["provider"] == "glm_ocr"
    assert len(chunks) == 1
    assert chunks[0].embedding is None


def test_worker_marks_job_failed_when_glm_ocr_errors(
    db_session: Session,
    session_factory: sessionmaker,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        filename="scan.pdf",
        content_type="application/pdf",
        payload=make_blank_pdf(),
    )

    with pytest.raises(RuntimeError, match="GLM-OCR request failed"):
        run_pdf_ingest_job(
            artifacts.job_id,
            session_factory=session_factory,
            extractor=PDFTextExtractor(ocr_backend=FailingOcrAdapter()),
        )

    job = db_session.get(Job, artifacts.job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "GLM-OCR request failed"
