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
            text="\uff34\uff41\uff42\uff4c\uff45\u3000\uff11\uff1a\u3000\uff21\uff24\uff28\uff24\uff0d\uff32\uff33\u3000\uff30\uff24\uff26",
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


def make_text_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    payload = doc.tobytes()
    doc.close()
    return payload


def test_worker_persists_ocr_fallback_result(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
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
    assert asset.raw_text == "Table 1: ADHD-RS PDF"
    assert asset.metadata_json is not None
    assert asset.metadata_json["extraction"]["used_ocr_pages"] == [1]
    assert asset.metadata_json["extraction"]["pages"][0]["ocr_metadata"]["provider"] == "glm_ocr"
    assert asset.metadata_json["extraction"]["pages"][0]["ocr_metadata"]["text_quality"]["normalization_applied"] is True
    assert asset.metadata_json["extraction"]["text_quality"]["normalization_applied"] is True
    assert asset.metadata_json["extraction"]["text_quality"]["normalized_ocr_pages"] == [1]
    assert len(chunks) == 1
    assert "Table 1: ADHD-RS PDF" in chunks[0].content


def test_worker_keeps_existing_text_layer_when_full_document_ocr_disabled(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.settings.ocr_min_chars_per_page", 1)
    monkeypatch.setattr("app.config.settings.ocr_min_alpha_ratio", 0.0)
    monkeypatch.setattr("app.config.settings.ocr_min_average_chars_per_page", 1)
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="embedded-text.pdf",
        content_type="application/pdf",
        payload=make_text_pdf("This embedded text layer should remain preferred when OCR fallback is disabled. " * 6),
    )

    run_pdf_ingest_job(
        artifacts.job_id,
        session_factory=session_factory,
        extractor=PDFTextExtractor(ocr_backend=MockOcrAdapter(), force_full_document_ocr=False),
    )

    asset = db_session.scalar(select(PaperAsset).where(PaperAsset.paper_id == artifacts.paper_id))

    assert asset is not None
    assert "This embedded text layer should remain preferred" in (asset.raw_text or "")
    assert asset.metadata_json is not None
    assert asset.metadata_json["extraction"]["force_full_document_ocr"] is False
    assert asset.metadata_json["extraction"]["used_ocr_pages"] == []


def test_worker_marks_job_failed_when_glm_ocr_errors(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
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


def test_worker_force_full_document_ocr_uses_ocr_even_with_embedded_text(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="embedded-text.pdf",
        content_type="application/pdf",
        payload=make_text_pdf("This text layer would normally be extracted directly."),
    )

    run_pdf_ingest_job(
        artifacts.job_id,
        session_factory=session_factory,
        extractor=PDFTextExtractor(
            ocr_backend=MockOcrAdapter(),
            force_full_document_ocr=True,
        ),
    )

    asset = db_session.scalar(select(PaperAsset).where(PaperAsset.paper_id == artifacts.paper_id))

    assert asset is not None
    assert asset.raw_text == "Table 1: ADHD-RS PDF"
    assert asset.metadata_json is not None
    assert asset.metadata_json["extraction"]["force_full_document_ocr"] is True
    assert asset.metadata_json["extraction"]["used_ocr_pages"] == [1]
    assert asset.metadata_json["extraction"]["pages"][0]["reasons"] == ["full_document_ocr_enabled"]
