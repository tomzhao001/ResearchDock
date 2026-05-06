import fitz
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job, Paper, PaperAsset
from app.routers import papers as papers_router
from app.services.papers import run_pdf_ingest_job
from app.services.pdf_extraction import DocumentExtractionResult, PDFTextExtractor


def make_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    payload = doc.tobytes()
    doc.close()
    return payload


class SuccessfulTextExtractor(PDFTextExtractor):
    def extract(self, pdf_path):
        return DocumentExtractionResult(
            raw_text="This is extracted directly from the embedded text layer.",
            metadata={
                "page_count": 1,
                "used_ocr_pages": [],
                "pages": [
                    {
                        "page_number": 1,
                        "char_count": 512,
                        "alpha_ratio": 0.92,
                        "continuous_line_ratio": 0.7,
                        "image_count": 0,
                        "suspected_double_column": False,
                        "needs_ocr": False,
                        "used_ocr": False,
                        "reasons": [],
                    }
                ],
            },
        )


def login(client) -> None:
    response = client.post("/api/auth/login", json={"username": "admin", "password": "123456"})
    assert response.status_code == 200


def test_upload_requires_auth(client, user) -> None:
    response = client.post(
        "/api/papers/upload",
        files={"file": ("sample.pdf", make_pdf_bytes("hello"), "application/pdf")},
    )

    assert response.status_code == 401


def test_upload_rejects_non_pdf(client, user) -> None:
    login(client)
    response = client.post(
        "/api/papers/upload",
        files={"file": ("sample.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only PDF files are supported"}


def test_upload_creates_records_and_completes_job(
    client,
    user,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay

    def eager_delay(job_id: int):
        run_pdf_ingest_job(
            job_id,
            session_factory=session_factory,
            extractor=SuccessfulTextExtractor(),
        )
        return None

    monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", eager_delay)

    try:
        response = client.post(
            "/api/papers/upload",
            files={
                "file": (
                    "research-paper.pdf",
                    make_pdf_bytes("This is a long enough paragraph to remain in the text layer."),
                    "application/pdf",
                )
            },
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "research-paper.pdf"

    job = db_session.get(Job, body["job_id"])
    paper = db_session.get(Paper, body["paper_id"])
    asset = db_session.scalar(select(PaperAsset).where(PaperAsset.paper_id == paper.id))

    assert job is not None
    assert job.status == "completed"
    assert paper is not None
    assert paper.status == "completed"
    assert asset is not None
    assert "embedded text layer" in (asset.raw_text or "")

    job_response = client.get(f"/api/jobs/{job.id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"
