import fitz
import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job, Paper, PaperAsset
from app.services import llm
from app.routers import papers as papers_router
from app.services.papers import run_paper_summary_job, run_pdf_ingest_job
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


def make_eager_upload_delay(
    session_factory: sessionmaker,
    *,
    extractor: PDFTextExtractor | None = None,
    swallow_errors: bool = True,
):
    def eager_delay(job_id: int):
        try:
            summary_job_id = run_pdf_ingest_job(
                job_id,
                session_factory=session_factory,
                extractor=extractor or SuccessfulTextExtractor(),
            )
            if summary_job_id is not None:
                run_paper_summary_job(summary_job_id, session_factory=session_factory)
        except Exception:
            if not swallow_errors:
                raise
        return None

    return eager_delay


def make_eager_summary_delay(session_factory: sessionmaker, *, swallow_errors: bool = True):
    def eager_delay(job_id: int):
        try:
            run_paper_summary_job(job_id, session_factory=session_factory)
        except Exception:
            if not swallow_errors:
                raise
        return None

    return eager_delay


@pytest.fixture(autouse=True)
def mock_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.openai_api_key", "")

    def fake_summary(_: str) -> dict:
        return {
            "abstract_cn": "这是一段中文摘要。",
            "key_points": ["要点一", "要点二", "要点三"],
            "research_question": "研究问题",
            "method": "研究方法",
            "findings": "主要发现",
            "limitations": "局限性",
        }

    monkeypatch.setattr(llm, "summarize_paper_text", fake_summary)
    monkeypatch.setattr("app.services.papers.summarize_paper_text", fake_summary)


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
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

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


def test_paper_list_and_detail_return_preview_data(
    client,
    user,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={
                "file": (
                    "milestone3.pdf",
                    make_pdf_bytes("Milestone 3 paper body."),
                    "application/pdf",
                )
            },
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    body = upload_response.json()

    list_response = client.get("/api/papers")
    assert list_response.status_code == 200
    items = list_response.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == body["paper_id"]
    assert items[0]["abstract_raw"] == "这是一段中文摘要。"
    assert items[0]["ocr_status"] == "completed"
    assert items[0]["summary_status"] == "completed"

    detail_response = client.get(f"/api/papers/{body['paper_id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert "embedded text layer" in detail["preview_text"]
    assert detail["original_filename"] == "milestone3.pdf"
    assert detail["structured_summary"]["method"] == "研究方法"
    assert detail["ocr_status"] == "completed"
    assert detail["summary_status"] == "completed"
    assert detail["latest_ocr_job"]["job_type"] == "pdf_ingest"
    assert detail["latest_summary_job"]["job_type"] == "paper_summary"


def test_summary_timeout_keeps_ocr_result_and_marks_only_summary_failed(
    client,
    user,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")

    original_delay = papers_router.process_uploaded_pdf.delay

    def timeout_summary(_: str) -> dict:
        raise httpx.ReadTimeout("The read operation timed out")

    monkeypatch.setattr(llm, "summarize_paper_text", timeout_summary)
    monkeypatch.setattr("app.services.papers.summarize_paper_text", timeout_summary)
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("timeout-summary.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    paper_id = upload_response.json()["paper_id"]

    detail_response = client.get(f"/api/papers/{paper_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert "embedded text layer" in detail["preview_text"]
    assert detail["ocr_status"] == "completed"
    assert detail["summary_status"] == "failed"
    assert detail["latest_ocr_job"]["status"] == "completed"
    assert detail["latest_summary_job"]["status"] == "failed"
    assert "timed out" in (detail["latest_summary_job"]["error_message"] or "")


def test_upload_rejects_duplicate_filename_without_overwrite(
    client,
    user,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        first = client.post(
            "/api/papers/upload",
            files={"file": ("duplicate.pdf", make_pdf_bytes("first"), "application/pdf")},
        )
        second = client.post(
            "/api/papers/upload",
            files={"file": ("duplicate.pdf", make_pdf_bytes("second"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["detail"]["filename"] == "duplicate.pdf"

    list_response = client.get("/api/papers")
    assert list_response.status_code == 200
    items = list_response.json()["items"]
    assert len(items) == 1
    assert items[0]["original_filename"] == "duplicate.pdf"


def test_upload_overwrite_soft_deletes_previous_paper(
    client,
    user,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        first = client.post(
            "/api/papers/upload",
            files={"file": ("replace.pdf", make_pdf_bytes("first"), "application/pdf")},
        )
        second = client.post(
            "/api/papers/upload",
            files={"file": ("replace.pdf", make_pdf_bytes("second"), "application/pdf"), "overwrite": (None, "true")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert first.status_code == 202
    assert second.status_code == 202

    first_paper = db_session.get(Paper, first.json()["paper_id"])
    second_paper = db_session.get(Paper, second.json()["paper_id"])
    assert first_paper is not None
    assert second_paper is not None
    assert first_paper.deleted_at is not None
    assert second_paper.deleted_at is None

    list_response = client.get("/api/papers")
    assert list_response.status_code == 200
    items = list_response.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == second.json()["paper_id"]


def test_patch_paper_title_allows_duplicate_display_name(
    client,
    user,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        first = client.post(
            "/api/papers/upload",
            files={"file": ("rename-a.pdf", make_pdf_bytes("first"), "application/pdf")},
        )
        second = client.post(
            "/api/papers/upload",
            files={"file": ("rename-b.pdf", make_pdf_bytes("second"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert first.status_code == 202
    assert second.status_code == 202

    first_title = client.patch(f"/api/papers/{first.json()['paper_id']}", json={"title": "统一展示名"})
    second_title = client.patch(f"/api/papers/{second.json()['paper_id']}", json={"title": "统一展示名"})

    assert first_title.status_code == 200
    assert second_title.status_code == 200
    assert first_title.json()["title"] == "统一展示名"
    assert second_title.json()["title"] == "统一展示名"


def test_patch_paper_metadata_updates_basic_fields(
    client,
    user,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("metadata.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    paper_id = upload_response.json()["paper_id"]

    update_response = client.patch(
        f"/api/papers/{paper_id}",
        json={
            "title": "更新后的文档名",
            "authors": "Alice; Bob",
            "doi": "10.1000/example",
            "source_url": "https://example.com/paper",
            "published_at": "2024-05-01T00:00:00Z",
        },
    )

    assert update_response.status_code == 200
    body = update_response.json()
    assert body["title"] == "更新后的文档名"
    assert body["authors"] == "Alice; Bob"
    assert body["doi"] == "10.1000/example"
    assert body["source_url"] == "https://example.com/paper"
    assert body["published_at"].startswith("2024-05-01")


def test_delete_completed_job(
    client,
    user,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("delete-job.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    job_id = upload_response.json()["job_id"]

    delete_response = client.delete(f"/api/jobs/{job_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"message": "Job deleted"}
    assert db_session.get(Job, job_id) is None


def test_delete_active_job_is_rejected(client, user, db_session: Session) -> None:
    login(client)

    paper = Paper(title="active paper", status="processing")
    db_session.add(paper)
    db_session.flush()
    job = Job(job_type="pdf_ingest", paper_id=paper.id, status="processing")
    db_session.add(job)
    db_session.commit()

    response = client.delete(f"/api/jobs/{job.id}")
    assert response.status_code == 409
    assert response.json()["detail"] == "Processing jobs cannot be deleted"


def test_delete_queued_job_is_allowed(client, user, db_session: Session) -> None:
    login(client)

    paper = Paper(title="queued paper", status="queued")
    db_session.add(paper)
    db_session.flush()
    job = Job(job_type="pdf_ingest", paper_id=paper.id, status="queued")
    db_session.add(job)
    db_session.commit()

    response = client.delete(f"/api/jobs/{job.id}")
    assert response.status_code == 200
    assert response.json() == {"message": "Job deleted"}


def test_delete_paper_soft_deletes_and_removes_jobs(
    client,
    user,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("delete-paper.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    paper_id = upload_response.json()["paper_id"]

    delete_response = client.delete(f"/api/papers/{paper_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"message": "Paper deleted"}

    paper = db_session.get(Paper, paper_id)
    assert paper is not None
    assert paper.deleted_at is not None
    assert db_session.scalars(select(Job).where(Job.paper_id == paper_id)).all() == []

    list_response = client.get("/api/papers")
    jobs_response = client.get("/api/jobs")
    assert list_response.status_code == 200
    assert jobs_response.status_code == 200
    assert list_response.json()["items"] == []
    assert jobs_response.json()["items"] == []


def test_delete_paper_with_active_job_is_rejected(client, user, db_session: Session) -> None:
    login(client)

    paper = Paper(title="active paper", status="processing")
    db_session.add(paper)
    db_session.flush()
    job = Job(job_type="pdf_ingest", paper_id=paper.id, status="queued")
    db_session.add(job)
    db_session.commit()

    response = client.delete(f"/api/papers/{paper.id}")
    assert response.status_code == 409
    assert response.json()["detail"] == "Papers with active jobs cannot be deleted"


def test_rerun_ocr_creates_new_job(
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
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("rerun-ocr.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
        assert upload_response.status_code == 202
        paper_id = upload_response.json()["paper_id"]

        rerun_response = client.post(f"/api/papers/{paper_id}/rerun-ocr")
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert rerun_response.status_code == 202
    rerun_job_id = rerun_response.json()["job_id"]
    rerun_job = db_session.get(Job, rerun_job_id)
    assert rerun_job is not None
    assert rerun_job.job_type == "pdf_ingest"
    assert rerun_job.status == "completed"
    assert len(db_session.scalars(select(Job).where(Job.paper_id == paper_id)).all()) == 2


def test_regenerate_summary_creates_summary_job_without_clearing_ocr_text(
    client,
    user,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")

    original_upload_delay = papers_router.process_uploaded_pdf.delay
    original_summary_delay = papers_router.process_paper_summary.delay

    def second_summary(_: str) -> dict:
        return {
            "abstract_cn": "重新生成后的摘要。",
            "key_points": ["新要点"],
            "research_question": "新研究问题",
            "method": "新方法",
            "findings": "新发现",
            "limitations": "新局限",
        }

    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("summary.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
        assert upload_response.status_code == 202
        paper_id = upload_response.json()["paper_id"]
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_upload_delay)

    before_detail = client.get(f"/api/papers/{paper_id}")
    assert before_detail.status_code == 200
    preview_text = before_detail.json()["preview_text"]

    monkeypatch.setattr(llm, "summarize_paper_text", second_summary)
    monkeypatch.setattr("app.services.papers.summarize_paper_text", second_summary)
    monkeypatch.setattr(papers_router.process_paper_summary, "delay", make_eager_summary_delay(session_factory))

    try:
        regenerate_response = client.post(f"/api/papers/{paper_id}/regenerate-summary")
    finally:
        monkeypatch.setattr(papers_router.process_paper_summary, "delay", original_summary_delay)

    assert regenerate_response.status_code == 202

    after_detail = client.get(f"/api/papers/{paper_id}")
    assert after_detail.status_code == 200
    body = after_detail.json()
    assert body["preview_text"] == preview_text
    assert body["abstract_raw"] == "重新生成后的摘要。"
    assert body["structured_summary"]["method"] == "新方法"
    assert body["latest_job"]["job_type"] == "paper_summary"
    assert body["ocr_status"] == "completed"
    assert body["summary_status"] == "completed"
