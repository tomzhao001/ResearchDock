import fitz
import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from types import SimpleNamespace

from app.models import Job, OrganizationSettings, Paper, PaperAsset, PaperChunk
from app.services import llm
from app.routers import papers as papers_router
from app.services.document_extraction import ExtractedBlock, ExtractedDocument, ExtractedPage
from app.services.papers import run_paper_question_set_job, run_paper_summary_job, run_pdf_ingest_job
from app.services.vision.base import NoopPictureDescriptionAdapter


def make_summary_payload(
    *,
    abstract_cn: str = "这是一段中文摘要。",
    key_points: list[str] | None = None,
    research_question: str = "研究问题",
    method: str = "研究方法",
    findings: str = "主要发现",
    limitations: str = "局限性",
    authors: str = "Alice Example; Bob Example",
    doi: str = "10.1000/researchdock",
    source_url: str = "https://example.com/papers/researchdock",
    published_at: str = "2024-05-01",
) -> dict:
    return {
        "abstract_cn": abstract_cn,
        "key_points": key_points or ["要点一", "要点二", "要点三"],
        "research_question": research_question,
        "method": method,
        "findings": findings,
        "limitations": limitations,
        "authors": authors,
        "doi": doi,
        "source_url": source_url,
        "published_at": published_at,
    }


def make_question_set_payload(
    *,
    questions: list[dict[str, str]] | None = None,
    model_name: str = "test-question-set-model",
) -> dict:
    return {
        "generated_at": "2026-05-16T00:00:00Z",
        "model_name": model_name,
        "questions": questions
        or [
            {"id": "q1", "question": "这篇论文研究了什么？", "answer": "研究了一个测试问题。"},
            {"id": "q2", "question": "方法是什么？", "answer": "使用测试方法。"},
        ],
    }


def make_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    payload = doc.tobytes()
    doc.close()
    return payload


class SuccessfulTextExtractor:
    def extract(self, pdf_path):
        return ExtractedDocument(
            markdown_text="This is extracted directly from the embedded text layer.",
            metadata={
                "engine": "docling",
                "page_count": 1,
            },
            pages=[ExtractedPage(page_number=1, text="This is extracted directly from the embedded text layer.")],
            blocks=[
                ExtractedBlock(
                    block_index=0,
                    text="This is extracted directly from the embedded text layer.",
                    block_type="paragraph",
                    page_number=1,
                    section_path="Document",
                )
            ],
        )


def login(client) -> None:
    response = client.post("/api/auth/login", json={"username": "admin", "password": "123456"})
    assert response.status_code == 200


def login_as(client, *, username: str, password: str) -> None:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200


def set_org_question_set(db_session: Session, *, organization_id: int, questions: list[dict[str, str]]) -> None:
    settings = db_session.scalar(select(OrganizationSettings).where(OrganizationSettings.organization_id == organization_id))
    if settings is None:
        settings = OrganizationSettings(organization_id=organization_id, auto_extraction_questions_json=questions)
        db_session.add(settings)
    else:
        settings.auto_extraction_questions_json = questions
    db_session.commit()


def make_eager_upload_delay(
    session_factory: sessionmaker,
    *,
    extractor: object | None = None,
    swallow_errors: bool = True,
):
    def eager_delay(job_id: int):
        try:
            summary_job_id = run_pdf_ingest_job(
                job_id,
                session_factory=session_factory,
                extractor=extractor or SuccessfulTextExtractor(),
                picture_adapter=NoopPictureDescriptionAdapter(),
            )
            if summary_job_id is not None:
                question_set_job_id = run_paper_summary_job(summary_job_id, session_factory=session_factory)
                if question_set_job_id is not None:
                    run_paper_question_set_job(question_set_job_id, session_factory=session_factory)
        except Exception:
            if not swallow_errors:
                raise
        return None

    return eager_delay


def make_eager_summary_delay(session_factory: sessionmaker, *, swallow_errors: bool = True):
    def eager_delay(job_id: int):
        try:
            question_set_job_id = run_paper_summary_job(job_id, session_factory=session_factory)
            if question_set_job_id is not None:
                run_paper_question_set_job(question_set_job_id, session_factory=session_factory)
        except Exception:
            if not swallow_errors:
                raise
        return None

    return eager_delay


def make_eager_question_set_delay(session_factory: sessionmaker, *, swallow_errors: bool = True):
    def eager_delay(job_id: int):
        try:
            run_paper_question_set_job(job_id, session_factory=session_factory)
        except Exception:
            if not swallow_errors:
                raise
        return None

    return eager_delay


def make_task_result(task_id: str):
    return SimpleNamespace(id=task_id)


@pytest.fixture(autouse=True)
def mock_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.openai_api_key", "")
    monkeypatch.setattr("app.config.settings.glm_api_key", "")
    monkeypatch.setattr("app.config.settings.llm_provider", "openai")

    def fake_summary(_: str) -> dict:
        return make_summary_payload()

    monkeypatch.setattr(llm, "summarize_paper_text", fake_summary)
    monkeypatch.setattr("app.services.papers.summarize_paper_text", fake_summary)

    def fake_question_set(_: str, *, structured_summary: dict | None, questions: list[dict[str, str]]) -> dict:
        question_payload = []
        for item in questions:
            question_payload.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": f"回答：{item['question']}",
                }
            )
        return make_question_set_payload(questions=question_payload)

    monkeypatch.setattr(llm, "answer_question_set_questions", fake_question_set)
    monkeypatch.setattr("app.services.papers.answer_question_set_questions", fake_question_set)

    def fake_embeddings(texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(len(text.split()))] for text in texts]

    monkeypatch.setattr(llm, "embed_texts", fake_embeddings)
    monkeypatch.setattr("app.services.rag.embed_texts", fake_embeddings)


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
    organization,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    set_org_question_set(
        db_session,
        organization_id=organization.id,
        questions=[
            {"id": "q1", "question": "这篇论文研究了什么？"},
            {"id": "q2", "question": "方法是什么？"},
        ],
    )

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
    chunks = db_session.scalars(select(PaperChunk).where(PaperChunk.paper_id == paper.id)).all()

    assert job is not None
    assert job.status == "completed"
    assert paper is not None
    assert paper.status == "completed"
    assert paper.authors == "Alice Example; Bob Example"
    assert paper.doi == "10.1000/researchdock"
    assert paper.source_url == "https://example.com/papers/researchdock"
    assert paper.published_at is not None
    assert paper.published_at.date().isoformat() == "2024-05-01"
    assert asset is not None
    assert "embedded text layer" in (asset.raw_text or "")
    assert isinstance(asset.metadata_json, dict)
    assert asset.metadata_json["extraction"]["engine"] == "docling"
    assert asset.metadata_json["extraction"]["block_count"] >= 1
    assert asset.metadata_json["structured_summary"]["doi"] == "10.1000/researchdock"
    assert asset.metadata_json["question_set_extraction"]["questions"][0]["id"] == "q1"
    assert len(chunks) > 0
    assert chunks[0].content
    assert chunks[0].page_from == 1
    assert isinstance(chunks[0].metadata_json, dict)
    assert chunks[0].metadata_json["section_title"] in {"Front Matter", "Document"}
    assert chunks[0].metadata_json["context_header"]

    job_response = client.get(f"/api/jobs/{job.id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"


def test_paper_list_and_detail_return_preview_data(
    client,
    user,
    organization,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    set_org_question_set(
        db_session,
        organization_id=organization.id,
        questions=[{"id": "q1", "question": "这篇论文研究了什么？"}],
    )

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
    assert items[0]["authors"] == "Alice Example; Bob Example"
    assert "published_at" in items[0]
    assert items[0]["published_at"].startswith("2024-05-01")
    assert items[0]["ocr_status"] == "completed"
    assert items[0]["summary_status"] == "completed"
    assert items[0]["question_set_status"] == "completed"

    detail_response = client.get(f"/api/papers/{body['paper_id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert "embedded text layer" in detail["preview_text"]
    assert detail["original_filename"] == "milestone3.pdf"
    assert detail["structured_summary"]["method"] == "研究方法"
    assert detail["structured_summary"]["authors"] == "Alice Example; Bob Example"
    assert detail["authors"] == "Alice Example; Bob Example"
    assert detail["doi"] == "10.1000/researchdock"
    assert detail["source_url"] == "https://example.com/papers/researchdock"
    assert detail["published_at"].startswith("2024-05-01")
    assert detail["ocr_status"] == "completed"
    assert detail["summary_status"] == "completed"
    assert detail["question_set_status"] == "completed"
    assert detail["latest_ocr_job"]["job_type"] == "pdf_ingest"
    assert detail["latest_summary_job"]["job_type"] == "paper_summary"
    assert detail["latest_question_set_job"]["job_type"] == "paper_question_set"
    assert detail["question_set_extraction"]["questions"][0]["id"] == "q1"


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
    assert detail["question_set_status"] is None
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


def test_same_filename_can_exist_in_different_organizations(
    client,
    user,
    second_user,
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
            files={"file": ("shared.pdf", make_pdf_bytes("first org"), "application/pdf")},
        )
        login_as(client, username=second_user.username, password="654321")
        second = client.post(
            "/api/papers/upload",
            files={"file": ("shared.pdf", make_pdf_bytes("second org"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert first.status_code == 202
    assert second.status_code == 202


def test_papers_and_jobs_are_scoped_to_current_organization(
    client,
    user,
    second_user,
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
            files={"file": ("org-a.pdf", make_pdf_bytes("org a paper"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    payload = upload_response.json()

    login_as(client, username=second_user.username, password="654321")

    list_response = client.get("/api/papers")
    assert list_response.status_code == 200
    assert list_response.json()["items"] == []

    detail_response = client.get(f"/api/papers/{payload['paper_id']}")
    assert detail_response.status_code == 404

    jobs_response = client.get("/api/jobs")
    assert jobs_response.status_code == 200
    assert jobs_response.json()["items"] == []

    job_detail_response = client.get(f"/api/jobs/{payload['job_id']}")
    assert job_detail_response.status_code == 404


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
    job = db_session.get(Job, job_id)
    assert job is not None
    assert job.deleted_at is not None

    jobs_response = client.get("/api/jobs")
    assert jobs_response.status_code == 200
    assert jobs_response.json()["items"] == []


def test_delete_active_job_is_rejected(client, user, db_session: Session) -> None:
    login(client)

    paper = Paper(organization_id=user.organization_id, title="active paper", status="processing")
    db_session.add(paper)
    db_session.flush()
    job = Job(job_type="pdf_ingest", paper_id=paper.id, status="processing")
    db_session.add(job)
    db_session.commit()

    response = client.delete(f"/api/jobs/{job.id}")
    assert response.status_code == 409
    assert response.json()["detail"] == "Only completed, failed, or cancelled jobs can be deleted"


def test_delete_queued_job_is_rejected(client, user, db_session: Session) -> None:
    login(client)

    paper = Paper(organization_id=user.organization_id, title="queued paper", status="queued")
    db_session.add(paper)
    db_session.flush()
    job = Job(job_type="pdf_ingest", paper_id=paper.id, status="queued")
    db_session.add(job)
    db_session.commit()

    response = client.delete(f"/api/jobs/{job.id}")
    assert response.status_code == 409
    assert response.json()["detail"] == "Only completed, failed, or cancelled jobs can be deleted"


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
    jobs = db_session.scalars(select(Job).where(Job.paper_id == paper_id)).all()
    assert jobs
    assert all(job.deleted_at is not None for job in jobs)

    list_response = client.get("/api/papers")
    jobs_response = client.get("/api/jobs")
    assert list_response.status_code == 200
    assert jobs_response.status_code == 200
    assert list_response.json()["items"] == []
    assert jobs_response.json()["items"] == []


def test_upload_persists_celery_task_id(client, user, db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    login(client)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", lambda job_id: make_task_result(f"celery-{job_id}"))

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("task-id.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    db_session.expire_all()
    job = db_session.get(Job, upload_response.json()["job_id"])
    assert job is not None
    assert job.celery_task_id == f"celery-{job.id}"


def test_cancel_queued_job_marks_cancelled_and_revokes(client, user, db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    login(client)

    paper = Paper(organization_id=user.organization_id, title="queued paper", status="queued")
    db_session.add(paper)
    db_session.flush()
    job = Job(job_type="pdf_ingest", paper_id=paper.id, status="queued", celery_task_id="celery-queued")
    db_session.add(job)
    db_session.commit()

    revoke_calls: list[str] = []
    monkeypatch.setattr("app.services.papers.celery_app.control.revoke", lambda task_id: revoke_calls.append(task_id))

    response = client.post(f"/api/jobs/{job.id}/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cancelled"
    assert body["cancel_requested_at"] is not None

    db_session.refresh(job)
    db_session.refresh(paper)
    assert job.status == "cancelled"
    assert job.finished_at is not None
    assert paper.status == "cancelled"
    assert revoke_calls == ["celery-queued"]


def test_cancel_processing_job_marks_cancel_requested_and_revokes(
    client,
    user,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    login(client)

    paper = Paper(organization_id=user.organization_id, title="processing paper", status="processing")
    db_session.add(paper)
    db_session.flush()
    job = Job(job_type="pdf_ingest", paper_id=paper.id, status="processing", celery_task_id="celery-processing")
    db_session.add(job)
    db_session.commit()

    revoke_calls: list[str] = []
    monkeypatch.setattr("app.services.papers.celery_app.control.revoke", lambda task_id: revoke_calls.append(task_id))

    response = client.post(f"/api/jobs/{job.id}/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cancel_requested"
    assert body["cancel_requested_at"] is not None

    db_session.refresh(job)
    db_session.refresh(paper)
    assert job.status == "cancel_requested"
    assert paper.status == "cancel_requested"
    assert revoke_calls == ["celery-processing"]


def test_delete_paper_with_active_job_is_rejected(client, user, db_session: Session) -> None:
    login(client)

    paper = Paper(organization_id=user.organization_id, title="active paper", status="processing")
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
            files={"file": ("reparse-document.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
        assert upload_response.status_code == 202
        paper_id = upload_response.json()["paper_id"]

        rerun_response = client.post(f"/api/papers/{paper_id}/reparse-document")
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
    organization,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    set_org_question_set(
        db_session,
        organization_id=organization.id,
        questions=[{"id": "q1", "question": "摘要重跑后仍要回答的问题"}],
    )

    original_upload_delay = papers_router.process_uploaded_pdf.delay
    original_summary_delay = papers_router.process_paper_summary.delay

    def second_summary(_: str) -> dict:
        return make_summary_payload(
            abstract_cn="重新生成后的摘要。",
            key_points=["新要点"],
            research_question="新研究问题",
            method="新方法",
            findings="新发现",
            limitations="新局限",
            authors="Regenerated Author",
            doi="10.2000/regenerated",
            source_url="https://example.com/regenerated",
            published_at="2025-01-02",
        )

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
    assert body["authors"] == "Alice Example; Bob Example"
    assert body["doi"] == "10.1000/researchdock"
    assert body["latest_job"]["job_type"] == "paper_question_set"
    assert body["ocr_status"] == "completed"
    assert body["summary_status"] == "completed"
    assert body["question_set_status"] == "completed"


def test_updating_org_question_set_does_not_change_existing_paper_results(
    client,
    user,
    organization,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    set_org_question_set(
        db_session,
        organization_id=organization.id,
        questions=[{"id": "q1", "question": "旧问题"}],
    )

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("question-set-stable.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    paper_id = upload_response.json()["paper_id"]

    set_org_question_set(
        db_session,
        organization_id=organization.id,
        questions=[{"id": "q2", "question": "新问题"}],
    )

    detail_response = client.get(f"/api/papers/{paper_id}")
    assert detail_response.status_code == 200
    body = detail_response.json()
    assert [item["id"] for item in body["question_set_extraction"]["questions"]] == ["q1"]


def test_regenerate_question_set_uses_latest_organization_questions(
    client,
    user,
    organization,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    set_org_question_set(
        db_session,
        organization_id=organization.id,
        questions=[{"id": "q1", "question": "旧问题"}],
    )

    original_upload_delay = papers_router.process_uploaded_pdf.delay
    original_question_set_delay = papers_router.process_paper_question_set.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("question-set-rerun.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
        assert upload_response.status_code == 202
        paper_id = upload_response.json()["paper_id"]
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_upload_delay)

    before_detail = client.get(f"/api/papers/{paper_id}")
    assert before_detail.status_code == 200
    assert [item["id"] for item in before_detail.json()["question_set_extraction"]["questions"]] == ["q1"]

    set_org_question_set(
        db_session,
        organization_id=organization.id,
        questions=[{"id": "q2", "question": "新问题"}],
    )

    monkeypatch.setattr(
        papers_router.process_paper_question_set,
        "delay",
        make_eager_question_set_delay(session_factory),
    )

    try:
        regenerate_response = client.post(f"/api/papers/{paper_id}/regenerate-question-set")
    finally:
        monkeypatch.setattr(papers_router.process_paper_question_set, "delay", original_question_set_delay)

    assert regenerate_response.status_code == 202
    detail_response = client.get(f"/api/papers/{paper_id}")
    assert detail_response.status_code == 200
    body = detail_response.json()
    assert [item["id"] for item in body["question_set_extraction"]["questions"]] == ["q2"]
    assert body["question_set_status"] == "completed"
    assert body["latest_question_set_job"]["job_type"] == "paper_question_set"


def test_summary_invalid_metadata_keeps_paper_fields_empty(
    client,
    user,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")

    def invalid_summary(_: str) -> dict:
        return make_summary_payload(
            authors="   ",
            doi="doi: not-a-doi",
            source_url="ftp://example.com/paper",
            published_at="not-a-date",
        )

    monkeypatch.setattr(llm, "summarize_paper_text", invalid_summary)
    monkeypatch.setattr("app.services.papers.summarize_paper_text", invalid_summary)

    original_delay = papers_router.process_uploaded_pdf.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("invalid-metadata.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_delay)

    assert upload_response.status_code == 202
    detail_response = client.get(f"/api/papers/{upload_response.json()['paper_id']}")
    assert detail_response.status_code == 200
    body = detail_response.json()
    assert body["authors"] is None
    assert body["doi"] is None
    assert body["source_url"] is None
    assert body["published_at"] is None
    assert body["structured_summary"]["authors"] == ""
    assert body["structured_summary"]["doi"] == ""
    assert body["structured_summary"]["source_url"] == ""
    assert body["structured_summary"]["published_at"] == ""


def test_regenerate_summary_does_not_override_manual_metadata(
    client,
    user,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> None:
    login(client)
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")

    original_upload_delay = papers_router.process_uploaded_pdf.delay
    original_summary_delay = papers_router.process_paper_summary.delay
    monkeypatch.setattr(
        papers_router.process_uploaded_pdf,
        "delay",
        make_eager_upload_delay(session_factory, extractor=SuccessfulTextExtractor()),
    )

    try:
        upload_response = client.post(
            "/api/papers/upload",
            files={"file": ("manual-metadata.pdf", make_pdf_bytes("paper body"), "application/pdf")},
        )
        assert upload_response.status_code == 202
        paper_id = upload_response.json()["paper_id"]
    finally:
        monkeypatch.setattr(papers_router.process_uploaded_pdf, "delay", original_upload_delay)

    manual_update = client.patch(
        f"/api/papers/{paper_id}",
        json={
            "authors": "Manual Author",
            "doi": "10.3000/manual",
            "source_url": "https://example.com/manual",
            "published_at": "2023-12-31T00:00:00Z",
        },
    )
    assert manual_update.status_code == 200

    def regenerated_summary(_: str) -> dict:
        return make_summary_payload(
            abstract_cn="人工信息保护后的摘要。",
            authors="Another Extracted Author",
            doi="10.4000/extracted",
            source_url="https://example.com/extracted",
            published_at="2025-06-01",
        )

    monkeypatch.setattr(llm, "summarize_paper_text", regenerated_summary)
    monkeypatch.setattr("app.services.papers.summarize_paper_text", regenerated_summary)
    monkeypatch.setattr(papers_router.process_paper_summary, "delay", make_eager_summary_delay(session_factory))

    try:
        regenerate_response = client.post(f"/api/papers/{paper_id}/regenerate-summary")
    finally:
        monkeypatch.setattr(papers_router.process_paper_summary, "delay", original_summary_delay)

    assert regenerate_response.status_code == 202
    detail_response = client.get(f"/api/papers/{paper_id}")
    assert detail_response.status_code == 200
    body = detail_response.json()
    assert body["abstract_raw"] == "人工信息保护后的摘要。"
    assert body["structured_summary"]["authors"] == "Another Extracted Author"
    assert body["structured_summary"]["doi"] == "10.4000/extracted"
    assert body["authors"] == "Manual Author"
    assert body["doi"] == "10.3000/manual"
    assert body["source_url"] == "https://example.com/manual"
    assert body["published_at"].startswith("2023-12-31")
