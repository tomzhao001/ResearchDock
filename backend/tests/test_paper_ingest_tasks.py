from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job, Paper
from app.services.papers import create_upload_artifacts, mark_job_failed, set_job_celery_task_id
from app.tasks.paper_ingest import process_paper_summary, process_uploaded_pdf


def make_pdf_payload() -> bytes:
    return b"%PDF-1.7\n% fake test payload\n"


@pytest.fixture(autouse=True)
def _disable_task_status_publish(monkeypatch) -> None:
    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr("app.services.papers.publish_task_status_event", noop)
    monkeypatch.setattr("app.services.task_events.publish_task_status_event", noop)


def _run_uploaded_pdf_with_routing_key(job_id: int, routing_key: str | None) -> None:
    delivery_info = {"routing_key": routing_key} if routing_key is not None else None
    process_uploaded_pdf.push_request(delivery_info=delivery_info)
    try:
        process_uploaded_pdf.run(job_id)
    finally:
        process_uploaded_pdf.pop_request()


def test_process_uploaded_pdf_persists_summary_celery_task_id(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
    monkeypatch,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="task-wrapper.pdf",
        content_type="application/pdf",
        payload=make_pdf_payload(),
    )
    summary_job = Job(job_type="paper_summary", paper_id=artifacts.paper_id, status="queued")
    db_session.add(summary_job)
    db_session.commit()
    db_session.refresh(summary_job)

    send_task_calls: list[tuple] = []
    monkeypatch.setattr("app.tasks.paper_ingest.run_pdf_ingest_job", lambda job_id: summary_job.id)
    monkeypatch.setattr(
        "app.tasks.paper_ingest.process_paper_summary.delay",
        lambda job_id: SimpleNamespace(id=f"summary-task-{job_id}"),
    )
    monkeypatch.setattr(
        "app.tasks.paper_ingest.set_job_celery_task_id",
        lambda job_id, task_id: set_job_celery_task_id(job_id, task_id, session_factory=session_factory),
    )
    monkeypatch.setattr(
        "app.tasks.paper_ingest.celery_app.send_task",
        lambda *args, **kwargs: send_task_calls.append((args, kwargs)) or SimpleNamespace(id="should-not-forward"),
    )

    process_uploaded_pdf(artifacts.job_id)

    assert send_task_calls == []
    db_session.expire_all()
    persisted_job = db_session.get(Job, summary_job.id)
    assert persisted_job is not None
    assert persisted_job.celery_task_id == f"summary-task-{summary_job.id}"


def test_process_uploaded_pdf_does_not_forward_when_routing_key_is_extract(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
    monkeypatch,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="extract-queue.pdf",
        content_type="application/pdf",
        payload=make_pdf_payload(),
    )

    ingest_calls: list[int] = []
    send_task_calls: list[tuple] = []
    monkeypatch.setattr(
        "app.tasks.paper_ingest.run_pdf_ingest_job",
        lambda job_id: ingest_calls.append(job_id) or None,
    )
    monkeypatch.setattr(
        "app.tasks.paper_ingest.celery_app.send_task",
        lambda *args, **kwargs: send_task_calls.append((args, kwargs)) or SimpleNamespace(id="should-not-forward"),
    )
    monkeypatch.setattr(process_uploaded_pdf.app.conf, "task_always_eager", False)

    _run_uploaded_pdf_with_routing_key(artifacts.job_id, "extract")

    assert ingest_calls == [artifacts.job_id]
    assert send_task_calls == []


def test_process_uploaded_pdf_forwards_legacy_celery_queue_and_persists_task_id(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
    monkeypatch,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="legacy-celery-queue.pdf",
        content_type="application/pdf",
        payload=make_pdf_payload(),
    )

    ingest_calls: list[int] = []
    send_task_calls: list[tuple] = []
    monkeypatch.setattr(
        "app.tasks.paper_ingest.run_pdf_ingest_job",
        lambda job_id: ingest_calls.append(job_id) or None,
    )
    monkeypatch.setattr(
        "app.tasks.paper_ingest.set_job_celery_task_id",
        lambda job_id, task_id: set_job_celery_task_id(job_id, task_id, session_factory=session_factory),
    )
    monkeypatch.setattr(
        "app.tasks.paper_ingest.celery_app.send_task",
        lambda *args, **kwargs: send_task_calls.append((args, kwargs)) or SimpleNamespace(id="forwarded-extract-task"),
    )
    monkeypatch.setattr(process_uploaded_pdf.app.conf, "task_always_eager", False)

    _run_uploaded_pdf_with_routing_key(artifacts.job_id, "celery")

    assert ingest_calls == []
    assert send_task_calls == [
        (
            ("app.tasks.paper_ingest.process_uploaded_pdf",),
            {"args": [artifacts.job_id], "queue": "extract"},
        )
    ]
    db_session.expire_all()
    persisted_job = db_session.get(Job, artifacts.job_id)
    assert persisted_job is not None
    assert persisted_job.status == "queued"
    assert persisted_job.celery_task_id == "forwarded-extract-task"


def test_process_uploaded_pdf_marks_job_failed_when_forward_raises(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
    monkeypatch,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="forward-failure.pdf",
        content_type="application/pdf",
        payload=make_pdf_payload(),
    )

    ingest_calls: list[int] = []
    monkeypatch.setattr(
        "app.tasks.paper_ingest.run_pdf_ingest_job",
        lambda job_id: ingest_calls.append(job_id) or None,
    )

    monkeypatch.setattr(
        "app.tasks.paper_ingest.mark_job_failed",
        lambda job_id, error: mark_job_failed(job_id, error, session_factory=session_factory),
    )
    monkeypatch.setattr(
        "app.tasks.paper_ingest.celery_app.send_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("broker unavailable")),
    )
    monkeypatch.setattr(process_uploaded_pdf.app.conf, "task_always_eager", False)

    with pytest.raises(ConnectionError, match="broker unavailable"):
        _run_uploaded_pdf_with_routing_key(artifacts.job_id, "celery")

    assert ingest_calls == []
    db_session.expire_all()
    persisted_job = db_session.get(Job, artifacts.job_id)
    persisted_paper = db_session.get(Paper, artifacts.paper_id)
    assert persisted_job is not None
    assert persisted_job.status == "failed"
    assert persisted_job.error_message == "broker unavailable"
    assert persisted_paper is not None
    assert persisted_paper.status == "failed"


def test_process_paper_summary_persists_question_set_celery_task_id(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
    monkeypatch,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="task-wrapper-summary.pdf",
        content_type="application/pdf",
        payload=make_pdf_payload(),
    )
    summary_job = Job(job_type="paper_summary", paper_id=artifacts.paper_id, status="queued")
    question_set_job = Job(job_type="paper_question_set", paper_id=artifacts.paper_id, status="queued")
    db_session.add(summary_job)
    db_session.add(question_set_job)
    db_session.commit()
    db_session.refresh(summary_job)
    db_session.refresh(question_set_job)

    monkeypatch.setattr("app.tasks.paper_ingest.run_paper_summary_job", lambda job_id: question_set_job.id)
    monkeypatch.setattr(
        "app.tasks.paper_ingest.process_paper_question_set.delay",
        lambda job_id: SimpleNamespace(id=f"question-set-task-{job_id}"),
    )
    monkeypatch.setattr(
        "app.tasks.paper_ingest.set_job_celery_task_id",
        lambda job_id, task_id: set_job_celery_task_id(job_id, task_id, session_factory=session_factory),
    )

    process_paper_summary(summary_job.id)

    db_session.expire_all()
    persisted_job = db_session.get(Job, question_set_job.id)
    assert persisted_job is not None
    assert persisted_job.celery_task_id == f"question-set-task-{question_set_job.id}"
