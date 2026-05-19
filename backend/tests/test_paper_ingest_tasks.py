from types import SimpleNamespace

from sqlalchemy.orm import Session, sessionmaker

from app.models import Job
from app.services.papers import create_upload_artifacts, set_job_celery_task_id
from app.tasks.paper_ingest import process_paper_summary, process_uploaded_pdf


def make_pdf_payload() -> bytes:
    return b"%PDF-1.7\n% fake test payload\n"


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

    monkeypatch.setattr("app.tasks.paper_ingest.run_pdf_ingest_job", lambda job_id: summary_job.id)
    monkeypatch.setattr(
        "app.tasks.paper_ingest.process_paper_summary.delay",
        lambda job_id: SimpleNamespace(id=f"summary-task-{job_id}"),
    )
    monkeypatch.setattr(
        "app.tasks.paper_ingest.set_job_celery_task_id",
        lambda job_id, task_id: set_job_celery_task_id(job_id, task_id, session_factory=session_factory),
    )

    process_uploaded_pdf(artifacts.job_id)

    db_session.expire_all()
    persisted_job = db_session.get(Job, summary_job.id)
    assert persisted_job is not None
    assert persisted_job.celery_task_id == f"summary-task-{summary_job.id}"


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
