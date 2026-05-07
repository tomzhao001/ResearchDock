from app.celery_app import celery_app
from app.services.papers import run_paper_summary_job, run_pdf_ingest_job


@celery_app.task(name="app.tasks.paper_ingest.process_uploaded_pdf")
def process_uploaded_pdf(job_id: int) -> None:
    run_pdf_ingest_job(job_id)


@celery_app.task(name="app.tasks.paper_ingest.process_paper_summary")
def process_paper_summary(job_id: int) -> None:
    run_paper_summary_job(job_id)
