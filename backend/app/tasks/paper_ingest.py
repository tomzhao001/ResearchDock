from app.celery_app import celery_app
from app.services.papers import run_pdf_ingest_job


@celery_app.task(name="app.tasks.paper_ingest.process_uploaded_pdf")
def process_uploaded_pdf(job_id: int) -> None:
    run_pdf_ingest_job(job_id)
