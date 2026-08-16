import os

from app.celery_app import celery_app
from app.services.papers import run_paper_question_set_job, run_paper_summary_job, run_pdf_ingest_job, set_job_celery_task_id


@celery_app.task(name="app.tasks.paper_ingest.process_uploaded_pdf", queue="extract")
def process_uploaded_pdf(job_id: int) -> None:
    if os.environ.get("WORKER_ROLE") != "celery-extract":
        # 部署过渡期：旧消息遗留 celery 队列被主 worker 消费时，转交 extract 队列执行，
        # 避免 1.5g 主 worker 执行 PDF 提取 OOM。extract worker 上 WORKER_ROLE=celery-extract 不会触发。
        celery_app.send_task(
            "app.tasks.paper_ingest.process_uploaded_pdf",
            args=[job_id],
            queue="extract",
        )
        return
    summary_job_id = run_pdf_ingest_job(job_id)
    if summary_job_id is not None:
        task = process_paper_summary.delay(summary_job_id)
        set_job_celery_task_id(summary_job_id, getattr(task, "id", None))


@celery_app.task(name="app.tasks.paper_ingest.process_paper_summary")
def process_paper_summary(job_id: int) -> None:
    question_set_job_id = run_paper_summary_job(job_id)
    if question_set_job_id is not None:
        task = process_paper_question_set.delay(question_set_job_id)
        set_job_celery_task_id(question_set_job_id, getattr(task, "id", None))


@celery_app.task(name="app.tasks.paper_ingest.process_paper_question_set")
def process_paper_question_set(job_id: int) -> None:
    run_paper_question_set_job(job_id)
