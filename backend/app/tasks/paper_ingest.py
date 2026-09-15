import logging

from app.celery_app import celery_app
from app.services.papers import mark_job_failed, run_paper_question_set_job, run_paper_summary_job, run_pdf_ingest_job, set_job_celery_task_id

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.paper_ingest.process_uploaded_pdf", queue="extract")
def process_uploaded_pdf(self, job_id: int) -> None:
    delivery = getattr(self.request, "delivery_info", None) or {}
    routing_key = delivery.get("routing_key")
    if routing_key == "celery" and not celery_app.conf.task_always_eager:
        # 部署过渡期：仅转发真正落在旧 celery 队列的遗留消息到 extract。
        # 直接调用 / .apply() 时 routing_key 为 None；extract worker 消费时为 extract，均执行任务体。
        # celery 队列清空且确认无旧版本消息后，本守卫段可整体删除。
        logger.warning(
            "Forwarding leftover PDF ingest job %s from routing_key=%s to extract queue",
            job_id,
            routing_key,
        )
        try:
            forwarded = celery_app.send_task(
                "app.tasks.paper_ingest.process_uploaded_pdf",
                args=[job_id],
                queue="extract",
            )
        except Exception as exc:
            logger.exception(
                "Failed to forward PDF ingest job %s from routing_key=%s: %s",
                job_id,
                routing_key,
                exc,
            )
            mark_job_failed(job_id, exc)
            raise
        set_job_celery_task_id(job_id, getattr(forwarded, "id", None))
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
