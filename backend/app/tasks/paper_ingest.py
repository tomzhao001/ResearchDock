from app.celery_app import celery_app
from app.services.papers import run_paper_question_set_job, run_paper_summary_job, run_pdf_ingest_job, set_job_celery_task_id


@celery_app.task(bind=True, name="app.tasks.paper_ingest.process_uploaded_pdf", queue="extract")
def process_uploaded_pdf(self, job_id: int) -> None:
    delivery = getattr(self.request, "delivery_info", None) or {}
    routing_key = delivery.get("routing_key")
    if routing_key != "extract" and not celery_app.conf.task_always_eager:
        # 部署过渡期：消息来自非 extract 队列（如旧版本遗留 celery 队列）时转交 extract 执行；
        # extract worker 消费时 routing_key 为 extract 不会触发；eager 模式直接执行避免递归。
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
