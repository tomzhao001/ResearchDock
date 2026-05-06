import sys

from celery import Celery
from celery.signals import worker_process_init
from celery.utils.nodenames import gethostname

from app.config import settings

celery_app = Celery(
    "researchdock",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.paper_ingest"],
)

_IS_WINDOWS = sys.platform.startswith("win")

celery_config = {
    "task_always_eager": settings.celery_task_always_eager,
    "task_eager_propagates": True,
    "accept_content": ["json"],
    "task_serializer": "json",
    "result_serializer": "json",
    "worker_prefetch_multiplier": 1,
}
if _IS_WINDOWS:
    celery_config.update(
        worker_pool="solo",
        worker_concurrency=1,
    )

celery_app.conf.update(**celery_config)


@worker_process_init.connect
def _ensure_worker_trace_locals(**_kwargs) -> None:
    """Prefork pool on Windows uses spawn; child processes skip setup_worker_optimizations.

    Then trace._localized stays empty while use_fast_trace_task is True, and
    fast_trace_task crashes with ValueError unpacking _loc.

    Celery upstream only calls setup_worker_optimizations in the child when
    FORKED_BY_MULTIPROCESSING is set (fork path).
    """
    from celery.app import trace

    if len(trace._localized) >= 3:
        return
    trace.setup_worker_optimizations(celery_app, gethostname())
