from __future__ import annotations

import logging
from datetime import datetime, timezone

from redis import Redis

from app.config import settings
from app.schemas import ChatProgressEvent

logger = logging.getLogger(__name__)
_publisher: Redis | None = None


def chat_progress_channel(*, user_id: int, topic_id: int) -> str:
    return f"researchdock:chat-progress:{user_id}:{topic_id}"


def _get_publisher() -> Redis:
    global _publisher
    if _publisher is None:
        _publisher = Redis.from_url(settings.redis_url, decode_responses=True)
    return _publisher


def publish_chat_progress_event(
    *,
    user_id: int,
    topic_id: int,
    phase: str,
    status: str,
    message: str,
    detail: str | None = None,
) -> None:
    event = ChatProgressEvent(
        topic_id=topic_id,
        phase=phase,
        status=status,
        message=message,
        detail=detail,
        created_at=datetime.now(timezone.utc),
    )
    try:
        _get_publisher().publish(
            chat_progress_channel(user_id=user_id, topic_id=topic_id),
            event.model_dump_json(),
        )
    except Exception:
        logger.exception("Failed to publish chat progress for topic %s", topic_id)
