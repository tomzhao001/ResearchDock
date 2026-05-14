import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import User
from app.schemas import (
    ChatMessageCreateRequest,
    ChatMessageListResponse,
    ChatMessagePublic,
    ChatTopicCreateRequest,
    ChatTopicListResponse,
    ChatTopicPublic,
    ChatTurnResponse,
)
from app.services.chat_events import publish_chat_progress_event
from app.services.rag import (
    build_topic_assistant_draft,
    create_topic,
    get_topic,
    list_topic_messages,
    list_topics,
    persist_topic_assistant_message,
    start_topic_message,
    send_topic_message,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _serialize_topic(topic_summary) -> ChatTopicPublic:
    return ChatTopicPublic(
        id=topic_summary.topic.id,
        title=topic_summary.topic.title,
        message_count=topic_summary.message_count,
        last_message_at=topic_summary.last_message_at,
        created_at=topic_summary.topic.created_at,
        updated_at=topic_summary.topic.updated_at,
    )


def _serialize_message(message) -> ChatMessagePublic:
    citations = message.citations_json if isinstance(message.citations_json, list) else []
    return ChatMessagePublic(
        id=message.id,
        topic_id=message.topic_id,
        role=message.role,
        content=message.content,
        model=message.model,
        answer_mode=message.answer_mode,
        used_knowledge_base=message.used_knowledge_base,
        citations=citations,
        created_at=message.created_at,
    )


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _iter_answer_chunks(text: str, *, chunk_size: int = 24):
    content = text or ""
    for start in range(0, len(content), chunk_size):
        yield content[start : start + chunk_size]


@router.get("/topics", response_model=ChatTopicListResponse)
def get_topics(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    items = [_serialize_topic(item) for item in list_topics(db, user_id=user.id)]
    return ChatTopicListResponse(items=items)


@router.post("/topics", response_model=ChatTopicPublic, status_code=status.HTTP_201_CREATED)
def post_topic(
    payload: ChatTopicCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    return _serialize_topic(create_topic(db, user=user, title=payload.title))


@router.get("/topics/{topic_id}/messages", response_model=ChatMessageListResponse)
def get_messages(
    topic_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    topic = get_topic(db, user_id=user.id, topic_id=topic_id)
    if topic is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found")
    items = [_serialize_message(item) for item in list_topic_messages(db, topic_id=topic_id)]
    return ChatMessageListResponse(items=items)


@router.post("/topics/{topic_id}/messages", response_model=ChatTurnResponse)
def create_chat_completion(
    topic_id: int,
    payload: ChatMessageCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        result = send_topic_message(db, user=user, topic_id=topic_id, prompt=payload.message)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc) or "Chat service is unavailable"
        status_code = status.HTTP_400_BAD_REQUEST if detail == "message is required" else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail=detail) from exc

    return ChatTurnResponse(
        topic=_serialize_topic(result.topic),
        user_message=_serialize_message(result.user_message),
        assistant_message=_serialize_message(result.assistant_message),
    )


@router.post("/topics/{topic_id}/messages/stream")
def stream_chat_completion(
    topic_id: int,
    payload: ChatMessageCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    def event_stream():
        try:
            started_turn = start_topic_message(
                db,
                user=user,
                topic_id=topic_id,
                prompt=payload.message,
            )
            yield _sse_event(
                "user_message",
                {"user_message": _serialize_message(started_turn.user_message).model_dump(mode="json")},
            )
            assistant_draft = build_topic_assistant_draft(
                db,
                user=user,
                started_turn=started_turn,
                progress_callback=lambda phase, progress_status, message, detail=None: publish_chat_progress_event(
                    user_id=user.id,
                    topic_id=topic_id,
                    phase=phase,
                    status=progress_status,
                    message=message,
                    detail=detail,
                ),
            )
            yield _sse_event(
                "assistant_start",
                {
                    "answer_mode": assistant_draft.answer_mode,
                    "used_knowledge_base": assistant_draft.used_knowledge_base,
                },
            )
            for chunk in _iter_answer_chunks(assistant_draft.content):
                yield _sse_event("assistant_delta", {"delta": chunk})
            assistant_message = persist_topic_assistant_message(
                db,
                topic=started_turn.topic,
                assistant_draft=assistant_draft,
            )
            yield _sse_event(
                "assistant_complete",
                {"assistant_message": _serialize_message(assistant_message).model_dump(mode="json")},
            )
            yield _sse_event("done", {"ok": True})
        except ValueError as exc:
            yield _sse_event("error", {"detail": str(exc) or "Topic not found"})
        except RuntimeError as exc:
            yield _sse_event("error", {"detail": str(exc) or "Chat service is unavailable"})
        except Exception:
            yield _sse_event("error", {"detail": "聊天流式输出失败"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
