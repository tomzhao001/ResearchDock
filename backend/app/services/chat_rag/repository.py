from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatTopic, User
from app.services import rag as legacy_rag


class ChatTopicRepository:
    def list_topics(self, db: Session, *, user_id: int) -> list[legacy_rag.TopicSummary]:
        return legacy_rag.list_topics(db, user_id=user_id)

    def create_topic(self, db: Session, *, user: User, title: str | None = None) -> legacy_rag.TopicSummary:
        now = datetime.now(timezone.utc)
        topic = ChatTopic(
            user_id=user.id,
            title=legacy_rag._normalize_title(title),
            created_at=now,
            updated_at=now,
        )
        db.add(topic)
        db.commit()
        db.refresh(topic)
        return legacy_rag._topic_summary(db, topic)

    def get_topic(self, db: Session, *, user_id: int, topic_id: int) -> ChatTopic | None:
        return legacy_rag.get_topic(db, user_id=user_id, topic_id=topic_id)

    def list_topic_messages(self, db: Session, *, topic_id: int) -> list[ChatMessage]:
        return legacy_rag.list_topic_messages(db, topic_id=topic_id)

    def start_topic_message(
        self,
        db: Session,
        *,
        user: User,
        topic_id: int,
        prompt: str,
    ) -> legacy_rag.StartedChatTurn:
        message = (prompt or "").strip()
        if not message:
            raise RuntimeError("message is required")

        topic = self.get_topic(db, user_id=user.id, topic_id=topic_id)
        if topic is None:
            raise ValueError("Topic not found")

        user_message = legacy_rag._create_message(db, topic=topic, role="user", content=message)
        legacy_rag._rename_topic_from_first_message(db, topic=topic, prompt=message)
        records = self.list_topic_messages(db, topic_id=topic.id)
        return legacy_rag.StartedChatTurn(
            topic=topic,
            user_message=user_message,
            records=records,
            prompt=message,
        )

    def persist_assistant_message(
        self,
        db: Session,
        *,
        topic: ChatTopic,
        assistant_draft: legacy_rag.AssistantMessageDraft,
    ) -> ChatMessage:
        return legacy_rag._create_message(
            db,
            topic=topic,
            role="assistant",
            content=assistant_draft.content,
            model=assistant_draft.model,
            answer_mode=assistant_draft.answer_mode,
            used_knowledge_base=assistant_draft.used_knowledge_base,
            citations_json=assistant_draft.citations_json,
            metadata_json=assistant_draft.metadata_json,
        )
