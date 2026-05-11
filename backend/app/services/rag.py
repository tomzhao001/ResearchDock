from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ChatMessage, ChatTopic, Paper, PaperChunk, User
from app.services.llm import chat_with_messages, embed_texts

logger = logging.getLogger(__name__)

DEFAULT_TOPIC_TITLE = "新话题"
MAX_HISTORY_MESSAGES = 8
MIN_RELEVANCE_SCORE = 0.12

RAG_SYSTEM_PROMPT = (
    "你是 ResearchDock 的论文知识库助理。"
    "请优先依据给定的知识库证据回答，避免编造论文内容。"
    "如果证据不足，请直接说明知识库中没有找到确切依据。"
)

FALLBACK_SYSTEM_PROMPT = (
    "你是 ResearchDock 的研究助理。"
    "当前知识库没有找到足够证据，请先明确告诉用户“知识库中未找到确切依据”。"
    "随后你可以基于通用知识给出简短补充，但必须避免伪造论文引用。"
)


@dataclass
class TopicSummary:
    topic: ChatTopic
    message_count: int
    last_message_at: datetime | None


@dataclass
class ChatTurnResult:
    topic: TopicSummary
    user_message: ChatMessage
    assistant_message: ChatMessage


@dataclass
class RetrievalResult:
    chunk: PaperChunk
    paper: Paper
    score: float


def _normalize_title(title: str | None) -> str:
    normalized = (title or "").strip()
    return normalized or DEFAULT_TOPIC_TITLE


def _excerpt(text: str, max_length: int = 24) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= max_length:
        return compact or DEFAULT_TOPIC_TITLE
    return f"{compact[:max_length].rstrip()}..."


def _tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    return re.findall(r"[\w\u4e00-\u9fff]+", lowered)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _lexical_score(query_tokens: set[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = set(_tokenize(content))
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def _history_messages(records: list[ChatMessage], *, include_last_user: bool = False) -> list[dict[str, str]]:
    history = records if include_last_user else records[:-1]
    return [{"role": record.role, "content": record.content} for record in history[-MAX_HISTORY_MESSAGES:]]


def _build_retrieval_query(records: list[ChatMessage], prompt: str) -> str:
    previous_user_messages = [record.content for record in records[:-1] if record.role == "user"][-2:]
    context = " ".join(previous_user_messages + [prompt]).strip()
    return context or prompt


def _topic_stats(db: Session, topic_ids: list[int]) -> dict[int, tuple[int, datetime | None]]:
    if not topic_ids:
        return {}
    rows = db.execute(
        select(
            ChatMessage.topic_id,
            func.count(ChatMessage.id),
            func.max(ChatMessage.created_at),
        )
        .where(ChatMessage.topic_id.in_(topic_ids))
        .group_by(ChatMessage.topic_id)
    ).all()
    return {
        int(topic_id): (int(message_count), last_message_at)
        for topic_id, message_count, last_message_at in rows
    }


def _topic_summary(db: Session, topic: ChatTopic) -> TopicSummary:
    stats = _topic_stats(db, [topic.id]).get(topic.id, (0, None))
    return TopicSummary(
        topic=topic,
        message_count=stats[0],
        last_message_at=stats[1],
    )


def list_topics(db: Session, *, user_id: int) -> list[TopicSummary]:
    topics = db.scalars(
        select(ChatTopic)
        .where(ChatTopic.user_id == user_id)
        .order_by(ChatTopic.updated_at.desc(), ChatTopic.id.desc())
    ).all()
    stats = _topic_stats(db, [topic.id for topic in topics])
    return [
        TopicSummary(
            topic=topic,
            message_count=stats.get(topic.id, (0, None))[0],
            last_message_at=stats.get(topic.id, (0, None))[1],
        )
        for topic in topics
    ]


def create_topic(db: Session, *, user: User, title: str | None = None) -> TopicSummary:
    now = datetime.now(timezone.utc)
    topic = ChatTopic(
        user_id=user.id,
        title=_normalize_title(title),
        created_at=now,
        updated_at=now,
    )
    db.add(topic)
    db.commit()
    db.refresh(topic)
    return _topic_summary(db, topic)


def get_topic(db: Session, *, user_id: int, topic_id: int) -> ChatTopic | None:
    return db.scalar(
        select(ChatTopic).where(
            ChatTopic.id == topic_id,
            ChatTopic.user_id == user_id,
        )
    )


def list_topic_messages(db: Session, *, topic_id: int) -> list[ChatMessage]:
    return db.scalars(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    ).all()


def _split_text(raw_text: str) -> list[dict]:
    text = re.sub(r"\s+", " ", (raw_text or "").strip())
    if not text:
        return []

    chunk_size = max(settings.rag_chunk_size, 200)
    overlap = max(min(settings.rag_chunk_overlap, chunk_size - 1), 0)
    step = max(chunk_size - overlap, 1)
    chunks: list[dict] = []
    cursor = 0
    index = 0
    while cursor < len(text):
        window = text[cursor : cursor + chunk_size]
        content = window.strip()
        if content:
            chunks.append(
                {
                    "chunk_index": index,
                    "content": content,
                    "token_count": len(_tokenize(content)),
                    "metadata_json": {
                        "char_start": cursor,
                        "char_end": min(cursor + chunk_size, len(text)),
                    },
                }
            )
            index += 1
        if cursor + chunk_size >= len(text):
            break
        cursor += step
    return chunks


def rebuild_paper_index(db: Session, *, paper_id: int, raw_text: str) -> int:
    chunks = _split_text(raw_text)
    embeddings: list[list[float]] = []
    if chunks and settings.openai_api_key.strip():
        try:
            embeddings = embed_texts([chunk["content"] for chunk in chunks])
        except Exception:
            embeddings = []

    db.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
    for index, chunk in enumerate(chunks):
        db.add(
            PaperChunk(
                paper_id=paper_id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                embedding=embeddings[index] if index < len(embeddings) else None,
                token_count=chunk["token_count"],
                page_from=None,
                page_to=None,
                metadata_json=chunk["metadata_json"],
            )
        )
    db.commit()
    return len(chunks)


def _search_chunks(db: Session, *, query: str, top_k: int | None = None) -> list[RetrievalResult]:
    rows = db.execute(
        select(PaperChunk, Paper)
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(Paper.deleted_at.is_(None))
        .order_by(PaperChunk.paper_id.asc(), PaperChunk.chunk_index.asc())
    ).all()
    if not rows:
        return []

    limit = top_k or settings.rag_top_k
    query_tokens = set(_tokenize(query))
    query_embedding: list[float] | None = None
    if settings.openai_api_key.strip() and any(chunk.embedding for chunk, _ in rows):
        try:
            query_embedding = embed_texts([query])[0]
        except Exception:
            query_embedding = None

    scored: list[RetrievalResult] = []
    for chunk, paper in rows:
        lexical_score = _lexical_score(query_tokens, chunk.content)
        embedding_score = (
            _cosine_similarity(query_embedding, chunk.embedding)
            if query_embedding is not None and isinstance(chunk.embedding, list)
            else 0.0
        )
        score = embedding_score if embedding_score > 0 else lexical_score
        if embedding_score > 0 and lexical_score > 0:
            score = embedding_score * 0.8 + lexical_score * 0.2
        if score < MIN_RELEVANCE_SCORE:
            continue
        scored.append(RetrievalResult(chunk=chunk, paper=paper, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]


def _serialize_citations(results: list[RetrievalResult]) -> list[dict]:
    citations: list[dict] = []
    for result in results:
        snippet = result.chunk.content
        if len(snippet) > 240:
            snippet = f"{snippet[:240].rstrip()}..."
        citations.append(
            {
                "chunk_id": result.chunk.id,
                "paper_id": result.paper.id,
                "paper_title": result.paper.title,
                "source_url": result.paper.source_url,
                "snippet": snippet,
                "score": round(result.score, 4),
                "page_from": result.chunk.page_from,
                "page_to": result.chunk.page_to,
            }
        )
    return citations


def _serialize_trace_candidates(results: list[RetrievalResult]) -> list[dict]:
    candidates: list[dict] = []
    for result in results:
        snippet = result.chunk.content
        if len(snippet) > 180:
            snippet = f"{snippet[:180].rstrip()}..."
        candidates.append(
            {
                "chunk_id": result.chunk.id,
                "chunk_index": result.chunk.chunk_index,
                "paper_id": result.paper.id,
                "paper_title": result.paper.title,
                "score": round(result.score, 4),
                "page_from": result.chunk.page_from,
                "page_to": result.chunk.page_to,
                "snippet": snippet,
            }
        )
    return candidates


def _create_message(
    db: Session,
    *,
    topic: ChatTopic,
    role: str,
    content: str,
    model: str | None = None,
    answer_mode: str | None = None,
    used_knowledge_base: bool = False,
    citations_json: list[dict] | None = None,
    metadata_json: dict | None = None,
) -> ChatMessage:
    now = datetime.now(timezone.utc)
    topic.updated_at = now
    message = ChatMessage(
        topic_id=topic.id,
        role=role,
        content=content,
        model=model,
        answer_mode=answer_mode,
        used_knowledge_base=used_knowledge_base,
        citations_json=citations_json,
        metadata_json=metadata_json,
        created_at=now,
    )
    db.add(message)
    db.commit()
    db.refresh(topic)
    db.refresh(message)
    return message


def _rename_topic_from_first_message(db: Session, *, topic: ChatTopic, prompt: str) -> None:
    if topic.title != DEFAULT_TOPIC_TITLE:
        return
    user_message_count = db.scalar(
        select(func.count(ChatMessage.id)).where(
            ChatMessage.topic_id == topic.id,
            ChatMessage.role == "user",
        )
    )
    if int(user_message_count or 0) != 1:
        return
    topic.title = _excerpt(prompt)
    topic.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(topic)


def send_topic_message(db: Session, *, user: User, topic_id: int, prompt: str) -> ChatTurnResult:
    message = (prompt or "").strip()
    if not message:
        raise RuntimeError("message is required")

    topic = get_topic(db, user_id=user.id, topic_id=topic_id)
    if topic is None:
        raise ValueError("Topic not found")

    user_message = _create_message(db, topic=topic, role="user", content=message)
    _rename_topic_from_first_message(db, topic=topic, prompt=message)

    records = list_topic_messages(db, topic_id=topic.id)
    started_at = time.perf_counter()
    retrieval_query = _build_retrieval_query(records, message)
    candidate_limit = max(settings.rag_top_k, 10)
    retrieval_candidates = _search_chunks(db, query=retrieval_query, top_k=candidate_limit)
    retrieval_results = retrieval_candidates[: settings.rag_top_k]
    citations = _serialize_citations(retrieval_results)
    retrieval_trace = {
        "original_user_query": message,
        "retrieval_query": retrieval_query,
        "first_pass_candidates": _serialize_trace_candidates(retrieval_candidates),
        "selected_citations": citations,
        "answer_mode": "knowledge_base" if citations else "fallback_general",
        "search_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }

    if citations:
        evidence_text = "\n\n".join(
            [
                (
                    f"[{index}] 论文标题: {citation['paper_title'] or '-'}\n"
                    f"来源链接: {citation['source_url'] or '-'}\n"
                    f"片段: {citation['snippet']}"
                )
                for index, citation in enumerate(citations, start=1)
            ]
        )
        answer, model = chat_with_messages(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                *_history_messages(records),
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{message}\n\n"
                        "请尽量只依据下面的知识库证据回答，并保持简洁。\n"
                        "如果证据仍然不足，请直接说明“知识库中未找到确切依据”。\n\n"
                        f"{evidence_text}"
                    ),
                },
            ],
            temperature=0.2,
        )
        retrieval_trace["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info("rag_trace %s", retrieval_trace)
        assistant_message = _create_message(
            db,
            topic=topic,
            role="assistant",
            content=answer,
            model=model,
            answer_mode="knowledge_base",
            used_knowledge_base=True,
            citations_json=citations,
            metadata_json={"retrieval": retrieval_trace},
        )
    else:
        answer, model = chat_with_messages(
            [
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                *_history_messages(records),
                {"role": "user", "content": message},
            ],
            temperature=0.3,
        )
        retrieval_trace["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info("rag_trace %s", retrieval_trace)
        assistant_message = _create_message(
            db,
            topic=topic,
            role="assistant",
            content=answer,
            model=model,
            answer_mode="fallback_general",
            used_knowledge_base=False,
            citations_json=[],
            metadata_json={"retrieval": retrieval_trace},
        )

    return ChatTurnResult(
        topic=_topic_summary(db, topic),
        user_message=user_message,
        assistant_message=assistant_message,
    )
