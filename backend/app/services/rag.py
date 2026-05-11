from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ChatMessage, ChatTopic, Paper, PaperChunk, User
from app.services.llm import chat_with_messages, embed_texts, rerank_documents

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


def _is_postgres_session(db: Session) -> bool:
    bind = db.get_bind()
    return bool(bind and bind.dialect.name == "postgresql")


def _normalize_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    if isinstance(value, tuple) and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    if hasattr(value, "tolist"):
        items = value.tolist()
        if isinstance(items, list) and all(isinstance(item, (int, float)) for item in items):
            return [float(item) for item in items]
    return None


def _search_chunks_legacy(db: Session, *, query: str, top_k: int) -> list[RetrievalResult]:
    rows = db.execute(
        select(PaperChunk, Paper)
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(Paper.deleted_at.is_(None))
        .order_by(PaperChunk.paper_id.asc(), PaperChunk.chunk_index.asc())
    ).all()
    if not rows:
        return []

    query_tokens = set(_tokenize(query))
    query_embedding: list[float] | None = None
    if (settings.glm_api_key.strip() or settings.openai_api_key.strip()) and any(chunk.embedding for chunk, _ in rows):
        try:
            query_embedding = embed_texts([query])[0]
        except Exception:
            query_embedding = None

    scored: list[RetrievalResult] = []
    for chunk, paper in rows:
        lexical_score = _lexical_score(query_tokens, chunk.content)
        chunk_embedding = _normalize_embedding(chunk.embedding)
        embedding_score = (
            _cosine_similarity(query_embedding, chunk_embedding)
            if query_embedding is not None and chunk_embedding is not None
            else 0.0
        )
        score = embedding_score if embedding_score > 0 else lexical_score
        if embedding_score > 0 and lexical_score > 0:
            score = embedding_score * 0.8 + lexical_score * 0.2
        if score < MIN_RELEVANCE_SCORE:
            continue
        scored.append(RetrievalResult(chunk=chunk, paper=paper, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def _search_sparse_chunks_postgres(db: Session, *, query: str, limit: int) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    search_config = re.sub(r"[^a-zA-Z0-9_]+", "", settings.rag_text_search_config) or "simple"
    rows = db.execute(
        text(
            f"""
            SELECT
                pc.id AS chunk_id,
                pc.paper_id AS paper_id,
                ts_rank_cd(pc.search_vector, plainto_tsquery('{search_config}', :query)) AS score
            FROM paper_chunks AS pc
            JOIN papers AS p ON p.id = pc.paper_id
            WHERE p.deleted_at IS NULL
              AND pc.search_vector @@ plainto_tsquery('{search_config}', :query)
            ORDER BY score DESC, pc.id ASC
            LIMIT :limit
            """
        ),
        {
            "query": query.strip(),
            "limit": limit,
        },
    ).mappings().all()
    return [
        {
            "chunk_id": int(row["chunk_id"]),
            "paper_id": int(row["paper_id"]),
            "score": float(row["score"] or 0.0),
        }
        for row in rows
    ]


def _search_dense_chunks_postgres(
    db: Session,
    *,
    query_embedding: list[float] | None,
    limit: int,
) -> list[dict[str, Any]]:
    if query_embedding is None:
        return []
    distance = PaperChunk.embedding.cosine_distance(query_embedding)
    rows = db.execute(
        select(
            PaperChunk.id.label("chunk_id"),
            PaperChunk.paper_id.label("paper_id"),
            (1 - distance).label("score"),
        )
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(Paper.deleted_at.is_(None), PaperChunk.embedding.is_not(None))
        .order_by(distance.asc(), PaperChunk.id.asc())
        .limit(limit)
    ).all()
    return [
        {
            "chunk_id": int(row.chunk_id),
            "paper_id": int(row.paper_id),
            "score": float(row.score or 0.0),
        }
        for row in rows
    ]


def _fuse_ranked_candidates(
    sparse_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for source, hits in (("sparse", sparse_hits), ("dense", dense_hits)):
        for rank, item in enumerate(hits, start=1):
            candidate = merged.setdefault(
                int(item["chunk_id"]),
                {
                    "chunk_id": int(item["chunk_id"]),
                    "paper_id": int(item["paper_id"]),
                    "source_scores": {},
                    "source_ranks": {},
                    "score": 0.0,
                },
            )
            candidate["source_scores"][source] = float(item["score"])
            candidate["source_ranks"][source] = rank
            candidate["score"] += 1.0 / (settings.rag_rrf_k + rank)
    fused = list(merged.values())
    fused.sort(
        key=lambda item: (
            float(item["score"]),
            max((float(score) for score in item["source_scores"].values()), default=0.0),
        ),
        reverse=True,
    )
    for rank, item in enumerate(fused, start=1):
        item["rank"] = rank
    return fused[:limit]


def _load_chunk_record_map(
    db: Session,
    *,
    chunk_ids: Iterable[int],
) -> dict[int, tuple[PaperChunk, Paper]]:
    ids = [int(chunk_id) for chunk_id in chunk_ids]
    if not ids:
        return {}
    rows = db.execute(
        select(PaperChunk, Paper)
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(Paper.deleted_at.is_(None), PaperChunk.id.in_(ids))
    ).all()
    return {int(chunk.id): (chunk, paper) for chunk, paper in rows}


def _serialize_ranked_trace_candidates(
    candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> list[dict]:
    serialized: list[dict] = []
    for item in candidates:
        record = record_map.get(int(item["chunk_id"]))
        if record is None:
            continue
        chunk, paper = record
        snippet = chunk.content
        if len(snippet) > 180:
            snippet = f"{snippet[:180].rstrip()}..."
        row = {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "paper_id": paper.id,
            "paper_title": paper.title,
            "score": round(float(item.get("score") or 0.0), 4),
            "page_from": chunk.page_from,
            "page_to": chunk.page_to,
            "snippet": snippet,
        }
        source_scores = item.get("source_scores")
        if isinstance(source_scores, dict) and source_scores:
            row["source_scores"] = {key: round(float(value), 4) for key, value in source_scores.items()}
        source_ranks = item.get("source_ranks")
        if isinstance(source_ranks, dict) and source_ranks:
            row["source_ranks"] = {key: int(value) for key, value in source_ranks.items()}
        if item.get("rerank_score") is not None:
            row["rerank_score"] = round(float(item["rerank_score"]), 4)
        if item.get("rerank_rank") is not None:
            row["rerank_rank"] = int(item["rerank_rank"])
        serialized.append(row)
    return serialized


def _build_retrieval_results(
    candidates: list[dict[str, Any]],
    *,
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for item in candidates:
        record = record_map.get(int(item["chunk_id"]))
        if record is None:
            continue
        chunk, paper = record
        results.append(RetrievalResult(chunk=chunk, paper=paper, score=float(item.get("score") or 0.0)))
    return results


def _apply_reranking(
    query: str,
    *,
    fused_candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
    limit: int,
) -> list[dict[str, Any]]:
    documents: list[str] = []
    aligned_candidates: list[dict[str, Any]] = []
    for candidate in fused_candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            continue
        documents.append(record[0].content)
        aligned_candidates.append(candidate)
    if not documents:
        return []
    rerank_results = rerank_documents(query, documents, top_n=limit)
    if not rerank_results:
        return aligned_candidates[:limit]

    reranked: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for rank, item in enumerate(rerank_results, start=1):
        if item.index < 0 or item.index >= len(aligned_candidates):
            continue
        candidate = dict(aligned_candidates[item.index])
        candidate["score"] = float(item.relevance_score)
        candidate["rerank_score"] = float(item.relevance_score)
        candidate["rerank_rank"] = rank
        reranked.append(candidate)
        seen_indexes.add(item.index)

    for index, candidate in enumerate(aligned_candidates):
        if index in seen_indexes:
            continue
        reranked.append(dict(candidate))

    return reranked[:limit]


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
    if _is_postgres_session(db):
        db.execute(
            update(PaperChunk)
            .where(PaperChunk.paper_id == paper_id)
            .values(
                search_vector=func.to_tsvector(
                    settings.rag_text_search_config,
                    PaperChunk.content,
                )
            )
        )
        db.commit()
    return len(chunks)


def _search_chunks(
    db: Session,
    *,
    query: str,
    top_k: int | None = None,
    trace: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    limit = top_k or settings.rag_top_k
    if not _is_postgres_session(db):
        results = _search_chunks_legacy(db, query=query, top_k=limit)
        if trace is not None:
            serialized = _serialize_trace_candidates(results)
            trace.update(
                {
                    "sparse_candidates": [],
                    "dense_candidates": [],
                    "fused_candidates": serialized,
                    "reranked_candidates": serialized,
                    "retrieval_backend": "legacy",
                }
            )
        return results

    query_embedding: list[float] | None = None
    if settings.glm_api_key.strip() or settings.openai_api_key.strip():
        try:
            query_embedding = embed_texts([query])[0]
        except Exception:
            query_embedding = None

    sparse_hits = _search_sparse_chunks_postgres(db, query=query, limit=max(settings.rag_sparse_top_k, limit))
    dense_hits = _search_dense_chunks_postgres(
        db,
        query_embedding=query_embedding,
        limit=max(settings.rag_dense_top_k, limit),
    )
    fusion_limit = max(settings.rag_fusion_window, settings.rag_rerank_top_n, limit)
    fused_candidates = _fuse_ranked_candidates(sparse_hits, dense_hits, limit=fusion_limit)
    record_map = _load_chunk_record_map(
        db,
        chunk_ids=[item["chunk_id"] for item in [*sparse_hits, *dense_hits, *fused_candidates]],
    )

    reranked_candidates = fused_candidates
    if fused_candidates:
        try:
            reranked_candidates = _apply_reranking(
                query,
                fused_candidates=fused_candidates,
                record_map=record_map,
                limit=max(settings.rag_rerank_top_n, limit),
            )
        except Exception:
            reranked_candidates = fused_candidates[: max(settings.rag_rerank_top_n, limit)]

    results = _build_retrieval_results(reranked_candidates[:limit], record_map=record_map)
    if trace is not None:
        trace.update(
            {
                "sparse_candidates": _serialize_ranked_trace_candidates(sparse_hits, record_map),
                "dense_candidates": _serialize_ranked_trace_candidates(dense_hits, record_map),
                "fused_candidates": _serialize_ranked_trace_candidates(fused_candidates, record_map),
                "reranked_candidates": _serialize_ranked_trace_candidates(reranked_candidates, record_map),
                "retrieval_backend": "postgres_hybrid",
            }
        )
    return results


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
    candidate_limit = max(settings.rag_top_k, settings.rag_rerank_top_n, 10)
    retrieval_debug: dict[str, Any] = {}
    retrieval_candidates = _search_chunks(db, query=retrieval_query, top_k=candidate_limit, trace=retrieval_debug)
    retrieval_results = retrieval_candidates[: settings.rag_top_k]
    citations = _serialize_citations(retrieval_results)
    retrieval_trace = {
        "original_user_query": message,
        "retrieval_query": retrieval_query,
        "retrieval_backend": retrieval_debug.get("retrieval_backend", "unknown"),
        "first_pass_candidates": _serialize_trace_candidates(retrieval_candidates),
        "sparse_candidates": retrieval_debug.get("sparse_candidates", []),
        "dense_candidates": retrieval_debug.get("dense_candidates", []),
        "fused_candidates": retrieval_debug.get("fused_candidates", []),
        "reranked_candidates": retrieval_debug.get("reranked_candidates", []),
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
