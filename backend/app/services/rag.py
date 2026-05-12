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


def _script_counts(text: str) -> tuple[int, int]:
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in text if char.isascii() and char.isalpha())
    return cjk_count, latin_count


def _is_cjk_dominant_text(text: str) -> bool:
    cjk_count, latin_count = _script_counts(text or "")
    return cjk_count > 0 and cjk_count >= max(latin_count * 2, 8)


def _is_table_or_figure_query(query: str) -> bool:
    compact = query or ""
    return bool(
        re.search(r"\b(?:table|tab\.?|figure|fig\.?)\s*\d+", compact, re.IGNORECASE)
        or re.search(r"(表|图)\s*\d+", compact)
    )


def _extract_exact_match_terms(query: str) -> list[str]:
    compact = re.sub(r"\s+", " ", (query or "").strip())
    if not compact:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    patterns = (
        (r"\b(?:table|tab\.?|figure|fig\.?)\s*\d+[a-z]?\b", re.IGNORECASE),
        (r"\b[A-Z]{2,}(?:[-/][A-Z0-9]+)*(?:\d+[A-Z0-9/-]*)?\b", 0),
        (r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)+\b", 0),
        (r"\b[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+\b", 0),
        (r"\b[A-Za-z]*\d+[A-Za-z0-9-]*\b", 0),
    )
    for pattern, flags in patterns:
        for match in re.finditer(pattern, query, flags):
            term = re.sub(r"\s+", " ", match.group(0).strip(" \t\r\n.,;:()[]{}"))
            lowered = term.lower()
            if len(lowered) < 2 or lowered in seen:
                continue
            candidates.append(term)
            seen.add(lowered)

    if re.search(r"[^\x00-\x7F]", query):
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", query):
            lowered = token.lower()
            if lowered in seen:
                continue
            candidates.append(token)
            seen.add(lowered)

    return sorted(candidates, key=len, reverse=True)


def _is_exact_match_heavy_query(query: str, exact_terms: list[str] | None = None) -> bool:
    terms = exact_terms if exact_terms is not None else _extract_exact_match_terms(query)
    if _is_table_or_figure_query(query):
        return True
    if _is_cjk_dominant_text(query):
        return False
    return len(terms) >= 2


def _chunk_text_payload(chunk: PaperChunk, *, include_supporting_context: bool = False) -> str:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    header = str(metadata.get("context_header") or "").strip()
    body_text = str(metadata.get("body_text") or chunk.content or "").strip()
    supporting_context = str(metadata.get("supporting_context") or "").strip()
    parts = [part for part in (header, body_text) if part]
    if include_supporting_context and supporting_context:
        parts.append(f"上下文补充: {supporting_context}")
    return "\n\n".join(parts) if parts else str(chunk.content or "")


def _clip_snippet(text: str, *, max_length: int) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length].rstrip()}..."


_TABLE_FOCUS_TERM_PATTERN = re.compile(
    r"(研究组|对照组|实验组|干预前|干预后|治疗前|治疗后|组别|stimulation|sham|baseline|post[- ]?(?:treatment|intervention)|pre[- ]?(?:treatment|intervention)|theta|beta|smr|frequency|score|p-?value|频率|评分|显著性)",
    re.IGNORECASE,
)
_TABLE_VALUE_PATTERN = re.compile(
    r"(?:[αβγθδμσχ]|SMR|P值|p-?value|频率|评分|均数|标准差|mean|sd|x±s|χ2|t检验)",
    re.IGNORECASE,
)


def _extract_table_focus_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in _extract_exact_match_terms(query):
        lowered = term.lower()
        if lowered in seen:
            continue
        terms.append(term)
        seen.add(lowered)
    for match in _TABLE_FOCUS_TERM_PATTERN.finditer(query or ""):
        term = match.group(0).strip()
        lowered = term.lower()
        if lowered in seen:
            continue
        terms.append(term)
        seen.add(lowered)
    return sorted(terms, key=len, reverse=True)


def _looks_like_table_body_text(text: str) -> bool:
    compact = str(text or "").strip()
    if not compact:
        return False
    number_count = len(re.findall(r"[-+]?\d+(?:\.\d+)?", compact))
    digit_count = sum(char.isdigit() for char in compact)
    body_term_hits = len(_TABLE_FOCUS_TERM_PATTERN.findall(compact))
    metric_hits = len(_TABLE_VALUE_PATTERN.findall(compact))
    return bool(
        (number_count >= 3 and body_term_hits >= 1)
        or (number_count >= 4 and metric_hits >= 1 and digit_count >= 8)
        or (digit_count >= 10 and "±" in compact)
    )


def _block_is_table_body_candidate(block: dict[str, Any]) -> bool:
    block_type = str(block.get("block_type") or "")
    if block_type == "table_like":
        return True
    return _looks_like_table_body_text(str(block.get("text") or ""))


def _chunk_has_table_body(chunk: PaperChunk) -> bool:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    block_types = {str(item).lower() for item in metadata.get("block_types", [])} if isinstance(metadata, dict) else set()
    if "table_like" in block_types:
        return True
    return _looks_like_table_body_text(str(metadata.get("body_text") or chunk.content or ""))


def _chunk_is_caption_only(chunk: PaperChunk) -> bool:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    block_types = {str(item).lower() for item in metadata.get("block_types", [])} if isinstance(metadata, dict) else set()
    if not block_types or block_types - {"table_caption"}:
        return False
    return not _looks_like_table_body_text(str(metadata.get("body_text") or chunk.content or ""))


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


def _search_sparse_chunks_postgres(
    db: Session,
    *,
    query: str,
    limit: int,
    exact_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    search_config = re.sub(r"[^a-zA-Z0-9_]+", "", settings.rag_text_search_config) or "simple"
    body_text_expr = "LOWER(COALESCE(pc.metadata_json->>'body_text', pc.content))"
    exact_terms = exact_terms if exact_terms is not None else _extract_exact_match_terms(query)
    exact_match_clauses: list[str] = []
    exact_bonus_clauses: list[str] = []
    params: dict[str, Any] = {"query": query.strip(), "limit": limit}
    for index, term in enumerate(exact_terms):
        term_key = f"exact_term_{index}"
        bonus_key = f"exact_bonus_{index}"
        params[term_key] = f"%{term.lower()}%"
        params[bonus_key] = round(min(0.35, 0.08 + len(term) * 0.01), 4)
        exact_match_clauses.append(f"{body_text_expr} LIKE :{term_key}")
        exact_bonus_clauses.append(f"CASE WHEN {body_text_expr} LIKE :{term_key} THEN :{bonus_key} ELSE 0 END")
    exact_match_sql = " OR ".join(exact_match_clauses) or "FALSE"
    exact_bonus_sql = " + ".join(exact_bonus_clauses) or "0"
    rows = db.execute(
        text(
            f"""
            SELECT
                pc.id AS chunk_id,
                pc.paper_id AS paper_id,
                (
                    ts_rank_cd(pc.search_vector, plainto_tsquery('{search_config}', :query))
                    + ({exact_bonus_sql})
                ) AS score
            FROM paper_chunks AS pc
            JOIN papers AS p ON p.id = pc.paper_id
            WHERE p.deleted_at IS NULL
              AND (
                    pc.search_vector @@ plainto_tsquery('{search_config}', :query)
                    OR {exact_match_sql}
              )
            ORDER BY score DESC, pc.id ASC
            LIMIT :limit
            """
        ),
        params,
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


def _boost_exact_match_candidates(
    query: str,
    *,
    candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> list[dict[str, Any]]:
    exact_terms = _extract_exact_match_terms(query)
    prefers_table = _is_table_or_figure_query(query)
    table_focus_terms = _extract_table_focus_terms(query) if prefers_table else []
    if not exact_terms and not table_focus_terms:
        return candidates

    boosted: list[dict[str, Any]] = []
    allow_general_exact_boost = not _is_cjk_dominant_text(query)
    for candidate in candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            boosted.append(dict(candidate))
            continue
        chunk, _ = record
        metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
        haystack = _chunk_text_payload(chunk, include_supporting_context=True).lower()
        matched_terms = [term for term in exact_terms if term.lower() in haystack]
        matched_focus_terms = [term for term in table_focus_terms if term.lower() in haystack]
        block_types = {str(item).lower() for item in metadata.get("block_types", [])} if isinstance(metadata, dict) else set()
        body_text = str(metadata.get("body_text") or chunk.content or "")
        has_table_body = _chunk_has_table_body(chunk)
        caption_only = _chunk_is_caption_only(chunk)
        bonus = 0.0
        if allow_general_exact_boost:
            bonus += sum(min(0.12, 0.04 + len(term) * 0.003) for term in matched_terms)
        if prefers_table:
            if has_table_body:
                bonus += 0.05
                bonus += sum(min(0.06, 0.02 + len(term) * 0.002) for term in matched_terms)
                bonus += sum(min(0.04, 0.015 + len(term) * 0.0015) for term in matched_focus_terms)
                if matched_focus_terms and re.search(r"\d", body_text):
                    bonus += 0.05
            elif "table_caption" in block_types and not caption_only:
                bonus += 0.03 + sum(min(0.03, 0.01 + len(term) * 0.0015) for term in matched_terms)
        boosted_candidate = dict(candidate)
        boosted_candidate["exact_match_terms"] = matched_terms
        boosted_candidate["exact_match_bonus"] = round(bonus, 4)
        boosted_candidate["score"] = float(candidate.get("score") or 0.0) + bonus
        boosted.append(boosted_candidate)

    boosted.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("exact_match_bonus") or 0.0),
            max((float(score) for score in item.get("source_scores", {}).values()), default=0.0),
        ),
        reverse=True,
    )
    for rank, item in enumerate(boosted, start=1):
        item["rank"] = rank
    return boosted


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
        snippet = _clip_snippet(_chunk_text_payload(chunk), max_length=180)
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
        if item.get("exact_match_bonus") is not None:
            row["exact_match_bonus"] = round(float(item["exact_match_bonus"]), 4)
        if item.get("exact_match_terms"):
            row["exact_match_terms"] = [str(term) for term in item["exact_match_terms"]]
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
    include_supporting_context = _is_table_or_figure_query(query)
    for candidate in fused_candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            continue
        documents.append(_chunk_text_payload(record[0], include_supporting_context=include_supporting_context))
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


def _build_context_header(
    *,
    paper_title: str | None,
    section_path: str | None,
    page_from: int | None,
    page_to: int | None,
) -> str:
    lines: list[str] = []
    if (paper_title or "").strip():
        lines.append(f"论文标题: {(paper_title or '').strip()}")
    if (section_path or "").strip():
        lines.append(f"章节: {(section_path or '').strip()}")
    if page_from is not None:
        if page_to is not None and page_to != page_from:
            lines.append(f"页码: {page_from}-{page_to}")
        else:
            lines.append(f"页码: {page_from}")
    return "\n".join(lines)


def _window_text(text: str, *, chunk_size: int, overlap: int) -> list[dict[str, int | str]]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    step = max(chunk_size - overlap, 1)
    windows: list[dict[str, int | str]] = []
    cursor = 0
    while cursor < len(text):
        window = text[cursor : cursor + chunk_size]
        content = window.strip()
        if content:
            trimmed_offset = window.find(content)
            windows.append(
                {
                    "content": content,
                    "char_start": cursor + max(trimmed_offset, 0),
                    "char_end": cursor + max(trimmed_offset, 0) + len(content),
                }
            )
        if cursor + chunk_size >= len(text):
            break
        cursor += step
    return windows


def _select_supporting_context(
    all_blocks: list[dict[str, Any]],
    selected_blocks: list[dict[str, Any]],
    *,
    max_chars: int = 420,
) -> list[dict[str, Any]]:
    if not all_blocks or not selected_blocks:
        return []

    selected_indexes = {int(block.get("block_index") or -1) for block in selected_blocks}
    anchor_indexes = sorted(index for index in selected_indexes if index >= 0)
    if not anchor_indexes:
        return []

    section_id = str(selected_blocks[0].get("section_id") or "")
    left = anchor_indexes[0] - 1
    right = anchor_indexes[-1] + 1
    supporting: list[dict[str, Any]] = []
    supporting_chars = 0

    while supporting_chars < max_chars and (left >= 0 or right < len(all_blocks)):
        progressed = False
        if left >= 0:
            candidate = all_blocks[left]
            if str(candidate.get("section_id") or "") == section_id and int(candidate.get("block_index") or -1) not in selected_indexes:
                supporting.insert(0, candidate)
                supporting_chars += len(str(candidate.get("text") or ""))
                progressed = True
            left -= 1
        if supporting_chars >= max_chars:
            break
        if right < len(all_blocks):
            candidate = all_blocks[right]
            if str(candidate.get("section_id") or "") == section_id and int(candidate.get("block_index") or -1) not in selected_indexes:
                supporting.append(candidate)
                supporting_chars += len(str(candidate.get("text") or ""))
                progressed = True
            right += 1
        if not progressed and left < 0 and right >= len(all_blocks):
            break
    return supporting


def _make_chunk_record(
    *,
    chunk_index: int,
    body_text: str,
    block_rows: list[dict[str, Any]],
    paper_title: str | None,
    supporting_context_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    page_numbers = sorted(
        {
            int(block.get("page_number"))
            for block in block_rows
            if isinstance(block.get("page_number"), int) or str(block.get("page_number") or "").isdigit()
        }
    )
    page_from = page_numbers[0] if page_numbers else None
    page_to = page_numbers[-1] if page_numbers else None
    section_path = next((str(block.get("section_path") or "").strip() for block in block_rows if str(block.get("section_path") or "").strip()), "")
    section_title = next((str(block.get("section_title") or "").strip() for block in block_rows if str(block.get("section_title") or "").strip()), "")
    heading_level = next(
        (
            int(block.get("heading_level"))
            for block in block_rows
            if isinstance(block.get("heading_level"), int) or str(block.get("heading_level") or "").isdigit()
        ),
        None,
    )
    context_header = _build_context_header(
        paper_title=paper_title,
        section_path=section_path or section_title,
        page_from=page_from,
        page_to=page_to,
    )
    supporting_context = "\n\n".join(
        str(block.get("text") or "").strip()
        for block in (supporting_context_rows or [])
        if str(block.get("text") or "").strip()
    )
    embedding_input = body_text
    content = body_text
    if context_header:
        embedding_input = f"{context_header}\n\n{body_text}"
        content = f"{context_header}\n\n{body_text}"
    return {
        "chunk_index": chunk_index,
        "content": content,
        "embedding_input": embedding_input,
        "token_count": len(_tokenize(body_text)),
        "page_from": page_from,
        "page_to": page_to,
        "metadata_json": {
            "char_start": min((int(block.get("char_start") or 0) for block in block_rows), default=0),
            "char_end": max((int(block.get("char_end") or 0) for block in block_rows), default=0),
            "context_header": context_header or None,
            "section_id": next((block.get("section_id") for block in block_rows if block.get("section_id")), None),
            "section_title": section_title or None,
            "section_path": section_path or None,
            "heading_level": heading_level,
            "block_types": sorted({str(block.get("block_type") or "paragraph") for block in block_rows}),
            "block_count": len(block_rows),
            "body_text": body_text,
            "supporting_context": supporting_context or None,
        },
    }


def _split_text_legacy(raw_text: str, *, paper_title: str | None = None) -> list[dict[str, Any]]:
    chunk_size = max(settings.rag_chunk_size, 200)
    overlap = max(min(settings.rag_chunk_overlap, chunk_size - 1), 0)
    chunks: list[dict[str, Any]] = []
    for index, window in enumerate(_window_text(raw_text, chunk_size=chunk_size, overlap=overlap)):
        body_text = str(window["content"])
        chunks.append(
            _make_chunk_record(
                chunk_index=index,
                body_text=body_text,
                block_rows=[
                    {
                        "char_start": int(window["char_start"]),
                        "char_end": int(window["char_end"]),
                        "page_number": 1,
                        "block_type": "window",
                        "section_title": "Document",
                        "section_path": "Document",
                        "heading_level": 1,
                    }
                ],
                paper_title=paper_title,
            )
        )
    return chunks


def _split_text(
    raw_text: str,
    *,
    preanalysis: dict[str, Any] | None = None,
    paper_title: str | None = None,
) -> list[dict[str, Any]]:
    chunk_size = max(settings.rag_chunk_size, 200)
    overlap = max(min(settings.rag_chunk_overlap, chunk_size - 1), 0)
    blocks = preanalysis.get("blocks") if isinstance(preanalysis, dict) else None
    if not isinstance(blocks, list) or not blocks:
        return _split_text_legacy(raw_text, paper_title=paper_title)

    normalized_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        raw_text = str(block.get("text") or "").strip()
        if not raw_text:
            continue
        block_type = str(block.get("block_type") or "")
        normalized_text = raw_text
        if block_type not in {"table_caption", "table_like"} and not _block_is_table_body_candidate(block):
            normalized_text = re.sub(r"\s+", " ", raw_text)
        normalized_blocks.append({**block, "text": normalized_text})
    if not normalized_blocks:
        return _split_text_legacy(raw_text, paper_title=paper_title)

    document_text = " ".join(str(block.get("text") or "") for block in normalized_blocks[:80])
    document_is_cjk_dominant = _is_cjk_dominant_text(document_text)
    child_chunk_size = (
        min(chunk_size, max(int(chunk_size * 0.85), chunk_size - overlap, 320))
        if document_is_cjk_dominant
        else min(chunk_size, max(min(chunk_size // 2, 500), 220))
    )
    chunks: list[dict[str, Any]] = []
    pending_blocks: list[dict[str, Any]] = []
    pending_length = 0

    def append_chunk(
        block_rows: list[dict[str, Any]],
        *,
        supporting_context_rows: list[dict[str, Any]] | None = None,
        body_text: str | None = None,
    ) -> None:
        chunk_body_text = body_text or "\n\n".join(
            str(block.get("text") or "").strip() for block in block_rows if str(block.get("text") or "").strip()
        )
        if not chunk_body_text:
            return
        chunks.append(
            _make_chunk_record(
                chunk_index=len(chunks),
                body_text=chunk_body_text,
                block_rows=list(block_rows),
                paper_title=paper_title,
                supporting_context_rows=supporting_context_rows,
            )
        )

    def flush_pending() -> None:
        nonlocal pending_blocks, pending_length
        if not pending_blocks:
            return
        has_structured_block = any(
            str(block.get("block_type") or "") in {"table_like", "table_caption", "figure_caption"}
            or _block_is_table_body_candidate(block)
            for block in pending_blocks
        )
        append_chunk(
            list(pending_blocks),
            supporting_context_rows=(
                _select_supporting_context(normalized_blocks, pending_blocks)
                if has_structured_block or not document_is_cjk_dominant
                else []
            )
        )
        pending_blocks = []
        pending_length = 0

    block_index = 0
    while block_index < len(normalized_blocks):
        block = normalized_blocks[block_index]
        block_text = str(block.get("text") or "").strip()
        block_length = len(block_text)
        section_id = str(block.get("section_id") or "")
        pending_section_id = str(pending_blocks[0].get("section_id") or "") if pending_blocks else ""
        block_type = str(block.get("block_type") or "paragraph")
        isolate_block = block_type in {"table_like", "table_caption", "figure_caption"}

        if pending_blocks and section_id != pending_section_id:
            flush_pending()

        next_block = normalized_blocks[block_index + 1] if block_index + 1 < len(normalized_blocks) else None
        if (
            block_type == "table_caption"
            and next_block is not None
            and str(next_block.get("section_id") or "") == section_id
            and _block_is_table_body_candidate(next_block)
        ):
            flush_pending()
            table_blocks = [{**block, "text": block_text}, {**next_block, "text": str(next_block.get("text") or "").strip()}]
            table_body_text = str(next_block.get("text") or "").strip()
            table_total_length = len(block_text) + len(table_body_text) + 2
            if table_total_length <= chunk_size:
                append_chunk(
                    table_blocks,
                    supporting_context_rows=_select_supporting_context(normalized_blocks, table_blocks),
                )
            else:
                body_piece_size = max(chunk_size - len(block_text) - 2, min(chunk_size // 2, 220))
                body_piece_size = max(body_piece_size, min(chunk_size, 220))
                windows = _window_text(table_body_text, chunk_size=body_piece_size, overlap=min(overlap, max(body_piece_size - 1, 0)))
                if not windows:
                    append_chunk(
                        table_blocks,
                        supporting_context_rows=_select_supporting_context(normalized_blocks, table_blocks),
                    )
                else:
                    next_char_start = int(next_block.get("char_start") or 0)
                    for window_index, piece in enumerate(windows):
                        piece_block = dict(next_block)
                        piece_block["text"] = str(piece["content"])
                        piece_block["char_start"] = next_char_start + int(piece["char_start"])
                        piece_block["char_end"] = next_char_start + int(piece["char_end"])
                        piece_body_text = str(piece["content"])
                        combined_blocks = [{**block, "text": block_text}, piece_block] if window_index == 0 else [piece_block]
                        combined_body_text = f"{block_text}\n\n{piece_body_text}" if window_index == 0 else piece_body_text
                        append_chunk(
                            combined_blocks,
                            supporting_context_rows=_select_supporting_context(normalized_blocks, table_blocks),
                            body_text=combined_body_text,
                        )
            block_index += 2
            continue

        if isolate_block and pending_blocks:
            flush_pending()

        if block_length > chunk_size:
            flush_pending()
            char_start = int(block.get("char_start") or 0)
            for piece in _window_text(block_text, chunk_size=chunk_size, overlap=overlap):
                piece_block = dict(block)
                piece_block["text"] = str(piece["content"])
                piece_block["char_start"] = char_start + int(piece["char_start"])
                piece_block["char_end"] = char_start + int(piece["char_end"])
                append_chunk(
                    [piece_block],
                    supporting_context_rows=_select_supporting_context(normalized_blocks, [block]),
                    body_text=str(piece["content"]),
                )
            block_index += 1
            continue

        projected_length = pending_length + block_length + (2 if pending_blocks else 0)
        if pending_blocks and projected_length > child_chunk_size:
            flush_pending()

        pending_blocks.append({**block, "text": block_text})
        pending_length += block_length + (2 if len(pending_blocks) > 1 else 0)
        if isolate_block:
            flush_pending()
        block_index += 1

    flush_pending()
    if not chunks:
        return _split_text_legacy(raw_text, paper_title=paper_title)
    return chunks


def rebuild_paper_index(
    db: Session,
    *,
    paper_id: int,
    raw_text: str,
    preanalysis: dict[str, Any] | None = None,
    paper_title: str | None = None,
) -> int:
    chunks = _split_text(raw_text, preanalysis=preanalysis, paper_title=paper_title)
    embeddings: list[list[float]] = []
    if chunks and (settings.glm_api_key.strip() or settings.openai_api_key.strip()):
        try:
            embeddings = embed_texts([str(chunk.get("embedding_input") or chunk["content"]) for chunk in chunks])
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
                page_from=chunk["page_from"],
                page_to=chunk["page_to"],
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
    exact_terms = _extract_exact_match_terms(query)
    exact_match_heavy = _is_exact_match_heavy_query(query, exact_terms)
    exact_terms_for_boost = exact_terms if exact_match_heavy or _is_table_or_figure_query(query) else []
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
                    "exact_match_terms": exact_terms,
                    "exact_match_terms_applied": exact_terms_for_boost,
                    "exact_match_heavy": exact_match_heavy,
                    "rerank_status": "not_applicable",
                }
            )
        return results

    query_embedding: list[float] | None = None
    if settings.glm_api_key.strip() or settings.openai_api_key.strip():
        try:
            query_embedding = embed_texts([query])[0]
        except Exception:
            query_embedding = None

    sparse_limit = max(settings.rag_sparse_top_k, limit)
    dense_limit = max(settings.rag_dense_top_k, limit)
    if exact_match_heavy:
        sparse_limit = max(sparse_limit, limit * 8, settings.rag_sparse_top_k * 2)
        dense_limit = max(dense_limit, limit * 4, settings.rag_dense_top_k)
    sparse_hits = _search_sparse_chunks_postgres(db, query=query, limit=sparse_limit, exact_terms=exact_terms_for_boost)
    dense_hits = _search_dense_chunks_postgres(
        db,
        query_embedding=query_embedding,
        limit=dense_limit,
    )
    fusion_limit = max(settings.rag_fusion_window, settings.rag_rerank_top_n, limit, sparse_limit if exact_match_heavy else 0)
    fused_candidates = _fuse_ranked_candidates(sparse_hits, dense_hits, limit=fusion_limit)
    record_map = _load_chunk_record_map(
        db,
        chunk_ids=[item["chunk_id"] for item in [*sparse_hits, *dense_hits, *fused_candidates]],
    )
    if exact_terms_for_boost:
        fused_candidates = _boost_exact_match_candidates(query, candidates=fused_candidates, record_map=record_map)[:fusion_limit]

    reranked_candidates = fused_candidates
    rerank_status = "skipped"
    rerank_error: str | None = None
    if fused_candidates:
        try:
            reranked_candidates = _apply_reranking(
                query,
                fused_candidates=fused_candidates,
                record_map=record_map,
                limit=max(settings.rag_rerank_top_n, limit),
            )
            rerank_status = "applied"
        except Exception:
            reranked_candidates = fused_candidates[: max(settings.rag_rerank_top_n, limit)]
            rerank_status = "fallback_to_fused"
            rerank_error = "rerank_failed"

    results = _build_retrieval_results(reranked_candidates[:limit], record_map=record_map)
    if trace is not None:
        trace.update(
            {
                "sparse_candidates": _serialize_ranked_trace_candidates(sparse_hits, record_map),
                "dense_candidates": _serialize_ranked_trace_candidates(dense_hits, record_map),
                "fused_candidates": _serialize_ranked_trace_candidates(fused_candidates, record_map),
                "reranked_candidates": _serialize_ranked_trace_candidates(reranked_candidates, record_map),
                "retrieval_backend": "postgres_hybrid",
                "exact_match_terms": exact_terms,
                "exact_match_terms_applied": exact_terms_for_boost,
                "exact_match_heavy": exact_match_heavy,
                "sparse_limit": sparse_limit,
                "dense_limit": dense_limit,
                "fusion_limit": fusion_limit,
                "rerank_status": rerank_status,
                "rerank_error": rerank_error,
            }
        )
    return results


def _serialize_citations(results: list[RetrievalResult]) -> list[dict]:
    citations: list[dict] = []
    for result in results:
        snippet = _clip_snippet(_chunk_text_payload(result.chunk), max_length=240)
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
        snippet = _clip_snippet(_chunk_text_payload(result.chunk), max_length=180)
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
