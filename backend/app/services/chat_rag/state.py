from __future__ import annotations

from typing import Any, TypedDict

from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatTopic, User


class ChatRagGraphState(TypedDict, total=False):
    db: Session
    user: User
    topic: ChatTopic
    records: list[ChatMessage]
    message: str
    progress_callback: Any
    relaxed_chat_rag: bool
    started_at: float
    selection_started_at: float
    chat_policy: Any
    route_plan: dict[str, Any]
    route_decision: Any
    engine_name: str
    engine_inputs: dict[str, Any]
    engine_result: dict[str, Any]
    router_debug: dict[str, Any]
    structured_answer: dict[str, Any]
    needs_rag_followup: bool
    metadata_query_plan: dict[str, Any]
    metadata_query_result: dict[str, Any]
    paper_scope_ids: list[int]
    retrieval_query_override: str
    retrieval_query: str
    candidate_limit: int
    retrieval_debug: dict[str, Any]
    retrieval_candidates: list[Any]
    retrieval_results: list[Any]
    evidence_candidates: list[dict[str, Any]]
    selection_result: Any
    selected_evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    draft: Any
    fallback_reason: str | None
