from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models import ChatMessage
from app.services import rag as legacy_rag
from app.services.chat_rag import metadata_query
from app.services.chat_rag import router_prompts


@dataclass(frozen=True)
class EngineCandidate:
    name: str
    description: str
    capabilities: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    priority: int = 0
    safety_mode: str = "default"


@dataclass(frozen=True)
class RouteDecision:
    engine_name: str
    confidence: float
    reason: str
    fallback_engine: str = "rag_engine"
    decision_source: str = "rules"
    low_confidence: bool = False
    selector_reason: str | None = None
    intent_family: str = "rag"
    answer_shape: str = "paragraph"
    candidate_scores: dict[str, float] | None = None
    conversation_scope_used: bool = False
    rag_followup_query: str | None = None
    required_operations: tuple[str, ...] = ()
    filters: dict[str, Any] | None = None
    aggregation: str | None = None


_COUNT_HINTS = ("有几个", "多少篇", "多少个文档", "几篇", "几份", "how many", "count")
_LIST_HINTS = ("有哪些", "列出", "哪些文档", "清单", "list", "show me", "which papers", "which documents")
_EXISTS_HINTS = ("有没有", "是否有", "exists", "is there", "are there")
_WORD_COUNT_HINTS = ("多少字", "字数", "多少词", "词数", "word count")
_SUMMARY_HINTS = ("总结", "概括", "综述", "overview", "summarize", "summary")
_COMPARISON_HINTS = ("比较", "区别", "差异", "对比", "compare", "difference")
_CONTENT_HINTS = (
    "是什么",
    "为什么",
    "如何",
    "研究什么",
    "讲了什么",
    "主要研究",
    "主要研究什么",
    "主要内容",
    "主要内容是什么",
    "研究结局",
    "结局变量",
    "终点",
    "指标",
    "变量",
    "结果",
    "主要结论",
    "主要发现",
    "研究发现",
    "主要结果",
    "研究内容",
    "机制",
    "outcome",
    "outcomes",
    "endpoint",
    "endpoints",
    "variable",
    "variables",
    "findings",
    "result",
    "results",
    "conclusion",
    "conclusions",
    "what is",
    "what are",
    "what do",
    "what does",
    "why",
    "how",
    "which",
    "explain",
    "describe",
)
_CONVERSATION_SCOPE_HINTS = ("这些论文", "这些文档", "这些研究", "上述论文", "上面这些论文", "these papers", "these documents", "those papers")


def build_engine_candidates() -> list[EngineCandidate]:
    return [
        EngineCandidate(
            name="rag_engine",
            description="适合回答论文正文、表格、图、术语、实验结果等需要证据引用的问题。",
            capabilities=("fact_lookup", "citation_grounded_answer", "content_extraction"),
            examples=("文中的 tES 指什么？", "Table 3 里报告了什么？"),
            priority=100,
            safety_mode="grounded",
        ),
        EngineCandidate(
            name="metadata_engine",
            description="适合计数、列举、过滤、是否存在、简单元数据统计类问题。",
            capabilities=("count", "list", "exists", "lightweight_aggregation"),
            examples=("有几个中文文档？", "列出 DOI 缺失的论文"),
            priority=80,
            safety_mode="structured",
        ),
        EngineCandidate(
            name="hybrid_sql_rag_engine",
            description="适合先做结构化筛选或统计，再继续回答这些文档内容或主要结论的问题。",
            capabilities=("structured_prefilter", "scoped_rag", "content_followup"),
            examples=("缺失 DOI 的论文主要研究什么？", "近三年英文论文主要结论是什么？"),
            priority=90,
            safety_mode="hybrid",
        ),
    ]


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in patterns)


def _infer_intent_family(query: str) -> str:
    normalized = legacy_rag._normalize_query_text(query).lower()
    if _contains_any(normalized, _WORD_COUNT_HINTS):
        return "single_paper_stats"
    if _contains_any(normalized, _COUNT_HINTS):
        return "count"
    if _contains_any(normalized, _EXISTS_HINTS):
        return "exists"
    if _contains_any(normalized, _COMPARISON_HINTS):
        return "comparison"
    if _contains_any(normalized, _SUMMARY_HINTS):
        return "summary"
    if _contains_any(normalized, _CONTENT_HINTS):
        return "content_extraction"
    if _contains_any(normalized, _LIST_HINTS):
        return "list"
    return "rag"


def _answer_shape_for_intent(intent_family: str) -> str:
    if intent_family in {"count", "exists", "single_paper_stats"}:
        return "scalar"
    if intent_family == "list":
        return "list"
    return "paragraph"


def _looks_like_content_followup(
    *,
    query: str,
    intent_family: str,
    metadata_plan: metadata_query.MetadataQueryPlan | None,
) -> bool:
    normalized = legacy_rag._normalize_query_text(query).lower()
    if intent_family in {"content_extraction", "summary", "comparison"}:
        return True
    if _contains_any(normalized, _CONTENT_HINTS) or _contains_any(normalized, _SUMMARY_HINTS) or _contains_any(normalized, _COMPARISON_HINTS):
        return True
    if metadata_plan is not None and metadata_plan.wants_content_followup:
        return True
    return False


def _extract_conversation_context(records: list[ChatMessage]) -> dict[str, Any]:
    for record in reversed(records):
        if record.role != "assistant" or not isinstance(record.metadata_json, dict):
            continue
        retrieval = record.metadata_json.get("retrieval")
        if not isinstance(retrieval, dict):
            continue
        paper_scope_ids = [
            int(item)
            for item in (
                retrieval.get("paper_scope_ids")
                or (retrieval.get("engine_execution") or {}).get("paper_ids")
                or ((retrieval.get("engine_execution") or {}).get("structured_phase") or {}).get("paper_ids")
                or ((retrieval.get("engine_execution") or {}).get("scoped_rag_phase") or {}).get("paper_scope_ids")
                or []
            )
            if str(item).isdigit()
        ]
        return {
            "last_engine_name": retrieval.get("engine_name"),
            "paper_scope_ids": paper_scope_ids,
            "last_answer_mode": retrieval.get("answer_mode"),
            "last_route_decision": retrieval.get("route_decision"),
        }
    return {"last_engine_name": None, "paper_scope_ids": [], "last_answer_mode": None, "last_route_decision": None}


def build_route_plan(query: str, *, records: list[ChatMessage] | None = None) -> dict[str, Any]:
    metadata_plan = metadata_query.build_metadata_query_plan(query)
    candidates = build_engine_candidates()
    intent_family = _infer_intent_family(query)
    answer_shape = _answer_shape_for_intent(intent_family)
    conversation_context = _extract_conversation_context(records or [])
    return {
        "query": query,
        "intent_family": intent_family,
        "answer_shape": answer_shape,
        "candidates": [serialize_engine_candidate(item) for item in candidates],
        "metadata_query_plan": metadata_query.serialize_metadata_query_plan(metadata_plan),
        "conversation_context": conversation_context,
    }


def deserialize_metadata_plan(plan_payload: dict[str, Any] | None) -> metadata_query.MetadataQueryPlan | None:
    if not isinstance(plan_payload, dict):
        return None
    return metadata_query.MetadataQueryPlan(
        operation=str(plan_payload.get("operation") or "list"),
        filters=dict(plan_payload.get("filters") or {}),
        limit=int(plan_payload.get("limit") or 8),
        wants_content_followup=bool(plan_payload.get("wants_content_followup")),
        followup_query=str(plan_payload.get("followup_query") or "") or None,
        title_hint=str(plan_payload.get("title_hint") or "") or None,
    )


def rule_route(query: str, *, route_plan: dict[str, Any] | None = None) -> RouteDecision:
    route_plan = route_plan if isinstance(route_plan, dict) else {}
    metadata_plan = metadata_query.build_metadata_query_plan(query) or deserialize_metadata_plan(route_plan.get("metadata_query_plan"))
    intent_family = str(route_plan.get("intent_family") or _infer_intent_family(query))
    answer_shape = str(route_plan.get("answer_shape") or _answer_shape_for_intent(intent_family))
    conversation_context = route_plan.get("conversation_context") if isinstance(route_plan.get("conversation_context"), dict) else {}
    paper_scope_ids = [int(item) for item in conversation_context.get("paper_scope_ids", []) if str(item).isdigit()]
    referential_followup = _contains_any(query, _CONVERSATION_SCOPE_HINTS)
    conversation_scope_used = bool(paper_scope_ids and referential_followup)
    content_followup = _looks_like_content_followup(
        query=query,
        intent_family=intent_family,
        metadata_plan=metadata_plan,
    )

    if conversation_scope_used and content_followup:
        return RouteDecision(
            engine_name="hybrid_sql_rag_engine",
            confidence=0.94,
            reason="conversation_scope_content_followup",
            decision_source="rules",
            intent_family=intent_family,
            answer_shape=answer_shape,
            conversation_scope_used=True,
            required_operations=("conversation_scope", "rag_followup"),
        )

    if metadata_plan is not None and metadata_plan.operation in {"count", "exists", "paper_word_count"}:
        return RouteDecision(
            engine_name="metadata_engine",
            confidence=0.96,
            reason="metadata_high_precision_operation",
            decision_source="rules",
            intent_family=intent_family,
            answer_shape=answer_shape,
            required_operations=("metadata_query",),
            filters=dict(metadata_plan.filters),
            aggregation=metadata_plan.operation,
        )

    if metadata_plan is not None:
        if metadata_plan.wants_content_followup:
            return RouteDecision(
                engine_name="hybrid_sql_rag_engine",
                confidence=0.82,
                reason="metadata_query_with_content_followup",
                decision_source="rules",
                intent_family=intent_family,
                answer_shape="paragraph",
                rag_followup_query=metadata_plan.followup_query,
                required_operations=("metadata_query", "rag_followup"),
                filters=dict(metadata_plan.filters),
                aggregation=metadata_plan.operation,
            )
        if metadata_plan.operation == "list" and intent_family == "list":
            return RouteDecision(
                engine_name="metadata_engine",
                confidence=0.91,
                reason="explicit_list_intent",
                decision_source="rules",
                intent_family=intent_family,
                answer_shape=answer_shape,
                required_operations=("metadata_query",),
                filters=dict(metadata_plan.filters),
                aggregation=metadata_plan.operation,
            )
        if metadata_plan.operation == "filter_scope":
            return RouteDecision(
                engine_name="hybrid_sql_rag_engine",
                confidence=0.58,
                reason="metadata_filter_scope_needs_selector",
                decision_source="rules",
                low_confidence=True,
                intent_family=intent_family,
                answer_shape=answer_shape,
                rag_followup_query=metadata_plan.followup_query,
                required_operations=("metadata_query",),
                filters=dict(metadata_plan.filters),
                aggregation=metadata_plan.operation,
            )

    if intent_family in {"content_extraction", "summary", "comparison"}:
        return RouteDecision(
            engine_name="rag_engine",
            confidence=0.92,
            reason="content_intent_to_rag",
            decision_source="rules",
            intent_family=intent_family,
            answer_shape=answer_shape,
            required_operations=("rag",),
        )

    return RouteDecision(
        engine_name="rag_engine",
        confidence=0.9,
        reason="default_rag_route",
        decision_source="rules",
        low_confidence=False,
        intent_family=intent_family,
        answer_shape=answer_shape,
        required_operations=("rag",),
    )


def selector_route(query: str, *, route_plan: dict[str, Any] | None = None, base_decision: RouteDecision | None = None) -> tuple[RouteDecision, dict[str, Any] | None]:
    route_plan = route_plan if isinstance(route_plan, dict) else {}
    base_decision = base_decision or rule_route(query, route_plan=route_plan)
    llm_config = legacy_rag.get_chat_llm_configuration()
    if not llm_config.get("configured"):
        fallback = RouteDecision(
            engine_name=base_decision.fallback_engine,
            confidence=0.0,
            reason="selector_config_unavailable",
            decision_source="fallback",
            low_confidence=True,
            selector_reason="selector_unavailable",
            intent_family=base_decision.intent_family,
            answer_shape=base_decision.answer_shape,
            candidate_scores=base_decision.candidate_scores,
            conversation_scope_used=base_decision.conversation_scope_used,
            required_operations=("rag",),
        )
        return fallback, {"status": "config_unavailable", "fallback_engine": fallback.engine_name}

    prompt_messages = router_prompts.build_selector_messages(
        query=query,
        route_plan=route_plan,
        base_decision=serialize_route_decision(base_decision) or {},
    )
    content, model = legacy_rag.chat_with_messages(prompt_messages, temperature=0.0)
    payload = legacy_rag._parse_json_object(content)
    selected_engine = str(payload.get("engine_name") or base_decision.engine_name).strip() or base_decision.engine_name
    confidence = float(payload.get("confidence") or 0.0)
    selector_reason = legacy_rag._normalize_query_text(str(payload.get("reason") or "")) or "selector_chosen"
    intent_family = str(payload.get("intent_family") or base_decision.intent_family or "rag")
    answer_shape = str(payload.get("answer_shape") or base_decision.answer_shape or "paragraph")
    low_confidence = confidence < 0.6
    resolved_engine = selected_engine if not low_confidence else base_decision.fallback_engine
    decision = RouteDecision(
        engine_name=resolved_engine,
        confidence=confidence,
        reason="selector_chosen" if not low_confidence else "selector_low_confidence_fallback",
        fallback_engine=base_decision.fallback_engine,
        decision_source="selector" if not low_confidence else "fallback",
        low_confidence=low_confidence,
        selector_reason=selector_reason,
        intent_family=intent_family,
        answer_shape=answer_shape,
        candidate_scores={
            str(item["name"]): float(item.get("priority") or 0.0)
            for item in route_plan.get("candidates", [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        },
        conversation_scope_used=base_decision.conversation_scope_used,
        rag_followup_query=base_decision.rag_followup_query,
        required_operations=base_decision.required_operations,
        filters=base_decision.filters,
        aggregation=base_decision.aggregation,
    )
    selector_result = {
        "status": "ok",
        "model": model,
        "raw": payload,
    }
    return decision, selector_result


def route_query(query: str, *, route_plan: dict[str, Any] | None = None) -> RouteDecision:
    base_decision = rule_route(query, route_plan=route_plan)
    if not base_decision.low_confidence:
        return base_decision
    decision, _selector_result = selector_route(query, route_plan=route_plan, base_decision=base_decision)
    return decision


def serialize_engine_candidate(candidate: EngineCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "description": candidate.description,
        "capabilities": list(candidate.capabilities),
        "examples": list(candidate.examples),
        "priority": candidate.priority,
        "safety_mode": candidate.safety_mode,
    }


def serialize_route_decision(decision: RouteDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "engine_name": decision.engine_name,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "fallback_engine": decision.fallback_engine,
        "decision_source": decision.decision_source,
        "low_confidence": decision.low_confidence,
        "selector_reason": decision.selector_reason,
        "intent_family": decision.intent_family,
        "answer_shape": decision.answer_shape,
        "candidate_scores": dict(decision.candidate_scores or {}),
        "conversation_scope_used": decision.conversation_scope_used,
        "rag_followup_query": decision.rag_followup_query,
        "required_operations": list(decision.required_operations),
        "filters": dict(decision.filters or {}),
        "aggregation": decision.aggregation,
    }
