from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.chat_rag import metadata_query


@dataclass(frozen=True)
class EngineCandidate:
    name: str
    description: str


@dataclass(frozen=True)
class RouteDecision:
    engine_name: str
    confidence: float
    reason: str
    fallback_engine: str = "rag_engine"
    rag_followup_query: str | None = None
    required_operations: tuple[str, ...] = ()
    filters: dict[str, Any] | None = None
    aggregation: str | None = None


def build_route_plan(query: str) -> dict[str, Any]:
    metadata_plan = metadata_query.build_metadata_query_plan(query)
    candidates = [
        EngineCandidate(
            name="rag_engine",
            description="适合回答论文正文、表格、图、术语、实验结果等需要证据引用的问题。",
        ),
        EngineCandidate(
            name="metadata_engine",
            description="适合计数、列举、过滤、是否存在、简单元数据统计类问题。",
        ),
        EngineCandidate(
            name="hybrid_sql_rag_engine",
            description="适合先做结构化筛选或统计，再继续回答这些文档内容或主要结论的问题。",
        ),
    ]
    return {
        "query": query,
        "candidates": [{"name": item.name, "description": item.description} for item in candidates],
        "metadata_query_plan": metadata_query.serialize_metadata_query_plan(metadata_plan),
    }


def route_query(query: str, *, route_plan: dict[str, Any] | None = None) -> RouteDecision:
    metadata_plan = metadata_query.build_metadata_query_plan(query)
    if metadata_plan is None and isinstance(route_plan, dict):
        plan_payload = route_plan.get("metadata_query_plan")
        if isinstance(plan_payload, dict):
            metadata_plan = metadata_query.MetadataQueryPlan(
                operation=str(plan_payload.get("operation") or "list"),
                filters=dict(plan_payload.get("filters") or {}),
                limit=int(plan_payload.get("limit") or 8),
                wants_content_followup=bool(plan_payload.get("wants_content_followup")),
                followup_query=str(plan_payload.get("followup_query") or "") or None,
                title_hint=str(plan_payload.get("title_hint") or "") or None,
            )

    if metadata_plan is not None:
        if metadata_plan.wants_content_followup:
            return RouteDecision(
                engine_name="hybrid_sql_rag_engine",
                confidence=0.82,
                reason="metadata_query_with_content_followup",
                rag_followup_query=metadata_plan.followup_query,
                required_operations=("metadata_query", "rag_followup"),
                filters=dict(metadata_plan.filters),
                aggregation=metadata_plan.operation,
            )
        return RouteDecision(
            engine_name="metadata_engine",
            confidence=0.9,
            reason="metadata_query_detected",
            required_operations=("metadata_query",),
            filters=dict(metadata_plan.filters),
            aggregation=metadata_plan.operation,
        )

    return RouteDecision(
        engine_name="rag_engine",
        confidence=0.95,
        reason="default_rag_route",
        required_operations=("rag",),
    )


def serialize_route_decision(decision: RouteDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "engine_name": decision.engine_name,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "fallback_engine": decision.fallback_engine,
        "rag_followup_query": decision.rag_followup_query,
        "required_operations": list(decision.required_operations),
        "filters": dict(decision.filters or {}),
        "aggregation": decision.aggregation,
    }
