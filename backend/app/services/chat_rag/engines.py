from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services import rag as legacy_rag
from app.services.chat_rag import metadata_query
from app.services.chat_rag.routing import RouteDecision, serialize_route_decision
from app.services.chat_rag.state import ChatRagGraphState


@dataclass(frozen=True)
class EngineExecutionResult:
    engine_name: str
    draft: legacy_rag.AssistantMessageDraft | None = None
    retrieval_trace: dict[str, Any] | None = None
    paper_scope_ids: tuple[int, ...] = ()
    retrieval_query_override: str | None = None
    metadata_query_result: dict[str, Any] | None = None


def _base_trace(
    state: ChatRagGraphState,
    *,
    engine_name: str,
    route_plan: dict[str, Any] | None,
    route_decision: RouteDecision | None,
) -> dict[str, Any]:
    return {
        "original_user_query": state["message"],
        "retrieval_backend": engine_name,
        "route_plan": route_plan,
        "route_decision": serialize_route_decision(route_decision),
        "engine_name": engine_name,
        "intent_family": state.get("intent_family"),
        "answer_shape": state.get("answer_shape"),
        "selector_result": state.get("selector_result"),
        "engine_candidates": state.get("engine_candidates"),
        "conversation_context": state.get("conversation_context"),
        "verifier_result": None,
    }


class MetadataQueryEngine:
    def run(self, state: ChatRagGraphState) -> dict[str, Any]:
        progress_callback = state.get("progress_callback")
        route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else None
        route_decision = state.get("route_decision")
        metadata_plan = metadata_query.deserialize_metadata_query_plan(state.get("metadata_query_plan"))
        if metadata_plan is None:
            metadata_plan = metadata_query.build_metadata_query_plan(state["message"])
        if metadata_plan is None:
            return {}

        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="engine_execution",
            status="started",
            message="正在执行元数据查询",
        )
        result = metadata_query.execute_metadata_query(
            state["db"],
            organization_id=state["user"].organization_id,
            plan=metadata_plan,
        )
        retrieval_trace = _base_trace(
            state,
            engine_name="metadata_engine",
            route_plan=route_plan,
            route_decision=route_decision if isinstance(route_decision, RouteDecision) else None,
        )
        retrieval_trace["metadata_query_plan"] = metadata_query.serialize_metadata_query_plan(metadata_plan)
        retrieval_trace["engine_execution"] = metadata_query.serialize_metadata_query_result(result)
        retrieval_trace["answer_mode"] = "metadata_query"
        retrieval_trace["fallback_used"] = False
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="engine_execution",
            status="finished",
            message="元数据查询完成",
            detail=f"匹配文档 {result.total_count} 篇",
        )
        draft = legacy_rag.AssistantMessageDraft(
            content=result.answer,
            model=None,
            answer_mode="metadata_query",
            used_knowledge_base=True,
            citations_json=[],
            metadata_json={"retrieval": retrieval_trace},
        )
        return {
            "draft": draft,
            "engine_name": "metadata_engine",
            "metadata_query_plan": metadata_query.serialize_metadata_query_plan(metadata_plan),
            "engine_result": metadata_query.serialize_metadata_query_result(result),
        }


class HybridScopedRagEngine:
    def prefilter(self, state: ChatRagGraphState) -> dict[str, Any]:
        progress_callback = state.get("progress_callback")
        route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else None
        route_decision = state.get("route_decision")
        metadata_plan = metadata_query.deserialize_metadata_query_plan(state.get("metadata_query_plan"))
        if metadata_plan is None:
            metadata_plan = metadata_query.build_metadata_query_plan(state["message"])
        conversation_context = state.get("conversation_context") if isinstance(state.get("conversation_context"), dict) else {}
        conversation_scope_ids = [int(item) for item in conversation_context.get("paper_scope_ids", []) if str(item).isdigit()]
        followup_query = (
            route_decision.rag_followup_query
            if isinstance(route_decision, RouteDecision) and route_decision.rag_followup_query
            else state["message"]
        )
        if metadata_plan is None and conversation_scope_ids:
            engine_result = {
                "structured_phase": {
                    "source": "conversation_context",
                    "paper_ids": list(conversation_scope_ids),
                },
                "scoped_rag_phase": {
                    "paper_scope_ids": list(conversation_scope_ids),
                    "retrieval_query_override": followup_query,
                },
            }
            return {
                "engine_name": "hybrid_sql_rag_engine",
                "engine_result": engine_result,
                "paper_scope_ids": list(conversation_scope_ids),
                "retrieval_query_override": followup_query,
            }
        if metadata_plan is None:
            return {}

        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="engine_execution",
            status="started",
            message="正在执行结构化筛选",
        )
        result = metadata_query.execute_metadata_query(
            state["db"],
            organization_id=state["user"].organization_id,
            plan=metadata_plan,
        )
        base_trace = _base_trace(
            state,
            engine_name="hybrid_sql_rag_engine",
            route_plan=route_plan,
            route_decision=route_decision if isinstance(route_decision, RouteDecision) else None,
        )
        base_trace["metadata_query_plan"] = metadata_query.serialize_metadata_query_plan(metadata_plan)
        followup_query = (
            route_decision.rag_followup_query
            if isinstance(route_decision, RouteDecision) and route_decision.rag_followup_query
            else metadata_plan.followup_query
            or state["message"]
        )
        engine_result = {
            "structured_phase": metadata_query.serialize_metadata_query_result(result),
            "scoped_rag_phase": {
                "paper_scope_ids": list(result.paper_ids),
                "retrieval_query_override": followup_query,
            },
        }
        base_trace["engine_execution"] = engine_result

        if not result.paper_ids:
            base_trace["answer_mode"] = "metadata_query"
            base_trace["fallback_used"] = False
            draft = legacy_rag.AssistantMessageDraft(
                content=f"{result.answer} 因为没有匹配文档，暂时无法继续生成基于论文内容的补充回答。",
                model=None,
                answer_mode="metadata_query",
                used_knowledge_base=True,
                citations_json=[],
                metadata_json={"retrieval": base_trace},
            )
            legacy_rag._emit_chat_progress(
                progress_callback,
                phase="engine_execution",
                status="finished",
                message="结构化筛选完成",
                detail="未找到匹配文档",
            )
            return {
                "draft": draft,
                "engine_name": "hybrid_sql_rag_engine",
                "metadata_query_plan": metadata_query.serialize_metadata_query_plan(metadata_plan),
                "engine_result": engine_result,
            }

        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="engine_execution",
            status="finished",
            message="结构化筛选完成，继续检索论文内容",
            detail=f"候选文档 {len(result.paper_ids)} 篇",
        )
        return {
            "engine_name": "hybrid_sql_rag_engine",
            "metadata_query_plan": metadata_query.serialize_metadata_query_plan(metadata_plan),
            "engine_result": engine_result,
            "paper_scope_ids": list(result.paper_ids),
            "retrieval_query_override": followup_query,
        }
