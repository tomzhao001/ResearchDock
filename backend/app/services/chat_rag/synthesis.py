from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.services import rag as legacy_rag
from app.services.chat_rag import routing
from app.services.chat_rag.state import ChatRagGraphState


def _infer_actual_answer_shape(*, answer_text: str, answer_mode: str | None) -> str:
    text = (answer_text or "").strip()
    if answer_mode == "knowledge_base":
        return "paragraph"
    if text.startswith("匹配文档如下") or "\n- " in text:
        return "list"
    if "共找到" in text or "未找到匹配文档" in text or "约 " in text:
        return "scalar"
    return "paragraph"


def normalize_engine_output(state: ChatRagGraphState) -> dict[str, Any]:
    draft = state.get("draft")
    engine_name = str(state.get("engine_name") or "rag_engine")
    engine_result = state.get("engine_result") if isinstance(state.get("engine_result"), dict) else {}
    if draft is None:
        return {
            "engine_name": engine_name,
            "answer_text": "",
            "answer_mode": None,
            "actual_answer_shape": "paragraph",
            "paper_scope_ids": [int(item) for item in state.get("paper_scope_ids", [])],
            "engine_result": engine_result,
        }
    return {
        "engine_name": engine_name,
        "answer_text": draft.content,
        "answer_mode": draft.answer_mode,
        "actual_answer_shape": _infer_actual_answer_shape(answer_text=draft.content, answer_mode=draft.answer_mode),
        "paper_scope_ids": [int(item) for item in state.get("paper_scope_ids", [])],
        "engine_result": engine_result,
    }


def validate_answer_shape(state: ChatRagGraphState) -> dict[str, Any]:
    normalized = state.get("normalized_engine_output") if isinstance(state.get("normalized_engine_output"), dict) else {}
    expected = str(state.get("answer_shape") or "paragraph")
    actual = str(normalized.get("actual_answer_shape") or "paragraph")
    engine_name = str(normalized.get("engine_name") or state.get("engine_name") or "rag_engine")
    intent_family = str(state.get("intent_family") or "rag")
    metadata_query_plan = state.get("metadata_query_plan") if isinstance(state.get("metadata_query_plan"), dict) else {}
    mismatch = expected == "paragraph" and actual in {"list", "scalar"} and engine_name == "metadata_engine"
    reroute_engine = None
    if mismatch:
        if (
            state.get("paper_scope_ids")
            or (state.get("conversation_context") or {}).get("paper_scope_ids")
            or metadata_query_plan.get("wants_content_followup")
            or str(metadata_query_plan.get("operation") or "") == "filter_scope"
        ):
            reroute_engine = "hybrid_sql_rag_engine"
        elif intent_family in {"content_extraction", "summary", "comparison"}:
            reroute_engine = "rag_engine"
        else:
            reroute_engine = "hybrid_sql_rag_engine"
    return {
        "expected_answer_shape": expected,
        "actual_answer_shape": actual,
        "mismatch": mismatch,
        "reroute_engine": reroute_engine,
    }


def synthesize_result(state: ChatRagGraphState) -> dict[str, Any]:
    draft = state.get("draft")
    normalized = state.get("normalized_engine_output") if isinstance(state.get("normalized_engine_output"), dict) else {}
    validation = state.get("answer_shape_validation") if isinstance(state.get("answer_shape_validation"), dict) else {}
    synthesis_result = {
        "engine_name": state.get("engine_name"),
        "normalized_engine_output": normalized,
        "answer_shape_validation": validation,
    }
    if draft is None:
        return {"synthesis_result": synthesis_result}
    metadata_json = dict(draft.metadata_json or {})
    retrieval = dict(metadata_json.get("retrieval") or {})
    retrieval["normalized_engine_output"] = normalized
    retrieval["answer_shape_validation"] = validation
    retrieval["selector_result"] = state.get("selector_result")
    retrieval["engine_candidates"] = state.get("engine_candidates")
    retrieval["router_debug"] = state.get("router_debug")
    retrieval["synthesis_result"] = synthesis_result
    retrieval["route_decision"] = routing.serialize_route_decision(state.get("route_decision"))
    metadata_json["retrieval"] = retrieval
    return {
        "draft": replace(draft, metadata_json=metadata_json),
        "synthesis_result": synthesis_result,
    }
