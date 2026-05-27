from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatTopic, User
from app.services import rag as legacy_rag
from app.services.chat_rag import evidence as evidence_ops
from app.services.chat_rag import engines as engine_ops
from app.services.chat_rag import generation as generation_ops
from app.services.chat_rag.repository import ChatTopicRepository
from app.services.chat_rag import retrieval as retrieval_ops
from app.services.chat_rag import routing as routing_ops
from app.services.chat_rag.state import ChatRagGraphState
from app.services.chat_rag import synthesis as synthesis_ops


@dataclass
class ChatRouteService:
    def plan(self, state: ChatRagGraphState) -> dict[str, Any]:
        route_plan = routing_ops.build_route_plan(state["message"], records=state["records"])
        return {
            "route_plan": route_plan,
            "intent_family": route_plan.get("intent_family"),
            "answer_shape": route_plan.get("answer_shape"),
            "engine_candidates": route_plan.get("candidates"),
            "conversation_context": route_plan.get("conversation_context"),
            "metadata_query_plan": route_plan.get("metadata_query_plan") if isinstance(route_plan.get("metadata_query_plan"), dict) else None,
        }

    def rule_route(self, state: ChatRagGraphState) -> dict[str, Any]:
        progress_callback = state.get("progress_callback")
        route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="routing",
            status="started",
            message="正在分析问题类型",
        )
        route_decision = routing_ops.rule_route(state["message"], route_plan=route_plan)
        engine_name_labels = {
            "rag_engine": "知识库检索回答",
            "metadata_engine": "元数据查询",
            "hybrid_sql_rag_engine": "结构化筛选后检索回答",
        }
        if not route_decision.low_confidence:
            legacy_rag._emit_chat_progress(
                progress_callback,
                phase="routing",
                status="finished",
                message="问题分流完成",
                detail=engine_name_labels.get(route_decision.engine_name, route_decision.engine_name),
            )
        return {
            "route_decision": route_decision,
            "engine_name": route_decision.engine_name,
            "intent_family": route_decision.intent_family,
            "answer_shape": route_decision.answer_shape,
            "router_debug": {"rule_route": routing_ops.serialize_route_decision(route_decision)},
        }

    def selector_route(self, state: ChatRagGraphState) -> dict[str, Any]:
        progress_callback = state.get("progress_callback")
        route_plan = state.get("route_plan") if isinstance(state.get("route_plan"), dict) else {}
        base_decision = state.get("route_decision")
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="selector",
            status="started",
            message="规则置信度不足，正在调用 selector 进一步判断",
        )
        route_decision, selector_result = routing_ops.selector_route(
            state["message"],
            route_plan=route_plan,
            base_decision=base_decision if isinstance(base_decision, routing_ops.RouteDecision) else None,
        )
        engine_name_labels = {
            "rag_engine": "知识库检索回答",
            "metadata_engine": "元数据查询",
            "hybrid_sql_rag_engine": "结构化筛选后检索回答",
        }
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="selector",
            status="finished",
            message="selector 判断完成",
            detail=engine_name_labels.get(route_decision.engine_name, route_decision.engine_name),
        )
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="routing",
            status="finished",
            message="问题分流完成",
            detail=engine_name_labels.get(route_decision.engine_name, route_decision.engine_name),
        )
        return {
            "route_decision": route_decision,
            "engine_name": route_decision.engine_name,
            "intent_family": route_decision.intent_family,
            "answer_shape": route_decision.answer_shape,
            "selector_result": selector_result,
            "router_debug": {
                "rule_route": routing_ops.serialize_route_decision(base_decision if isinstance(base_decision, routing_ops.RouteDecision) else None),
                "selector_route": routing_ops.serialize_route_decision(route_decision),
            },
        }


@dataclass
class ChatSynthesisService:
    def normalize(self, state: ChatRagGraphState) -> dict[str, Any]:
        return {"normalized_engine_output": synthesis_ops.normalize_engine_output(state)}

    def validate(self, state: ChatRagGraphState) -> dict[str, Any]:
        validation = synthesis_ops.validate_answer_shape(state)
        router_debug = dict(state.get("router_debug") or {})
        if validation.get("mismatch"):
            router_debug["answer_shape_guard"] = {
                "triggered": True,
                "expected_answer_shape": validation.get("expected_answer_shape"),
                "actual_answer_shape": validation.get("actual_answer_shape"),
                "reroute_engine": validation.get("reroute_engine"),
            }
        payload: dict[str, Any] = {"answer_shape_validation": validation, "router_debug": router_debug}
        if validation.get("mismatch"):
            reroute_engine = str(validation.get("reroute_engine") or "rag_engine")
            existing_decision = state.get("route_decision")
            payload["engine_name"] = reroute_engine
            payload["draft"] = None
            if isinstance(existing_decision, routing_ops.RouteDecision):
                payload["route_decision"] = routing_ops.RouteDecision(
                    engine_name=reroute_engine,
                    confidence=existing_decision.confidence,
                    reason="answer_shape_mismatch_reroute",
                    fallback_engine=existing_decision.fallback_engine,
                    decision_source="fallback",
                    low_confidence=False,
                    selector_reason=existing_decision.selector_reason,
                    intent_family=existing_decision.intent_family,
                    answer_shape=existing_decision.answer_shape,
                    candidate_scores=existing_decision.candidate_scores,
                    conversation_scope_used=existing_decision.conversation_scope_used,
                    rag_followup_query=existing_decision.rag_followup_query,
                    required_operations=existing_decision.required_operations,
                    filters=existing_decision.filters,
                    aggregation=existing_decision.aggregation,
                )
        return payload

    def synthesize(self, state: ChatRagGraphState) -> dict[str, Any]:
        return synthesis_ops.synthesize_result(state)


@dataclass
class ChatRetrievalService:
    def run(self, state: ChatRagGraphState) -> dict[str, Any]:
        db = state["db"]
        user = state["user"]
        topic = state["topic"]
        records = state["records"]
        message = state["message"]
        progress_callback = state.get("progress_callback")

        retrieval_query_input = str(state.get("retrieval_query_override") or message)
        retrieval_query = retrieval_ops.build_retrieval_query(records, retrieval_query_input)
        candidate_limit = max(legacy_rag.settings.rag_top_k, legacy_rag.settings.rag_rerank_top_n, 10)
        retrieval_debug: dict[str, Any] = {}
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="retrieval",
            status="started",
            message="正在检索知识库",
            detail=legacy_rag._preview_log_text(retrieval_query),
        )
        legacy_rag.logger.info(
            "RAG retrieval started: topic_id=%s candidate_limit=%s query=%s",
            topic.id,
            candidate_limit,
            retrieval_query,
        )
        retrieval_candidates = retrieval_ops.search_chunks(
            db,
            query=retrieval_query,
            organization_id=user.organization_id,
            top_k=candidate_limit,
            trace=retrieval_debug,
            paper_ids=[int(item) for item in state.get("paper_scope_ids", [])],
        )
        retrieval_results = retrieval_candidates[: legacy_rag.settings.rag_top_k]
        evidence_candidates = retrieval_ops.build_evidence_candidates(retrieval_results, query=message)
        legacy_rag.logger.info(
            "RAG evidence candidate preparation finished: topic_id=%s retrieval_results=%s evidence_candidates=%s elapsed=%.2fs",
            topic.id,
            len(retrieval_results),
            len(evidence_candidates),
            time.perf_counter() - state["started_at"],
        )
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="retrieval",
            status="finished",
            message="知识库检索完成",
            detail=f"候选证据 {len(evidence_candidates)} 条",
        )
        return {
            "retrieval_query": retrieval_query,
            "candidate_limit": candidate_limit,
            "retrieval_debug": retrieval_debug,
            "retrieval_candidates": retrieval_candidates,
            "retrieval_results": retrieval_results,
            "evidence_candidates": evidence_candidates,
            "selection_started_at": time.perf_counter(),
        }


@dataclass
class ChatEvidenceService:
    def select(self, state: ChatRagGraphState) -> dict[str, Any]:
        progress_callback = state.get("progress_callback")
        retrieval_debug = state["retrieval_debug"]
        evidence_candidates = state["evidence_candidates"]
        message = state["message"]
        topic = state["topic"]

        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="evidence_selection",
            status="started",
            message="正在筛选可支撑答案的证据",
        )
        selection_result = evidence_ops.select_claim_supporting_evidence(
            question=message,
            query_plan=retrieval_debug.get("query_plan") if isinstance(retrieval_debug.get("query_plan"), dict) else {},
            evidence_candidates=evidence_candidates,
            policy=state["chat_policy"],
        )
        selected_evidence = [dict(item) for item in selection_result.selected_evidence]
        citations = evidence_ops.serialize_evidence_list(selected_evidence)
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="evidence_selection",
            status="finished",
            message="证据筛选完成",
            detail=f"选中证据 {len(selected_evidence)} 条",
        )
        legacy_rag.logger.info(
            "RAG retrieval finished: topic_id=%s backend=%s candidates=%s selected_evidence=%s elapsed=%.2fs",
            topic.id,
            retrieval_debug.get("retrieval_backend", "unknown"),
            len(state["retrieval_candidates"]),
            len(selected_evidence),
            time.perf_counter() - state["started_at"],
        )
        return {
            "selection_result": selection_result,
            "selected_evidence": selected_evidence,
            "citations": citations,
        }


@dataclass
class RetrievalTraceBuilder:
    def build(self, state: ChatRagGraphState) -> dict[str, Any]:
        retrieval_debug = state["retrieval_debug"]
        selection_result = state["selection_result"]
        evidence_candidates = state["evidence_candidates"]
        citations = state["citations"]
        retrieval_trace = {
            "original_user_query": state["message"],
            "retrieval_query": state["retrieval_query"],
            "route_plan": state.get("route_plan"),
            "route_decision": routing_ops.serialize_route_decision(state.get("route_decision")),
            "engine_candidates": state.get("engine_candidates"),
            "selector_result": state.get("selector_result"),
            "router_debug": state.get("router_debug"),
            "conversation_context": state.get("conversation_context"),
            "intent_family": state.get("intent_family"),
            "answer_shape": state.get("answer_shape"),
            "engine_name": state.get("engine_name") or "rag_engine",
            "query_plan": retrieval_debug.get("query_plan"),
            "retrieval_backend": retrieval_debug.get("retrieval_backend", "unknown"),
            "metadata_query_plan": state.get("metadata_query_plan"),
            "engine_execution": state.get("engine_result"),
            "first_pass_candidates": legacy_rag._serialize_trace_candidates(state["retrieval_candidates"]),
            "variant_candidates": retrieval_debug.get("variant_candidates", {}),
            "sparse_candidates": retrieval_debug.get("sparse_candidates", []),
            "dense_candidates": retrieval_debug.get("dense_candidates", []),
            "fused_candidates": retrieval_debug.get("fused_candidates", []),
            "reranked_candidates": retrieval_debug.get("reranked_candidates", []),
            "paper_scope_ids": [int(item) for item in state.get("paper_scope_ids", [])],
            "selected_citations": citations,
            "selected_evidence": citations,
            "evidence_selection_trace": {
                "method": selection_result.method,
                "candidate_evidence": evidence_ops.serialize_evidence_list(evidence_candidates),
                "claims": list(selection_result.claims),
                "overall_support_score": selection_result.overall_support_score,
                "missing_information": selection_result.missing_information,
            },
            "sufficiency_decision": selection_result.sufficiency_decision,
            "verifier_result": None,
            "answer_mode": "knowledge_base" if selection_result.sufficiency_decision.get("is_sufficient") else "kb_insufficient_evidence",
            "chat_response_policy": {
                "name": state["chat_policy"].name,
                "allow_fallback_generation": state["chat_policy"].allow_fallback_generation,
                "llm_insufficient_hard_gate": state["chat_policy"].llm_insufficient_hard_gate,
                "verifier_min_support_score": state["chat_policy"].verifier_min_support_score,
            },
            "generation_instruction": retrieval_debug.get("generation_instruction"),
            "rerank_query": retrieval_debug.get("rerank_query"),
            "search_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
            "evidence_selection_ms": round((time.perf_counter() - state["selection_started_at"]) * 1000, 2),
        }
        return {"retrieval_trace": retrieval_trace}


@dataclass
class ChatGenerationService:
    def generate(self, state: ChatRagGraphState) -> dict[str, Any]:
        progress_callback = state.get("progress_callback")
        selected_evidence = state["selected_evidence"]
        citations = state["citations"]
        retrieval_debug = state["retrieval_debug"]
        retrieval_trace = dict(state["retrieval_trace"])
        records = state["records"]
        message = state["message"]
        topic = state["topic"]
        selection_result = state["selection_result"]
        chat_policy = state["chat_policy"]

        generation_started_at = time.perf_counter()
        evidence_text = legacy_rag._build_evidence_prompt_text(selected_evidence)
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="generation",
            status="started",
            message="正在生成答案草稿",
        )
        legacy_rag.logger.info(
            "RAG generation started: topic_id=%s mode=knowledge_base citations=%s",
            topic.id,
            len(selected_evidence),
        )
        model: str | None = None
        final_model: str | None = None
        fallback_reason: str | None = None
        try:
            answer_started_at = time.perf_counter()
            legacy_rag.logger.info(
                "RAG answer draft started: topic_id=%s evidence=%s",
                topic.id,
                len(selected_evidence),
            )
            answer, model = legacy_rag.chat_with_messages(
                [
                    {"role": "system", "content": legacy_rag.RAG_SYSTEM_PROMPT},
                    *legacy_rag._history_messages(records),
                    {
                        "role": "user",
                        "content": (
                            f"用户问题：{message}\n\n"
                            f"回答要求：{retrieval_debug.get('generation_instruction') or '请用中文回答，保留关键英文术语原文，并引用英文证据。'}\n"
                            "请先从证据中抽取可以直接支持回答的 claim，再生成最终答案。\n"
                            "只允许使用下列证据支持的内容，避免补充证据中没有的结论。\n"
                            "如果证据仍然不足，请直接说明“知识库中未找到确切依据”。\n\n"
                            f"{evidence_text}"
                        ),
                    },
                ],
                temperature=0.2,
            )
            legacy_rag.logger.info(
                "RAG answer draft finished: topic_id=%s model=%s elapsed=%.2fs",
                topic.id,
                model,
                time.perf_counter() - answer_started_at,
            )
            legacy_rag._emit_chat_progress(
                progress_callback,
                phase="generation",
                status="finished",
                message="答案草稿已生成",
                detail=model,
            )
            legacy_rag._emit_chat_progress(
                progress_callback,
                phase="verification",
                status="started",
                message="正在校验答案可靠性",
            )
            verifier_result = evidence_ops.verify_grounded_answer(
                question=message,
                answer=answer,
                query_plan=retrieval_debug.get("query_plan") if isinstance(retrieval_debug.get("query_plan"), dict) else {},
                selected_evidence=selected_evidence,
                selection_result=selection_result,
            )
            retrieval_trace["verifier_result"] = verifier_result
            use_abstain_path = (
                "知识库中未找到确切依据" in (answer or "")
                or not verifier_result.get("supported")
                or float(verifier_result.get("support_score") or 0.0) < chat_policy.verifier_min_support_score
            )
            if "知识库中未找到确切依据" in (answer or ""):
                fallback_reason = "generation_abstained"
            elif not verifier_result.get("supported"):
                fallback_reason = "verifier_rejected"
            elif float(verifier_result.get("support_score") or 0.0) < chat_policy.verifier_min_support_score:
                fallback_reason = "verifier_low_support"
            legacy_rag._emit_chat_progress(
                progress_callback,
                phase="verification",
                status="finished",
                message="答案校验完成",
                detail=f"supported={bool(verifier_result.get('supported'))}",
            )
            legacy_rag.logger.info(
                "RAG verifier decision: topic_id=%s supported=%s support_score=%s use_abstain=%s",
                topic.id,
                verifier_result.get("supported"),
                verifier_result.get("support_score"),
                use_abstain_path,
            )
        except Exception as exc:
            legacy_rag.logger.warning("RAG generation failed, falling back to abstain: topic_id=%s error=%s", topic.id, exc)
            answer = "知识库中未找到确切依据。"
            retrieval_trace["verifier_result"] = {
                "method": "system",
                "supported": False,
                "support_score": 0.0,
                "unsupported_claims": [message],
                "notes": "generation_failed",
            }
            legacy_rag._emit_chat_progress(
                progress_callback,
                phase="verification",
                status="failed",
                message="答案生成或校验失败，已回退为保守回答",
                detail=str(exc),
            )
            use_abstain_path = True
            fallback_reason = "generation_failed"
        if use_abstain_path and chat_policy.allow_fallback_generation:
            fallback_answer, fallback_model = generation_ops.generate_fallback_chat_answer(
                records=records,
                message=message,
                selected_evidence=selected_evidence,
                selection_result=selection_result,
                retrieval_debug=retrieval_debug,
                fallback_reason=fallback_reason or "fallback_requested",
                prior_answer=answer,
                progress_callback=progress_callback,
            )
            final_answer = fallback_answer
            final_citations = citations if selected_evidence else []
            final_model = fallback_model
            answer_mode = "kb_insufficient_evidence"
            used_knowledge_base = False
            retrieval_trace["fallback_reason"] = fallback_reason or "fallback_requested"
            retrieval_trace["fallback_model"] = fallback_model
            retrieval_trace["fallback_used"] = True
        else:
            final_answer = answer if not use_abstain_path else "知识库中未找到确切依据。"
            final_citations = citations if not use_abstain_path else []
            final_model = model
            answer_mode = "knowledge_base" if not use_abstain_path else "kb_insufficient_evidence"
            used_knowledge_base = not use_abstain_path
            retrieval_trace["fallback_used"] = False
        legacy_rag.logger.info(
            "RAG generation finished: topic_id=%s mode=knowledge_base model=%s elapsed=%.2fs",
            topic.id,
            final_model,
            time.perf_counter() - generation_started_at,
        )
        retrieval_trace["generation_ms"] = round((time.perf_counter() - state["started_at"]) * 1000, 2)
        retrieval_trace["answer_mode"] = answer_mode
        legacy_rag.logger.info("rag_trace %s", retrieval_trace)
        return {
            "draft": legacy_rag.AssistantMessageDraft(
                content=final_answer,
                model=final_model,
                answer_mode=answer_mode,
                used_knowledge_base=used_knowledge_base,
                citations_json=final_citations,
                metadata_json={"retrieval": retrieval_trace},
            )
        }

    def build_insufficient_draft(self, state: ChatRagGraphState) -> dict[str, Any]:
        retrieval_trace = dict(state["retrieval_trace"])
        chat_policy = state["chat_policy"]
        progress_callback = state.get("progress_callback")
        selected_evidence = state["selected_evidence"]
        citations = state["citations"]

        legacy_rag.logger.info("RAG abstain path: topic_id=%s selected_evidence=%s", state["topic"].id, len(selected_evidence))
        retrieval_trace["generation_ms"] = round((time.perf_counter() - state["started_at"]) * 1000, 2)
        retrieval_trace["fallback_used"] = bool(chat_policy.allow_fallback_generation)
        retrieval_trace["fallback_reason"] = "insufficient_evidence"
        if chat_policy.allow_fallback_generation:
            fallback_answer, fallback_model = generation_ops.generate_fallback_chat_answer(
                records=state["records"],
                message=state["message"],
                selected_evidence=selected_evidence,
                selection_result=state["selection_result"],
                retrieval_debug=state["retrieval_debug"],
                fallback_reason="insufficient_evidence",
                progress_callback=progress_callback,
            )
            retrieval_trace["fallback_model"] = fallback_model
            retrieval_trace["answer_mode"] = "kb_insufficient_evidence"
            legacy_rag.logger.info("rag_trace %s", retrieval_trace)
            draft = legacy_rag.AssistantMessageDraft(
                content=fallback_answer,
                model=fallback_model,
                answer_mode="kb_insufficient_evidence",
                used_knowledge_base=False,
                citations_json=citations if selected_evidence else [],
                metadata_json={"retrieval": retrieval_trace},
            )
        else:
            retrieval_trace["answer_mode"] = "kb_insufficient_evidence"
            legacy_rag.logger.info("rag_trace %s", retrieval_trace)
            draft = legacy_rag.AssistantMessageDraft(
                content="知识库中未找到确切依据。",
                model=None,
                answer_mode="kb_insufficient_evidence",
                used_knowledge_base=False,
                citations_json=[],
                metadata_json={"retrieval": retrieval_trace},
            )
        return {"draft": draft}


class ChatDraftGraphRunner:
    def __init__(self) -> None:
        self.route_service = ChatRouteService()
        self.retrieval_service = ChatRetrievalService()
        self.evidence_service = ChatEvidenceService()
        self.trace_builder = RetrievalTraceBuilder()
        self.generation_service = ChatGenerationService()
        self.synthesis_service = ChatSynthesisService()
        self.metadata_engine = engine_ops.MetadataQueryEngine()
        self.hybrid_engine = engine_ops.HybridScopedRagEngine()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatRagGraphState)
        graph.add_node("plan_route", self.route_service.plan)
        graph.add_node("rule_route", self.route_service.rule_route)
        graph.add_node("selector_route", self.route_service.selector_route)
        graph.add_node("run_metadata_engine", self.metadata_engine.run)
        graph.add_node("run_hybrid_prefilter", self.hybrid_engine.prefilter)
        graph.add_node("run_retrieval", self.retrieval_service.run)
        graph.add_node("select_evidence", self.evidence_service.select)
        graph.add_node("build_trace", self.trace_builder.build)
        graph.add_node("generate_answer", self.generation_service.generate)
        graph.add_node("build_insufficient_draft", self.generation_service.build_insufficient_draft)
        graph.add_node("normalize_output", self.synthesis_service.normalize)
        graph.add_node("validate_answer_shape", self.synthesis_service.validate)
        graph.add_node("synthesize_result", self.synthesis_service.synthesize)
        graph.add_node("finalize_draft", self._finalize_draft)
        graph.add_edge(START, "plan_route")
        graph.add_edge("plan_route", "rule_route")
        graph.add_conditional_edges(
            "rule_route",
            self._route_after_rule_route,
            {
                "selector_route": "selector_route",
                "run_retrieval": "run_retrieval",
                "run_metadata_engine": "run_metadata_engine",
                "run_hybrid_prefilter": "run_hybrid_prefilter",
            },
        )
        graph.add_conditional_edges(
            "selector_route",
            self._route_after_query,
            {
                "run_retrieval": "run_retrieval",
                "run_metadata_engine": "run_metadata_engine",
                "run_hybrid_prefilter": "run_hybrid_prefilter",
            },
        )
        graph.add_conditional_edges(
            "run_hybrid_prefilter",
            self._route_after_hybrid_prefilter,
            {
                "run_retrieval": "run_retrieval",
                "normalize_output": "normalize_output",
            },
        )
        graph.add_edge("run_metadata_engine", "normalize_output")
        graph.add_conditional_edges(
            "validate_answer_shape",
            self._route_after_answer_shape_validation,
            {
                "run_retrieval": "run_retrieval",
                "run_hybrid_prefilter": "run_hybrid_prefilter",
                "synthesize_result": "synthesize_result",
            },
        )
        graph.add_edge("run_retrieval", "select_evidence")
        graph.add_edge("select_evidence", "build_trace")
        graph.add_conditional_edges(
            "build_trace",
            self._route_after_trace,
            {
                "generate_answer": "generate_answer",
                "build_insufficient_draft": "build_insufficient_draft",
            },
        )
        graph.add_edge("generate_answer", "normalize_output")
        graph.add_edge("build_insufficient_draft", "normalize_output")
        graph.add_conditional_edges(
            "normalize_output",
            self._route_after_normalize,
            {
                "validate_answer_shape": "validate_answer_shape",
                "synthesize_result": "synthesize_result",
            },
        )
        graph.add_edge("synthesize_result", "finalize_draft")
        graph.add_edge("finalize_draft", END)
        return graph.compile()

    @staticmethod
    def _route_after_rule_route(state: ChatRagGraphState) -> str:
        route_decision = state.get("route_decision")
        if getattr(route_decision, "low_confidence", False):
            return "selector_route"
        return ChatDraftGraphRunner._route_after_query(state)

    @staticmethod
    def _route_after_query(state: ChatRagGraphState) -> str:
        engine_name = getattr(state.get("route_decision"), "engine_name", None) or state.get("engine_name")
        if engine_name == "metadata_engine":
            return "run_metadata_engine"
        if engine_name == "hybrid_sql_rag_engine":
            return "run_hybrid_prefilter"
        return "run_retrieval"

    @staticmethod
    def _route_after_hybrid_prefilter(state: ChatRagGraphState) -> str:
        if state.get("draft") is not None:
            return "normalize_output"
        return "run_retrieval"

    @staticmethod
    def _route_after_trace(state: ChatRagGraphState) -> str:
        if state["selection_result"].sufficiency_decision.get("is_sufficient") and state["selected_evidence"]:
            return "generate_answer"
        return "build_insufficient_draft"

    @staticmethod
    def _route_after_normalize(state: ChatRagGraphState) -> str:
        engine_name = str(state.get("engine_name") or "")
        if engine_name in {"metadata_engine", "hybrid_sql_rag_engine"}:
            return "validate_answer_shape"
        return "synthesize_result"

    @staticmethod
    def _route_after_answer_shape_validation(state: ChatRagGraphState) -> str:
        validation = state.get("answer_shape_validation") if isinstance(state.get("answer_shape_validation"), dict) else {}
        if not validation.get("mismatch"):
            return "synthesize_result"
        reroute_engine = str(validation.get("reroute_engine") or "rag_engine")
        return "run_hybrid_prefilter" if reroute_engine == "hybrid_sql_rag_engine" else "run_retrieval"

    @staticmethod
    def _finalize_draft(state: ChatRagGraphState) -> dict[str, Any]:
        legacy_rag._emit_chat_progress(
            state.get("progress_callback"),
            phase="answer_ready",
            status="finished",
            message="最终答案已准备完成",
        )
        return {}

    def build_assistant_draft(
        self,
        db: Session,
        *,
        user: User,
        topic: ChatTopic,
        records: list[ChatMessage],
        message: str,
        progress_callback: legacy_rag.ChatProgressCallback | None = None,
        relaxed_chat_rag: bool = False,
    ) -> legacy_rag.AssistantMessageDraft:
        chat_policy = (
            legacy_rag.RELAXED_CHAT_ATTRIBUTION_POLICY
            if relaxed_chat_rag
            else legacy_rag.STRICT_CHAT_ATTRIBUTION_POLICY
        )
        result = self._graph.invoke(
            {
                "db": db,
                "user": user,
                "topic": topic,
                "records": records,
                "message": message,
                "progress_callback": progress_callback,
                "relaxed_chat_rag": relaxed_chat_rag,
                "started_at": time.perf_counter(),
                "chat_policy": chat_policy,
            }
        )
        if "draft" not in result:
            raise KeyError(
                "draft missing; "
                f"keys={sorted(result.keys())}; "
                f"engine_name={result.get('engine_name')}; "
                f"route_decision={routing_ops.serialize_route_decision(result.get('route_decision'))}; "
                f"answer_shape_validation={result.get('answer_shape_validation')}"
            )
        return result["draft"]


class ChatTurnService:
    def __init__(
        self,
        *,
        repository: ChatTopicRepository | None = None,
        draft_runner: ChatDraftGraphRunner | None = None,
    ) -> None:
        self.repository = repository or ChatTopicRepository()
        self.draft_runner = draft_runner or ChatDraftGraphRunner()

    def build_topic_assistant_draft(
        self,
        db: Session,
        *,
        user: User,
        started_turn: legacy_rag.StartedChatTurn,
        progress_callback: legacy_rag.ChatProgressCallback | None = None,
        relaxed_chat_rag: bool = False,
    ) -> legacy_rag.AssistantMessageDraft:
        return self.draft_runner.build_assistant_draft(
            db,
            user=user,
            topic=started_turn.topic,
            records=started_turn.records,
            message=started_turn.prompt,
            progress_callback=progress_callback,
            relaxed_chat_rag=relaxed_chat_rag,
        )

    def prepare_topic_message(
        self,
        db: Session,
        *,
        user: User,
        topic_id: int,
        prompt: str,
        progress_callback: legacy_rag.ChatProgressCallback | None = None,
        relaxed_chat_rag: bool = False,
    ) -> legacy_rag.PreparedChatTurn:
        started_turn = self.repository.start_topic_message(db, user=user, topic_id=topic_id, prompt=prompt)
        assistant_draft = self.build_topic_assistant_draft(
            db,
            user=user,
            started_turn=started_turn,
            progress_callback=progress_callback,
            relaxed_chat_rag=relaxed_chat_rag,
        )
        return legacy_rag.PreparedChatTurn(
            topic=started_turn.topic,
            user_message=started_turn.user_message,
            assistant_draft=assistant_draft,
        )

    def send_topic_message(
        self,
        db: Session,
        *,
        user: User,
        topic_id: int,
        prompt: str,
        relaxed_chat_rag: bool = False,
    ) -> legacy_rag.ChatTurnResult:
        prepared_turn = self.prepare_topic_message(
            db,
            user=user,
            topic_id=topic_id,
            prompt=prompt,
            relaxed_chat_rag=relaxed_chat_rag,
        )
        assistant_message = self.repository.persist_assistant_message(
            db,
            topic=prepared_turn.topic,
            assistant_draft=prepared_turn.assistant_draft,
        )
        return legacy_rag.ChatTurnResult(
            topic=legacy_rag._topic_summary(db, prepared_turn.topic),
            user_message=prepared_turn.user_message,
            assistant_message=assistant_message,
        )


_CHAT_TURN_SERVICE: ChatTurnService | None = None


def get_chat_turn_service() -> ChatTurnService:
    global _CHAT_TURN_SERVICE
    if _CHAT_TURN_SERVICE is None:
        _CHAT_TURN_SERVICE = ChatTurnService()
    return _CHAT_TURN_SERVICE
