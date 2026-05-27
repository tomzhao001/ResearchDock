from __future__ import annotations

from typing import Any

from app.models import ChatMessage
from app.services import rag as legacy_rag


EVIDENCE_BACKED_FALLBACK_SYSTEM_PROMPT = (
    "你是 ResearchDock 的论文阅读助理。"
    "当前已有部分知识库证据，但不足以支撑一个完全确定的答案。"
    "请严格基于给定证据做保守总结，明确哪些内容是证据支持的，哪些部分暂时无法确认。"
    "不要补充通用知识，不要伪造论文引用。"
)


def classify_fallback_mode(*, selected_evidence: list[dict[str, Any]], fallback_reason: str) -> str:
    if selected_evidence:
        return "evidence_backed_fallback"
    return "general_fallback"


def build_status_detail(
    *,
    fallback_mode: str,
    fallback_reason: str,
    missing_information: str,
) -> str | None:
    if fallback_mode == "general_fallback":
        return "未找到可直接支撑当前问题的知识库材料，以下仅提供通用参考。"
    if fallback_reason in {"generation_abstained", "verifier_rejected", "verifier_low_support"}:
        return "已找到相关材料，但完整答案未通过最终校验，以下仅保留可由材料支持的部分。"
    if missing_information:
        return missing_information
    return "已找到部分相关材料，但不足以完整回答，以下为基于现有材料的保守总结。"


def build_status_message(*, response_kind: str, attribution_status: str) -> str:
    if response_kind == "metadata_answer":
        return "这是元数据/文档范围回答，不是正文证据归因回答。"
    if attribution_status == "grounded":
        return "以下回答可由知识库材料完整支撑，请结合引用核验。"
    if attribution_status == "verification_failed":
        return "已找到相关材料，但完整答案未通过最终校验。"
    if attribution_status == "partial_evidence":
        return "已找到部分相关材料，以下为基于现有材料的归纳总结。"
    if attribution_status == "scope_empty":
        return "按当前筛选条件未匹配到论文。"
    if attribution_status == "no_usable_evidence":
        return "未找到可直接支撑当前问题的知识库材料。"
    return "以下回答请结合材料自行判断。"


def derive_response_semantics(
    *,
    answer_mode: str | None,
    used_knowledge_base: bool,
    retrieval_trace: dict[str, Any],
    citations_json: list[dict[str, Any]],
) -> dict[str, str]:
    engine_name = str(retrieval_trace.get("engine_name") or "")
    fallback_mode = str(retrieval_trace.get("fallback_mode") or "")
    fallback_reason = str(retrieval_trace.get("fallback_reason") or "")
    selected_evidence = citations_json or []
    if answer_mode == "metadata_query":
        response_kind = "metadata_answer"
        engine_execution = retrieval_trace.get("engine_execution") if isinstance(retrieval_trace.get("engine_execution"), dict) else {}
        total_count = 0
        if isinstance(engine_execution.get("structured_phase"), dict):
            total_count = int(engine_execution["structured_phase"].get("total_count") or 0)
        elif str(engine_name or retrieval_trace.get("retrieval_backend") or "") == "metadata_engine":
            total_count = int(engine_execution.get("total_count") or 0)
        attribution_status = "scope_empty" if total_count == 0 else "metadata_only"
    elif answer_mode == "knowledge_base" and used_knowledge_base:
        response_kind = "grounded_rag"
        attribution_status = "grounded"
    elif fallback_mode == "evidence_backed_fallback" or selected_evidence:
        response_kind = "evidence_backed_fallback"
        if fallback_reason in {"generation_abstained", "verifier_rejected", "verifier_low_support"}:
            attribution_status = "verification_failed"
        else:
            attribution_status = "partial_evidence"
    else:
        response_kind = "general_fallback"
        attribution_status = "scope_empty" if fallback_reason == "scope_empty" else "no_usable_evidence"
    if response_kind == "metadata_answer":
        status_detail = "该回答基于文档范围、数量或筛选结果生成，不包含正文引用。"
    elif attribution_status == "grounded":
        status_detail = "回答内容已限制在当前检索到并通过校验的材料范围内。"
    else:
        status_detail = (
            build_status_detail(
                fallback_mode=fallback_mode or ("general_fallback" if not selected_evidence else "evidence_backed_fallback"),
                fallback_reason=fallback_reason,
                missing_information=str(
                    ((retrieval_trace.get("evidence_selection_trace") or {}).get("missing_information") if isinstance(retrieval_trace.get("evidence_selection_trace"), dict) else "")
                    or ""
                ),
            )
            or ""
        )
    return {
        "response_kind": response_kind,
        "attribution_status": attribution_status,
        "status_message": build_status_message(response_kind=response_kind, attribution_status=attribution_status),
        "status_detail": status_detail,
    }


def generate_fallback_chat_answer(
    *,
    records: list[ChatMessage],
    message: str,
    selected_evidence: list[dict[str, Any]],
    selection_result: legacy_rag.EvidenceSelectionResult,
    retrieval_debug: dict[str, Any],
    fallback_reason: str,
    prior_answer: str | None = None,
    progress_callback: legacy_rag.ChatProgressCallback | None = None,
) -> tuple[str, str | None]:
    has_evidence = bool(selected_evidence)
    fallback_mode = classify_fallback_mode(selected_evidence=selected_evidence, fallback_reason=fallback_reason)
    evidence_text = legacy_rag._build_evidence_prompt_text(selected_evidence) if has_evidence else "无可引用知识库证据。"
    sufficiency = selection_result.sufficiency_decision or {}
    reason_codes = ", ".join(str(code) for code in (sufficiency.get("reason_codes") or []) if str(code).strip()) or "none"
    missing_information = selection_result.missing_information or "未提供"
    prior_answer_text = (prior_answer or "").strip() or "无"
    generation_instruction = retrieval_debug.get("generation_instruction") or "请用中文回答，保留关键英文术语原文。"
    mode_instruction = (
        "请基于给定证据总结已经能确定的内容，并明确标注暂时无法确认的部分。\n"
        "不要输出“知识库中未找到确切依据”，不要补充通用知识，不要伪造结论或引用。\n\n"
        if fallback_mode == "evidence_backed_fallback"
        else "请先明确说明“知识库中未找到确切依据”。\n"
        "随后给出简短通用补充，并避免伪造具体论文结论、数值或引用。\n\n"
    )
    legacy_rag._emit_chat_progress(
        progress_callback,
        phase="fallback_generation",
        status="started",
        message="正在生成保守补充回答",
        detail=fallback_reason,
    )
    try:
        answer, model = legacy_rag.chat_with_messages(
            [
                {
                    "role": "system",
                    "content": (
                        EVIDENCE_BACKED_FALLBACK_SYSTEM_PROMPT
                        if fallback_mode == "evidence_backed_fallback"
                        else legacy_rag.FALLBACK_SYSTEM_PROMPT
                    ),
                },
                *legacy_rag._history_messages(records),
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{message}\n\n"
                        f"回答要求：{generation_instruction}\n"
                        f"fallback_reason: {fallback_reason}\n"
                        f"fallback_mode: {fallback_mode}\n"
                        f"is_sufficient: {bool(sufficiency.get('is_sufficient'))}\n"
                        f"reason_codes: {reason_codes}\n"
                        f"missing_information: {missing_information}\n\n"
                        f"{mode_instruction}"
                        f"可参考的知识库证据（可能不足）：\n{evidence_text}\n\n"
                        f"上一阶段答案草稿（如有，可用于纠偏，不可照搬无依据内容）：\n{prior_answer_text}"
                    ),
                },
            ],
            temperature=0.3,
        )
        normalized = answer.strip() if fallback_mode == "evidence_backed_fallback" else legacy_rag._normalize_fallback_answer(answer, has_evidence=has_evidence)
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="fallback_generation",
            status="finished",
            message="保守补充回答已生成",
            detail=model,
        )
        return normalized, model
    except Exception as exc:
        legacy_rag.logger.warning("Fallback generation failed: reason=%s error=%s", fallback_reason, exc)
        legacy_rag._emit_chat_progress(
            progress_callback,
            phase="fallback_generation",
            status="failed",
            message="保守补充回答生成失败，已使用默认提示",
            detail=str(exc),
        )
        if fallback_mode == "evidence_backed_fallback":
            detail = build_status_detail(
                fallback_mode=fallback_mode,
                fallback_reason=fallback_reason,
                missing_information=missing_information,
            ) or "已找到部分相关材料，但暂时无法生成更完整的总结。"
            return f"基于已检索到的材料，可先确认以下内容：\n\n{detail}", None
        return legacy_rag._normalize_fallback_answer("", has_evidence=has_evidence), None
