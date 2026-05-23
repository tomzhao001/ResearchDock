from __future__ import annotations

from typing import Any

from app.models import ChatMessage
from app.services import rag as legacy_rag


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
    evidence_text = legacy_rag._build_evidence_prompt_text(selected_evidence) if has_evidence else "无可引用知识库证据。"
    sufficiency = selection_result.sufficiency_decision or {}
    reason_codes = ", ".join(str(code) for code in (sufficiency.get("reason_codes") or []) if str(code).strip()) or "none"
    missing_information = selection_result.missing_information or "未提供"
    prior_answer_text = (prior_answer or "").strip() or "无"
    generation_instruction = retrieval_debug.get("generation_instruction") or "请用中文回答，保留关键英文术语原文。"
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
                {"role": "system", "content": legacy_rag.FALLBACK_SYSTEM_PROMPT},
                *legacy_rag._history_messages(records),
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{message}\n\n"
                        f"回答要求：{generation_instruction}\n"
                        f"fallback_reason: {fallback_reason}\n"
                        f"is_sufficient: {bool(sufficiency.get('is_sufficient'))}\n"
                        f"reason_codes: {reason_codes}\n"
                        f"missing_information: {missing_information}\n\n"
                        "请先明确说明“知识库中未找到确切依据”。\n"
                        "如果存在部分相关证据，可说明这些证据只是线索，不能当作充分结论。\n"
                        "随后给出简短通用补充，并避免伪造具体论文结论、数值或引用。\n\n"
                        f"可参考的知识库证据（可能不足）：\n{evidence_text}\n\n"
                        f"上一阶段答案草稿（如有，可用于纠偏，不可照搬无依据内容）：\n{prior_answer_text}"
                    ),
                },
            ],
            temperature=0.3,
        )
        normalized = legacy_rag._normalize_fallback_answer(answer, has_evidence=has_evidence)
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
        return legacy_rag._normalize_fallback_answer("", has_evidence=has_evidence), None
