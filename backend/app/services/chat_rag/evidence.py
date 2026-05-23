from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

from app.services import rag as legacy_rag


def serialize_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    row = {
        "evidence_id": str(item.get("evidence_id") or ""),
        "chunk_id": int(item.get("chunk_id") or 0),
        "paper_id": int(item.get("paper_id") or 0),
        "paper_title": item.get("paper_title"),
        "source_url": item.get("source_url"),
        "snippet": str(item.get("snippet") or ""),
        "score": round(float(item.get("score") or 0.0), 4),
        "support_score": round(float(item.get("support_score") or 0.0), 4),
        "page_from": item.get("page_from"),
        "page_to": item.get("page_to"),
        "section_path": item.get("section_path"),
    }
    if item.get("selection_reason"):
        row["selection_reason"] = str(item["selection_reason"])
    claim_texts = [str(text) for text in item.get("claim_texts") or [] if str(text).strip()]
    if claim_texts:
        row["claim_texts"] = claim_texts
    return row


def serialize_evidence_list(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serialize_evidence_item(item) for item in items]


def build_sufficiency_decision(
    selected_evidence: list[dict[str, Any]],
    *,
    overall_support_score: float,
    llm_sufficient: bool | None,
    policy: legacy_rag.ChatAttributionPolicy = legacy_rag.STRICT_CHAT_ATTRIBUTION_POLICY,
) -> dict[str, Any]:
    support_scores = [float(item.get("support_score") or 0.0) for item in selected_evidence]
    top_support = max(support_scores, default=0.0)
    total_support = sum(support_scores)
    threshold_passed = bool(selected_evidence) and (
        top_support >= policy.min_support_score
        and total_support >= policy.min_total_support_score
    )
    if llm_sufficient is None:
        is_sufficient = threshold_passed
    elif policy.llm_insufficient_hard_gate:
        is_sufficient = bool(llm_sufficient and threshold_passed)
    else:
        is_sufficient = threshold_passed
    reason_codes: list[str] = []
    if not selected_evidence:
        reason_codes.append("no_evidence_selected")
    if top_support < policy.min_support_score:
        reason_codes.append("top_support_below_threshold")
    if total_support < policy.min_total_support_score:
        reason_codes.append("total_support_below_threshold")
    if llm_sufficient is False:
        reason_codes.append("llm_marked_insufficient")
    if is_sufficient:
        reason_codes = ["sufficient"]
        if llm_sufficient is False and not policy.llm_insufficient_hard_gate:
            reason_codes.append("llm_marked_insufficient_advisory")
        if policy.name != "strict":
            reason_codes.append("relaxed_chat_policy")
    return {
        "is_sufficient": is_sufficient,
        "llm_sufficient": llm_sufficient,
        "evidence_count": len(selected_evidence),
        "top_support_score": round(top_support, 4),
        "total_support_score": round(total_support, 4),
        "overall_support_score": round(float(overall_support_score or 0.0), 4),
        "min_support_score_threshold": policy.min_support_score,
        "min_total_support_score_threshold": policy.min_total_support_score,
        "policy_name": policy.name,
        "reason_codes": reason_codes,
    }


def heuristic_select_claim_supporting_evidence(
    *,
    question: str,
    query_plan: dict[str, Any] | None,
    evidence_candidates: list[dict[str, Any]],
    policy: legacy_rag.ChatAttributionPolicy = legacy_rag.STRICT_CHAT_ATTRIBUTION_POLICY,
    reason_suffix: str = "",
) -> legacy_rag.EvidenceSelectionResult:
    selected: list[dict[str, Any]] = []
    max_evidence = max(1, legacy_rag.settings.rag_attribution_max_evidence)
    for candidate in evidence_candidates[:max_evidence]:
        evidence = dict(candidate)
        reasons = ["高排序检索证据"]
        if evidence.get("section_path"):
            reasons.append(f"章节 {evidence['section_path']}")
        detected_language = str((query_plan or {}).get("detected_language") or "")
        if detected_language in {"zh", "mixed"}:
            reasons.append("兼容跨语言检索改写")
        if reason_suffix:
            reasons.append(reason_suffix)
        evidence["selection_reason"] = "；".join(reasons)
        evidence["claim_texts"] = [question]
        selected.append(evidence)
        if (
            len(selected) >= 2
            and sum(float(item.get("support_score") or 0.0) for item in selected) >= policy.min_total_support_score
        ):
            break
    overall_support_score = round(legacy_rag._mean([float(item.get("support_score") or 0.0) for item in selected]), 4)
    claims = (
        [
            {
                "claim_text": question,
                "supporting_evidence_ids": [str(item.get("evidence_id") or "") for item in selected],
                "support_score": overall_support_score,
                "selection_reason": "基于高排序证据的回退选择",
            }
        ]
        if selected
        else []
    )
    sufficiency_decision = build_sufficiency_decision(
        selected,
        overall_support_score=overall_support_score,
        llm_sufficient=None,
        policy=policy,
    )
    missing_information = "" if sufficiency_decision["is_sufficient"] else "未找到足以稳定支撑回答的知识库证据。"
    return legacy_rag.EvidenceSelectionResult(
        selected_evidence=tuple(selected),
        claims=tuple(claims),
        overall_support_score=overall_support_score,
        sufficiency_decision=sufficiency_decision,
        missing_information=missing_information,
        method="heuristic",
    )


def select_claim_supporting_evidence(
    *,
    question: str,
    query_plan: dict[str, Any] | None,
    evidence_candidates: list[dict[str, Any]],
    policy: legacy_rag.ChatAttributionPolicy = legacy_rag.STRICT_CHAT_ATTRIBUTION_POLICY,
) -> legacy_rag.EvidenceSelectionResult:
    fallback = heuristic_select_claim_supporting_evidence(
        question=question,
        query_plan=query_plan,
        evidence_candidates=evidence_candidates,
        policy=policy,
    )
    if not evidence_candidates or not legacy_rag._llm_available_for_grounding():
        legacy_rag.logger.info(
            "RAG evidence selection skipped: candidates=%s method=%s question=%s",
            len(evidence_candidates),
            fallback.method,
            legacy_rag._preview_log_text(question),
        )
        return fallback

    selection_started_at = time.perf_counter()
    legacy_rag.logger.info(
        "RAG evidence selection started: candidates=%s question=%s",
        len(evidence_candidates),
        legacy_rag._preview_log_text(question),
    )
    evidence_text = "\n\n".join(
        [
            (
                f"evidence_id: {item['evidence_id']}\n"
                f"paper_title: {item['paper_title'] or '-'}\n"
                f"section_path: {item.get('section_path') or '-'}\n"
                f"page_range: {item.get('page_from') or '-'}-{item.get('page_to') or '-'}\n"
                f"retrieval_score: {item.get('score')}\n"
                f"candidate_support_score: {item.get('support_score')}\n"
                f"snippet: {item.get('snippet') or ''}"
            )
            for item in evidence_candidates
        ]
    )
    prompt = (
        "请阅读用户问题与候选证据，返回 JSON 对象，字段必须包含：\n"
        "- claims: 数组。每项包含 claim_text, supporting_evidence_ids, support_score, selection_reason\n"
        "- selected_evidence: 数组。每项包含 evidence_id, support_score, selection_reason\n"
        "- overall_support_score: 0 到 1 之间的数字\n"
        "- is_sufficient: boolean，表示这些证据是否足以支持一个保守答案\n"
        "- missing_information: 字符串，不足时说明缺什么\n\n"
        "规则：\n"
        "1. 只允许使用给定 evidence_id。\n"
        "2. 同一条证据可以支持多个 claim。\n"
        "3. 支持关系按语义判断，允许中文问题对应英文论文证据。\n"
        "4. 如果证据不足，is_sufficient 必须为 false。\n\n"
        f"用户问题：{question}\n\n"
        f"Query plan：{json.dumps(query_plan or {}, ensure_ascii=False)}\n\n"
        f"候选证据：\n{evidence_text}"
    )
    try:
        content, _ = legacy_rag.chat_with_messages(
            [
                {"role": "system", "content": legacy_rag.EVIDENCE_SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        payload = legacy_rag._parse_json_object(content)
        candidate_map = {str(item["evidence_id"]): item for item in evidence_candidates}
        support_by_id: dict[str, float] = {}
        reasons_by_id: dict[str, list[str]] = {}
        claims_by_id: dict[str, list[str]] = {}
        claims: list[dict[str, Any]] = []
        for claim in payload.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_text = legacy_rag._normalize_query_text(str(claim.get("claim_text") or ""))
            evidence_ids = [
                str(evidence_id)
                for evidence_id in (claim.get("supporting_evidence_ids") or [])
                if str(evidence_id) in candidate_map
            ]
            support_score = float(claim.get("support_score") or 0.0)
            selection_reason = legacy_rag._normalize_query_text(str(claim.get("selection_reason") or ""))
            if not claim_text and not evidence_ids:
                continue
            claims.append(
                {
                    "claim_text": claim_text or question,
                    "supporting_evidence_ids": evidence_ids,
                    "support_score": round(support_score, 4),
                    "selection_reason": selection_reason,
                }
            )
            for evidence_id in evidence_ids:
                support_by_id[evidence_id] = max(support_by_id.get(evidence_id, 0.0), support_score)
                if selection_reason:
                    reasons_by_id.setdefault(evidence_id, []).append(selection_reason)
                if claim_text:
                    claims_by_id.setdefault(evidence_id, []).append(claim_text)
        explicit_order: list[str] = []
        for row in payload.get("selected_evidence") or []:
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id") or "")
            if evidence_id not in candidate_map:
                continue
            explicit_order.append(evidence_id)
            support_by_id[evidence_id] = max(support_by_id.get(evidence_id, 0.0), float(row.get("support_score") or 0.0))
            reason = legacy_rag._normalize_query_text(str(row.get("selection_reason") or ""))
            if reason:
                reasons_by_id.setdefault(evidence_id, []).append(reason)
        ordered_ids = explicit_order + [
            str(item["evidence_id"])
            for item in evidence_candidates
            if str(item["evidence_id"]) not in explicit_order and str(item["evidence_id"]) in support_by_id
        ]
        seen_ids: set[str] = set()
        selected: list[dict[str, Any]] = []
        for evidence_id in ordered_ids:
            if evidence_id in seen_ids:
                continue
            seen_ids.add(evidence_id)
            base = dict(candidate_map[evidence_id])
            base["support_score"] = round(
                max(float(base.get("support_score") or 0.0), support_by_id.get(evidence_id, 0.0)),
                4,
            )
            reason_text = "；".join(reasons_by_id.get(evidence_id, []))
            if reason_text:
                base["selection_reason"] = reason_text
            base["claim_texts"] = claims_by_id.get(evidence_id, [])
            selected.append(base)
            if len(selected) >= legacy_rag.settings.rag_attribution_max_evidence:
                break
        if not selected:
            legacy_rag.logger.warning(
                "RAG evidence selection returned empty result, using heuristic fallback: candidates=%s elapsed=%.2fs",
                len(evidence_candidates),
                time.perf_counter() - selection_started_at,
            )
            return heuristic_select_claim_supporting_evidence(
                question=question,
                query_plan=query_plan,
                evidence_candidates=evidence_candidates,
                policy=policy,
                reason_suffix="LLM 选择为空，已回退到启发式",
            )
        overall_support_score = round(
            float(payload.get("overall_support_score") or legacy_rag._mean([float(item.get("support_score") or 0.0) for item in selected])),
            4,
        )
        llm_sufficient = payload.get("is_sufficient")
        sufficiency_decision = build_sufficiency_decision(
            selected,
            overall_support_score=overall_support_score,
            llm_sufficient=bool(llm_sufficient) if llm_sufficient is not None else None,
            policy=policy,
        )
        missing_information = legacy_rag._normalize_query_text(str(payload.get("missing_information") or ""))
        return legacy_rag.EvidenceSelectionResult(
            selected_evidence=tuple(selected),
            claims=tuple(claims),
            overall_support_score=overall_support_score,
            sufficiency_decision=sufficiency_decision,
            missing_information=missing_information,
            method="llm",
        )
    except Exception as exc:
        legacy_rag.logger.warning(
            "RAG evidence selection failed, using heuristic fallback: candidates=%s error=%s elapsed=%.2fs",
            len(evidence_candidates),
            exc,
            time.perf_counter() - selection_started_at,
        )
        return heuristic_select_claim_supporting_evidence(
            question=question,
            query_plan=query_plan,
            evidence_candidates=evidence_candidates,
            policy=policy,
            reason_suffix="LLM 选择失败，已回退到启发式",
        )
    finally:
        legacy_rag.logger.info(
            "RAG evidence selection finished: candidates=%s elapsed=%.2fs",
            len(evidence_candidates),
            time.perf_counter() - selection_started_at,
        )


def build_evidence_prompt_text(selected_evidence: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            (
                f"[{index}] evidence_id: {item['evidence_id']}\n"
                f"论文标题: {item['paper_title'] or '-'}\n"
                f"章节路径: {item.get('section_path') or '-'}\n"
                f"页码: {item.get('page_from') or '-'}-{item.get('page_to') or '-'}\n"
                f"support_score: {round(float(item.get('support_score') or 0.0), 4)}\n"
                f"片段: {item.get('snippet') or ''}"
            )
            for index, item in enumerate(selected_evidence, start=1)
        ]
    )


def heuristic_verify_answer(
    *,
    question: str,
    answer: str,
    selection_result: legacy_rag.EvidenceSelectionResult,
) -> dict[str, Any]:
    abstained = "知识库中未找到确切依据" in (answer or "")
    supported = selection_result.sufficiency_decision.get("is_sufficient", False) and not abstained
    if abstained and not selection_result.sufficiency_decision.get("is_sufficient", False):
        supported = True
    support_score = (
        selection_result.overall_support_score
        if supported
        else (1.0 if abstained and not selection_result.sufficiency_decision.get("is_sufficient", False) else 0.0)
    )
    return {
        "method": "heuristic",
        "supported": bool(supported),
        "support_score": round(float(support_score or 0.0), 4),
        "unsupported_claims": [] if supported else [question],
        "notes": "heuristic_verifier",
    }


def verify_grounded_answer(
    *,
    question: str,
    answer: str,
    query_plan: dict[str, Any] | None,
    selected_evidence: list[dict[str, Any]],
    selection_result: legacy_rag.EvidenceSelectionResult,
) -> dict[str, Any]:
    fallback = heuristic_verify_answer(
        question=question,
        answer=answer,
        selection_result=selection_result,
    )
    if not selected_evidence or not legacy_rag._llm_available_for_grounding():
        legacy_rag.logger.info(
            "RAG grounding verifier skipped: evidence=%s method=%s",
            len(selected_evidence),
            fallback.get("method"),
        )
        return fallback
    verifier_started_at = time.perf_counter()
    legacy_rag.logger.info(
        "RAG grounding verifier started: evidence=%s question=%s",
        len(selected_evidence),
        legacy_rag._preview_log_text(question),
    )
    prompt = (
        "请检查最终答案是否被证据支持，并返回 JSON 对象，字段必须包含：\n"
        "- supported: boolean\n"
        "- support_score: 0 到 1 的数字\n"
        "- unsupported_claims: 字符串数组\n"
        "- notes: 字符串\n\n"
        "规则：\n"
        "1. 只根据给定证据判断，不要补充外部知识。\n"
        "2. 要允许跨语言支持关系，例如中文回答由英文证据支持。\n"
        "3. 若答案包含任何证据中不存在的关键结论，应标到 unsupported_claims。\n\n"
        f"用户问题：{question}\n\n"
        f"最终答案：{answer}\n\n"
        f"Query plan：{json.dumps(query_plan or {}, ensure_ascii=False)}\n\n"
        f"选中证据：\n{build_evidence_prompt_text(selected_evidence)}"
    )
    try:
        content, _ = legacy_rag.chat_with_messages(
            [
                {"role": "system", "content": legacy_rag.EVIDENCE_VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        payload = legacy_rag._parse_json_object(content)
        return {
            "method": "llm",
            "supported": bool(payload.get("supported")),
            "support_score": round(float(payload.get("support_score") or 0.0), 4),
            "unsupported_claims": [str(item) for item in (payload.get("unsupported_claims") or []) if str(item).strip()],
            "notes": str(payload.get("notes") or ""),
        }
    except Exception as exc:
        legacy_rag.logger.warning(
            "RAG grounding verifier failed, using heuristic fallback: error=%s elapsed=%.2fs",
            exc,
            time.perf_counter() - verifier_started_at,
        )
        return fallback
    finally:
        legacy_rag.logger.info(
            "RAG grounding verifier finished: evidence=%s elapsed=%.2fs",
            len(selected_evidence),
            time.perf_counter() - verifier_started_at,
        )
