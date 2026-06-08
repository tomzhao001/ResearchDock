from __future__ import annotations

import json
import re
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


def _covers_exact_terms(query_plan: dict[str, Any] | None, selected_evidence: list[dict[str, Any]]) -> bool | None:
    exact_terms = [
        legacy_rag._normalize_query_text(str(term or "")).lower()
        for term in ((query_plan or {}).get("exact_terms") or [])
        if legacy_rag._normalize_query_text(str(term or ""))
    ]
    if not exact_terms or not selected_evidence:
        return None
    combined_text = " ".join(
        legacy_rag._normalize_query_text(str(item.get("snippet") or "")).lower()
        for item in selected_evidence
    )
    return any(term in combined_text for term in exact_terms)


def _apply_exact_term_guardrail(
    sufficiency_decision: dict[str, Any],
    *,
    query_plan: dict[str, Any] | None,
    selected_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    query_type = str((query_plan or {}).get("query_type") or "")
    exact_term_covered = _covers_exact_terms(query_plan, selected_evidence)
    if query_type != "exact_heavy_short" or exact_term_covered is not False:
        return sufficiency_decision
    guarded = dict(sufficiency_decision)
    guarded["is_sufficient"] = False
    guarded["is_partially_sufficient"] = False
    guarded["should_generate_answer"] = False
    reason_codes = [code for code in guarded.get("reason_codes") or [] if code != "sufficient"]
    if "exact_terms_not_covered" not in reason_codes:
        reason_codes.append("exact_terms_not_covered")
    guarded["reason_codes"] = reason_codes
    return guarded


_ABSTENTION_LIKE_QUERY_PATTERN = re.compile(
    r"((是否|有没有|有无).*(报告|提及|说明|比较|差异|分层|结果|显著))|(\b(?:did|does|was|were|is|are|has|have)\b.{0,48}\b(report|mention|describe|compare|stratified)\b)",
    re.IGNORECASE,
)
_NEGATIVE_EVIDENCE_ANSWER_PATTERN = re.compile(
    r"(未提及|未报告|未说明|无法确认|不能确认|没有足够依据|当前材料.*不能确认|当前证据.*不能确认|current evidence does not mention|not reported|not mentioned|cannot be confirmed|insufficient evidence)",
    re.IGNORECASE,
)
_UNCERTAINTY_MARKER_PATTERN = re.compile(
    r"(暂时无法确认|不能确认|尚不清楚|证据不足|仅能说明|只能确认|未提及|未报告|cannot be confirmed|insufficient evidence|not reported|not mentioned)",
    re.IGNORECASE,
)


def is_abstention_like_query(question: str, query_plan: dict[str, Any] | None = None) -> bool:
    exact_terms = [str(item).strip() for item in ((query_plan or {}).get("exact_terms") or []) if str(item).strip()]
    query_type = str((query_plan or {}).get("query_type") or "")
    return bool(
        _ABSTENTION_LIKE_QUERY_PATTERN.search(question or "")
        or (query_type == "exact_heavy_short" and bool(exact_terms))
    )


def is_negative_evidence_answer(answer: str) -> bool:
    return bool(_NEGATIVE_EVIDENCE_ANSWER_PATTERN.search(answer or ""))


def has_explicit_uncertainty_marker(answer: str) -> bool:
    return bool(_UNCERTAINTY_MARKER_PATTERN.search(answer or ""))


def _build_missing_claims(
    claims: tuple[dict[str, Any], ...],
    *,
    partial_answer_mode: bool,
    answer: str,
    claim_coverage: float,
) -> list[str]:
    if not claims:
        return []
    if claim_coverage >= 0.999:
        return []
    if partial_answer_mode and has_explicit_uncertainty_marker(answer):
        return []
    return [str(claim.get("claim_text") or "") for claim in claims if str(claim.get("claim_text") or "").strip()]


def _derive_heuristic_claim_coverage(
    *,
    selection_result: legacy_rag.EvidenceSelectionResult,
    partial_answer_mode: bool,
    negative_answer: bool,
    exact_term_gap: bool,
    answer: str,
) -> float:
    if exact_term_gap or negative_answer:
        return 0.0
    if not selection_result.claims:
        return 1.0 if selection_result.selected_evidence else 0.0
    if partial_answer_mode:
        return 0.75 if has_explicit_uncertainty_marker(answer) else 0.5
    return 1.0


def _derive_verifier_failure_mode(
    *,
    policy: legacy_rag.ChatAttributionPolicy,
    supported: bool,
    support_score: float,
    claim_coverage: float,
    abstention_recommended: bool,
    exact_term_gap: bool,
    negative_answer: bool,
) -> str | None:
    if exact_term_gap:
        return "exact_terms_not_covered"
    if abstention_recommended:
        return "abstention_like_negative_answer" if negative_answer else "verifier_abstention_recommended"
    if not supported:
        return "verifier_rejected"
    if policy.verifier_requires_claim_coverage and claim_coverage < policy.verifier_min_claim_coverage:
        return "verifier_claim_coverage_low"
    if support_score < policy.verifier_min_support_score:
        return "verifier_low_support"
    return None


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
    partial_threshold_passed = bool(selected_evidence) and policy.allow_partial_answer and (
        top_support >= policy.partial_min_support_score
        and total_support >= policy.partial_min_total_support_score
    )
    if llm_sufficient is None:
        is_sufficient = threshold_passed
    elif policy.llm_insufficient_hard_gate:
        is_sufficient = bool(llm_sufficient and threshold_passed)
    else:
        is_sufficient = threshold_passed
    is_partially_sufficient = (
        not is_sufficient
        and partial_threshold_passed
        and (llm_sufficient is not False or not policy.llm_insufficient_hard_gate)
    )
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
        if policy.name != "strict_factual_strict":
            reason_codes.append("relaxed_chat_policy")
    elif is_partially_sufficient:
        if llm_sufficient is False and not policy.llm_insufficient_hard_gate:
            reason_codes.append("llm_marked_insufficient_advisory")
        reason_codes.append("partial_answer_allowed")
        if policy.name != "strict_factual_strict":
            reason_codes.append("relaxed_chat_policy")
    return {
        "is_sufficient": is_sufficient,
        "is_partially_sufficient": is_partially_sufficient,
        "should_generate_answer": bool(is_sufficient or is_partially_sufficient),
        "llm_sufficient": llm_sufficient,
        "evidence_count": len(selected_evidence),
        "top_support_score": round(top_support, 4),
        "total_support_score": round(total_support, 4),
        "overall_support_score": round(float(overall_support_score or 0.0), 4),
        "min_support_score_threshold": policy.min_support_score,
        "min_total_support_score_threshold": policy.min_total_support_score,
        "partial_min_support_score_threshold": policy.partial_min_support_score,
        "partial_min_total_support_score_threshold": policy.partial_min_total_support_score,
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
    sufficiency_decision = _apply_exact_term_guardrail(
        sufficiency_decision,
        query_plan=query_plan,
        selected_evidence=selected,
    )
    missing_information = (
        ""
        if sufficiency_decision["should_generate_answer"]
        else "未找到足以稳定支撑回答的知识库证据。"
    )
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
        sufficiency_decision = _apply_exact_term_guardrail(
            sufficiency_decision,
            query_plan=query_plan,
            selected_evidence=selected,
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
    policy: legacy_rag.ChatAttributionPolicy,
    verifier_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verifier_context = verifier_context or {}
    abstained = "知识库中未找到确切依据" in (answer or "")
    expected_abstention_like_query = bool(verifier_context.get("expected_abstention_like_query"))
    partial_answer_mode = bool(verifier_context.get("partial_answer_mode"))
    reason_codes = {str(code) for code in (selection_result.sufficiency_decision.get("reason_codes") or []) if str(code).strip()}
    exact_term_gap = "exact_terms_not_covered" in reason_codes
    negative_answer = expected_abstention_like_query and is_negative_evidence_answer(answer)
    claim_coverage = _derive_heuristic_claim_coverage(
        selection_result=selection_result,
        partial_answer_mode=partial_answer_mode,
        negative_answer=negative_answer,
        exact_term_gap=exact_term_gap,
        answer=answer,
    )
    abstention_recommended = exact_term_gap or (expected_abstention_like_query and (abstained or negative_answer))
    partial_answer_safe = partial_answer_mode and has_explicit_uncertainty_marker(answer)
    supported = bool(
        selection_result.selected_evidence
        and not abstained
        and not abstention_recommended
        and (
            not policy.verifier_requires_claim_coverage
            or claim_coverage >= policy.verifier_min_claim_coverage
        )
        and (
            not partial_answer_mode
            or policy.verifier_partial_answer_strictness == "off"
            or partial_answer_safe
        )
    )
    support_score = float(selection_result.overall_support_score or 0.0)
    if abstention_recommended:
        support_score = 0.0
    elif partial_answer_mode and not partial_answer_safe and policy.verifier_partial_answer_strictness == "strict":
        support_score = min(support_score, max(0.0, policy.verifier_min_support_score - 0.1))
    failure_mode = _derive_verifier_failure_mode(
        policy=policy,
        supported=supported,
        support_score=support_score,
        claim_coverage=claim_coverage,
        abstention_recommended=abstention_recommended,
        exact_term_gap=exact_term_gap,
        negative_answer=negative_answer,
    )
    return {
        "method": "heuristic",
        "supported": bool(supported),
        "support_score": round(float(support_score or 0.0), 4),
        "claim_coverage": round(float(claim_coverage), 4),
        "unsupported_claims": [] if supported else [question],
        "missing_claims": _build_missing_claims(
            selection_result.claims,
            partial_answer_mode=partial_answer_mode,
            answer=answer,
            claim_coverage=claim_coverage,
        ),
        "abstention_recommended": bool(abstention_recommended),
        "failure_mode": failure_mode,
        "notes": "heuristic_verifier",
    }


def verify_grounded_answer(
    *,
    question: str,
    answer: str,
    query_plan: dict[str, Any] | None,
    selected_evidence: list[dict[str, Any]],
    selection_result: legacy_rag.EvidenceSelectionResult,
    policy: legacy_rag.ChatAttributionPolicy,
    verifier_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verifier_context = verifier_context or {}
    fallback = heuristic_verify_answer(
        question=question,
        answer=answer,
        selection_result=selection_result,
        policy=policy,
        verifier_context=verifier_context,
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
        "- claim_coverage: 0 到 1 的数字，表示答案覆盖关键 claim 的程度\n"
        "- unsupported_claims: 字符串数组\n"
        "- missing_claims: 字符串数组，表示答案未覆盖但问题仍要求回答的 claim\n"
        "- abstention_recommended: boolean，如果答案本质是在说明证据不足/未报告，应为 true\n"
        "- failure_mode: 字符串，可选值优先使用 abstention_like_negative_answer、exact_terms_not_covered、verifier_claim_coverage_low、verifier_rejected、verifier_low_support\n"
        "- notes: 字符串\n\n"
        "规则：\n"
        "1. 只根据给定证据判断，不要补充外部知识。\n"
        "2. 要允许跨语言支持关系，例如中文回答由英文证据支持。\n"
        "3. 若答案包含任何证据中不存在的关键结论，应标到 unsupported_claims。\n"
        "4. 如果这是 partial answer，只能接受“已回答部分被证据支持，未覆盖部分被明确标注为暂不能确认”的答案。\n"
        "5. 如果问题本质是在问论文是否报告/是否提及某个条件，而答案是在说“未报告/无法确认/证据不足”，应推荐 abstention，而不是把它当成 knowledge_base 正向回答。\n"
        "6. multi_span / multi_turn 问题若只覆盖了部分 claim，不应判为 fully supported。\n\n"
        f"用户问题：{question}\n\n"
        f"最终答案：{answer}\n\n"
        f"Query plan：{json.dumps(query_plan or {}, ensure_ascii=False)}\n\n"
        f"Verifier context：{json.dumps(verifier_context, ensure_ascii=False)}\n\n"
        f"Policy：{json.dumps({'name': policy.name, 'verifier_min_support_score': policy.verifier_min_support_score, 'verifier_min_claim_coverage': policy.verifier_min_claim_coverage, 'verifier_requires_claim_coverage': policy.verifier_requires_claim_coverage, 'verifier_negative_answer_guard': policy.verifier_negative_answer_guard, 'verifier_partial_answer_strictness': policy.verifier_partial_answer_strictness}, ensure_ascii=False)}\n\n"
        f"Selector claims：{json.dumps(list(selection_result.claims), ensure_ascii=False)}\n\n"
        f"Sufficiency decision：{json.dumps(selection_result.sufficiency_decision, ensure_ascii=False)}\n\n"
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
        supported = bool(payload.get("supported"))
        support_score = round(float(payload.get("support_score") or 0.0), 4)
        claim_coverage = round(float(payload.get("claim_coverage") or (1.0 if supported else 0.0)), 4)
        abstention_recommended = bool(payload.get("abstention_recommended"))
        failure_mode = str(payload.get("failure_mode") or "").strip() or _derive_verifier_failure_mode(
            policy=policy,
            supported=supported,
            support_score=support_score,
            claim_coverage=claim_coverage,
            abstention_recommended=abstention_recommended,
            exact_term_gap="exact_terms_not_covered" in {
                str(code) for code in (selection_result.sufficiency_decision.get("reason_codes") or []) if str(code).strip()
            },
            negative_answer=is_negative_evidence_answer(answer),
        )
        return {
            "method": "llm",
            "supported": supported,
            "support_score": support_score,
            "claim_coverage": claim_coverage,
            "unsupported_claims": [str(item) for item in (payload.get("unsupported_claims") or []) if str(item).strip()],
            "missing_claims": [str(item) for item in (payload.get("missing_claims") or []) if str(item).strip()],
            "abstention_recommended": abstention_recommended,
            "failure_mode": failure_mode,
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
