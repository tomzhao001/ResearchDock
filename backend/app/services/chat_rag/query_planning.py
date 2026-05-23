from __future__ import annotations

import time
from typing import Any

from app.services import rag as legacy_rag


def serialize_query_plan(plan: legacy_rag.RetrievalQueryPlan) -> dict[str, Any]:
    return {
        "original_query": plan.original_query,
        "detected_language": plan.detected_language,
        "query_type": plan.query_type,
        "generation_instruction": plan.generation_instruction,
        "rerank_query": plan.rerank_query,
        "exact_terms": list(plan.exact_terms),
        "retrieval_query_en": plan.retrieval_query_en,
        "exact_guardrail_query_en": plan.exact_guardrail_query_en,
        "subqueries_en": list(plan.subqueries_en),
        "rewrite_status": plan.rewrite_status,
        "llm_rewrite_status": plan.llm_rewrite_status,
        "used_llm": plan.used_llm,
        "llm_attempted": plan.llm_attempted,
        "rewrite_provider": plan.rewrite_provider,
        "rewrite_model": plan.rewrite_model,
        "fallback_source": plan.fallback_source,
        "rewrite_error": plan.rewrite_error,
        "rewrite_backfilled_terms": list(plan.rewrite_backfilled_terms),
        "variants": [
            {
                "name": variant.name,
                "query": variant.query,
                "language": variant.language,
                "use_sparse": variant.use_sparse,
                "use_dense": variant.use_dense,
                "role": variant.role,
            }
            for variant in plan.variants
        ],
    }


def build_crosslingual_query_plan(query: str) -> legacy_rag.RetrievalQueryPlan:
    original_query = legacy_rag._normalize_query_text(query)
    detected_language = legacy_rag._detect_query_language(original_query)
    retrieval_query_en = ""
    exact_guardrail_query_en = ""
    subqueries_en: list[str] = []
    exact_terms = legacy_rag._sanitize_exact_terms(legacy_rag._extract_exact_match_terms(original_query))
    query_type = legacy_rag._classify_query_type(original_query, exact_terms)
    generation_instruction = legacy_rag._default_generation_instruction(detected_language)
    rewrite_status = "not_needed"
    llm_rewrite_status = "not_needed"
    used_llm = False
    llm_attempted = False
    rewrite_provider: str | None = None
    rewrite_model: str | None = None
    fallback_source: str | None = None
    rewrite_error: str | None = None
    rewrite_backfilled_terms: list[str] = []

    if detected_language in {"zh", "mixed"}:
        exact_guardrail_query_en, fallback_source = legacy_rag._build_heuristic_retrieval_query_en(original_query, exact_terms)
        llm_config = legacy_rag.get_chat_llm_configuration()
        rewrite_provider = str(llm_config.get("provider") or "") or None
        llm_rewrite_status = "not_attempted_config_unavailable"
        if legacy_rag._llm_available_for_query_rewrite():
            llm_attempted = True
            rewrite_started_at = time.perf_counter()
            legacy_rag.logger.info(
                "RAG query rewrite started: language=%s provider=%s query=%s",
                detected_language,
                rewrite_provider or "-",
                legacy_rag._preview_log_text(original_query),
            )
            try:
                content, rewrite_model = legacy_rag.chat_with_messages(
                    [
                        {"role": "system", "content": legacy_rag.QUERY_PLAN_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": legacy_rag.QUERY_PLAN_USER_TEMPLATE.format(
                                query=original_query,
                                max_subqueries=legacy_rag.settings.rag_crosslingual_max_subqueries,
                            ),
                        },
                    ],
                    temperature=0.0,
                )
            except Exception as exc:
                llm_rewrite_status = "llm_failed_http_or_provider"
                rewrite_error = legacy_rag._summarize_exception_message(exc)
                legacy_rag.logger.warning(
                    "RAG query rewrite request failed: language=%s provider=%s error=%s elapsed=%.2fs",
                    detected_language,
                    rewrite_provider or "-",
                    rewrite_error,
                    time.perf_counter() - rewrite_started_at,
                )
            else:
                try:
                    payload = legacy_rag._parse_json_object(content)
                except Exception as exc:
                    llm_rewrite_status = "llm_failed_parse"
                    rewrite_error = legacy_rag._summarize_exception_message(exc)
                    legacy_rag.logger.warning(
                        "RAG query rewrite parse failed: language=%s provider=%s error=%s elapsed=%.2fs",
                        detected_language,
                        rewrite_provider or "-",
                        rewrite_error,
                        time.perf_counter() - rewrite_started_at,
                    )
                else:
                    retrieval_query_en = legacy_rag._sanitize_sparse_query_text(str(payload.get("retrieval_query_en") or ""))
                    subqueries_en = [
                        legacy_rag._sanitize_sparse_query_text(item)
                        for item in legacy_rag._unique_strings(payload.get("subqueries_en") or [])
                    ][: legacy_rag.settings.rag_crosslingual_max_subqueries]
                    exact_terms = legacy_rag._sanitize_exact_terms(
                        [
                            *exact_terms,
                            *(payload.get("exact_terms") or []),
                            *legacy_rag._extract_exact_match_terms(retrieval_query_en),
                        ]
                    )
                    query_type = legacy_rag._classify_query_type(original_query, exact_terms)
                    if query_type != "complex_multi_query":
                        subqueries_en = []
                    retrieval_query_en, rewrite_backfilled_terms = legacy_rag._backfill_exact_terms_into_query(
                        retrieval_query_en,
                        exact_terms=exact_terms,
                        query_type=query_type,
                    )
                    exact_guardrail_query_en, fallback_source = legacy_rag._build_heuristic_retrieval_query_en(original_query, exact_terms)
                    generation_instruction = (
                        legacy_rag._normalize_query_text(str(payload.get("generation_instruction") or generation_instruction))
                        or generation_instruction
                    )
                    rewrite_status = "llm_rewritten"
                    llm_rewrite_status = "llm_rewritten"
                    used_llm = True
                    rewrite_error = None
                    legacy_rag.logger.info(
                        "RAG query rewrite finished: language=%s provider=%s status=%s retrieval_query_en=%s subqueries=%s elapsed=%.2fs",
                        detected_language,
                        rewrite_provider or "-",
                        rewrite_status,
                        legacy_rag._preview_log_text(retrieval_query_en),
                        len(subqueries_en),
                        time.perf_counter() - rewrite_started_at,
                    )
        if not retrieval_query_en:
            retrieval_query_en = exact_guardrail_query_en
            if retrieval_query_en:
                rewrite_status = "heuristic_template" if fallback_source == "template_rules" else "heuristic_terms"
            else:
                fallback_source = "none"
                rewrite_status = "fallback_empty"

    rerank_query = retrieval_query_en or original_query
    guardrail_query_en = ""
    if detected_language in {"zh", "mixed"} and exact_guardrail_query_en:
        if not retrieval_query_en or exact_guardrail_query_en.lower() != retrieval_query_en.lower():
            guardrail_query_en = exact_guardrail_query_en
    variants = legacy_rag._query_variants_for_plan(
        original_query=original_query,
        detected_language=detected_language,
        query_type=query_type,
        exact_guardrail_query_en=guardrail_query_en,
        retrieval_query_en=retrieval_query_en,
        subqueries_en=subqueries_en,
    )
    return legacy_rag.RetrievalQueryPlan(
        original_query=original_query,
        detected_language=detected_language,
        query_type=query_type,
        generation_instruction=generation_instruction,
        rerank_query=rerank_query,
        exact_terms=tuple(exact_terms),
        retrieval_query_en=retrieval_query_en or None,
        exact_guardrail_query_en=guardrail_query_en or None,
        subqueries_en=tuple(subqueries_en),
        variants=tuple(variants),
        rewrite_status=rewrite_status,
        llm_rewrite_status=llm_rewrite_status,
        used_llm=used_llm,
        llm_attempted=llm_attempted,
        rewrite_provider=rewrite_provider,
        rewrite_model=rewrite_model,
        fallback_source=fallback_source,
        rewrite_error=rewrite_error,
        rewrite_backfilled_terms=tuple(rewrite_backfilled_terms),
    )
