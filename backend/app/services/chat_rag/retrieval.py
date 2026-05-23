from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models import ChatMessage, PaperChunk
from app.services import rag as legacy_rag
from app.services.chat_rag import query_planning
from app.services.chat_rag import retrieval_low_level as low_level
from app.services.chat_rag import tracing as retrieval_tracing


def build_retrieval_query(records: list[ChatMessage], prompt: str) -> str:
    previous_user_messages = [record.content for record in records[:-1] if record.role == "user"][-2:]
    context = " ".join(previous_user_messages + [prompt]).strip()
    return context or prompt


def search_chunks(
    db: Session,
    *,
    query: str,
    organization_id: int,
    top_k: int | None = None,
    trace: dict[str, Any] | None = None,
) -> list[legacy_rag.RetrievalResult]:
    limit = top_k or legacy_rag.settings.rag_top_k
    query_plan = query_planning.build_crosslingual_query_plan(query)
    exact_terms = list(query_plan.exact_terms)
    exact_match_heavy = query_plan.query_type == "exact_heavy_short" or legacy_rag._is_exact_match_heavy_query(
        query_plan.original_query,
        exact_terms,
    )
    exact_terms_for_boost = exact_terms if exact_match_heavy or legacy_rag._is_table_or_figure_query(query) else []
    if not low_level.is_postgres_session(db):
        variant_results: dict[str, list[legacy_rag.RetrievalResult]] = {}
        for variant in query_plan.variants:
            variant_results[variant.name] = low_level.search_chunks_legacy(
                db,
                query=variant.query,
                top_k=max(legacy_rag.settings.rag_rerank_top_n, limit),
                organization_id=organization_id,
            )
        source_hits: dict[str, list[dict[str, Any]]] = {}
        variant_traces: dict[str, dict[str, Any]] = {}
        for variant in query_plan.variants:
            hits = retrieval_tracing.results_to_ranked_hits(variant_results.get(variant.name, []))
            local_sources: dict[str, list[dict[str, Any]]] = {}
            if variant.use_sparse:
                local_sources[f"{variant.name}_sparse"] = hits
                source_hits[f"{variant.name}_sparse"] = hits
            if variant.use_dense:
                local_sources[f"{variant.name}_dense"] = hits
                source_hits[f"{variant.name}_dense"] = hits
            variant_traces[variant.name] = {
                "query": variant.query,
                "language": variant.language,
                "role": variant.role,
                "use_sparse": variant.use_sparse,
                "use_dense": variant.use_dense,
                "sparse_hits": hits if variant.use_sparse else [],
                "dense_hits": hits if variant.use_dense else [],
                "fused_hits": low_level.fuse_ranked_candidate_sources(
                    local_sources,
                    limit=max(legacy_rag.settings.rag_rerank_top_n, limit),
                )
                if local_sources
                else [],
            }
        fused_hits = low_level.fuse_ranked_candidate_sources(
            source_hits,
            limit=max(legacy_rag.settings.rag_rerank_top_n, limit),
        ) if source_hits else []
        record_map = {
            int(result.chunk.id): (result.chunk, result.paper)
            for results in variant_results.values()
            for result in results
        }
        results = retrieval_tracing.build_retrieval_results(fused_hits[:limit], record_map=record_map)
        if trace is not None:
            serialized = retrieval_tracing.serialize_ranked_trace_candidates(fused_hits, record_map)
            sparse_sources = {name: hits for name, hits in source_hits.items() if name.endswith("_sparse")}
            dense_sources = {name: hits for name, hits in source_hits.items() if name.endswith("_dense")}
            trace.update(
                {
                    "query_plan": query_planning.serialize_query_plan(query_plan),
                    "variant_candidates": retrieval_tracing.serialize_variant_traces(variant_traces, record_map=record_map),
                    "sparse_candidates": retrieval_tracing.serialize_ranked_trace_candidates(
                        low_level.fuse_ranked_candidate_sources(
                            sparse_sources,
                            limit=max(legacy_rag.settings.rag_rerank_top_n, limit),
                        )
                        if sparse_sources
                        else [],
                        record_map,
                    ),
                    "dense_candidates": retrieval_tracing.serialize_ranked_trace_candidates(
                        low_level.fuse_ranked_candidate_sources(
                            dense_sources,
                            limit=max(legacy_rag.settings.rag_rerank_top_n, limit),
                        )
                        if dense_sources
                        else [],
                        record_map,
                    ),
                    "fused_candidates": serialized,
                    "reranked_candidates": serialized,
                    "retrieval_backend": "legacy",
                    "exact_match_terms": exact_terms,
                    "exact_match_terms_applied": exact_terms_for_boost,
                    "exact_match_heavy": exact_match_heavy,
                    "generation_instruction": query_plan.generation_instruction,
                    "rerank_query": query_plan.rerank_query,
                    "rerank_status": "not_applicable",
                }
            )
        return results

    sparse_limit = max(legacy_rag.settings.rag_sparse_top_k, limit)
    dense_limit = max(legacy_rag.settings.rag_dense_top_k, limit)
    if exact_match_heavy:
        sparse_limit = max(sparse_limit, limit * 8, legacy_rag.settings.rag_sparse_top_k * 2)
        dense_limit = max(dense_limit, limit * 4, legacy_rag.settings.rag_dense_top_k)

    dense_query_texts = legacy_rag._unique_strings([variant.query for variant in query_plan.variants if variant.use_dense])
    dense_embeddings: dict[str, list[float] | None] = {}
    if dense_query_texts and (legacy_rag.settings.glm_api_key.strip() or legacy_rag.settings.openai_api_key.strip()):
        try:
            embedding_rows = legacy_rag.embed_texts(dense_query_texts)
            dense_embeddings = {
                dense_query_texts[index]: embedding_rows[index]
                for index in range(min(len(dense_query_texts), len(embedding_rows)))
            }
        except Exception:
            dense_embeddings = {}

    sparse_source_hits: dict[str, list[dict[str, Any]]] = {}
    dense_source_hits: dict[str, list[dict[str, Any]]] = {}
    all_source_hits: dict[str, list[dict[str, Any]]] = {}
    variant_traces: dict[str, dict[str, Any]] = {}
    fusion_limit = max(
        legacy_rag.settings.rag_fusion_window,
        legacy_rag.settings.rag_rerank_top_n,
        limit,
        sparse_limit if exact_match_heavy else 0,
    )

    for variant in query_plan.variants:
        variant_sparse_hits: list[dict[str, Any]] = []
        variant_dense_hits: list[dict[str, Any]] = []
        if variant.use_sparse:
            variant_sparse_hits = low_level.search_sparse_chunks_postgres(
                db,
                query=variant.query,
                limit=sparse_limit,
                organization_id=organization_id,
                exact_terms=exact_terms_for_boost if variant.language == "en" else [],
            )
            if variant_sparse_hits:
                source_name = f"{variant.name}_sparse"
                sparse_source_hits[source_name] = variant_sparse_hits
                all_source_hits[source_name] = variant_sparse_hits
        if variant.use_dense:
            variant_dense_hits = low_level.search_dense_chunks_postgres(
                db,
                query_embedding=dense_embeddings.get(variant.query),
                limit=dense_limit,
                organization_id=organization_id,
            )
            if variant_dense_hits:
                source_name = f"{variant.name}_dense"
                dense_source_hits[source_name] = variant_dense_hits
                all_source_hits[source_name] = variant_dense_hits
        local_sources = {
            **({f"{variant.name}_sparse": variant_sparse_hits} if variant_sparse_hits else {}),
            **({f"{variant.name}_dense": variant_dense_hits} if variant_dense_hits else {}),
        }
        variant_traces[variant.name] = {
            "query": variant.query,
            "language": variant.language,
            "role": variant.role,
            "use_sparse": variant.use_sparse,
            "use_dense": variant.use_dense,
            "sparse_hits": variant_sparse_hits,
            "dense_hits": variant_dense_hits,
            "fused_hits": low_level.fuse_ranked_candidate_sources(local_sources, limit=fusion_limit) if local_sources else [],
        }

    sparse_hits = low_level.fuse_ranked_candidate_sources(sparse_source_hits, limit=fusion_limit) if sparse_source_hits else []
    dense_hits = low_level.fuse_ranked_candidate_sources(dense_source_hits, limit=fusion_limit) if dense_source_hits else []
    fused_candidates = low_level.fuse_ranked_candidate_sources(all_source_hits, limit=fusion_limit) if all_source_hits else []
    record_map = low_level.load_chunk_record_map(
        db,
        chunk_ids=[
            item["chunk_id"]
            for item in [
                *sparse_hits,
                *dense_hits,
                *fused_candidates,
                *[candidate for hits in all_source_hits.values() for candidate in hits],
            ]
        ],
        organization_id=organization_id,
    )
    expanded_candidates = fused_candidates
    expansion_stats: dict[str, Any] = {
        "enabled": False,
        "expanded_candidate_count": 0,
        "expansion_added_ids": [],
    }
    if fused_candidates:
        expanded_candidates, expansion_stats = low_level.expand_retrieval_candidates(
            db,
            query=query,
            candidates=fused_candidates,
            record_map=record_map,
            organization_id=organization_id,
            query_plan=query_plan,
            limit=fusion_limit,
        )
    if exact_terms_for_boost:
        expanded_candidates = low_level.boost_exact_match_candidates(
            query,
            candidates=expanded_candidates,
            record_map=record_map,
        )[: max(fusion_limit, len(expanded_candidates))]

    reranked_candidates = expanded_candidates
    rerank_status = "skipped"
    rerank_error: str | None = None
    rerank_context_stats: dict[str, Any] = {
        "query_chars": len(query_plan.rerank_query or ""),
        "documents_count": 0,
        "max_document_chars": 0,
        "over_budget_truncated_count": 0,
    }
    if expanded_candidates:
        try:
            reranked_candidates, rerank_context_stats = low_level.apply_reranking(
                db,
                query_plan.rerank_query,
                fused_candidates=expanded_candidates,
                record_map=record_map,
                limit=max(legacy_rag.settings.rag_rerank_top_n, limit),
            )
            rerank_status = "applied"
        except Exception as exc:
            legacy_rag.logger.warning(
                "Rerank failed, falling back to fused order: user_query=%s rerank_query=%s fused_count=%s rerank_limit=%s rerank_context_stats=%s error=%s",
                legacy_rag._preview_log_text(query, limit=200),
                legacy_rag._preview_log_text(query_plan.rerank_query, limit=200),
                len(expanded_candidates),
                max(legacy_rag.settings.rag_rerank_top_n, limit),
                rerank_context_stats,
                exc,
                exc_info=True,
            )
            reranked_candidates = expanded_candidates[: max(legacy_rag.settings.rag_rerank_top_n, limit)]
            rerank_status = "fallback_to_fused"
            rerank_error = "rerank_failed"

    results = retrieval_tracing.build_retrieval_results(reranked_candidates[:limit], record_map=record_map)
    if trace is not None:
        trace.update(
            {
                "query_plan": query_planning.serialize_query_plan(query_plan),
                "variant_candidates": retrieval_tracing.serialize_variant_traces(variant_traces, record_map=record_map),
                "sparse_candidates": retrieval_tracing.serialize_ranked_trace_candidates(sparse_hits, record_map),
                "dense_candidates": retrieval_tracing.serialize_ranked_trace_candidates(dense_hits, record_map),
                "fused_candidates": retrieval_tracing.serialize_ranked_trace_candidates(fused_candidates, record_map),
                "expanded_candidates": retrieval_tracing.serialize_ranked_trace_candidates(expanded_candidates, record_map),
                "reranked_candidates": retrieval_tracing.serialize_ranked_trace_candidates(reranked_candidates, record_map),
                "retrieval_backend": "postgres_hybrid",
                "exact_match_terms": exact_terms,
                "exact_match_terms_applied": exact_terms_for_boost,
                "exact_match_heavy": exact_match_heavy,
                "generation_instruction": query_plan.generation_instruction,
                "rerank_query": query_plan.rerank_query,
                "sparse_limit": sparse_limit,
                "dense_limit": dense_limit,
                "fusion_limit": fusion_limit,
                "expansion_stats": expansion_stats,
                "rerank_status": rerank_status,
                "rerank_error": rerank_error,
                "rerank_context_stats": rerank_context_stats,
            }
        )
    return results


def chunk_section_path(chunk: PaperChunk) -> str | None:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    section_path = str(metadata.get("section_path") or metadata.get("section_title") or "").strip()
    return section_path or None


def build_evidence_candidates(
    results: list[legacy_rag.RetrievalResult],
    *,
    query: str,
) -> list[dict[str, Any]]:
    if not results:
        return []
    max_score = max((float(result.score) for result in results), default=0.0)
    total = len(results)
    candidates: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        relative_score = (float(result.score) / max_score) if max_score > 0 else 0.0
        rank_score = 1.0 if total == 1 else 1.0 - ((rank - 1) / max(total - 1, 1))
        support_score = round(min(0.99, max(0.05, relative_score * 0.6 + rank_score * 0.4)), 4)
        full_text = legacy_rag._chunk_text_payload(result.chunk, include_supporting_context=True)
        candidates.append(
            {
                "evidence_id": f"chunk-{result.chunk.id}",
                "chunk_id": int(result.chunk.id),
                "paper_id": int(result.paper.id),
                "paper_title": result.paper.title,
                "source_url": result.paper.source_url,
                "snippet": legacy_rag._clip_snippet(full_text, max_length=320),
                "full_text": full_text,
                "score": round(float(result.score), 4),
                "support_score": support_score,
                "page_from": result.chunk.page_from,
                "page_to": result.chunk.page_to,
                "section_path": chunk_section_path(result.chunk),
                "selection_reason": "",
                "claim_texts": [],
                "rank": rank,
            }
        )
    return candidates
