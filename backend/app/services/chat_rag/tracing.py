from __future__ import annotations

from typing import Any

from app.models import Paper
from app.services import rag as legacy_rag


def serialize_ranked_trace_candidates(
    candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[legacy_rag.PaperChunk, Paper]],
) -> list[dict]:
    serialized: list[dict] = []
    for item in candidates:
        record = record_map.get(int(item["chunk_id"]))
        if record is None:
            continue
        chunk, paper = record
        snippet = legacy_rag._clip_snippet(legacy_rag._chunk_text_payload(chunk), max_length=180)
        row = {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "paper_id": paper.id,
            "paper_title": paper.title,
            "score": round(float(item.get("score") or 0.0), 4),
            "page_from": chunk.page_from,
            "page_to": chunk.page_to,
            "snippet": snippet,
            "chunk_role": legacy_rag._chunk_role(chunk),
            "granularity": legacy_rag._chunk_granularity(chunk),
        }
        source_kind = legacy_rag._chunk_source_kind(chunk)
        if source_kind:
            row["source_kind"] = source_kind
        source_scores = item.get("source_scores")
        if isinstance(source_scores, dict) and source_scores:
            row["source_scores"] = {key: round(float(value), 4) for key, value in source_scores.items()}
        source_ranks = item.get("source_ranks")
        if isinstance(source_ranks, dict) and source_ranks:
            row["source_ranks"] = {key: int(value) for key, value in source_ranks.items()}
        if item.get("rerank_score") is not None:
            row["rerank_score"] = round(float(item["rerank_score"]), 4)
        if item.get("rerank_rank") is not None:
            row["rerank_rank"] = int(item["rerank_rank"])
        if item.get("exact_match_bonus") is not None:
            row["exact_match_bonus"] = round(float(item["exact_match_bonus"]), 4)
        if item.get("exact_match_terms"):
            row["exact_match_terms"] = [str(term) for term in item["exact_match_terms"]]
        if item.get("expansion_sources"):
            row["expansion_sources"] = [str(source) for source in item["expansion_sources"]]
        if item.get("expansion_anchor_ids"):
            row["expansion_anchor_ids"] = [int(anchor_id) for anchor_id in item["expansion_anchor_ids"]]
        if item.get("rerank_context_chars") is not None:
            row["rerank_context_chars"] = int(item["rerank_context_chars"])
        if item.get("rerank_evidence_blocks") is not None:
            row["rerank_evidence_blocks"] = int(item["rerank_evidence_blocks"])
        if item.get("rerank_context_truncated") is not None:
            row["rerank_context_truncated"] = bool(item["rerank_context_truncated"])
        if item.get("rerank_evidence_types"):
            row["rerank_evidence_types"] = [str(kind) for kind in item["rerank_evidence_types"]]
        serialized.append(row)
    return serialized


def serialize_variant_traces(
    variant_traces: dict[str, dict[str, Any]],
    *,
    record_map: dict[int, tuple[legacy_rag.PaperChunk, Paper]],
) -> dict[str, dict[str, Any]]:
    serialized: dict[str, dict[str, Any]] = {}
    for name, variant in variant_traces.items():
        serialized[name] = {
            "query": variant.get("query"),
            "language": variant.get("language"),
            "role": variant.get("role"),
            "use_sparse": bool(variant.get("use_sparse")),
            "use_dense": bool(variant.get("use_dense")),
            "sparse_candidates": serialize_ranked_trace_candidates(variant.get("sparse_hits", []), record_map),
            "dense_candidates": serialize_ranked_trace_candidates(variant.get("dense_hits", []), record_map),
            "fused_candidates": serialize_ranked_trace_candidates(variant.get("fused_hits", []), record_map),
        }
    return serialized


def resolve_candidate_chunk_id(
    item: dict[str, Any],
    *,
    record_map: dict[int, tuple[legacy_rag.PaperChunk, Paper]],
) -> int:
    chunk_id = int(item["chunk_id"])
    record = record_map.get(chunk_id)
    if record is None:
        return chunk_id
    chunk, _paper = record
    if not legacy_rag._is_summary_chunk_role(legacy_rag._chunk_role(chunk)):
        return chunk_id
    metadata = legacy_rag._chunk_metadata(chunk)
    resolved_chunk_id = metadata.get("resolved_chunk_id")
    if str(resolved_chunk_id or "").isdigit() and int(resolved_chunk_id) in record_map:
        return int(resolved_chunk_id)
    for anchor_chunk_id in legacy_rag._chunk_anchor_ids(chunk, limit=legacy_rag.settings.rag_summary_anchor_limit):
        if int(anchor_chunk_id) in record_map:
            return int(anchor_chunk_id)
    return chunk_id


def build_retrieval_results(
    candidates: list[dict[str, Any]],
    *,
    record_map: dict[int, tuple[legacy_rag.PaperChunk, Paper]],
) -> list[legacy_rag.RetrievalResult]:
    results: list[legacy_rag.RetrievalResult] = []
    seen_chunk_ids: set[int] = set()
    for item in candidates:
        resolved_chunk_id = resolve_candidate_chunk_id(item, record_map=record_map)
        if resolved_chunk_id in seen_chunk_ids:
            continue
        record = record_map.get(resolved_chunk_id) or record_map.get(int(item["chunk_id"]))
        if record is None:
            continue
        chunk, paper = record
        seen_chunk_ids.add(int(chunk.id))
        results.append(legacy_rag.RetrievalResult(chunk=chunk, paper=paper, score=float(item.get("score") or 0.0)))
    return results


def results_to_ranked_hits(results: list[legacy_rag.RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": int(result.chunk.id),
            "paper_id": int(result.paper.id),
            "score": float(result.score),
        }
        for result in results
    ]


def serialize_trace_candidates(results: list[legacy_rag.RetrievalResult]) -> list[dict]:
    candidates: list[dict] = []
    for result in results:
        snippet = legacy_rag._clip_snippet(legacy_rag._chunk_text_payload(result.chunk), max_length=180)
        candidates.append(
            {
                "chunk_id": result.chunk.id,
                "chunk_index": result.chunk.chunk_index,
                "paper_id": result.paper.id,
                "paper_title": result.paper.title,
                "score": round(result.score, 4),
                "page_from": result.chunk.page_from,
                "page_to": result.chunk.page_to,
                "snippet": snippet,
            }
        )
    return candidates
