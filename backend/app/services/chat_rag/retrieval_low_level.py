from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import Paper, PaperChunk
from app.services import rag as legacy_rag


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def lexical_score(query_tokens: set[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = set(legacy_rag._tokenize(content))
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def is_postgres_session(db: Session) -> bool:
    bind = db.get_bind()
    return bool(bind and bind.dialect.name == "postgresql")


def normalize_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    if isinstance(value, tuple) and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    if hasattr(value, "tolist"):
        items = value.tolist()
        if isinstance(items, list) and all(isinstance(item, (int, float)) for item in items):
            return [float(item) for item in items]
    return None


def search_chunks_legacy(
    db: Session,
    *,
    query: str,
    top_k: int,
    organization_id: int,
    paper_ids: list[int] | None = None,
) -> list[legacy_rag.RetrievalResult]:
    searchable_roles = legacy_rag._searchable_chunk_roles()
    statement = (
        select(PaperChunk, Paper)
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(
            Paper.deleted_at.is_(None),
            Paper.organization_id == organization_id,
            PaperChunk.chunk_role.in_(searchable_roles),
        )
        .order_by(PaperChunk.paper_id.asc(), PaperChunk.chunk_index.asc())
    )
    if paper_ids:
        statement = statement.where(Paper.id.in_([int(paper_id) for paper_id in paper_ids]))
    rows = db.execute(statement).all()
    if not rows:
        return []

    query_tokens = set(legacy_rag._tokenize(query))
    query_embedding: list[float] | None = None
    if (legacy_rag.settings.glm_api_key.strip() or legacy_rag.settings.openai_api_key.strip()) and any(chunk.embedding for chunk, _ in rows):
        try:
            query_embedding = legacy_rag.embed_texts([query])[0]
        except Exception:
            query_embedding = None

    scored: list[legacy_rag.RetrievalResult] = []
    for chunk, paper in rows:
        lexical = lexical_score(query_tokens, chunk.content)
        chunk_embedding = normalize_embedding(chunk.embedding)
        embedding_score = cosine_similarity(query_embedding, chunk_embedding) if query_embedding is not None and chunk_embedding is not None else 0.0
        score = embedding_score if embedding_score > 0 else lexical
        if embedding_score > 0 and lexical > 0:
            score = embedding_score * 0.8 + lexical * 0.2
        if score < legacy_rag.MIN_RELEVANCE_SCORE:
            continue
        scored.append(legacy_rag.RetrievalResult(chunk=chunk, paper=paper, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def search_sparse_chunks_postgres(
    db: Session,
    *,
    query: str,
    limit: int,
    organization_id: int,
    exact_terms: list[str] | None = None,
    paper_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    searchable_roles = legacy_rag._searchable_chunk_roles()
    role_sql = ", ".join(f"'{role}'" for role in searchable_roles)
    search_config = re.sub(r"[^a-zA-Z0-9_]+", "", legacy_rag.settings.rag_text_search_config) or "simple"
    body_text_expr = "LOWER(COALESCE(pc.metadata_json->>'body_text', pc.content))"
    exact_terms = exact_terms if exact_terms is not None else legacy_rag._extract_exact_match_terms(query)
    exact_match_clauses: list[str] = []
    exact_bonus_clauses: list[str] = []
    params: dict[str, Any] = {"query": query.strip(), "limit": limit, "organization_id": organization_id}
    for index, term in enumerate(exact_terms):
        term_key = f"exact_term_{index}"
        bonus_key = f"exact_bonus_{index}"
        params[term_key] = f"%{term.lower()}%"
        params[bonus_key] = round(min(0.35, 0.08 + len(term) * 0.01), 4)
        exact_match_clauses.append(f"{body_text_expr} LIKE :{term_key}")
        exact_bonus_clauses.append(f"CASE WHEN {body_text_expr} LIKE :{term_key} THEN :{bonus_key} ELSE 0 END")
    exact_match_sql = " OR ".join(exact_match_clauses) or "FALSE"
    exact_bonus_sql = " + ".join(exact_bonus_clauses) or "0"
    paper_scope_sql = ""
    if paper_ids:
        placeholders: list[str] = []
        for index, paper_id in enumerate([int(item) for item in paper_ids]):
            key = f"paper_id_{index}"
            params[key] = paper_id
            placeholders.append(f":{key}")
        if placeholders:
            paper_scope_sql = f" AND p.id IN ({', '.join(placeholders)}) "
    rows = db.execute(
        text(
            f"""
            SELECT
                pc.id AS chunk_id,
                pc.paper_id AS paper_id,
                (
                    ts_rank_cd(pc.search_vector, plainto_tsquery('{search_config}', :query))
                    + ({exact_bonus_sql})
                ) AS score
            FROM paper_chunks AS pc
            JOIN papers AS p ON p.id = pc.paper_id
            WHERE p.deleted_at IS NULL
              AND p.organization_id = :organization_id
              {paper_scope_sql}
              AND pc.chunk_role IN ({role_sql})
              AND (
                    pc.search_vector @@ plainto_tsquery('{search_config}', :query)
                    OR {exact_match_sql}
              )
            ORDER BY score DESC, pc.id ASC
            LIMIT :limit
            """
        ),
        params,
    ).mappings().all()
    return [
        {
            "chunk_id": int(row["chunk_id"]),
            "paper_id": int(row["paper_id"]),
            "score": float(row["score"] or 0.0),
        }
        for row in rows
    ]


def search_dense_chunks_postgres(
    db: Session,
    *,
    query_embedding: list[float] | None,
    limit: int,
    organization_id: int,
    paper_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    if query_embedding is None:
        return []
    searchable_roles = legacy_rag._searchable_chunk_roles()
    distance = PaperChunk.embedding.cosine_distance(query_embedding)
    statement = (
        select(
            PaperChunk.id.label("chunk_id"),
            PaperChunk.paper_id.label("paper_id"),
            (1 - distance).label("score"),
        )
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(
            Paper.deleted_at.is_(None),
            Paper.organization_id == organization_id,
            PaperChunk.embedding.is_not(None),
            PaperChunk.chunk_role.in_(searchable_roles),
        )
        .order_by(distance.asc(), PaperChunk.id.asc())
        .limit(limit)
    )
    if paper_ids:
        statement = statement.where(Paper.id.in_([int(paper_id) for paper_id in paper_ids]))
    rows = db.execute(statement).all()
    return [
        {
            "chunk_id": int(row.chunk_id),
            "paper_id": int(row.paper_id),
            "score": float(row.score or 0.0),
        }
        for row in rows
    ]


def fuse_ranked_candidate_sources(
    source_hits: dict[str, list[dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for source_name, hits in source_hits.items():
        for rank, item in enumerate(hits, start=1):
            candidate = merged.setdefault(
                int(item["chunk_id"]),
                {
                    "chunk_id": int(item["chunk_id"]),
                    "paper_id": int(item["paper_id"]),
                    "source_scores": {},
                    "source_ranks": {},
                    "score": 0.0,
                },
            )
            candidate["source_scores"][source_name] = float(item["score"])
            candidate["source_ranks"][source_name] = rank
            candidate["score"] += 1.0 / (legacy_rag.settings.rag_rrf_k + rank)
    fused = list(merged.values())
    fused.sort(
        key=lambda item: (
            float(item["score"]),
            max((float(score) for score in item["source_scores"].values()), default=0.0),
        ),
        reverse=True,
    )
    for rank, item in enumerate(fused, start=1):
        item["rank"] = rank
    return fused[:limit]


def fuse_ranked_candidates(
    sparse_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return fuse_ranked_candidate_sources({"sparse": sparse_hits, "dense": dense_hits}, limit=limit)


def boost_exact_match_candidates(
    query: str,
    *,
    candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> list[dict[str, Any]]:
    exact_terms = legacy_rag._extract_exact_match_terms(query)
    prefers_table = legacy_rag._is_table_or_figure_query(query)
    table_focus_terms = legacy_rag._extract_table_focus_terms(query) if prefers_table else []
    if not exact_terms and not table_focus_terms:
        return candidates

    boosted: list[dict[str, Any]] = []
    allow_general_exact_boost = not legacy_rag._is_cjk_dominant_text(query)
    for candidate in candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            boosted.append(dict(candidate))
            continue
        chunk, _ = record
        metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
        haystack = legacy_rag._chunk_text_payload(chunk, include_supporting_context=True).lower()
        matched_terms = [term for term in exact_terms if term.lower() in haystack]
        matched_focus_terms = [term for term in table_focus_terms if term.lower() in haystack]
        block_types = {str(item).lower() for item in metadata.get("block_types", [])} if isinstance(metadata, dict) else set()
        body_text = str(metadata.get("body_text") or chunk.content or "")
        has_table_body = legacy_rag._chunk_has_table_body(chunk)
        caption_only = legacy_rag._chunk_is_caption_only(chunk)
        bonus = 0.0
        if allow_general_exact_boost:
            bonus += sum(min(0.12, 0.04 + len(term) * 0.003) for term in matched_terms)
        if prefers_table:
            if has_table_body:
                bonus += 0.05
                bonus += sum(min(0.06, 0.02 + len(term) * 0.002) for term in matched_terms)
                bonus += sum(min(0.04, 0.015 + len(term) * 0.0015) for term in matched_focus_terms)
                if matched_focus_terms and re.search(r"\d", body_text):
                    bonus += 0.05
            elif "table_caption" in block_types and not caption_only:
                bonus += 0.03 + sum(min(0.03, 0.01 + len(term) * 0.0015) for term in matched_terms)
        boosted_candidate = dict(candidate)
        boosted_candidate["exact_match_terms"] = matched_terms
        boosted_candidate["exact_match_bonus"] = round(bonus, 4)
        boosted_candidate["score"] = float(candidate.get("score") or 0.0) + bonus
        boosted.append(boosted_candidate)

    boosted.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("exact_match_bonus") or 0.0),
            max((float(score) for score in item.get("source_scores", {}).values()), default=0.0),
        ),
        reverse=True,
    )
    for rank, item in enumerate(boosted, start=1):
        item["rank"] = rank
    return boosted


def load_chunk_record_map(
    db: Session,
    *,
    chunk_ids: Iterable[int],
    organization_id: int,
) -> dict[int, tuple[PaperChunk, Paper]]:
    ids = [int(chunk_id) for chunk_id in chunk_ids]
    if not ids:
        return {}
    rows = db.execute(
        select(PaperChunk, Paper)
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(Paper.deleted_at.is_(None), Paper.organization_id == organization_id, PaperChunk.id.in_(ids))
    ).all()
    return {int(chunk.id): (chunk, paper) for chunk, paper in rows}


def should_expand_multi_span_candidates(query: str, *, query_plan: legacy_rag.RetrievalQueryPlan) -> bool:
    if not query.strip() or legacy_rag._is_table_or_figure_query(query):
        return False
    if query_plan.query_type in {"exact_heavy_short", "decontextualization_short"}:
        return False
    if query_plan.subqueries_en:
        return True
    normalized_query = re.sub(r"\s+", " ", query).strip().lower()
    patterns = (
        r"分别",
        r"各自",
        r"哪些",
        r"以及",
        r"同时",
        r"原因",
        r"退出",
        r"why .* reasons?",
        r"reasons? for",
        r"respectively",
        r"\bboth\b",
    )
    return any(re.search(pattern, normalized_query, re.IGNORECASE) for pattern in patterns)


def merge_ranked_candidates(
    primary: list[dict[str, Any]],
    additional: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {int(item["chunk_id"]): dict(item) for item in primary}
    for item in additional:
        chunk_id = int(item["chunk_id"])
        existing = merged.get(chunk_id)
        if existing is None:
            merged[chunk_id] = dict(item)
            continue
        existing["score"] = max(float(existing.get("score") or 0.0), float(item.get("score") or 0.0))
        existing_scores = existing.setdefault("source_scores", {})
        new_scores = item.get("source_scores") if isinstance(item.get("source_scores"), dict) else {}
        for key, value in new_scores.items():
            existing_scores[key] = max(float(existing_scores.get(key) or 0.0), float(value))
        existing_ranks = existing.setdefault("source_ranks", {})
        new_ranks = item.get("source_ranks") if isinstance(item.get("source_ranks"), dict) else {}
        for key, value in new_ranks.items():
            if key not in existing_ranks:
                existing_ranks[key] = int(value)
        existing_anchor_ids = {int(item_id) for item_id in existing.get("expansion_anchor_ids", [])}
        existing_anchor_ids.update(int(item_id) for item_id in item.get("expansion_anchor_ids", []) if str(item_id).isdigit())
        if existing_anchor_ids:
            existing["expansion_anchor_ids"] = sorted(existing_anchor_ids)
        existing_sources = {str(source) for source in existing.get("expansion_sources", [])}
        existing_sources.update(str(source) for source in item.get("expansion_sources", []) if str(source).strip())
        if existing_sources:
            existing["expansion_sources"] = sorted(existing_sources)
    ranked = list(merged.values())
    ranked.sort(
        key=lambda candidate: (
            float(candidate.get("score") or 0.0),
            max((float(score) for score in candidate.get("source_scores", {}).values()), default=0.0),
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked[:limit]


def expand_retrieval_candidates(
    db: Session,
    *,
    query: str,
    candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
    organization_id: int,
    query_plan: legacy_rag.RetrievalQueryPlan,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": False,
        "expanded_candidate_count": 0,
        "expansion_added_ids": [],
    }
    if not candidates:
        return candidates, stats

    enable_multi_span_expansion = should_expand_multi_span_candidates(query, query_plan=query_plan)
    anchor_candidates: list[tuple[dict[str, Any], PaperChunk]] = []
    parent_ids: set[int] = set()
    summary_candidates: list[tuple[dict[str, Any], PaperChunk, list[int]]] = []
    summary_anchor_ids: set[int] = set()
    for candidate in candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            continue
        chunk, _paper = record
        chunk_role = legacy_rag._chunk_role(chunk)
        if chunk_role == "child" and enable_multi_span_expansion:
            anchor_candidates.append((candidate, chunk))
            if chunk.parent_chunk_id is not None:
                parent_ids.add(int(chunk.parent_chunk_id))
            continue
        if not legacy_rag._is_summary_chunk_role(chunk_role):
            continue
        anchor_ids = legacy_rag._chunk_anchor_ids(chunk, limit=legacy_rag.settings.rag_summary_anchor_limit)
        if not anchor_ids:
            continue
        summary_candidates.append((candidate, chunk, anchor_ids))
        summary_anchor_ids.update(anchor_ids)

    if (not anchor_candidates or not parent_ids) and not summary_anchor_ids:
        return candidates, stats

    stats["enabled"] = True
    expanded_limit = min(max(limit + 6, limit), 24)
    max_siblings_per_anchor = 2

    related_rows = db.execute(
        select(PaperChunk, Paper)
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(
            Paper.deleted_at.is_(None),
            Paper.organization_id == organization_id,
            (PaperChunk.parent_chunk_id.in_(parent_ids)) | (PaperChunk.id.in_(parent_ids | summary_anchor_ids)),
        )
    ).all()
    related_record_map = {int(chunk.id): (chunk, paper) for chunk, paper in related_rows}

    parent_rows: dict[int, PaperChunk] = {}
    sibling_rows_by_parent: dict[int, list[PaperChunk]] = {}
    for chunk, _paper in related_rows:
        if int(chunk.id) in parent_ids and str(getattr(chunk, "chunk_role", "child") or "child") == "parent":
            parent_rows[int(chunk.id)] = chunk
            continue
        if chunk.parent_chunk_id is None:
            continue
        sibling_rows_by_parent.setdefault(int(chunk.parent_chunk_id), []).append(chunk)

    anchor_ids = {int(chunk.id) for _, chunk in anchor_candidates}
    additional_candidates: list[dict[str, Any]] = []
    if enable_multi_span_expansion:
        for candidate, anchor_chunk in anchor_candidates:
            anchor_score = float(candidate.get("score") or 0.0)
            anchor_rank = int(candidate.get("rank") or 0)
            parent_id = int(anchor_chunk.parent_chunk_id or 0)
            if parent_id <= 0:
                continue
            sibling_rows = [
                sibling
                for sibling in sibling_rows_by_parent.get(parent_id, [])
                if int(sibling.id) not in anchor_ids and int(sibling.id) != int(anchor_chunk.id)
            ]
            sibling_rows.sort(
                key=lambda sibling: (
                    abs(int(sibling.chunk_index or 0) - int(anchor_chunk.chunk_index or 0)),
                    int(sibling.chunk_index or 0),
                )
            )
            for sibling_rank, sibling in enumerate(sibling_rows[:max_siblings_per_anchor], start=1):
                decay = max(0.62, 0.88 - (sibling_rank - 1) * 0.12)
                additional_candidates.append(
                    {
                        "chunk_id": int(sibling.id),
                        "paper_id": int(sibling.paper_id),
                        "score": anchor_score * decay,
                        "source_scores": {"expansion_sibling": anchor_score * decay},
                        "source_ranks": {"expansion_sibling": max(anchor_rank, 1)},
                        "expansion_anchor_ids": [int(anchor_chunk.id)],
                        "expansion_sources": ["sibling"],
                    }
                )
            parent_row = parent_rows.get(parent_id)
            if parent_row is not None:
                parent_score = anchor_score * 0.72
                additional_candidates.append(
                    {
                        "chunk_id": int(parent_row.id),
                        "paper_id": int(parent_row.paper_id),
                        "score": parent_score,
                        "source_scores": {"expansion_parent": parent_score},
                        "source_ranks": {"expansion_parent": max(anchor_rank, 1)},
                        "expansion_anchor_ids": [int(anchor_chunk.id)],
                        "expansion_sources": ["parent"],
                    }
                )

    for candidate, summary_chunk, anchor_chunk_ids in summary_candidates:
        summary_score = float(candidate.get("score") or 0.0)
        summary_rank = int(candidate.get("rank") or 0)
        for anchor_rank, anchor_chunk_id in enumerate(anchor_chunk_ids[: legacy_rag.settings.rag_summary_anchor_limit], start=1):
            record = related_record_map.get(int(anchor_chunk_id)) or record_map.get(int(anchor_chunk_id))
            if record is None:
                continue
            anchor_chunk, _paper = record
            decay = max(0.58, 0.92 - (anchor_rank - 1) * 0.12)
            additional_candidates.append(
                {
                    "chunk_id": int(anchor_chunk.id),
                    "paper_id": int(anchor_chunk.paper_id),
                    "score": summary_score * decay,
                    "source_scores": {"expansion_summary_anchor": summary_score * decay},
                    "source_ranks": {"expansion_summary_anchor": max(summary_rank, 1)},
                    "expansion_anchor_ids": [int(summary_chunk.id)],
                    "expansion_sources": ["summary_anchor"],
                }
            )

    if not additional_candidates:
        return candidates, stats

    merged = merge_ranked_candidates(candidates, additional_candidates, limit=expanded_limit)
    added_ids = [int(item["chunk_id"]) for item in merged if int(item["chunk_id"]) not in {int(candidate["chunk_id"]) for candidate in candidates}]
    record_map.update(related_record_map)
    stats["expanded_candidate_count"] = len(added_ids)
    stats["expansion_added_ids"] = added_ids
    return merged, stats


def extract_query_numbers(text: str) -> tuple[str, ...]:
    return tuple(sorted({match.group(0) for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", text or "")}))


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def truncate_for_budget(text: str, *, max_chars: int) -> str:
    compact = compact_text(text)
    if len(compact) <= max_chars:
        return compact
    if max_chars <= 3:
        return compact[:max_chars]
    return f"{compact[: max_chars - 3].rstrip()}..."


def summarize_text_excerpt(text: str, *, max_sentences: int, max_chars: int) -> str:
    compact = compact_text(text)
    if not compact:
        return ""
    sentence_candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？.!?])\s+", compact)
        if sentence.strip()
    ]
    if not sentence_candidates:
        sentence_candidates = [compact]
    selected = " ".join(sentence_candidates[: max(max_sentences, 1)])
    return truncate_for_budget(selected or compact, max_chars=max_chars)


def serialize_structured_summary_for_chunk(summary: dict[str, Any] | None) -> str:
    if not isinstance(summary, dict):
        return ""
    parts: list[str] = []
    abstract_cn = compact_text(str(summary.get("abstract_cn") or ""))
    if abstract_cn:
        parts.append(f"摘要：{abstract_cn}")
    research_question = compact_text(str(summary.get("research_question") or ""))
    if research_question:
        parts.append(f"研究问题：{research_question}")
    method = compact_text(str(summary.get("method") or ""))
    if method:
        parts.append(f"方法：{method}")
    findings = compact_text(str(summary.get("findings") or ""))
    if findings:
        parts.append(f"发现：{findings}")
    limitations = compact_text(str(summary.get("limitations") or ""))
    if limitations:
        parts.append(f"局限：{limitations}")
    key_points = [
        compact_text(str(item))
        for item in (summary.get("key_points") if isinstance(summary.get("key_points"), list) else [])
        if compact_text(str(item))
    ]
    if key_points:
        parts.append("要点：" + "；".join(key_points[:4]))
    return truncate_for_budget("\n".join(parts), max_chars=legacy_rag.settings.rag_paper_summary_max_chars) if parts else ""


def build_fallback_paper_summary_text(
    section_summaries: list[str],
    *,
    normalized_blocks: list[dict[str, Any]],
) -> str:
    compact_summaries = [compact_text(item) for item in section_summaries if compact_text(item)]
    if compact_summaries:
        return truncate_for_budget("\n".join(compact_summaries[:3]), max_chars=legacy_rag.settings.rag_paper_summary_max_chars)
    fallback_text = "\n".join(
        compact_text(str(block.get("text") or ""))
        for block in normalized_blocks[:6]
        if compact_text(str(block.get("text") or ""))
    )
    return truncate_for_budget(fallback_text, max_chars=legacy_rag.settings.rag_paper_summary_max_chars) if fallback_text else ""


def build_rerank_block_catalog(
    db: Session,
    *,
    paper_ids: Iterable[int],
) -> dict[int, dict[str, Any]]:
    catalogs: dict[int, dict[str, Any]] = {}
    for paper_id in sorted({int(item) for item in paper_ids if int(item)}):
        preanalysis = legacy_rag._build_preanalysis_from_document_structure(db, paper_id=paper_id) or {}
        blocks = preanalysis.get("blocks") if isinstance(preanalysis, dict) else []
        if not isinstance(blocks, list):
            blocks = []
        catalogs[paper_id] = {
            "blocks": blocks,
            "by_block_index": {
                int(block.get("block_index")): block
                for block in blocks
                if isinstance(block, dict) and (isinstance(block.get("block_index"), int) or str(block.get("block_index") or "").isdigit())
            },
            "by_source_block_id": {
                int(block.get("source_block_id")): block
                for block in blocks
                if isinstance(block, dict) and (isinstance(block.get("source_block_id"), int) or str(block.get("source_block_id") or "").isdigit())
            },
            "by_source_table_id": {
                int(block.get("source_table_id")): block
                for block in blocks
                if isinstance(block, dict) and (isinstance(block.get("source_table_id"), int) or str(block.get("source_table_id") or "").isdigit())
            },
            "by_source_picture_id": {
                int(block.get("source_picture_id")): block
                for block in blocks
                if isinstance(block, dict) and (isinstance(block.get("source_picture_id"), int) or str(block.get("source_picture_id") or "").isdigit())
            },
        }
    return catalogs


def score_rerank_text(
    text: str,
    *,
    query_tokens: list[str],
    exact_terms: list[str],
    query_numbers: tuple[str, ...],
    prefers_table: bool,
    unit_type: str,
    is_primary: bool,
    section_path: str,
) -> float:
    compact = compact_text(text).lower()
    if not compact:
        return 0.0
    score = 0.0
    query_token_set = set(query_tokens)
    unit_token_set = set(legacy_rag._tokenize(compact))
    score += float(len(query_token_set & unit_token_set))
    exact_hits = sum(1 for term in exact_terms if term.lower() in compact)
    score += exact_hits * 4.0
    if section_path:
        section_compact = section_path.lower()
        score += sum(1.5 for term in exact_terms if term.lower() in section_compact)
    if query_numbers:
        text_numbers = {match.group(0) for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", compact)}
        score += float(len(set(query_numbers) & text_numbers)) * 3.0
    if prefers_table and unit_type in {"table_caption", "table_row"}:
        score += 1.5
    if is_primary:
        score += 2.0
    if unit_type == "neighbor_context":
        score -= 0.25
    return score


def neighbor_blocks_for_rerank(
    block: dict[str, Any],
    *,
    catalog: dict[str, Any],
    neighbor_count: int,
) -> list[dict[str, Any]]:
    if neighbor_count <= 0:
        return []
    block_index = int(block.get("block_index") or -1)
    if block_index < 0:
        return []
    by_block_index = catalog.get("by_block_index") if isinstance(catalog, dict) else {}
    section_id = str(block.get("section_id") or "")
    neighbors: list[dict[str, Any]] = []
    for distance in range(1, neighbor_count + 1):
        for candidate_index in (block_index - distance, block_index + distance):
            candidate = by_block_index.get(candidate_index) if isinstance(by_block_index, dict) else None
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("section_id") or "") != section_id:
                continue
            neighbors.append(candidate)
    return neighbors


def table_evidence_units(block: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    table_text = str(block.get("text") or "").strip()
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if lines:
        units.append({"text": lines[0], "unit_type": "table_caption"})
    serialized_rows = legacy_rag._serialize_table_rows(block.get("table_data_json"), max_rows=16)
    row_lines = [line.strip() for line in serialized_rows.splitlines() if line.strip()]
    if row_lines:
        for row_line in row_lines:
            units.append({"text": row_line, "unit_type": "table_row"})
    elif len(lines) > 1:
        for row_line in lines[1:9]:
            units.append({"text": row_line, "unit_type": "table_row"})
    if not units and table_text:
        units.append({"text": table_text, "unit_type": "table_row"})
    return units


def candidate_rerank_units(
    chunk: PaperChunk,
    *,
    catalog: dict[str, Any],
    query: str,
) -> list[dict[str, Any]]:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    source_blocks = metadata.get("source_block_ids") if isinstance(metadata.get("source_block_ids"), list) else []
    source_tables = metadata.get("source_table_ids") if isinstance(metadata.get("source_table_ids"), list) else []
    source_pictures = metadata.get("source_picture_ids") if isinstance(metadata.get("source_picture_ids"), list) else []
    prefers_table = legacy_rag._is_table_or_figure_query(query)
    query_tokens = legacy_rag._tokenize(query)
    exact_terms = legacy_rag._extract_exact_match_terms(query)
    query_numbers = extract_query_numbers(query)
    units: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    source_candidates: list[tuple[dict[str, Any], bool]] = []
    for block_id in source_blocks:
        block = catalog.get("by_source_block_id", {}).get(int(block_id))
        if isinstance(block, dict):
            source_candidates.append((block, True))
    for table_id in source_tables:
        block = catalog.get("by_source_table_id", {}).get(int(table_id))
        if isinstance(block, dict):
            source_candidates.append((block, True))
    for picture_id in source_pictures:
        block = catalog.get("by_source_picture_id", {}).get(int(picture_id))
        if isinstance(block, dict):
            source_candidates.append((block, True))
    for block, is_primary in list(source_candidates):
        for neighbor in neighbor_blocks_for_rerank(
            block,
            catalog=catalog,
            neighbor_count=legacy_rag.settings.rerank_neighbor_blocks,
        ):
            source_candidates.append((neighbor, False))
    for block, is_primary in source_candidates:
        block_type = str(block.get("block_type") or "paragraph")
        unit_specs = (
            table_evidence_units(block)
            if block_type == "table_like"
            else [{"text": str(block.get("text") or "").strip(), "unit_type": "primary_block" if is_primary else "neighbor_context"}]
        )
        for unit_spec in unit_specs:
            text = compact_text(unit_spec.get("text") or "")
            if not text or text in seen_texts:
                continue
            unit_type = str(unit_spec.get("unit_type") or "primary_block")
            section_path = str(block.get("section_path") or "")
            score = score_rerank_text(
                text,
                query_tokens=query_tokens,
                exact_terms=exact_terms,
                query_numbers=query_numbers,
                prefers_table=prefers_table,
                unit_type=unit_type,
                is_primary=is_primary,
                section_path=section_path,
            )
            units.append(
                {
                    "text": text,
                    "unit_type": unit_type,
                    "score": score,
                    "is_primary": is_primary,
                    "block_index": int(block.get("block_index") or -1),
                    "section_path": section_path,
                    "source_key": (
                        f"block:{block.get('source_block_id')}"
                        if block.get("source_block_id") is not None
                        else f"table:{block.get('source_table_id')}"
                        if block.get("source_table_id") is not None
                        else f"picture:{block.get('source_picture_id')}"
                        if block.get("source_picture_id") is not None
                        else f"idx:{int(block.get('block_index') or -1)}"
                    ),
                }
            )
            seen_texts.add(text)
    return units


def build_rerank_context_for_chunk(
    query: str,
    *,
    chunk: PaperChunk,
    catalog: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    header = str(metadata.get("context_header") or "").strip()
    if not isinstance(catalog, dict):
        raw_chunk_text = legacy_rag._chunk_text_payload(chunk, include_supporting_context=False)
        fallback = truncate_for_budget(raw_chunk_text, max_chars=legacy_rag.settings.rerank_max_context_chars)
        return fallback, {
            "rerank_context_chars": len(fallback),
            "rerank_evidence_blocks": 0,
            "rerank_context_truncated": len(fallback) < len(compact_text(raw_chunk_text)),
            "rerank_evidence_types": [],
        }

    units = candidate_rerank_units(chunk, catalog=catalog, query=query)
    grouped_primary: dict[str, dict[str, Any]] = {}
    others: list[dict[str, Any]] = []
    for unit in sorted(
        units,
        key=lambda item: (-float(item.get("score") or 0.0), not bool(item.get("is_primary")), int(item.get("block_index") or -1)),
    ):
        source_key = str(unit.get("source_key") or "")
        if unit.get("is_primary") and source_key and source_key not in grouped_primary:
            grouped_primary[source_key] = unit
            continue
        others.append(unit)
    ordered_units = list(grouped_primary.values()) + others

    pieces: list[str] = []
    used_types: list[str] = []
    total_chars = 0
    max_chars = max(int(legacy_rag.settings.rerank_max_context_chars), 200)
    if header:
        pieces.append(header)
        total_chars = len(header)
    selected_count = 0
    truncated = False
    for unit in ordered_units:
        if selected_count >= max(int(legacy_rag.settings.rerank_max_evidence_blocks), 1):
            truncated = True
            break
        label = {
            "table_caption": "表格标题",
            "table_row": "表格证据",
            "neighbor_context": "相邻上下文",
        }.get(str(unit.get("unit_type") or ""), "正文证据")
        formatted = f"{label}: {str(unit.get('text') or '').strip()}"
        if not formatted.strip():
            continue
        remaining = max_chars - total_chars - (2 if pieces else 0)
        if remaining <= 40:
            truncated = True
            break
        if len(formatted) > remaining:
            formatted = truncate_for_budget(formatted, max_chars=remaining)
            truncated = True
        pieces.append(formatted)
        total_chars += len(formatted) + (2 if len(pieces) > 1 else 0)
        used_types.append(str(unit.get("unit_type") or "primary_block"))
        selected_count += 1

    if len(pieces) <= 1:
        fallback_body = str(metadata.get("body_text") or chunk.content or "").strip()
        fallback = "\n\n".join(part for part in (header, truncate_for_budget(fallback_body, max_chars=max_chars // 2)) if part)
        return fallback, {
            "rerank_context_chars": len(fallback),
            "rerank_evidence_blocks": 0,
            "rerank_context_truncated": truncated or len(fallback) >= max_chars,
            "rerank_evidence_types": [],
        }

    context = "\n\n".join(part for part in pieces if part)
    return context, {
        "rerank_context_chars": len(context),
        "rerank_evidence_blocks": selected_count,
        "rerank_context_truncated": truncated,
        "rerank_evidence_types": sorted(set(used_types)),
    }


def apply_reranking(
    db: Session,
    query: str,
    *,
    fused_candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents: list[str] = []
    aligned_candidates: list[dict[str, Any]] = []
    rerank_stats = {
        "query_chars": len(query or ""),
        "documents_count": 0,
        "max_document_chars": 0,
        "over_budget_truncated_count": 0,
    }
    block_catalogs = build_rerank_block_catalog(
        db,
        paper_ids=[paper.id for _, paper in record_map.values()],
    )
    for candidate in fused_candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            continue
        chunk, paper = record
        rerank_context, context_meta = build_rerank_context_for_chunk(
            query,
            chunk=chunk,
            catalog=block_catalogs.get(int(paper.id)),
        )
        if not rerank_context:
            continue
        candidate_with_meta = dict(candidate)
        candidate_with_meta.update(context_meta)
        documents.append(rerank_context)
        aligned_candidates.append(candidate_with_meta)
        rerank_stats["documents_count"] = len(documents)
        rerank_stats["max_document_chars"] = max(int(rerank_stats["max_document_chars"]), len(rerank_context))
        if bool(context_meta.get("rerank_context_truncated")):
            rerank_stats["over_budget_truncated_count"] = int(rerank_stats["over_budget_truncated_count"]) + 1
    if not documents:
        return [], rerank_stats
    rerank_results = legacy_rag.rerank_documents(query, documents, top_n=limit)
    if not rerank_results:
        return aligned_candidates[:limit], rerank_stats

    reranked: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for rank, item in enumerate(rerank_results, start=1):
        if item.index < 0 or item.index >= len(aligned_candidates):
            continue
        candidate = dict(aligned_candidates[item.index])
        candidate["score"] = float(item.relevance_score)
        candidate["rerank_score"] = float(item.relevance_score)
        candidate["rerank_rank"] = rank
        reranked.append(candidate)
        seen_indexes.add(item.index)

    for index, candidate in enumerate(aligned_candidates):
        if index in seen_indexes:
            continue
        reranked.append(dict(candidate))

    return reranked[:limit], rerank_stats
