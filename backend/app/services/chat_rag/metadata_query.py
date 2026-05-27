from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.models import Paper, PaperAsset
from app.services import rag as legacy_rag
from app.services.paper_pipeline.rendering import render_paper_text_from_structure


@dataclass(frozen=True)
class MetadataQueryPlan:
    operation: str
    filters: dict[str, Any]
    limit: int = 8
    wants_content_followup: bool = False
    followup_query: str | None = None
    title_hint: str | None = None


@dataclass(frozen=True)
class MetadataQueryResult:
    answer: str
    total_count: int
    items: tuple[dict[str, Any], ...]
    paper_ids: tuple[int, ...]
    trace: dict[str, Any]


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in patterns)


def _extract_years(text: str) -> list[int]:
    return [int(match.group(0)) for match in re.finditer(r"\b(19|20)\d{2}\b", text or "")]


def _extract_title_hint(text: str) -> str | None:
    quoted_match = re.search(r"[\"'“”‘’《](.+?)[\"'“”‘’》]", text or "")
    if quoted_match:
        hint = str(quoted_match.group(1) or "").strip()
        return hint or None
    return None


def _build_followup_query(text: str) -> str | None:
    segments = [segment.strip() for segment in re.split(r"[?？!！。；;]", text or "") if segment.strip()]
    followup_patterns = (
        "主要结论",
        "主要研究",
        "研究什么",
        "讲了什么",
        "内容是什么",
        "总结",
        "研究结局",
        "结局变量",
        "主要结果",
        "主要发现",
        "what do they study",
        "what are they about",
        "main findings",
        "main conclusion",
        "main conclusions",
        "about what",
    )
    for segment in reversed(segments):
        if _contains_any(segment, followup_patterns):
            return segment
    return None


def build_metadata_query_plan(query: str) -> MetadataQueryPlan | None:
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    if not normalized:
        return None

    count_patterns = ("有几个", "多少篇", "多少个文档", "几篇", "几份", "how many", "count")
    list_patterns = ("有哪些", "列出", "哪些文档", "list", "show me", "which papers", "which documents")
    exists_patterns = ("有没有", "是否有", "exists", "is there", "are there")
    word_count_patterns = ("多少字", "字数", "多少词", "词数", "word count")
    group_by_patterns = ("按语言", "按状态", "分语言", "分状态", "各语言", "各种状态", "by language", "by status")
    content_followup_patterns = (
        "主要结论",
        "主要研究",
        "研究什么",
        "讲了什么",
        "内容是什么",
        "总结",
        "研究结局",
        "结局变量",
        "主要结果",
        "主要发现",
        "what do they study",
        "what are they about",
        "main findings",
        "main conclusion",
        "main conclusions",
        "about what",
    )
    operation: str | None = None
    if _contains_any(normalized, word_count_patterns):
        operation = "paper_word_count"
    elif _contains_any(normalized, count_patterns):
        operation = "count"
    elif _contains_any(normalized, group_by_patterns):
        operation = "group_by_small_enum"
    elif _contains_any(normalized, list_patterns):
        operation = "list"
    elif _contains_any(normalized, exists_patterns):
        operation = "exists"

    filters: dict[str, Any] = {}
    lowered = normalized.lower()
    if "中文" in normalized or " chinese " in f" {lowered} ":
        filters["language"] = "zh"
    elif "英文" in normalized or " english " in f" {lowered} ":
        filters["language"] = "en"

    years = _extract_years(normalized)
    if len(years) >= 2:
        filters["published_year_from"] = min(years)
        filters["published_year_to"] = max(years)
    elif years:
        filters["published_year"] = years[0]
    if _contains_any(normalized, ("doi 缺失", "没有 doi", "无 doi", "missing doi", "without doi")):
        filters["doi_missing"] = True
    if _contains_any(normalized, ("source url 缺失", "来源链接缺失", "没有 source url", "missing source url", "without source url")):
        filters["source_url_missing"] = True
    if _contains_any(normalized, ("按语言", "分语言", "各语言", "by language")):
        filters["group_by"] = "language"
    elif _contains_any(normalized, ("按状态", "分状态", "各种状态", "by status")):
        filters["group_by"] = "status"
    status_filters: list[str] = []
    if _contains_any(normalized, ("已完成", "完成", "completed")):
        status_filters.append("completed")
    if _contains_any(normalized, ("处理中", "processing")):
        status_filters.append("processing")
    if _contains_any(normalized, ("失败", "failed")):
        status_filters.append("failed")
    if status_filters:
        filters["status_in"] = sorted(set(status_filters))

    wants_content_followup = _contains_any(normalized, content_followup_patterns)
    followup_query = _build_followup_query(normalized) if wants_content_followup else None
    title_hint = _extract_title_hint(normalized)
    if title_hint:
        filters["title_hint"] = title_hint
    has_structured_filters = bool(filters)
    if operation is None:
        if wants_content_followup and has_structured_filters:
            operation = "filter_scope"
        elif has_structured_filters:
            operation = "filter_scope"
        else:
            return None
    return MetadataQueryPlan(
        operation=operation,
        filters=filters,
        limit=8,
        wants_content_followup=wants_content_followup,
        followup_query=followup_query,
        title_hint=title_hint,
    )


def _infer_paper_language(paper: Paper, asset: PaperAsset | None) -> str:
    metadata = asset.metadata_json if asset and isinstance(asset.metadata_json, dict) else {}
    structured_summary = metadata.get("structured_summary") if isinstance(metadata, dict) else None
    explicit_language = (
        metadata.get("document_language")
        or (structured_summary.get("language") if isinstance(structured_summary, dict) else None)
        or paper.metadata_json.get("language") if hasattr(paper, "metadata_json") and isinstance(getattr(paper, "metadata_json", None), dict) else None
    )
    if isinstance(explicit_language, str) and explicit_language.strip():
        normalized = explicit_language.strip().lower()
        if normalized.startswith("zh"):
            return "zh"
        if normalized.startswith("en"):
            return "en"
    sample_text = " ".join(part for part in [paper.title or "", paper.abstract_raw or ""] if part).strip()
    detected = legacy_rag._detect_query_language(sample_text)
    if detected in {"zh", "en"}:
        return detected
    if detected == "mixed":
        return "zh" if legacy_rag._is_cjk_dominant_text(sample_text) else "en"
    return "unknown"


def _paper_rows(db: Session, *, organization_id: int) -> list[tuple[Paper, PaperAsset | None]]:
    rows = db.execute(
        select(Paper, PaperAsset)
        .outerjoin(
            PaperAsset,
            and_(PaperAsset.paper_id == Paper.id, PaperAsset.asset_type == "original_pdf"),
        )
        .where(Paper.deleted_at.is_(None), Paper.organization_id == organization_id)
        .order_by(Paper.id.asc())
    ).all()
    return [(paper, asset) for paper, asset in rows]


def _apply_filters(rows: list[tuple[Paper, PaperAsset | None]], *, filters: dict[str, Any]) -> list[tuple[Paper, PaperAsset | None]]:
    filtered = rows
    language = str(filters.get("language") or "").strip()
    if language:
        filtered = [(paper, asset) for paper, asset in filtered if _infer_paper_language(paper, asset) == language]

    published_year = filters.get("published_year")
    if isinstance(published_year, int):
        filtered = [
            (paper, asset)
            for paper, asset in filtered
            if isinstance(paper.published_at, datetime) and paper.published_at.year == published_year
        ]
    year_from = filters.get("published_year_from")
    if isinstance(year_from, int):
        filtered = [
            (paper, asset)
            for paper, asset in filtered
            if isinstance(paper.published_at, datetime) and paper.published_at.year >= year_from
        ]
    year_to = filters.get("published_year_to")
    if isinstance(year_to, int):
        filtered = [
            (paper, asset)
            for paper, asset in filtered
            if isinstance(paper.published_at, datetime) and paper.published_at.year <= year_to
        ]

    if filters.get("doi_missing"):
        filtered = [(paper, asset) for paper, asset in filtered if not str(paper.doi or "").strip()]
    if filters.get("source_url_missing"):
        filtered = [(paper, asset) for paper, asset in filtered if not str(paper.source_url or "").strip()]
    status_in = [str(item).strip() for item in filters.get("status_in", []) if str(item).strip()]
    if status_in:
        allowed = set(status_in)
        filtered = [(paper, asset) for paper, asset in filtered if str(paper.status or "").strip() in allowed]
    title_hint = str(filters.get("title_hint") or "").strip().lower()
    if title_hint:
        filtered = [
            (paper, asset)
            for paper, asset in filtered
            if title_hint in str(paper.title or "").strip().lower()
        ]
    return filtered


def _match_title_hint(rows: list[tuple[Paper, PaperAsset | None]], *, title_hint: str | None) -> tuple[Paper, PaperAsset | None] | None:
    hint = str(title_hint or "").strip().lower()
    if not hint:
        return None
    for paper, asset in rows:
        title = str(paper.title or "").strip().lower()
        if title and (hint in title or title in hint):
            return paper, asset
    return None


def _paper_item(paper: Paper, asset: PaperAsset | None) -> dict[str, Any]:
    return {
        "paper_id": int(paper.id),
        "title": paper.title,
        "published_year": paper.published_at.year if isinstance(paper.published_at, datetime) else None,
        "language": _infer_paper_language(paper, asset),
        "doi": paper.doi,
        "source_url": paper.source_url,
    }


def _format_list_answer(items: list[dict[str, Any]], *, filters: dict[str, Any]) -> str:
    if not items:
        return "按当前条件未找到匹配文档。"
    prefix_parts: list[str] = []
    if filters.get("language") == "zh":
        prefix_parts.append("中文")
    elif filters.get("language") == "en":
        prefix_parts.append("英文")
    if isinstance(filters.get("published_year"), int):
        prefix_parts.append(f"{filters['published_year']} 年")
    year_from = filters.get("published_year_from")
    year_to = filters.get("published_year_to")
    if isinstance(year_from, int) and isinstance(year_to, int):
        prefix_parts.append(f"{year_from}-{year_to} 年")
    prefix = "".join(prefix_parts)
    lines = [f"{prefix}匹配文档如下：" if prefix else "匹配文档如下："]
    for item in items:
        year_text = f"（{item['published_year']}）" if item.get("published_year") else ""
        lines.append(f"- {item.get('title') or '未命名论文'}{year_text}")
    return "\n".join(lines)


def execute_metadata_query(db: Session, *, organization_id: int, plan: MetadataQueryPlan) -> MetadataQueryResult:
    rows = _paper_rows(db, organization_id=organization_id)
    filtered_rows = _apply_filters(rows, filters=plan.filters)

    if plan.operation == "paper_word_count":
        matched = _match_title_hint(filtered_rows or rows, title_hint=plan.title_hint)
        if matched is None:
            answer = "未能根据问题中的论文标题定位到目标文档，暂时无法计算字数。"
            return MetadataQueryResult(
                answer=answer,
                total_count=0,
                items=(),
                paper_ids=(),
                trace={"plan": _serialize_plan(plan), "matched_paper_ids": [], "operation": plan.operation},
            )
        paper, _asset = matched
        paper_text = render_paper_text_from_structure(db, paper.id)
        char_count = len((paper_text or "").strip())
        word_count = len(legacy_rag._tokenize(paper_text or ""))
        answer = f"《{paper.title or '未命名论文'}》当前结构化文本约 {char_count} 个字符，约 {word_count} 个词。"
        return MetadataQueryResult(
            answer=answer,
            total_count=1,
            items=(_paper_item(paper, _asset),),
            paper_ids=(int(paper.id),),
            trace={
                "plan": _serialize_plan(plan),
                "matched_paper_ids": [int(paper.id)],
                "operation": plan.operation,
                "char_count": char_count,
                "word_count": word_count,
            },
        )

    items = [_paper_item(paper, asset) for paper, asset in filtered_rows[: plan.limit]]
    paper_ids = tuple(int(item["paper_id"]) for item in items)
    total_count = len(filtered_rows)
    if plan.operation == "group_by_small_enum":
        group_key = str(plan.filters.get("group_by") or "language")
        bucket: dict[str, int] = {}
        for paper, asset in filtered_rows:
            if group_key == "language":
                key = _infer_paper_language(paper, asset) or "unknown"
            else:
                key = str(paper.status or "unknown").strip() or "unknown"
            bucket[key] = bucket.get(key, 0) + 1
        sorted_items = [{"group": key, "count": value} for key, value in sorted(bucket.items(), key=lambda item: (-item[1], item[0]))]
        answer = "\n".join([f"按{group_key}统计：", *[f"- {item['group']}: {item['count']} 篇" for item in sorted_items]]) if sorted_items else "按当前条件未找到匹配文档。"
        return MetadataQueryResult(
            answer=answer,
            total_count=total_count,
            items=tuple(sorted_items),
            paper_ids=paper_ids,
            trace={
                "plan": _serialize_plan(plan),
                "matched_paper_ids": list(paper_ids),
                "operation": plan.operation,
                "group_key": group_key,
                "total_count": total_count,
                "items": sorted_items,
            },
        )
    if plan.operation == "count":
        answer = f"按当前条件，共找到 {total_count} 篇文档。"
    elif plan.operation == "exists":
        answer = "按当前条件找到了匹配文档。" if total_count > 0 else "按当前条件未找到匹配文档。"
    else:
        answer = _format_list_answer(items, filters=plan.filters)

    return MetadataQueryResult(
        answer=answer,
        total_count=total_count,
        items=tuple(items),
        paper_ids=paper_ids,
        trace={
            "plan": _serialize_plan(plan),
            "matched_paper_ids": list(paper_ids),
            "operation": plan.operation,
            "total_count": total_count,
            "items": list(items),
        },
    )


def _serialize_plan(plan: MetadataQueryPlan) -> dict[str, Any]:
    return {
        "operation": plan.operation,
        "filters": dict(plan.filters),
        "limit": plan.limit,
        "wants_content_followup": plan.wants_content_followup,
        "followup_query": plan.followup_query,
        "title_hint": plan.title_hint,
    }


def deserialize_metadata_query_plan(payload: dict[str, Any] | None) -> MetadataQueryPlan | None:
    if not isinstance(payload, dict):
        return None
    return MetadataQueryPlan(
        operation=str(payload.get("operation") or "list"),
        filters=dict(payload.get("filters") or {}),
        limit=int(payload.get("limit") or 8),
        wants_content_followup=bool(payload.get("wants_content_followup")),
        followup_query=str(payload.get("followup_query") or "") or None,
        title_hint=str(payload.get("title_hint") or "") or None,
    )


def serialize_metadata_query_plan(plan: MetadataQueryPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return _serialize_plan(plan)


def serialize_metadata_query_result(result: MetadataQueryResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "answer": result.answer,
        "total_count": result.total_count,
        "items": list(result.items),
        "paper_ids": list(result.paper_ids),
        "trace": dict(result.trace),
    }
