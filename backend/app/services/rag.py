from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ChatMessage, ChatTopic, Paper, PaperChunk, PaperDocumentBlock, PaperDocumentPicture, PaperDocumentTable, User
from app.services.llm import chat_with_messages, embed_texts, get_chat_llm_configuration, rerank_documents

logger = logging.getLogger(__name__)

DEFAULT_TOPIC_TITLE = "新话题"
MAX_HISTORY_MESSAGES = 8
MIN_RELEVANCE_SCORE = 0.12
SUMMARY_CHUNK_ROLES = {"section_summary", "paper_summary"}
DEFAULT_SEARCHABLE_CHUNK_ROLES = ("child", "section_summary", "paper_summary")

RAG_SYSTEM_PROMPT = (
    "你是 ResearchDock 的论文知识库助理。"
    "请优先依据给定的知识库证据回答，避免编造论文内容。"
    "如果证据不足，请直接说明知识库中没有找到确切依据。"
)

FALLBACK_SYSTEM_PROMPT = (
    "你是 ResearchDock 的研究助理。"
    "当前知识库没有找到足够证据，请先明确告诉用户“知识库中未找到确切依据”。"
    "随后你可以基于通用知识给出简短补充，但必须避免伪造论文引用。"
)


def _preview_log_text(text: str, *, limit: int = 100) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."

QUERY_PLAN_SYSTEM_PROMPT = (
    "你是跨语言学术检索规划器。"
    "请把用户问题转成适合英文论文检索的 JSON 计划。"
    "必须返回 JSON 对象，不要输出额外解释。"
)

QUERY_PLAN_USER_TEMPLATE = """
请根据下面的问题，输出一个 JSON 对象，字段必须严格包含：
- detected_language: 字符串，只能是 zh、en 或 mixed
- retrieval_query_en: 用于检索英文论文的英文 query；如果不需要可返回空字符串
- exact_terms: 字符串数组，保留应被精确匹配的术语、缩写、表号、图号、指标名
- subqueries_en: 字符串数组，只有在复杂问题确实需要拆分时才返回，最多 {max_subqueries} 条
- generation_instruction: 面向回答阶段的中文指令，要求中文作答、保留关键英文术语原文并引用英文证据

规则：
1. 用户原问题如果是中文，retrieval_query_en 必须是适合英文论文检索的英文表达，而不是逐字翻译。
2. 保留论文中的关键英文术语，例如 tES、ADHD-RS、Table 3、theta wave、within-subject design。
3. 只有在问题明显包含两个独立信息点时，才拆成 subqueries_en。
4. 不要编造论文中不存在的新术语。
5. 如果问题很短，或包含缩写、图表号、量表名、术语名、数值单位，不要把这些精确术语泛化掉。
6. retrieval_query_en 必须是适合 sparse 检索的自然关键词串，不要输出 OR/AND/NOT 这类布尔模板。

用户问题：
{query}
""".strip()

_QUERY_TEMPLATE_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"(研究设计|试验设计|研究方法)"), ("study design",)),
    (re.compile(r"(主要结局|主要终点|主要结局指标)"), ("primary outcome measure",)),
    (re.compile(r"(年龄范围|年龄段|纳入.*年龄)"), ("age range",)),
    (re.compile(r"(样本量|受试者|参与者)"), ("sample size", "participants")),
    (re.compile(r"(完成研究|最终完成)"), ("completed participants",)),
    (re.compile(r"(干预|刺激|治疗)"), ("intervention",)),
    (re.compile(r"(对照组|对照条件|安慰剂|假刺激)"), ("control group",)),
    (re.compile(r"(量表)"), ("scale",)),
    (re.compile(r"(副作用|不良反应|安全性)"), ("adverse events", "side effects")),
    (re.compile(r"(显著性|p值|p-value|P值)"), ("p-value", "statistical significance")),
    (re.compile(r"(基线)"), ("baseline",)),
)

EVIDENCE_SELECTION_SYSTEM_PROMPT = (
    "你是 ResearchDock 的归因证据选择器。"
    "你需要先拆出用户问题中的核心 claim，再从给定证据中挑出真正能支持这些 claim 的证据。"
    "用户问题、最终回答和证据可以是不同语言，例如中文问题对应英文论文证据。"
    "必须只返回 JSON 对象，不要输出额外解释。"
)

EVIDENCE_VERIFIER_SYSTEM_PROMPT = (
    "你是 ResearchDock 的 groundedness verifier。"
    "请判断最终答案是否被给定证据真实支持。"
    "要允许跨语言支持关系，例如中文回答可由英文证据支持。"
    "必须只返回 JSON 对象，不要输出额外解释。"
)


@dataclass
class TopicSummary:
    topic: ChatTopic
    message_count: int
    last_message_at: datetime | None


@dataclass
class ChatTurnResult:
    topic: TopicSummary
    user_message: ChatMessage
    assistant_message: ChatMessage


@dataclass(frozen=True)
class AssistantMessageDraft:
    content: str
    model: str | None
    answer_mode: str | None
    used_knowledge_base: bool
    citations_json: list[dict[str, Any]]
    metadata_json: dict[str, Any]


@dataclass
class PreparedChatTurn:
    topic: ChatTopic
    user_message: ChatMessage
    assistant_draft: AssistantMessageDraft


@dataclass
class StartedChatTurn:
    topic: ChatTopic
    user_message: ChatMessage
    records: list[ChatMessage]
    prompt: str


@dataclass
class RetrievalResult:
    chunk: PaperChunk
    paper: Paper
    score: float


@dataclass(frozen=True)
class RetrievalQueryVariant:
    name: str
    query: str
    language: str
    use_sparse: bool
    use_dense: bool
    role: str


@dataclass(frozen=True)
class RetrievalQueryPlan:
    original_query: str
    detected_language: str
    query_type: str
    generation_instruction: str
    rerank_query: str
    exact_terms: tuple[str, ...]
    retrieval_query_en: str | None
    exact_guardrail_query_en: str | None
    subqueries_en: tuple[str, ...]
    variants: tuple[RetrievalQueryVariant, ...]
    rewrite_status: str
    llm_rewrite_status: str
    used_llm: bool
    llm_attempted: bool
    rewrite_provider: str | None
    rewrite_model: str | None
    fallback_source: str | None
    rewrite_error: str | None
    rewrite_backfilled_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceSelectionResult:
    selected_evidence: tuple[dict[str, Any], ...]
    claims: tuple[dict[str, Any], ...]
    overall_support_score: float
    sufficiency_decision: dict[str, Any]
    missing_information: str
    method: str


@dataclass(frozen=True)
class ChatAttributionPolicy:
    name: str
    min_support_score: float
    min_total_support_score: float
    verifier_min_support_score: float
    llm_insufficient_hard_gate: bool
    allow_fallback_generation: bool


ChatProgressCallback = Callable[[str, str, str, str | None], None]


STRICT_CHAT_ATTRIBUTION_POLICY = ChatAttributionPolicy(
    name="strict",
    min_support_score=settings.rag_attribution_min_support_score,
    min_total_support_score=settings.rag_attribution_min_total_support_score,
    verifier_min_support_score=settings.rag_attribution_verifier_min_support_score,
    llm_insufficient_hard_gate=True,
    allow_fallback_generation=False,
)

RELAXED_CHAT_ATTRIBUTION_POLICY = ChatAttributionPolicy(
    name="relaxed_chat",
    min_support_score=0.45,
    min_total_support_score=0.75,
    verifier_min_support_score=0.35,
    llm_insufficient_hard_gate=True,
    allow_fallback_generation=True,
)


def _normalize_title(title: str | None) -> str:
    normalized = (title or "").strip()
    return normalized or DEFAULT_TOPIC_TITLE


def _excerpt(text: str, max_length: int = 24) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= max_length:
        return compact or DEFAULT_TOPIC_TITLE
    return f"{compact[:max_length].rstrip()}..."


def _tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    return re.findall(r"[\w\u4e00-\u9fff]+", lowered)


def _script_counts(text: str) -> tuple[int, int]:
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in text if char.isascii() and char.isalpha())
    return cjk_count, latin_count


def _is_cjk_dominant_text(text: str) -> bool:
    cjk_count, latin_count = _script_counts(text or "")
    return cjk_count > 0 and cjk_count >= max(latin_count * 2, 8)


def _is_table_or_figure_query(query: str) -> bool:
    compact = query or ""
    return bool(
        re.search(r"\b(?:table|tab\.?|figure|fig\.?)\s*\d+", compact, re.IGNORECASE)
        or re.search(r"(表|图)\s*\d+", compact)
    )


def _extract_exact_match_terms(query: str) -> list[str]:
    compact = re.sub(r"\s+", " ", (query or "").strip())
    if not compact:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    patterns = (
        (r"\b(?:table|tab\.?|figure|fig\.?)\s*\d+[a-z]?\b", re.IGNORECASE),
        (r"\b[A-Z]{2,}(?:[-/][A-Z0-9]+)*(?:\d+[A-Z0-9/-]*)?\b", 0),
        (r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)+\b", 0),
        (r"\b[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+\b", 0),
        (r"\b[A-Za-z]*\d+[A-Za-z0-9-]*\b", 0),
    )
    for pattern, flags in patterns:
        for match in re.finditer(pattern, query, flags):
            term = re.sub(r"\s+", " ", match.group(0).strip(" \t\r\n.,;:()[]{}"))
            lowered = term.lower()
            if len(lowered) < 2 or lowered in seen:
                continue
            candidates.append(term)
            seen.add(lowered)

    if re.search(r"[^\x00-\x7F]", query):
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", query):
            lowered = token.lower()
            if lowered in seen:
                continue
            candidates.append(token)
            seen.add(lowered)

    return sorted(candidates, key=len, reverse=True)


_SHORT_QUERY_DECONTEXT_PATTERN = re.compile(
    r"(这里|这个|那个|上面|上一问|上一轮|前面|刚才|这篇|那篇|这里的|那个重新校准后的|这里说的)",
    re.IGNORECASE,
)
_COMPLEX_QUERY_PATTERN = re.compile(
    r"(为什么|如何|概括|总结|比较|对比|分别|以及|并且|同时|结合|影响|原因|差别|区别|优缺点)",
    re.IGNORECASE,
)
_BOOLEAN_OPERATOR_PATTERN = re.compile(r"\b(?:OR|AND|NOT)\b", re.IGNORECASE)
_NUMERIC_UNIT_PATTERN = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*(?:hz|khz|mhz|ma|mg/d|mg|%|week|weeks|month|months|year|years|分钟|小时|周|个月|例|岁|分)",
    re.IGNORECASE,
)


def _is_short_query_text(query: str) -> bool:
    compact = _normalize_query_text(query)
    if not compact:
        return False
    token_count = len(_tokenize(compact))
    return len(compact) <= 96 or token_count <= 10


def _has_strong_exact_term_signal(query: str, exact_terms: list[str]) -> bool:
    if _is_table_or_figure_query(query):
        return True
    if _NUMERIC_UNIT_PATTERN.search(query or ""):
        return True
    for term in exact_terms:
        normalized = _normalize_query_text(term)
        if not normalized:
            continue
        if re.search(r"[A-Z]", normalized) or re.search(r"\d", normalized) or any(char in normalized for char in "-/()"):
            return True
    return False


def _looks_like_decontextualization_short_query(query: str) -> bool:
    compact = _normalize_query_text(query)
    if not compact or not _is_short_query_text(compact):
        return False
    return bool(_SHORT_QUERY_DECONTEXT_PATTERN.search(compact))


def _looks_like_complex_multi_query(query: str) -> bool:
    compact = _normalize_query_text(query)
    if not compact:
        return False
    token_count = len(_tokenize(compact))
    return token_count >= 12 or bool(_COMPLEX_QUERY_PATTERN.search(compact))


def _classify_query_type(query: str, exact_terms: list[str]) -> str:
    if _looks_like_decontextualization_short_query(query):
        return "decontextualization_short"
    if _is_short_query_text(query) and _has_strong_exact_term_signal(query, exact_terms):
        return "exact_heavy_short"
    if _looks_like_complex_multi_query(query):
        return "complex_multi_query"
    return "general"


def _sanitize_exact_terms(values: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_query_text(str(value).strip(" \t\r\n\"'`[]{}"))
        if not normalized:
            continue
        if _BOOLEAN_OPERATOR_PATTERN.fullmatch(normalized):
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        cleaned.append(normalized)
        seen.add(lowered)
    return cleaned


def _sanitize_sparse_query_text(text: str) -> str:
    normalized = _normalize_query_text(text)
    if not normalized:
        return ""
    normalized = re.sub(r"[\"'`]+", " ", normalized)
    normalized = _BOOLEAN_OPERATOR_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"[(){}\[\]]+", " ", normalized)
    return _normalize_query_text(normalized)


def _backfill_exact_terms_into_query(
    query: str,
    *,
    exact_terms: list[str],
    query_type: str,
) -> tuple[str, list[str]]:
    normalized_query = _sanitize_sparse_query_text(query)
    if not normalized_query or query_type not in {"exact_heavy_short", "decontextualization_short"}:
        return normalized_query, []
    protected_terms = [
        term
        for term in exact_terms
        if re.search(r"[A-Za-z]", term) or re.search(r"\d", term) or any(char in term for char in "-/()")
    ]
    missing_terms = [term for term in protected_terms if term.lower() not in normalized_query.lower()]
    if not missing_terms:
        return normalized_query, []
    backfilled_terms = missing_terms[:4]
    return _normalize_query_text(f"{normalized_query} {' '.join(backfilled_terms)}"), backfilled_terms


def _is_exact_match_heavy_query(query: str, exact_terms: list[str] | None = None) -> bool:
    terms = exact_terms if exact_terms is not None else _extract_exact_match_terms(query)
    query_type = _classify_query_type(query, terms)
    if query_type == "exact_heavy_short":
        return True
    if _is_table_or_figure_query(query):
        return True
    if _is_cjk_dominant_text(query):
        return False
    return len(terms) >= 2 or _has_strong_exact_term_signal(query, terms)


def _chunk_text_payload(chunk: PaperChunk, *, include_supporting_context: bool = False) -> str:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    header = str(metadata.get("context_header") or "").strip()
    body_text = str(metadata.get("body_text") or chunk.content or "").strip()
    supporting_context = str(metadata.get("supporting_context") or "").strip()
    parent_context = str(metadata.get("parent_text") or "").strip()
    parts = [part for part in (header, body_text) if part]
    if include_supporting_context and supporting_context:
        parts.append(f"上下文补充: {supporting_context}")
    if include_supporting_context and parent_context:
        parts.append(f"章节上下文: {parent_context}")
    return "\n\n".join(parts) if parts else str(chunk.content or "")


def _chunk_metadata(chunk: PaperChunk) -> dict[str, Any]:
    return chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}


def _chunk_role(chunk: PaperChunk) -> str:
    metadata = _chunk_metadata(chunk)
    return str(metadata.get("chunk_role") or getattr(chunk, "chunk_role", "child") or "child")


def _chunk_granularity(chunk: PaperChunk) -> str:
    metadata = _chunk_metadata(chunk)
    return str(metadata.get("granularity") or _chunk_role(chunk) or "child")


def _chunk_source_kind(chunk: PaperChunk) -> str | None:
    metadata = _chunk_metadata(chunk)
    source_kind = str(metadata.get("source_kind") or "").strip()
    return source_kind or None


def _is_summary_chunk_role(role: str | None) -> bool:
    return str(role or "").strip() in SUMMARY_CHUNK_ROLES


def _searchable_chunk_roles() -> tuple[str, ...]:
    configured = [
        item.strip()
        for item in str(settings.rag_searchable_chunk_roles or "").split(",")
        if item.strip()
    ]
    sanitized: list[str] = []
    for role in configured or list(DEFAULT_SEARCHABLE_CHUNK_ROLES):
        normalized = str(role).strip()
        if normalized not in {"child", *SUMMARY_CHUNK_ROLES}:
            continue
        if normalized not in sanitized:
            sanitized.append(normalized)
    if "child" not in sanitized:
        sanitized.insert(0, "child")
    return tuple(sanitized)


def _chunk_anchor_ids(chunk: PaperChunk, *, limit: int | None = None) -> list[int]:
    metadata = _chunk_metadata(chunk)
    raw_ids = metadata.get("anchor_chunk_ids")
    if not isinstance(raw_ids, list):
        resolved = metadata.get("resolved_chunk_id")
        return [int(resolved)] if str(resolved or "").isdigit() else []
    anchors = [int(item) for item in raw_ids if str(item or "").isdigit()]
    if limit is not None:
        return anchors[: max(limit, 0)]
    return anchors


def _clip_snippet(text: str, *, max_length: int) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length].rstrip()}..."


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


_TABLE_FOCUS_TERM_PATTERN = re.compile(
    r"(研究组|对照组|实验组|干预前|干预后|治疗前|治疗后|组别|stimulation|sham|baseline|post[- ]?(?:treatment|intervention)|pre[- ]?(?:treatment|intervention)|theta|beta|smr|frequency|score|p-?value|频率|评分|显著性)",
    re.IGNORECASE,
)
_TABLE_VALUE_PATTERN = re.compile(
    r"(?:[αβγθδμσχ]|SMR|P值|p-?value|频率|评分|均数|标准差|mean|sd|x±s|χ2|t检验)",
    re.IGNORECASE,
)


def _extract_table_focus_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in _extract_exact_match_terms(query):
        lowered = term.lower()
        if lowered in seen:
            continue
        terms.append(term)
        seen.add(lowered)
    for match in _TABLE_FOCUS_TERM_PATTERN.finditer(query or ""):
        term = match.group(0).strip()
        lowered = term.lower()
        if lowered in seen:
            continue
        terms.append(term)
        seen.add(lowered)
    return sorted(terms, key=len, reverse=True)


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _unique_strings(values: Iterable[str]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_query_text(str(value))
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        items.append(normalized)
        seen.add(lowered)
    return items


def _detect_query_language(text: str) -> str:
    compact = _normalize_query_text(text)
    if not compact:
        return "unknown"
    cjk_count, latin_count = _script_counts(compact)
    if cjk_count > latin_count:
        return "zh"
    if cjk_count and latin_count:
        return "mixed"
    if cjk_count:
        return "zh"
    return "en"


def _llm_available_for_query_rewrite() -> bool:
    if not settings.rag_crosslingual_query_rewrite_enabled:
        return False
    llm_config = get_chat_llm_configuration()
    return bool(llm_config.get("configured"))


def _parse_json_object(text: str) -> dict[str, Any]:
    payload = (text or "").strip()
    if not payload:
        raise RuntimeError("query rewrite 返回为空")
    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 3:
            payload = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise RuntimeError("query rewrite 未返回合法 JSON") from None
        data = json.loads(payload[start : end + 1])
    if not isinstance(data, dict):
        raise RuntimeError("query rewrite JSON 必须是对象")
    return data


def _default_generation_instruction(language: str) -> str:
    if language in {"zh", "mixed"}:
        return "请用中文回答，保留关键英文术语原文，并引用英文证据。"
    return "请使用与用户提问一致的语言回答，并在必要时保留英文术语原文。"


def _summarize_exception_message(exc: Exception, *, limit: int = 160) -> str:
    message = f"{exc.__class__.__name__}: {exc}"
    compact = re.sub(r"\s+", " ", message).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _extract_bilingual_structural_terms(query: str) -> list[str]:
    terms: list[str] = []
    if re.search(r"(显著性|p值|P值)", query or ""):
        terms.append("p-value")
    if re.search(r"(基线)", query or ""):
        terms.append("baseline")
    if re.search(r"(量表)", query or ""):
        terms.append("scale")
    for match in re.finditer(r"表\s*(\d+)", query or ""):
        terms.append(f"Table {match.group(1)}")
    for match in re.finditer(r"图\s*(\d+)", query or ""):
        terms.append(f"Figure {match.group(1)}")
    return _unique_strings(terms)


def _extract_template_rule_terms(query: str) -> list[str]:
    terms: list[str] = []
    for pattern, mapped_terms in _QUERY_TEMPLATE_RULES:
        if pattern.search(query or ""):
            terms.extend(mapped_terms)
    return _unique_strings(terms)


def _build_heuristic_retrieval_query_en(query: str, exact_terms: list[str]) -> tuple[str, str | None]:
    english_terms = [term for term in exact_terms if re.search(r"[A-Za-z]", term)]
    template_terms = _extract_template_rule_terms(query)
    structural_terms = _extract_bilingual_structural_terms(query)
    combined_terms = _unique_strings([*english_terms, *template_terms, *structural_terms])
    if not combined_terms:
        return "", None
    fallback_source = "template_rules" if template_terms or structural_terms else "exact_terms"
    return _sanitize_sparse_query_text(" ".join(combined_terms[:8])), fallback_source


def _llm_available_for_grounding() -> bool:
    return bool(settings.glm_api_key.strip() or settings.openai_api_key.strip())


def _query_variants_for_plan(
    *,
    original_query: str,
    detected_language: str,
    query_type: str,
    exact_guardrail_query_en: str,
    retrieval_query_en: str,
    subqueries_en: list[str],
) -> list[RetrievalQueryVariant]:
    variants: list[RetrievalQueryVariant] = []
    if detected_language in {"zh", "mixed"}:
        variants.append(
            RetrievalQueryVariant(
                name="zh_original",
                query=original_query,
                language="zh",
                use_sparse=False,
                use_dense=True,
                role="original",
            )
        )
        if exact_guardrail_query_en:
            variants.append(
                RetrievalQueryVariant(
                    name="en_exact_terms",
                    query=exact_guardrail_query_en,
                    language="en",
                    use_sparse=True,
                    use_dense=False,
                    role="guardrail",
                )
            )
        if retrieval_query_en:
            variants.append(
                RetrievalQueryVariant(
                    name="en_rewrite",
                    query=retrieval_query_en,
                    language="en",
                    use_sparse=True,
                    use_dense=True,
                    role="rewrite",
                )
            )
        allow_subqueries = query_type == "complex_multi_query"
        for index, subquery in enumerate(subqueries_en if allow_subqueries else [], start=1):
            variants.append(
                RetrievalQueryVariant(
                    name=f"en_subquery_{index}",
                    query=subquery,
                    language="en",
                    use_sparse=True,
                    use_dense=True,
                    role="subquery",
                )
            )
        return variants
    variants.append(
        RetrievalQueryVariant(
            name="default",
            query=original_query,
            language="en",
            use_sparse=True,
            use_dense=True,
            role="original",
        )
    )
    return variants


def _serialize_query_plan(plan: RetrievalQueryPlan) -> dict[str, Any]:
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


def _build_crosslingual_query_plan(query: str) -> RetrievalQueryPlan:
    original_query = _normalize_query_text(query)
    detected_language = _detect_query_language(original_query)
    retrieval_query_en = ""
    exact_guardrail_query_en = ""
    subqueries_en: list[str] = []
    exact_terms = _sanitize_exact_terms(_extract_exact_match_terms(original_query))
    query_type = _classify_query_type(original_query, exact_terms)
    generation_instruction = _default_generation_instruction(detected_language)
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
        exact_guardrail_query_en, fallback_source = _build_heuristic_retrieval_query_en(original_query, exact_terms)
        llm_config = get_chat_llm_configuration()
        rewrite_provider = str(llm_config.get("provider") or "") or None
        llm_rewrite_status = "not_attempted_config_unavailable"
        if _llm_available_for_query_rewrite():
            llm_attempted = True
            rewrite_started_at = time.perf_counter()
            logger.info(
                "RAG query rewrite started: language=%s provider=%s query=%s",
                detected_language,
                rewrite_provider or "-",
                _preview_log_text(original_query),
            )
            try:
                content, rewrite_model = chat_with_messages(
                    [
                        {"role": "system", "content": QUERY_PLAN_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": QUERY_PLAN_USER_TEMPLATE.format(
                                query=original_query,
                                max_subqueries=settings.rag_crosslingual_max_subqueries,
                            ),
                        },
                    ],
                    temperature=0.0,
                )
            except Exception as exc:
                llm_rewrite_status = "llm_failed_http_or_provider"
                rewrite_error = _summarize_exception_message(exc)
                logger.warning(
                    "RAG query rewrite request failed: language=%s provider=%s error=%s elapsed=%.2fs",
                    detected_language,
                    rewrite_provider or "-",
                    rewrite_error,
                    time.perf_counter() - rewrite_started_at,
                )
            else:
                try:
                    payload = _parse_json_object(content)
                except Exception as exc:
                    llm_rewrite_status = "llm_failed_parse"
                    rewrite_error = _summarize_exception_message(exc)
                    logger.warning(
                        "RAG query rewrite parse failed: language=%s provider=%s error=%s elapsed=%.2fs",
                        detected_language,
                        rewrite_provider or "-",
                        rewrite_error,
                        time.perf_counter() - rewrite_started_at,
                    )
                else:
                    retrieval_query_en = _sanitize_sparse_query_text(str(payload.get("retrieval_query_en") or ""))
                    subqueries_en = [_sanitize_sparse_query_text(item) for item in _unique_strings(payload.get("subqueries_en") or [])][
                        : settings.rag_crosslingual_max_subqueries
                    ]
                    exact_terms = _sanitize_exact_terms(
                        [
                            *exact_terms,
                            *(payload.get("exact_terms") or []),
                            *_extract_exact_match_terms(retrieval_query_en),
                        ]
                    )
                    query_type = _classify_query_type(original_query, exact_terms)
                    if query_type != "complex_multi_query":
                        subqueries_en = []
                    retrieval_query_en, rewrite_backfilled_terms = _backfill_exact_terms_into_query(
                        retrieval_query_en,
                        exact_terms=exact_terms,
                        query_type=query_type,
                    )
                    exact_guardrail_query_en, fallback_source = _build_heuristic_retrieval_query_en(original_query, exact_terms)
                    generation_instruction = _normalize_query_text(
                        str(payload.get("generation_instruction") or generation_instruction)
                    ) or generation_instruction
                    rewrite_status = "llm_rewritten"
                    llm_rewrite_status = "llm_rewritten"
                    used_llm = True
                    rewrite_error = None
                    logger.info(
                        "RAG query rewrite finished: language=%s provider=%s status=%s retrieval_query_en=%s subqueries=%s elapsed=%.2fs",
                        detected_language,
                        rewrite_provider or "-",
                        rewrite_status,
                        _preview_log_text(retrieval_query_en),
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
    variants = _query_variants_for_plan(
        original_query=original_query,
        detected_language=detected_language,
        query_type=query_type,
        exact_guardrail_query_en=guardrail_query_en,
        retrieval_query_en=retrieval_query_en,
        subqueries_en=subqueries_en,
    )
    return RetrievalQueryPlan(
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


def _looks_like_table_body_text(text: str) -> bool:
    compact = str(text or "").strip()
    if not compact:
        return False
    number_count = len(re.findall(r"[-+]?\d+(?:\.\d+)?", compact))
    digit_count = sum(char.isdigit() for char in compact)
    body_term_hits = len(_TABLE_FOCUS_TERM_PATTERN.findall(compact))
    metric_hits = len(_TABLE_VALUE_PATTERN.findall(compact))
    return bool(
        (number_count >= 3 and body_term_hits >= 1)
        or (number_count >= 4 and metric_hits >= 1 and digit_count >= 8)
        or (digit_count >= 10 and "±" in compact)
    )


def _block_is_table_body_candidate(block: dict[str, Any]) -> bool:
    block_type = str(block.get("block_type") or "")
    if block_type == "table_like":
        return True
    return _looks_like_table_body_text(str(block.get("text") or ""))


def _chunk_has_table_body(chunk: PaperChunk) -> bool:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    block_types = {str(item).lower() for item in metadata.get("block_types", [])} if isinstance(metadata, dict) else set()
    if "table_like" in block_types:
        return True
    return _looks_like_table_body_text(str(metadata.get("body_text") or chunk.content or ""))


def _chunk_is_caption_only(chunk: PaperChunk) -> bool:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    block_types = {str(item).lower() for item in metadata.get("block_types", [])} if isinstance(metadata, dict) else set()
    if not block_types or block_types - {"table_caption"}:
        return False
    return not _looks_like_table_body_text(str(metadata.get("body_text") or chunk.content or ""))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _lexical_score(query_tokens: set[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = set(_tokenize(content))
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def _is_postgres_session(db: Session) -> bool:
    bind = db.get_bind()
    return bool(bind and bind.dialect.name == "postgresql")


def _normalize_embedding(value: Any) -> list[float] | None:
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


def _search_chunks_legacy(db: Session, *, query: str, top_k: int, organization_id: int) -> list[RetrievalResult]:
    searchable_roles = _searchable_chunk_roles()
    rows = db.execute(
        select(PaperChunk, Paper)
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(
            Paper.deleted_at.is_(None),
            Paper.organization_id == organization_id,
            PaperChunk.chunk_role.in_(searchable_roles),
        )
        .order_by(PaperChunk.paper_id.asc(), PaperChunk.chunk_index.asc())
    ).all()
    if not rows:
        return []

    query_tokens = set(_tokenize(query))
    query_embedding: list[float] | None = None
    if (settings.glm_api_key.strip() or settings.openai_api_key.strip()) and any(chunk.embedding for chunk, _ in rows):
        try:
            query_embedding = embed_texts([query])[0]
        except Exception:
            query_embedding = None

    scored: list[RetrievalResult] = []
    for chunk, paper in rows:
        lexical_score = _lexical_score(query_tokens, chunk.content)
        chunk_embedding = _normalize_embedding(chunk.embedding)
        embedding_score = (
            _cosine_similarity(query_embedding, chunk_embedding)
            if query_embedding is not None and chunk_embedding is not None
            else 0.0
        )
        score = embedding_score if embedding_score > 0 else lexical_score
        if embedding_score > 0 and lexical_score > 0:
            score = embedding_score * 0.8 + lexical_score * 0.2
        if score < MIN_RELEVANCE_SCORE:
            continue
        scored.append(RetrievalResult(chunk=chunk, paper=paper, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def _search_sparse_chunks_postgres(
    db: Session,
    *,
    query: str,
    limit: int,
    organization_id: int,
    exact_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    searchable_roles = _searchable_chunk_roles()
    role_sql = ", ".join(f"'{role}'" for role in searchable_roles)
    search_config = re.sub(r"[^a-zA-Z0-9_]+", "", settings.rag_text_search_config) or "simple"
    body_text_expr = "LOWER(COALESCE(pc.metadata_json->>'body_text', pc.content))"
    exact_terms = exact_terms if exact_terms is not None else _extract_exact_match_terms(query)
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


def _search_dense_chunks_postgres(
    db: Session,
    *,
    query_embedding: list[float] | None,
    limit: int,
    organization_id: int,
) -> list[dict[str, Any]]:
    if query_embedding is None:
        return []
    searchable_roles = _searchable_chunk_roles()
    distance = PaperChunk.embedding.cosine_distance(query_embedding)
    rows = db.execute(
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
    ).all()
    return [
        {
            "chunk_id": int(row.chunk_id),
            "paper_id": int(row.paper_id),
            "score": float(row.score or 0.0),
        }
        for row in rows
    ]


def _fuse_ranked_candidate_sources(
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
            candidate["score"] += 1.0 / (settings.rag_rrf_k + rank)
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


def _fuse_ranked_candidates(
    sparse_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return _fuse_ranked_candidate_sources({"sparse": sparse_hits, "dense": dense_hits}, limit=limit)


def _boost_exact_match_candidates(
    query: str,
    *,
    candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> list[dict[str, Any]]:
    exact_terms = _extract_exact_match_terms(query)
    prefers_table = _is_table_or_figure_query(query)
    table_focus_terms = _extract_table_focus_terms(query) if prefers_table else []
    if not exact_terms and not table_focus_terms:
        return candidates

    boosted: list[dict[str, Any]] = []
    allow_general_exact_boost = not _is_cjk_dominant_text(query)
    for candidate in candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            boosted.append(dict(candidate))
            continue
        chunk, _ = record
        metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
        haystack = _chunk_text_payload(chunk, include_supporting_context=True).lower()
        matched_terms = [term for term in exact_terms if term.lower() in haystack]
        matched_focus_terms = [term for term in table_focus_terms if term.lower() in haystack]
        block_types = {str(item).lower() for item in metadata.get("block_types", [])} if isinstance(metadata, dict) else set()
        body_text = str(metadata.get("body_text") or chunk.content or "")
        has_table_body = _chunk_has_table_body(chunk)
        caption_only = _chunk_is_caption_only(chunk)
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


def _load_chunk_record_map(
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


def _should_expand_multi_span_candidates(query: str, *, query_plan: RetrievalQueryPlan) -> bool:
    if not query.strip() or _is_table_or_figure_query(query):
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


def _merge_ranked_candidates(
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


def _expand_retrieval_candidates(
    db: Session,
    *,
    query: str,
    candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
    organization_id: int,
    query_plan: RetrievalQueryPlan,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": False,
        "expanded_candidate_count": 0,
        "expansion_added_ids": [],
    }
    if not candidates:
        return candidates, stats

    enable_multi_span_expansion = _should_expand_multi_span_candidates(query, query_plan=query_plan)
    anchor_candidates: list[tuple[dict[str, Any], PaperChunk]] = []
    parent_ids: set[int] = set()
    summary_candidates: list[tuple[dict[str, Any], PaperChunk, list[int]]] = []
    summary_anchor_ids: set[int] = set()
    for candidate in candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            continue
        chunk, _paper = record
        chunk_role = _chunk_role(chunk)
        if chunk_role == "child" and enable_multi_span_expansion:
            anchor_candidates.append((candidate, chunk))
            if chunk.parent_chunk_id is not None:
                parent_ids.add(int(chunk.parent_chunk_id))
            continue
        if not _is_summary_chunk_role(chunk_role):
            continue
        anchor_ids = _chunk_anchor_ids(chunk, limit=settings.rag_summary_anchor_limit)
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
        for anchor_rank, anchor_chunk_id in enumerate(anchor_chunk_ids[: settings.rag_summary_anchor_limit], start=1):
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

    merged = _merge_ranked_candidates(candidates, additional_candidates, limit=expanded_limit)
    added_ids = [int(item["chunk_id"]) for item in merged if int(item["chunk_id"]) not in {int(candidate["chunk_id"]) for candidate in candidates}]
    record_map.update(related_record_map)
    stats["expanded_candidate_count"] = len(added_ids)
    stats["expansion_added_ids"] = added_ids
    return merged, stats


def _serialize_ranked_trace_candidates(
    candidates: list[dict[str, Any]],
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> list[dict]:
    serialized: list[dict] = []
    for item in candidates:
        record = record_map.get(int(item["chunk_id"]))
        if record is None:
            continue
        chunk, paper = record
        snippet = _clip_snippet(_chunk_text_payload(chunk), max_length=180)
        row = {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "paper_id": paper.id,
            "paper_title": paper.title,
            "score": round(float(item.get("score") or 0.0), 4),
            "page_from": chunk.page_from,
            "page_to": chunk.page_to,
            "snippet": snippet,
            "chunk_role": _chunk_role(chunk),
            "granularity": _chunk_granularity(chunk),
        }
        source_kind = _chunk_source_kind(chunk)
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


def _serialize_variant_traces(
    variant_traces: dict[str, dict[str, Any]],
    *,
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> dict[str, dict[str, Any]]:
    serialized: dict[str, dict[str, Any]] = {}
    for name, variant in variant_traces.items():
        serialized[name] = {
            "query": variant.get("query"),
            "language": variant.get("language"),
            "role": variant.get("role"),
            "use_sparse": bool(variant.get("use_sparse")),
            "use_dense": bool(variant.get("use_dense")),
            "sparse_candidates": _serialize_ranked_trace_candidates(variant.get("sparse_hits", []), record_map),
            "dense_candidates": _serialize_ranked_trace_candidates(variant.get("dense_hits", []), record_map),
            "fused_candidates": _serialize_ranked_trace_candidates(variant.get("fused_hits", []), record_map),
        }
    return serialized


def _build_retrieval_results(
    candidates: list[dict[str, Any]],
    *,
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    seen_chunk_ids: set[int] = set()
    for item in candidates:
        resolved_chunk_id = _resolve_candidate_chunk_id(item, record_map=record_map)
        if resolved_chunk_id in seen_chunk_ids:
            continue
        record = record_map.get(resolved_chunk_id) or record_map.get(int(item["chunk_id"]))
        if record is None:
            continue
        chunk, paper = record
        seen_chunk_ids.add(int(chunk.id))
        results.append(RetrievalResult(chunk=chunk, paper=paper, score=float(item.get("score") or 0.0)))
    return results


def _results_to_ranked_hits(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": int(result.chunk.id),
            "paper_id": int(result.paper.id),
            "score": float(result.score),
        }
        for result in results
    ]


def _resolve_candidate_chunk_id(
    item: dict[str, Any],
    *,
    record_map: dict[int, tuple[PaperChunk, Paper]],
) -> int:
    chunk_id = int(item["chunk_id"])
    record = record_map.get(chunk_id)
    if record is None:
        return chunk_id
    chunk, _paper = record
    if not _is_summary_chunk_role(_chunk_role(chunk)):
        return chunk_id
    metadata = _chunk_metadata(chunk)
    resolved_chunk_id = metadata.get("resolved_chunk_id")
    if str(resolved_chunk_id or "").isdigit() and int(resolved_chunk_id) in record_map:
        return int(resolved_chunk_id)
    for anchor_chunk_id in _chunk_anchor_ids(chunk, limit=settings.rag_summary_anchor_limit):
        if int(anchor_chunk_id) in record_map:
            return int(anchor_chunk_id)
    return chunk_id


def _extract_query_numbers(text: str) -> tuple[str, ...]:
    return tuple(sorted({match.group(0) for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", text or "")}))


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _truncate_for_budget(text: str, *, max_chars: int) -> str:
    compact = _compact_text(text)
    if len(compact) <= max_chars:
        return compact
    if max_chars <= 3:
        return compact[:max_chars]
    return f"{compact[: max_chars - 3].rstrip()}..."


def _summarize_text_excerpt(text: str, *, max_sentences: int, max_chars: int) -> str:
    compact = _compact_text(text)
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
    return _truncate_for_budget(selected or compact, max_chars=max_chars)


def _serialize_structured_summary_for_chunk(summary: dict[str, Any] | None) -> str:
    if not isinstance(summary, dict):
        return ""
    parts: list[str] = []
    abstract_cn = _compact_text(str(summary.get("abstract_cn") or ""))
    if abstract_cn:
        parts.append(f"摘要：{abstract_cn}")
    research_question = _compact_text(str(summary.get("research_question") or ""))
    if research_question:
        parts.append(f"研究问题：{research_question}")
    method = _compact_text(str(summary.get("method") or ""))
    if method:
        parts.append(f"方法：{method}")
    findings = _compact_text(str(summary.get("findings") or ""))
    if findings:
        parts.append(f"发现：{findings}")
    limitations = _compact_text(str(summary.get("limitations") or ""))
    if limitations:
        parts.append(f"局限：{limitations}")
    key_points = [
        _compact_text(str(item))
        for item in (summary.get("key_points") if isinstance(summary.get("key_points"), list) else [])
        if _compact_text(str(item))
    ]
    if key_points:
        parts.append("要点：" + "；".join(key_points[:4]))
    return _truncate_for_budget("\n".join(parts), max_chars=settings.rag_paper_summary_max_chars) if parts else ""


def _build_fallback_paper_summary_text(
    section_summaries: list[str],
    *,
    normalized_blocks: list[dict[str, Any]],
) -> str:
    compact_summaries = [_compact_text(item) for item in section_summaries if _compact_text(item)]
    if compact_summaries:
        return _truncate_for_budget("\n".join(compact_summaries[:3]), max_chars=settings.rag_paper_summary_max_chars)
    fallback_text = "\n".join(
        _compact_text(str(block.get("text") or ""))
        for block in normalized_blocks[:6]
        if _compact_text(str(block.get("text") or ""))
    )
    return _truncate_for_budget(fallback_text, max_chars=settings.rag_paper_summary_max_chars) if fallback_text else ""


def _build_rerank_block_catalog(
    db: Session,
    *,
    paper_ids: Iterable[int],
) -> dict[int, dict[str, Any]]:
    catalogs: dict[int, dict[str, Any]] = {}
    for paper_id in sorted({int(item) for item in paper_ids if int(item)}):
        preanalysis = _build_preanalysis_from_document_structure(db, paper_id=paper_id) or {}
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


def _score_rerank_text(
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
    compact = _compact_text(text).lower()
    if not compact:
        return 0.0
    score = 0.0
    query_token_set = set(query_tokens)
    unit_token_set = set(_tokenize(compact))
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


def _neighbor_blocks_for_rerank(
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


def _table_evidence_units(block: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    table_text = str(block.get("text") or "").strip()
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if lines:
        units.append({"text": lines[0], "unit_type": "table_caption"})
    serialized_rows = _serialize_table_rows(block.get("table_data_json"), max_rows=16)
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


def _candidate_rerank_units(
    chunk: PaperChunk,
    *,
    catalog: dict[str, Any],
    query: str,
) -> list[dict[str, Any]]:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    source_blocks = metadata.get("source_block_ids") if isinstance(metadata.get("source_block_ids"), list) else []
    source_tables = metadata.get("source_table_ids") if isinstance(metadata.get("source_table_ids"), list) else []
    source_pictures = metadata.get("source_picture_ids") if isinstance(metadata.get("source_picture_ids"), list) else []
    prefers_table = _is_table_or_figure_query(query)
    query_tokens = _tokenize(query)
    exact_terms = _extract_exact_match_terms(query)
    query_numbers = _extract_query_numbers(query)
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
        for neighbor in _neighbor_blocks_for_rerank(
            block,
            catalog=catalog,
            neighbor_count=settings.rerank_neighbor_blocks,
        ):
            source_candidates.append((neighbor, False))
    for block, is_primary in source_candidates:
        block_type = str(block.get("block_type") or "paragraph")
        unit_specs = (
            _table_evidence_units(block)
            if block_type == "table_like"
            else [{"text": str(block.get("text") or "").strip(), "unit_type": "primary_block" if is_primary else "neighbor_context"}]
        )
        for unit_spec in unit_specs:
            text = _compact_text(unit_spec.get("text") or "")
            if not text or text in seen_texts:
                continue
            unit_type = str(unit_spec.get("unit_type") or "primary_block")
            section_path = str(block.get("section_path") or "")
            score = _score_rerank_text(
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


def _build_rerank_context_for_chunk(
    query: str,
    *,
    chunk: PaperChunk,
    catalog: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    header = str(metadata.get("context_header") or "").strip()
    if not isinstance(catalog, dict):
        fallback = _truncate_for_budget(
            _chunk_text_payload(chunk, include_supporting_context=False),
            max_chars=settings.rerank_max_context_chars,
        )
        return fallback, {
            "rerank_context_chars": len(fallback),
            "rerank_evidence_blocks": 0,
            "rerank_context_truncated": len(fallback) < len(_compact_text(_chunk_text_payload(chunk, include_supporting_context=False))),
            "rerank_evidence_types": [],
        }

    units = _candidate_rerank_units(chunk, catalog=catalog, query=query)
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
    max_chars = max(int(settings.rerank_max_context_chars), 200)
    if header:
        pieces.append(header)
        total_chars = len(header)
    selected_count = 0
    truncated = False
    for unit in ordered_units:
        if selected_count >= max(int(settings.rerank_max_evidence_blocks), 1):
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
            formatted = _truncate_for_budget(formatted, max_chars=remaining)
            truncated = True
        pieces.append(formatted)
        total_chars += len(formatted) + (2 if len(pieces) > 1 else 0)
        used_types.append(str(unit.get("unit_type") or "primary_block"))
        selected_count += 1

    if len(pieces) <= 1:
        fallback_body = str(metadata.get("body_text") or chunk.content or "").strip()
        fallback = "\n\n".join(part for part in (header, _truncate_for_budget(fallback_body, max_chars=max_chars // 2)) if part)
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


def _apply_reranking(
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
    block_catalogs = _build_rerank_block_catalog(
        db,
        paper_ids=[paper.id for _, paper in record_map.values()],
    )
    for candidate in fused_candidates:
        record = record_map.get(int(candidate["chunk_id"]))
        if record is None:
            continue
        chunk, paper = record
        rerank_context, context_meta = _build_rerank_context_for_chunk(
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
    rerank_results = rerank_documents(query, documents, top_n=limit)
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


def _history_messages(records: list[ChatMessage], *, include_last_user: bool = False) -> list[dict[str, str]]:
    history = records if include_last_user else records[:-1]
    return [{"role": record.role, "content": record.content} for record in history[-MAX_HISTORY_MESSAGES:]]


def _build_retrieval_query(records: list[ChatMessage], prompt: str) -> str:
    previous_user_messages = [record.content for record in records[:-1] if record.role == "user"][-2:]
    context = " ".join(previous_user_messages + [prompt]).strip()
    return context or prompt


def _topic_stats(db: Session, topic_ids: list[int]) -> dict[int, tuple[int, datetime | None]]:
    if not topic_ids:
        return {}
    rows = db.execute(
        select(
            ChatMessage.topic_id,
            func.count(ChatMessage.id),
            func.max(ChatMessage.created_at),
        )
        .where(ChatMessage.topic_id.in_(topic_ids))
        .group_by(ChatMessage.topic_id)
    ).all()
    return {
        int(topic_id): (int(message_count), last_message_at)
        for topic_id, message_count, last_message_at in rows
    }


def _topic_summary(db: Session, topic: ChatTopic) -> TopicSummary:
    stats = _topic_stats(db, [topic.id]).get(topic.id, (0, None))
    return TopicSummary(
        topic=topic,
        message_count=stats[0],
        last_message_at=stats[1],
    )


def list_topics(db: Session, *, user_id: int) -> list[TopicSummary]:
    topics = db.scalars(
        select(ChatTopic)
        .where(ChatTopic.user_id == user_id)
        .order_by(ChatTopic.updated_at.desc(), ChatTopic.id.desc())
    ).all()
    stats = _topic_stats(db, [topic.id for topic in topics])
    return [
        TopicSummary(
            topic=topic,
            message_count=stats.get(topic.id, (0, None))[0],
            last_message_at=stats.get(topic.id, (0, None))[1],
        )
        for topic in topics
    ]


def create_topic(db: Session, *, user: User, title: str | None = None) -> TopicSummary:
    now = datetime.now(timezone.utc)
    topic = ChatTopic(
        user_id=user.id,
        title=_normalize_title(title),
        created_at=now,
        updated_at=now,
    )
    db.add(topic)
    db.commit()
    db.refresh(topic)
    return _topic_summary(db, topic)


def get_topic(db: Session, *, user_id: int, topic_id: int) -> ChatTopic | None:
    return db.scalar(
        select(ChatTopic).where(
            ChatTopic.id == topic_id,
            ChatTopic.user_id == user_id,
        )
    )


def list_topic_messages(db: Session, *, topic_id: int) -> list[ChatMessage]:
    return db.scalars(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    ).all()


def _build_context_header(
    *,
    paper_title: str | None,
    section_path: str | None,
    page_from: int | None,
    page_to: int | None,
) -> str:
    lines: list[str] = []
    if (paper_title or "").strip():
        lines.append(f"论文标题: {(paper_title or '').strip()}")
    if (section_path or "").strip():
        lines.append(f"章节: {(section_path or '').strip()}")
    if page_from is not None:
        if page_to is not None and page_to != page_from:
            lines.append(f"页码: {page_from}-{page_to}")
        else:
            lines.append(f"页码: {page_from}")
    return "\n".join(lines)


def _serialize_table_rows(data: object, *, max_rows: int = 8) -> str:
    if not isinstance(data, list):
        return ""
    lines: list[str] = []
    for row in data[:max_rows]:
        if not isinstance(row, dict):
            continue
        cells = [f"{str(key).strip()}: {str(value).strip()}" for key, value in row.items() if str(value).strip()]
        if cells:
            lines.append("; ".join(cells))
    return "\n".join(lines)


def _compose_table_text(*, caption: str | None, markdown: str | None, data: object) -> str:
    return _compose_table_text_with_mode(caption=caption, markdown=markdown, data=data)[0]


def _compose_table_text_with_mode(*, caption: str | None, markdown: str | None, data: object) -> tuple[str, str]:
    parts = [str(caption or "").strip()]
    serialized_rows = _serialize_table_rows(data)
    if serialized_rows:
        parts.append(serialized_rows)
        mode = "rows"
    elif str(markdown or "").strip():
        parts.append(str(markdown or "").strip())
        mode = "markdown"
    else:
        mode = "caption_only"
    return "\n".join(part for part in parts if part), mode


def _structure_sort_key(*, reading_order: int | None, page_number: int | None, fallback_group: int, fallback_index: int) -> tuple[int, int, int, int]:
    if reading_order is not None:
        return (0, int(reading_order), fallback_group, fallback_index)
    return (1, int(page_number or 0), fallback_group, fallback_index)


def _window_text(text: str, *, chunk_size: int, overlap: int) -> list[dict[str, int | str]]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    step = max(chunk_size - overlap, 1)
    windows: list[dict[str, int | str]] = []
    cursor = 0
    while cursor < len(text):
        window = text[cursor : cursor + chunk_size]
        content = window.strip()
        if content:
            trimmed_offset = window.find(content)
            windows.append(
                {
                    "content": content,
                    "char_start": cursor + max(trimmed_offset, 0),
                    "char_end": cursor + max(trimmed_offset, 0) + len(content),
                }
            )
        if cursor + chunk_size >= len(text):
            break
        cursor += step
    return windows


def _select_supporting_context(
    all_blocks: list[dict[str, Any]],
    selected_blocks: list[dict[str, Any]],
    *,
    max_chars: int = 420,
) -> list[dict[str, Any]]:
    if not all_blocks or not selected_blocks:
        return []

    selected_indexes = {int(block.get("block_index") or -1) for block in selected_blocks}
    anchor_indexes = sorted(index for index in selected_indexes if index >= 0)
    if not anchor_indexes:
        return []

    section_id = str(selected_blocks[0].get("section_id") or "")
    left = anchor_indexes[0] - 1
    right = anchor_indexes[-1] + 1
    supporting: list[dict[str, Any]] = []
    supporting_chars = 0

    while supporting_chars < max_chars and (left >= 0 or right < len(all_blocks)):
        progressed = False
        if left >= 0:
            candidate = all_blocks[left]
            if str(candidate.get("section_id") or "") == section_id and int(candidate.get("block_index") or -1) not in selected_indexes:
                supporting.insert(0, candidate)
                supporting_chars += len(str(candidate.get("text") or ""))
                progressed = True
            left -= 1
        if supporting_chars >= max_chars:
            break
        if right < len(all_blocks):
            candidate = all_blocks[right]
            if str(candidate.get("section_id") or "") == section_id and int(candidate.get("block_index") or -1) not in selected_indexes:
                supporting.append(candidate)
                supporting_chars += len(str(candidate.get("text") or ""))
                progressed = True
            right += 1
        if not progressed and left < 0 and right >= len(all_blocks):
            break
    return supporting


def _make_chunk_record(
    *,
    chunk_index: int,
    body_text: str,
    block_rows: list[dict[str, Any]],
    paper_title: str | None,
    supporting_context_rows: list[dict[str, Any]] | None = None,
    chunk_role: str = "child",
    granularity: str | None = None,
    source_kind: str | None = None,
    parent_key: str | None = None,
    parent_text: str | None = None,
) -> dict[str, Any]:
    page_numbers = sorted(
        {
            int(block.get("page_number"))
            for block in block_rows
            if isinstance(block.get("page_number"), int) or str(block.get("page_number") or "").isdigit()
        }
    )
    page_from = page_numbers[0] if page_numbers else None
    page_to = page_numbers[-1] if page_numbers else None
    section_path = next((str(block.get("section_path") or "").strip() for block in block_rows if str(block.get("section_path") or "").strip()), "")
    section_title = next((str(block.get("section_title") or "").strip() for block in block_rows if str(block.get("section_title") or "").strip()), "")
    heading_level = next(
        (
            int(block.get("heading_level"))
            for block in block_rows
            if isinstance(block.get("heading_level"), int) or str(block.get("heading_level") or "").isdigit()
        ),
        None,
    )
    context_header = _build_context_header(
        paper_title=paper_title,
        section_path=section_path or section_title,
        page_from=page_from,
        page_to=page_to,
    )
    supporting_context = "\n\n".join(
        str(block.get("text") or "").strip()
        for block in (supporting_context_rows or [])
        if str(block.get("text") or "").strip()
    )
    embedding_input = body_text
    content = body_text
    if context_header:
        embedding_input = f"{context_header}\n\n{body_text}"
        content = f"{context_header}\n\n{body_text}"
    return {
        "chunk_index": chunk_index,
        "content": content,
        "embedding_input": embedding_input,
        "token_count": len(_tokenize(body_text)),
        "page_from": page_from,
        "page_to": page_to,
        "chunk_role": chunk_role,
        "parent_key": parent_key,
        "metadata_json": {
            "char_start": min((int(block.get("char_start") or 0) for block in block_rows), default=0),
            "char_end": max((int(block.get("char_end") or 0) for block in block_rows), default=0),
            "context_header": context_header or None,
            "section_id": next((block.get("section_id") for block in block_rows if block.get("section_id")), None),
            "section_title": section_title or None,
            "section_path": section_path or None,
            "heading_level": heading_level,
            "block_types": sorted({str(block.get("block_type") or "paragraph") for block in block_rows}),
            "block_count": len(block_rows),
            "chunk_role": chunk_role,
            "granularity": granularity or chunk_role,
            "source_kind": source_kind or None,
            "body_text": body_text,
            "supporting_context": supporting_context or None,
            "parent_text": parent_text or None,
            "resolved_chunk_id": None,
            "anchor_chunk_ids": [],
            "reading_order_start": min(
                (int(block.get("reading_order")) for block in block_rows if isinstance(block.get("reading_order"), int) or str(block.get("reading_order") or "").isdigit()),
                default=None,
            ),
            "reading_order_end": max(
                (int(block.get("reading_order")) for block in block_rows if isinstance(block.get("reading_order"), int) or str(block.get("reading_order") or "").isdigit()),
                default=None,
            ),
            "bbox": next((block.get("bbox") for block in block_rows if isinstance(block.get("bbox"), dict)), None),
            "provenance": next((block.get("provenance") for block in block_rows if isinstance(block.get("provenance"), dict)), None),
            "source_block_ids": sorted({int(block["source_block_id"]) for block in block_rows if str(block.get("source_block_id") or "").isdigit()}),
            "source_table_ids": sorted({int(block["source_table_id"]) for block in block_rows if str(block.get("source_table_id") or "").isdigit()}),
            "source_picture_ids": sorted({int(block["source_picture_id"]) for block in block_rows if str(block.get("source_picture_id") or "").isdigit()}),
            "docling_labels": sorted({str(block.get("docling_label")) for block in block_rows if str(block.get("docling_label") or "").strip()}),
            "table_markdown": next((block.get("table_markdown") for block in block_rows if block.get("table_markdown")), None),
            "table_data_json": next((block.get("table_data_json") for block in block_rows if block.get("table_data_json") is not None), None),
            "table_text_mode": next((str(block.get("table_text_mode")) for block in block_rows if str(block.get("table_text_mode") or "").strip()), None),
            "picture_description": next((block.get("picture_description") for block in block_rows if block.get("picture_description")), None),
        },
    }


def _split_text(
    *,
    preanalysis: dict[str, Any] | None = None,
    paper_title: str | None = None,
    structured_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chunk_size = max(settings.rag_chunk_size, 200)
    overlap = max(min(settings.rag_chunk_overlap, chunk_size - 1), 0)
    blocks = preanalysis.get("blocks") if isinstance(preanalysis, dict) else None
    if not isinstance(blocks, list) or not blocks:
        raise RuntimeError("Structured blocks unavailable for chunking")

    normalized_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        raw_block_text = str(block.get("text") or "").strip()
        if not raw_block_text:
            continue
        block_type = str(block.get("block_type") or "")
        normalized_text = raw_block_text
        if block_type not in {"table_caption", "table_like"} and not _block_is_table_body_candidate(block):
            normalized_text = re.sub(r"\s+", " ", raw_block_text)
        section_id = str(block.get("section_id") or block.get("section_path") or "Document").strip() or "Document"
        normalized_blocks.append({**block, "section_id": section_id, "text": normalized_text})
    if not normalized_blocks:
        raise RuntimeError("Structured blocks unavailable for chunking")

    document_text = " ".join(str(block.get("text") or "") for block in normalized_blocks[:80] if str(block.get("block_type") or "") != "heading")
    document_is_cjk_dominant = _is_cjk_dominant_text(document_text)
    child_chunk_size = (
        min(chunk_size, max(int(chunk_size * 0.85), chunk_size - overlap, 360))
        if document_is_cjk_dominant
        else min(chunk_size, max(int(chunk_size * 0.65), 420))
    )
    chunks: list[dict[str, Any]] = []
    parent_chunks_by_section: dict[str, dict[str, Any]] = {}
    section_blocks: dict[str, list[dict[str, Any]]] = {}
    for block in normalized_blocks:
        section_blocks.setdefault(str(block.get("section_id") or "Document"), []).append(block)
    pending_blocks: list[dict[str, Any]] = []
    pending_length = 0

    def append_chunk(
        block_rows: list[dict[str, Any]],
        *,
        supporting_context_rows: list[dict[str, Any]] | None = None,
        body_text: str | None = None,
    ) -> None:
        chunk_body_text = body_text or "\n\n".join(
            str(block.get("text") or "").strip() for block in block_rows if str(block.get("text") or "").strip()
        )
        if not chunk_body_text:
            return
        parent_key = str(block_rows[0].get("section_id") or "Document") if block_rows else "Document"
        chunks.append(
            _make_chunk_record(
                chunk_index=len(chunks),
                body_text=chunk_body_text,
                block_rows=list(block_rows),
                paper_title=paper_title,
                supporting_context_rows=supporting_context_rows,
                parent_key=parent_key,
            )
        )

    def flush_pending() -> None:
        nonlocal pending_blocks, pending_length
        if not pending_blocks:
            return
        has_structured_block = any(
            str(block.get("block_type") or "") in {"table_like", "table_caption", "figure_caption"}
            or _block_is_table_body_candidate(block)
            for block in pending_blocks
        )
        append_chunk(
            list(pending_blocks),
            supporting_context_rows=(
                _select_supporting_context(normalized_blocks, pending_blocks)
                if has_structured_block or not document_is_cjk_dominant
                else []
            )
        )
        pending_blocks = []
        pending_length = 0

    block_index = 0
    while block_index < len(normalized_blocks):
        block = normalized_blocks[block_index]
        block_text = str(block.get("text") or "").strip()
        block_length = len(block_text)
        section_id = str(block.get("section_id") or "")
        pending_section_id = str(pending_blocks[0].get("section_id") or "") if pending_blocks else ""
        block_type = str(block.get("block_type") or "paragraph")
        if block_type == "heading":
            flush_pending()
            block_index += 1
            continue
        isolate_block = block_type in {"table_like", "table_caption", "figure_caption"}

        if pending_blocks and section_id != pending_section_id:
            flush_pending()

        next_block = normalized_blocks[block_index + 1] if block_index + 1 < len(normalized_blocks) else None
        if (
            block_type == "table_caption"
            and next_block is not None
            and str(next_block.get("section_id") or "") == section_id
            and _block_is_table_body_candidate(next_block)
        ):
            flush_pending()
            table_blocks = [{**block, "text": block_text}, {**next_block, "text": str(next_block.get("text") or "").strip()}]
            table_body_text = str(next_block.get("text") or "").strip()
            table_total_length = len(block_text) + len(table_body_text) + 2
            if table_total_length <= chunk_size:
                append_chunk(
                    table_blocks,
                    supporting_context_rows=_select_supporting_context(normalized_blocks, table_blocks),
                )
            else:
                available_with_caption = max(chunk_size - len(block_text) - 2, 0)
                body_piece_size = max(
                    min(max(available_with_caption, child_chunk_size), chunk_size),
                    min(chunk_size, 420),
                )
                windows = _window_text(table_body_text, chunk_size=body_piece_size, overlap=min(overlap, max(body_piece_size - 1, 0)))
                if not windows:
                    append_chunk(
                        table_blocks,
                        supporting_context_rows=_select_supporting_context(normalized_blocks, table_blocks),
                    )
                else:
                    next_char_start = int(next_block.get("char_start") or 0)
                    for window_index, piece in enumerate(windows):
                        piece_block = dict(next_block)
                        piece_block["text"] = str(piece["content"])
                        piece_block["char_start"] = next_char_start + int(piece["char_start"])
                        piece_block["char_end"] = next_char_start + int(piece["char_end"])
                        piece_body_text = str(piece["content"])
                        combined_blocks = [{**block, "text": block_text}, piece_block]
                        combined_body_text = f"{block_text}\n\n{piece_body_text}"
                        append_chunk(
                            combined_blocks,
                            supporting_context_rows=_select_supporting_context(normalized_blocks, table_blocks),
                            body_text=combined_body_text,
                        )
            block_index += 2
            continue

        if isolate_block and pending_blocks:
            flush_pending()

        if block_length > chunk_size:
            flush_pending()
            char_start = int(block.get("char_start") or 0)
            for piece in _window_text(block_text, chunk_size=chunk_size, overlap=overlap):
                piece_block = dict(block)
                piece_block["text"] = str(piece["content"])
                piece_block["char_start"] = char_start + int(piece["char_start"])
                piece_block["char_end"] = char_start + int(piece["char_end"])
                append_chunk(
                    [piece_block],
                    supporting_context_rows=_select_supporting_context(normalized_blocks, [block]),
                    body_text=str(piece["content"]),
                )
            block_index += 1
            continue

        projected_length = pending_length + block_length + (2 if pending_blocks else 0)
        if pending_blocks and projected_length > child_chunk_size:
            flush_pending()

        pending_blocks.append({**block, "text": block_text})
        pending_length += block_length + (2 if len(pending_blocks) > 1 else 0)
        if isolate_block:
            flush_pending()
        block_index += 1

    flush_pending()
    if not chunks:
        raise RuntimeError("Structured blocks unavailable for chunking")
    parent_records: list[dict[str, Any]] = []
    section_summary_records: list[dict[str, Any]] = []
    section_summary_texts: list[str] = []
    for section_id, rows in section_blocks.items():
        parent_rows = [row for row in rows if str(row.get("text") or "").strip()]
        if not parent_rows:
            continue
        parent_body_text = "\n\n".join(str(row.get("text") or "").strip() for row in parent_rows if str(row.get("text") or "").strip())
        if not parent_body_text:
            continue
        parent_record = _make_chunk_record(
            chunk_index=len(chunks) + len(parent_records),
            body_text=parent_body_text,
            block_rows=parent_rows,
            paper_title=paper_title,
            chunk_role="parent",
        )
        parent_record["section_key"] = section_id
        parent_chunks_by_section[section_id] = parent_record
        parent_records.append(parent_record)
        section_summary_text = _summarize_text_excerpt(
            parent_body_text,
            max_sentences=settings.rag_section_summary_max_sentences,
            max_chars=settings.rag_section_summary_max_chars,
        )
        if section_summary_text and (len(parent_rows) > 1 or len(parent_body_text) > settings.rag_section_summary_max_chars):
            section_summary_record = _make_chunk_record(
                chunk_index=len(chunks) + len(parent_records) + len(section_summary_records),
                body_text=section_summary_text,
                block_rows=parent_rows,
                paper_title=paper_title,
                chunk_role="section_summary",
                granularity="section_summary",
                source_kind="heuristic_section_summary",
            )
            section_summary_record["section_key"] = section_id
            section_summary_records.append(section_summary_record)
            section_summary_texts.append(section_summary_text)
    parent_text_by_section = {
        section_id: str(parent_record["metadata_json"].get("body_text") or "")[:2200]
        for section_id, parent_record in parent_chunks_by_section.items()
    }
    for chunk in chunks:
        section_key = str(chunk.get("parent_key") or "")
        if not section_key:
            continue
        parent_text = parent_text_by_section.get(section_key)
        if parent_text:
            chunk["metadata_json"]["parent_text"] = parent_text
    chunks.extend(parent_records)
    chunks.extend(section_summary_records)

    paper_summary_text = _serialize_structured_summary_for_chunk(structured_summary)
    paper_summary_source_kind = "structured_summary"
    if not paper_summary_text:
        paper_summary_text = _build_fallback_paper_summary_text(
            section_summary_texts,
            normalized_blocks=normalized_blocks,
        )
        paper_summary_source_kind = "heuristic_paper_summary"
    if paper_summary_text:
        representative_rows = [rows[0] for rows in section_blocks.values() if rows] or normalized_blocks[:1]
        paper_summary_record = _make_chunk_record(
            chunk_index=len(chunks),
            body_text=paper_summary_text,
            block_rows=representative_rows,
            paper_title=paper_title,
            chunk_role="paper_summary",
            granularity="paper_summary",
            source_kind=paper_summary_source_kind,
        )
        chunks.append(paper_summary_record)
    return chunks


def _build_preanalysis_from_document_structure(db: Session, *, paper_id: int) -> dict[str, Any] | None:
    blocks: list[dict[str, Any]] = []
    char_cursor = 0
    db_blocks = db.scalars(
        select(PaperDocumentBlock)
        .where(PaperDocumentBlock.paper_id == paper_id)
        .order_by(PaperDocumentBlock.block_index.asc(), PaperDocumentBlock.id.asc())
    ).all()
    for fallback_index, row in enumerate(db_blocks):
        text_value = (row.text or "").strip()
        if not text_value:
            continue
        provenance = row.provenance_json if isinstance(row.provenance_json, dict) else {}
        page_number = int(provenance.get("page_number") or 1)
        char_start = char_cursor
        char_end = char_start + len(text_value)
        char_cursor = char_end + 2
        blocks.append(
            {
                "block_index": len(blocks),
                "source_block_id": row.id,
                "page_number": page_number,
                "section_id": row.section_path or "Document",
                "section_title": row.section_path or "Document",
                "section_path": row.section_path or "Document",
                "heading_level": row.heading_level or 1,
                "block_type": row.block_type or "paragraph",
                "docling_label": row.docling_label,
                "reading_order": row.reading_order,
                "char_start": char_start,
                "char_end": char_end,
                "text": text_value,
                "bbox": row.bbox_json if isinstance(row.bbox_json, dict) else None,
                "provenance": provenance or None,
                "_sort_key": _structure_sort_key(
                    reading_order=row.reading_order,
                    page_number=page_number,
                    fallback_group=0,
                    fallback_index=fallback_index,
                ),
            }
        )

    tables = db.scalars(
        select(PaperDocumentTable)
        .where(PaperDocumentTable.paper_id == paper_id)
        .order_by(PaperDocumentTable.table_index.asc(), PaperDocumentTable.id.asc())
    ).all()
    for fallback_index, table in enumerate(tables):
        text_value, table_text_mode = _compose_table_text_with_mode(
            caption=table.caption,
            markdown=table.markdown,
            data=table.data_json,
        )
        if not text_value:
            continue
        char_start = char_cursor
        char_end = char_start + len(text_value)
        char_cursor = char_end + 2
        blocks.append(
            {
                "block_index": len(blocks),
                "source_table_id": table.id,
                "page_number": table.page_from or 1,
                "section_id": table.section_path or "Document",
                "section_title": table.section_path or "Document",
                "section_path": table.section_path or "Document",
                "heading_level": table.heading_level or 1,
                "block_type": "table_like",
                "reading_order": table.reading_order,
                "char_start": char_start,
                "char_end": char_end,
                "text": text_value,
                "table_markdown": table.markdown,
                "table_data_json": table.data_json,
                "table_text_mode": table_text_mode,
                "bbox": table.bbox_json if isinstance(table.bbox_json, dict) else None,
                "provenance": table.provenance_json if isinstance(table.provenance_json, dict) else None,
                "_sort_key": _structure_sort_key(
                    reading_order=table.reading_order,
                    page_number=table.page_from,
                    fallback_group=1,
                    fallback_index=fallback_index,
                ),
            }
        )

    pictures = db.scalars(
        select(PaperDocumentPicture)
        .where(PaperDocumentPicture.paper_id == paper_id)
        .order_by(PaperDocumentPicture.picture_index.asc(), PaperDocumentPicture.id.asc())
    ).all()
    for fallback_index, picture in enumerate(pictures):
        parts = [str(picture.caption or "").strip(), str(picture.description or "").strip()]
        text_value = "\n\n".join(part for part in parts if part)
        if not text_value:
            continue
        char_start = char_cursor
        char_end = char_start + len(text_value)
        char_cursor = char_end + 2
        blocks.append(
            {
                "block_index": len(blocks),
                "source_picture_id": picture.id,
                "page_number": picture.page_number or 1,
                "section_id": picture.section_path or "Document",
                "section_title": picture.section_path or "Document",
                "section_path": picture.section_path or "Document",
                "heading_level": picture.heading_level or 1,
                "block_type": "figure_caption",
                "reading_order": picture.reading_order,
                "char_start": char_start,
                "char_end": char_end,
                "text": text_value,
                "picture_description": picture.description,
                "bbox": picture.bbox_json if isinstance(picture.bbox_json, dict) else None,
                "provenance": picture.provenance_json if isinstance(picture.provenance_json, dict) else None,
                "_sort_key": _structure_sort_key(
                    reading_order=picture.reading_order,
                    page_number=picture.page_number,
                    fallback_group=2,
                    fallback_index=fallback_index,
                ),
            }
        )
    sorted_blocks = sorted(blocks, key=lambda item: item.get("_sort_key", (1, 0, 99, int(item.get("block_index") or 0))))
    for index, block in enumerate(sorted_blocks):
        block["block_index"] = index
        block.pop("_sort_key", None)
    return {"blocks": sorted_blocks} if sorted_blocks else None


def rebuild_paper_index(
    db: Session,
    *,
    paper_id: int,
    preanalysis: dict[str, Any] | None = None,
    paper_title: str | None = None,
    structured_summary: dict[str, Any] | None = None,
) -> int:
    chunks = _split_text(
        preanalysis=preanalysis,
        paper_title=paper_title,
        structured_summary=structured_summary,
    )
    embeddings: list[list[float]] = []
    if chunks and (settings.glm_api_key.strip() or settings.openai_api_key.strip()):
        try:
            embeddings = embed_texts([str(chunk.get("embedding_input") or chunk["content"]) for chunk in chunks])
        except Exception:
            embeddings = []

    db.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
    parent_chunk_ids_by_section: dict[str, int] = {}
    child_chunk_ids_by_section: dict[str, list[int]] = {}
    child_rows: list[tuple[PaperChunk, str]] = []
    section_summary_rows: list[tuple[PaperChunk, str]] = []
    paper_summary_rows: list[PaperChunk] = []
    for index, chunk in enumerate(chunks):
        record = PaperChunk(
            paper_id=paper_id,
            parent_chunk_id=None,
            chunk_index=chunk["chunk_index"],
            chunk_role=str(chunk.get("chunk_role") or "child"),
            content=chunk["content"],
            embedding=embeddings[index] if index < len(embeddings) else None,
            token_count=chunk["token_count"],
            page_from=chunk["page_from"],
            page_to=chunk["page_to"],
            metadata_json=chunk["metadata_json"],
        )
        db.add(record)
        db.flush()
        if record.chunk_role == "parent":
            section_key = str(chunk.get("section_key") or chunk["metadata_json"].get("section_id") or "")
            if section_key:
                parent_chunk_ids_by_section[section_key] = int(record.id)
        elif record.chunk_role == "child":
            child_rows.append((record, str(chunk.get("parent_key") or "")))
            section_key = str(chunk.get("parent_key") or chunk["metadata_json"].get("section_id") or "")
            if section_key:
                child_chunk_ids_by_section.setdefault(section_key, []).append(int(record.id))
        elif record.chunk_role == "section_summary":
            section_summary_rows.append((record, str(chunk.get("section_key") or chunk["metadata_json"].get("section_id") or "")))
        elif record.chunk_role == "paper_summary":
            paper_summary_rows.append(record)
    for record, section_key in child_rows:
        parent_chunk_id = parent_chunk_ids_by_section.get(section_key)
        if parent_chunk_id is not None:
            record.parent_chunk_id = parent_chunk_id
    anchor_limit = max(settings.rag_summary_anchor_limit, 1)
    for record, section_key in section_summary_rows:
        metadata = dict(record.metadata_json or {})
        parent_chunk_id = parent_chunk_ids_by_section.get(section_key)
        anchor_chunk_ids: list[int] = []
        if parent_chunk_id is not None:
            anchor_chunk_ids.append(parent_chunk_id)
            metadata["resolved_chunk_id"] = parent_chunk_id
        anchor_chunk_ids.extend(child_chunk_ids_by_section.get(section_key, [])[:anchor_limit])
        metadata["anchor_chunk_ids"] = anchor_chunk_ids
        metadata["anchor_section_id"] = section_key or None
        record.metadata_json = metadata
    ordered_parent_ids = [
        chunk_id
        for section_key, chunk_id in parent_chunk_ids_by_section.items()
        if section_key
    ]
    for record in paper_summary_rows:
        metadata = dict(record.metadata_json or {})
        metadata["anchor_chunk_ids"] = ordered_parent_ids[:anchor_limit]
        metadata["resolved_chunk_id"] = ordered_parent_ids[0] if ordered_parent_ids else None
        metadata["anchor_section_ids"] = list(parent_chunk_ids_by_section.keys())[:anchor_limit]
        record.metadata_json = metadata
    db.commit()
    if _is_postgres_session(db):
        db.execute(
            update(PaperChunk)
            .where(PaperChunk.paper_id == paper_id)
            .values(
                search_vector=func.to_tsvector(
                    settings.rag_text_search_config,
                    PaperChunk.content,
                )
            )
        )
        db.commit()
    return len(chunks)


def rebuild_paper_index_from_document_structure(
    db: Session,
    *,
    paper_id: int,
    paper_title: str | None = None,
    structured_summary: dict[str, Any] | None = None,
) -> int:
    preanalysis = _build_preanalysis_from_document_structure(db, paper_id=paper_id)
    return rebuild_paper_index(
        db,
        paper_id=paper_id,
        preanalysis=preanalysis,
        paper_title=paper_title,
        structured_summary=structured_summary,
    )


def _search_chunks(
    db: Session,
    *,
    query: str,
    organization_id: int,
    top_k: int | None = None,
    trace: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    limit = top_k or settings.rag_top_k
    query_plan = _build_crosslingual_query_plan(query)
    exact_terms = list(query_plan.exact_terms)
    exact_match_heavy = query_plan.query_type == "exact_heavy_short" or _is_exact_match_heavy_query(
        query_plan.original_query,
        exact_terms,
    )
    exact_terms_for_boost = exact_terms if exact_match_heavy or _is_table_or_figure_query(query) else []
    if not _is_postgres_session(db):
        variant_results: dict[str, list[RetrievalResult]] = {}
        for variant in query_plan.variants:
            variant_results[variant.name] = _search_chunks_legacy(
                db,
                query=variant.query,
                top_k=max(settings.rag_rerank_top_n, limit),
                organization_id=organization_id,
            )
        source_hits: dict[str, list[dict[str, Any]]] = {}
        variant_traces: dict[str, dict[str, Any]] = {}
        for variant in query_plan.variants:
            hits = _results_to_ranked_hits(variant_results.get(variant.name, []))
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
                "fused_hits": _fuse_ranked_candidate_sources(local_sources, limit=max(settings.rag_rerank_top_n, limit))
                if local_sources
                else [],
            }
        fused_hits = _fuse_ranked_candidate_sources(source_hits, limit=max(settings.rag_rerank_top_n, limit)) if source_hits else []
        record_map = {
            int(result.chunk.id): (result.chunk, result.paper)
            for results in variant_results.values()
            for result in results
        }
        results = _build_retrieval_results(fused_hits[:limit], record_map=record_map)
        if trace is not None:
            serialized = _serialize_ranked_trace_candidates(fused_hits, record_map)
            sparse_sources = {
                name: hits
                for name, hits in source_hits.items()
                if name.endswith("_sparse")
            }
            dense_sources = {
                name: hits
                for name, hits in source_hits.items()
                if name.endswith("_dense")
            }
            trace.update(
                {
                    "query_plan": _serialize_query_plan(query_plan),
                    "variant_candidates": _serialize_variant_traces(variant_traces, record_map=record_map),
                    "sparse_candidates": _serialize_ranked_trace_candidates(
                        _fuse_ranked_candidate_sources(sparse_sources, limit=max(settings.rag_rerank_top_n, limit))
                        if sparse_sources
                        else [],
                        record_map,
                    ),
                    "dense_candidates": _serialize_ranked_trace_candidates(
                        _fuse_ranked_candidate_sources(dense_sources, limit=max(settings.rag_rerank_top_n, limit))
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

    sparse_limit = max(settings.rag_sparse_top_k, limit)
    dense_limit = max(settings.rag_dense_top_k, limit)
    if exact_match_heavy:
        sparse_limit = max(sparse_limit, limit * 8, settings.rag_sparse_top_k * 2)
        dense_limit = max(dense_limit, limit * 4, settings.rag_dense_top_k)

    dense_query_texts = _unique_strings([variant.query for variant in query_plan.variants if variant.use_dense])
    dense_embeddings: dict[str, list[float] | None] = {}
    if dense_query_texts and (settings.glm_api_key.strip() or settings.openai_api_key.strip()):
        try:
            embedding_rows = embed_texts(dense_query_texts)
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
    fusion_limit = max(settings.rag_fusion_window, settings.rag_rerank_top_n, limit, sparse_limit if exact_match_heavy else 0)

    for variant in query_plan.variants:
        variant_sparse_hits: list[dict[str, Any]] = []
        variant_dense_hits: list[dict[str, Any]] = []
        if variant.use_sparse:
            variant_sparse_hits = _search_sparse_chunks_postgres(
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
            variant_dense_hits = _search_dense_chunks_postgres(
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
            "fused_hits": _fuse_ranked_candidate_sources(local_sources, limit=fusion_limit) if local_sources else [],
        }

    sparse_hits = _fuse_ranked_candidate_sources(sparse_source_hits, limit=fusion_limit) if sparse_source_hits else []
    dense_hits = _fuse_ranked_candidate_sources(dense_source_hits, limit=fusion_limit) if dense_source_hits else []
    fused_candidates = _fuse_ranked_candidate_sources(all_source_hits, limit=fusion_limit) if all_source_hits else []
    record_map = _load_chunk_record_map(
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
        expanded_candidates, expansion_stats = _expand_retrieval_candidates(
            db,
            query=query,
            candidates=fused_candidates,
            record_map=record_map,
            organization_id=organization_id,
            query_plan=query_plan,
            limit=fusion_limit,
        )
    if exact_terms_for_boost:
        expanded_candidates = _boost_exact_match_candidates(query, candidates=expanded_candidates, record_map=record_map)[: max(fusion_limit, len(expanded_candidates))]

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
            reranked_candidates, rerank_context_stats = _apply_reranking(
                db,
                query_plan.rerank_query,
                fused_candidates=expanded_candidates,
                record_map=record_map,
                limit=max(settings.rag_rerank_top_n, limit),
            )
            rerank_status = "applied"
        except Exception as exc:
            logger.warning(
                "Rerank failed, falling back to fused order: user_query=%s rerank_query=%s fused_count=%s rerank_limit=%s rerank_context_stats=%s error=%s",
                _preview_log_text(query, limit=200),
                _preview_log_text(query_plan.rerank_query, limit=200),
                len(expanded_candidates),
                max(settings.rag_rerank_top_n, limit),
                rerank_context_stats,
                exc,
                exc_info=True,
            )
            reranked_candidates = expanded_candidates[: max(settings.rag_rerank_top_n, limit)]
            rerank_status = "fallback_to_fused"
            rerank_error = "rerank_failed"

    results = _build_retrieval_results(reranked_candidates[:limit], record_map=record_map)
    if trace is not None:
        trace.update(
            {
                "query_plan": _serialize_query_plan(query_plan),
                "variant_candidates": _serialize_variant_traces(variant_traces, record_map=record_map),
                "sparse_candidates": _serialize_ranked_trace_candidates(sparse_hits, record_map),
                "dense_candidates": _serialize_ranked_trace_candidates(dense_hits, record_map),
                "fused_candidates": _serialize_ranked_trace_candidates(fused_candidates, record_map),
                "expanded_candidates": _serialize_ranked_trace_candidates(expanded_candidates, record_map),
                "reranked_candidates": _serialize_ranked_trace_candidates(reranked_candidates, record_map),
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


def _serialize_citations(results: list[RetrievalResult]) -> list[dict]:
    citations: list[dict] = []
    for result in results:
        snippet = _clip_snippet(_chunk_text_payload(result.chunk), max_length=240)
        citations.append(
            {
                "chunk_id": result.chunk.id,
                "paper_id": result.paper.id,
                "paper_title": result.paper.title,
                "source_url": result.paper.source_url,
                "snippet": snippet,
                "score": round(result.score, 4),
                "page_from": result.chunk.page_from,
                "page_to": result.chunk.page_to,
            }
        )
    return citations


def _chunk_section_path(chunk: PaperChunk) -> str | None:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    section_path = str(metadata.get("section_path") or metadata.get("section_title") or "").strip()
    return section_path or None


def _build_evidence_candidates(
    results: list[RetrievalResult],
    *,
    query: str,
) -> list[dict[str, Any]]:
    if not results:
        return []
    max_score = max((float(result.score) for result in results), default=0.0)
    include_supporting_context = True
    total = len(results)
    candidates: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        relative_score = (float(result.score) / max_score) if max_score > 0 else 0.0
        rank_score = 1.0 if total == 1 else 1.0 - ((rank - 1) / max(total - 1, 1))
        support_score = round(min(0.99, max(0.05, relative_score * 0.6 + rank_score * 0.4)), 4)
        full_text = _chunk_text_payload(result.chunk, include_supporting_context=include_supporting_context)
        candidates.append(
            {
                "evidence_id": f"chunk-{result.chunk.id}",
                "chunk_id": int(result.chunk.id),
                "paper_id": int(result.paper.id),
                "paper_title": result.paper.title,
                "source_url": result.paper.source_url,
                "snippet": _clip_snippet(full_text, max_length=320),
                "full_text": full_text,
                "score": round(float(result.score), 4),
                "support_score": support_score,
                "page_from": result.chunk.page_from,
                "page_to": result.chunk.page_to,
                "section_path": _chunk_section_path(result.chunk),
                "selection_reason": "",
                "claim_texts": [],
                "rank": rank,
            }
        )
    return candidates


def _serialize_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
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


def _serialize_evidence_list(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_serialize_evidence_item(item) for item in items]


def _build_sufficiency_decision(
    selected_evidence: list[dict[str, Any]],
    *,
    overall_support_score: float,
    llm_sufficient: bool | None,
    policy: ChatAttributionPolicy = STRICT_CHAT_ATTRIBUTION_POLICY,
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


def _heuristic_select_claim_supporting_evidence(
    *,
    question: str,
    query_plan: dict[str, Any] | None,
    evidence_candidates: list[dict[str, Any]],
    policy: ChatAttributionPolicy = STRICT_CHAT_ATTRIBUTION_POLICY,
    reason_suffix: str = "",
) -> EvidenceSelectionResult:
    selected: list[dict[str, Any]] = []
    max_evidence = max(1, settings.rag_attribution_max_evidence)
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
            and sum(float(item.get("support_score") or 0.0) for item in selected)
            >= policy.min_total_support_score
        ):
            break
    overall_support_score = round(_mean([float(item.get("support_score") or 0.0) for item in selected]), 4)
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
    sufficiency_decision = _build_sufficiency_decision(
        selected,
        overall_support_score=overall_support_score,
        llm_sufficient=None,
        policy=policy,
    )
    missing_information = "" if sufficiency_decision["is_sufficient"] else "未找到足以稳定支撑回答的知识库证据。"
    return EvidenceSelectionResult(
        selected_evidence=tuple(selected),
        claims=tuple(claims),
        overall_support_score=overall_support_score,
        sufficiency_decision=sufficiency_decision,
        missing_information=missing_information,
        method="heuristic",
    )


def _select_claim_supporting_evidence(
    *,
    question: str,
    query_plan: dict[str, Any] | None,
    evidence_candidates: list[dict[str, Any]],
    policy: ChatAttributionPolicy = STRICT_CHAT_ATTRIBUTION_POLICY,
) -> EvidenceSelectionResult:
    fallback = _heuristic_select_claim_supporting_evidence(
        question=question,
        query_plan=query_plan,
        evidence_candidates=evidence_candidates,
        policy=policy,
    )
    if not evidence_candidates or not _llm_available_for_grounding():
        logger.info(
            "RAG evidence selection skipped: candidates=%s method=%s question=%s",
            len(evidence_candidates),
            fallback.method,
            _preview_log_text(question),
        )
        return fallback

    selection_started_at = time.perf_counter()
    logger.info(
        "RAG evidence selection started: candidates=%s question=%s",
        len(evidence_candidates),
        _preview_log_text(question),
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
        content, _ = chat_with_messages(
            [
                {"role": "system", "content": EVIDENCE_SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        payload = _parse_json_object(content)
        candidate_map = {str(item["evidence_id"]): item for item in evidence_candidates}
        support_by_id: dict[str, float] = {}
        reasons_by_id: dict[str, list[str]] = {}
        claims_by_id: dict[str, list[str]] = {}
        claims: list[dict[str, Any]] = []
        for claim in payload.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_text = _normalize_query_text(str(claim.get("claim_text") or ""))
            evidence_ids = [
                str(evidence_id)
                for evidence_id in (claim.get("supporting_evidence_ids") or [])
                if str(evidence_id) in candidate_map
            ]
            support_score = float(claim.get("support_score") or 0.0)
            selection_reason = _normalize_query_text(str(claim.get("selection_reason") or ""))
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
            reason = _normalize_query_text(str(row.get("selection_reason") or ""))
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
            base["support_score"] = round(max(float(base.get("support_score") or 0.0), support_by_id.get(evidence_id, 0.0)), 4)
            reason_text = "；".join(reasons_by_id.get(evidence_id, []))
            if reason_text:
                base["selection_reason"] = reason_text
            base["claim_texts"] = claims_by_id.get(evidence_id, [])
            selected.append(base)
            if len(selected) >= settings.rag_attribution_max_evidence:
                break
        if not selected:
            logger.warning(
                "RAG evidence selection returned empty result, using heuristic fallback: candidates=%s elapsed=%.2fs",
                len(evidence_candidates),
                time.perf_counter() - selection_started_at,
            )
            return _heuristic_select_claim_supporting_evidence(
                question=question,
                query_plan=query_plan,
                evidence_candidates=evidence_candidates,
                policy=policy,
                reason_suffix="LLM 选择为空，已回退到启发式",
            )
        overall_support_score = round(
            float(payload.get("overall_support_score") or _mean([float(item.get("support_score") or 0.0) for item in selected])),
            4,
        )
        llm_sufficient = payload.get("is_sufficient")
        sufficiency_decision = _build_sufficiency_decision(
            selected,
            overall_support_score=overall_support_score,
            llm_sufficient=bool(llm_sufficient) if llm_sufficient is not None else None,
            policy=policy,
        )
        missing_information = _normalize_query_text(str(payload.get("missing_information") or ""))
        return EvidenceSelectionResult(
            selected_evidence=tuple(selected),
            claims=tuple(claims),
            overall_support_score=overall_support_score,
            sufficiency_decision=sufficiency_decision,
            missing_information=missing_information,
            method="llm",
        )
    except Exception as exc:
        logger.warning(
            "RAG evidence selection failed, using heuristic fallback: candidates=%s error=%s elapsed=%.2fs",
            len(evidence_candidates),
            exc,
            time.perf_counter() - selection_started_at,
        )
        return _heuristic_select_claim_supporting_evidence(
            question=question,
            query_plan=query_plan,
            evidence_candidates=evidence_candidates,
            reason_suffix="LLM 选择失败，已回退到启发式",
        )
    finally:
        logger.info(
            "RAG evidence selection finished: candidates=%s elapsed=%.2fs",
            len(evidence_candidates),
            time.perf_counter() - selection_started_at,
        )


def _build_evidence_prompt_text(selected_evidence: list[dict[str, Any]]) -> str:
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


def _heuristic_verify_answer(
    *,
    question: str,
    answer: str,
    selection_result: EvidenceSelectionResult,
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


def _verify_grounded_answer(
    *,
    question: str,
    answer: str,
    query_plan: dict[str, Any] | None,
    selected_evidence: list[dict[str, Any]],
    selection_result: EvidenceSelectionResult,
) -> dict[str, Any]:
    fallback = _heuristic_verify_answer(
        question=question,
        answer=answer,
        selection_result=selection_result,
    )
    if not selected_evidence or not _llm_available_for_grounding():
        logger.info(
            "RAG grounding verifier skipped: evidence=%s method=%s",
            len(selected_evidence),
            fallback.get("method"),
        )
        return fallback
    verifier_started_at = time.perf_counter()
    logger.info(
        "RAG grounding verifier started: evidence=%s question=%s",
        len(selected_evidence),
        _preview_log_text(question),
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
        f"选中证据：\n{_build_evidence_prompt_text(selected_evidence)}"
    )
    try:
        content, _ = chat_with_messages(
            [
                {"role": "system", "content": EVIDENCE_VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        payload = _parse_json_object(content)
        return {
            "method": "llm",
            "supported": bool(payload.get("supported")),
            "support_score": round(float(payload.get("support_score") or 0.0), 4),
            "unsupported_claims": [str(item) for item in (payload.get("unsupported_claims") or []) if str(item).strip()],
            "notes": str(payload.get("notes") or ""),
        }
    except Exception as exc:
        logger.warning(
            "RAG grounding verifier failed, using heuristic fallback: error=%s elapsed=%.2fs",
            exc,
            time.perf_counter() - verifier_started_at,
        )
        return fallback
    finally:
        logger.info(
            "RAG grounding verifier finished: evidence=%s elapsed=%.2fs",
            len(selected_evidence),
            time.perf_counter() - verifier_started_at,
        )


def _serialize_trace_candidates(results: list[RetrievalResult]) -> list[dict]:
    candidates: list[dict] = []
    for result in results:
        snippet = _clip_snippet(_chunk_text_payload(result.chunk), max_length=180)
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


def _create_message(
    db: Session,
    *,
    topic: ChatTopic,
    role: str,
    content: str,
    model: str | None = None,
    answer_mode: str | None = None,
    used_knowledge_base: bool = False,
    citations_json: list[dict] | None = None,
    metadata_json: dict | None = None,
) -> ChatMessage:
    now = datetime.now(timezone.utc)
    topic.updated_at = now
    message = ChatMessage(
        topic_id=topic.id,
        role=role,
        content=content,
        model=model,
        answer_mode=answer_mode,
        used_knowledge_base=used_knowledge_base,
        citations_json=citations_json,
        metadata_json=metadata_json,
        created_at=now,
    )
    db.add(message)
    db.commit()
    db.refresh(topic)
    db.refresh(message)
    return message


def _rename_topic_from_first_message(db: Session, *, topic: ChatTopic, prompt: str) -> None:
    if topic.title != DEFAULT_TOPIC_TITLE:
        return
    user_message_count = db.scalar(
        select(func.count(ChatMessage.id)).where(
            ChatMessage.topic_id == topic.id,
            ChatMessage.role == "user",
        )
    )
    if int(user_message_count or 0) != 1:
        return
    topic.title = _excerpt(prompt)
    topic.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(topic)


def _emit_chat_progress(
    progress_callback: ChatProgressCallback | None,
    *,
    phase: str,
    status: str,
    message: str,
    detail: str | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(phase, status, message, detail)


def _fallback_intro_text(*, has_evidence: bool) -> str:
    if has_evidence:
        return "知识库中未找到确切依据。以下补充结合了通用知识与部分相关证据线索，请自行判断。"
    return "知识库中未找到确切依据。以下为通用补充，请自行判断。"


def _normalize_fallback_answer(answer: str, *, has_evidence: bool) -> str:
    normalized = (answer or "").strip()
    if not normalized:
        return _fallback_intro_text(has_evidence=has_evidence)
    if "知识库中未找到确切依据" in normalized:
        return normalized
    return f"{_fallback_intro_text(has_evidence=has_evidence)}\n\n{normalized}"


def _generate_fallback_chat_answer(
    *,
    records: list[ChatMessage],
    message: str,
    selected_evidence: list[dict[str, Any]],
    selection_result: EvidenceSelectionResult,
    retrieval_debug: dict[str, Any],
    fallback_reason: str,
    prior_answer: str | None = None,
    progress_callback: ChatProgressCallback | None = None,
) -> tuple[str, str | None]:
    has_evidence = bool(selected_evidence)
    evidence_text = _build_evidence_prompt_text(selected_evidence) if has_evidence else "无可引用知识库证据。"
    sufficiency = selection_result.sufficiency_decision or {}
    reason_codes = ", ".join(str(code) for code in (sufficiency.get("reason_codes") or []) if str(code).strip()) or "none"
    missing_information = selection_result.missing_information or "未提供"
    prior_answer_text = (prior_answer or "").strip() or "无"
    generation_instruction = retrieval_debug.get("generation_instruction") or "请用中文回答，保留关键英文术语原文。"
    _emit_chat_progress(
        progress_callback,
        phase="fallback_generation",
        status="started",
        message="正在生成保守补充回答",
        detail=fallback_reason,
    )
    try:
        answer, model = chat_with_messages(
            [
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                *_history_messages(records),
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
        normalized = _normalize_fallback_answer(answer, has_evidence=has_evidence)
        _emit_chat_progress(
            progress_callback,
            phase="fallback_generation",
            status="finished",
            message="保守补充回答已生成",
            detail=model,
        )
        return normalized, model
    except Exception as exc:
        logger.warning("Fallback generation failed: reason=%s error=%s", fallback_reason, exc)
        _emit_chat_progress(
            progress_callback,
            phase="fallback_generation",
            status="failed",
            message="保守补充回答生成失败，已使用默认提示",
            detail=str(exc),
        )
        return _normalize_fallback_answer("", has_evidence=has_evidence), None


def _build_assistant_message_draft(
    db: Session,
    *,
    user: User,
    topic: ChatTopic,
    records: list[ChatMessage],
    message: str,
    progress_callback: ChatProgressCallback | None = None,
    relaxed_chat_rag: bool = False,
) -> AssistantMessageDraft:
    started_at = time.perf_counter()
    chat_policy = RELAXED_CHAT_ATTRIBUTION_POLICY if relaxed_chat_rag else STRICT_CHAT_ATTRIBUTION_POLICY
    retrieval_query = _build_retrieval_query(records, message)
    candidate_limit = max(settings.rag_top_k, settings.rag_rerank_top_n, 10)
    retrieval_debug: dict[str, Any] = {}
    _emit_chat_progress(
        progress_callback,
        phase="retrieval",
        status="started",
        message="正在检索知识库",
        detail=_preview_log_text(retrieval_query),
    )
    logger.info(
        "RAG retrieval started: topic_id=%s candidate_limit=%s query=%s",
        topic.id,
        candidate_limit,
        retrieval_query,
    )
    retrieval_candidates = _search_chunks(
        db,
        query=retrieval_query,
        organization_id=user.organization_id,
        top_k=candidate_limit,
        trace=retrieval_debug,
    )
    retrieval_results = retrieval_candidates[: settings.rag_top_k]
    evidence_candidates = _build_evidence_candidates(retrieval_results, query=message)
    selection_started_at = time.perf_counter()
    logger.info(
        "RAG evidence candidate preparation finished: topic_id=%s retrieval_results=%s evidence_candidates=%s elapsed=%.2fs",
        topic.id,
        len(retrieval_results),
        len(evidence_candidates),
        time.perf_counter() - started_at,
    )
    _emit_chat_progress(
        progress_callback,
        phase="retrieval",
        status="finished",
        message="知识库检索完成",
        detail=f"候选证据 {len(evidence_candidates)} 条",
    )
    _emit_chat_progress(
        progress_callback,
        phase="evidence_selection",
        status="started",
        message="正在筛选可支撑答案的证据",
    )
    selection_result = _select_claim_supporting_evidence(
        question=message,
        query_plan=retrieval_debug.get("query_plan") if isinstance(retrieval_debug.get("query_plan"), dict) else {},
        evidence_candidates=evidence_candidates,
        policy=chat_policy,
    )
    selected_evidence = [dict(item) for item in selection_result.selected_evidence]
    citations = _serialize_evidence_list(selected_evidence)
    _emit_chat_progress(
        progress_callback,
        phase="evidence_selection",
        status="finished",
        message="证据筛选完成",
        detail=f"选中证据 {len(selected_evidence)} 条",
    )
    logger.info(
        "RAG retrieval finished: topic_id=%s backend=%s candidates=%s selected_evidence=%s elapsed=%.2fs",
        topic.id,
        retrieval_debug.get("retrieval_backend", "unknown"),
        len(retrieval_candidates),
        len(selected_evidence),
        time.perf_counter() - started_at,
    )
    retrieval_trace = {
        "original_user_query": message,
        "retrieval_query": retrieval_query,
        "query_plan": retrieval_debug.get("query_plan"),
        "retrieval_backend": retrieval_debug.get("retrieval_backend", "unknown"),
        "first_pass_candidates": _serialize_trace_candidates(retrieval_candidates),
        "variant_candidates": retrieval_debug.get("variant_candidates", {}),
        "sparse_candidates": retrieval_debug.get("sparse_candidates", []),
        "dense_candidates": retrieval_debug.get("dense_candidates", []),
        "fused_candidates": retrieval_debug.get("fused_candidates", []),
        "reranked_candidates": retrieval_debug.get("reranked_candidates", []),
        "selected_citations": citations,
        "selected_evidence": citations,
        "evidence_selection_trace": {
            "method": selection_result.method,
            "candidate_evidence": _serialize_evidence_list(evidence_candidates),
            "claims": list(selection_result.claims),
            "overall_support_score": selection_result.overall_support_score,
            "missing_information": selection_result.missing_information,
        },
        "sufficiency_decision": selection_result.sufficiency_decision,
        "verifier_result": None,
        "answer_mode": "knowledge_base" if selection_result.sufficiency_decision.get("is_sufficient") else "kb_insufficient_evidence",
        "chat_response_policy": {
            "name": chat_policy.name,
            "allow_fallback_generation": chat_policy.allow_fallback_generation,
            "llm_insufficient_hard_gate": chat_policy.llm_insufficient_hard_gate,
            "verifier_min_support_score": chat_policy.verifier_min_support_score,
        },
        "generation_instruction": retrieval_debug.get("generation_instruction"),
        "rerank_query": retrieval_debug.get("rerank_query"),
        "search_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "evidence_selection_ms": round((time.perf_counter() - selection_started_at) * 1000, 2),
    }

    if selection_result.sufficiency_decision.get("is_sufficient") and selected_evidence:
        generation_started_at = time.perf_counter()
        evidence_text = _build_evidence_prompt_text(selected_evidence)
        _emit_chat_progress(
            progress_callback,
            phase="generation",
            status="started",
            message="正在生成答案草稿",
        )
        logger.info(
            "RAG generation started: topic_id=%s mode=knowledge_base citations=%s",
            topic.id,
            len(selected_evidence),
        )
        model: str | None = None
        final_model: str | None = None
        fallback_reason: str | None = None
        try:
            answer_started_at = time.perf_counter()
            logger.info(
                "RAG answer draft started: topic_id=%s evidence=%s",
                topic.id,
                len(selected_evidence),
            )
            answer, model = chat_with_messages(
                [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    *_history_messages(records),
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
            logger.info(
                "RAG answer draft finished: topic_id=%s model=%s elapsed=%.2fs",
                topic.id,
                model,
                time.perf_counter() - answer_started_at,
            )
            _emit_chat_progress(
                progress_callback,
                phase="generation",
                status="finished",
                message="答案草稿已生成",
                detail=model,
            )
            _emit_chat_progress(
                progress_callback,
                phase="verification",
                status="started",
                message="正在校验答案可靠性",
            )
            verifier_result = _verify_grounded_answer(
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
            _emit_chat_progress(
                progress_callback,
                phase="verification",
                status="finished",
                message="答案校验完成",
                detail=f"supported={bool(verifier_result.get('supported'))}",
            )
            logger.info(
                "RAG verifier decision: topic_id=%s supported=%s support_score=%s use_abstain=%s",
                topic.id,
                verifier_result.get("supported"),
                verifier_result.get("support_score"),
                use_abstain_path,
            )
        except Exception as exc:
            logger.warning("RAG generation failed, falling back to abstain: topic_id=%s error=%s", topic.id, exc)
            answer = "知识库中未找到确切依据。"
            retrieval_trace["verifier_result"] = {
                "method": "system",
                "supported": False,
                "support_score": 0.0,
                "unsupported_claims": [message],
                "notes": "generation_failed",
            }
            _emit_chat_progress(
                progress_callback,
                phase="verification",
                status="failed",
                message="答案生成或校验失败，已回退为保守回答",
                detail=str(exc),
            )
            use_abstain_path = True
            fallback_reason = "generation_failed"
        if use_abstain_path and chat_policy.allow_fallback_generation:
            fallback_answer, fallback_model = _generate_fallback_chat_answer(
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
        logger.info(
            "RAG generation finished: topic_id=%s mode=knowledge_base model=%s elapsed=%.2fs",
            topic.id,
            final_model,
            time.perf_counter() - generation_started_at,
        )
        retrieval_trace["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        retrieval_trace["answer_mode"] = answer_mode
        logger.info("rag_trace %s", retrieval_trace)
        draft = AssistantMessageDraft(
            content=final_answer,
            model=final_model,
            answer_mode=answer_mode,
            used_knowledge_base=used_knowledge_base,
            citations_json=final_citations,
            metadata_json={"retrieval": retrieval_trace},
        )
    else:
        logger.info("RAG abstain path: topic_id=%s selected_evidence=%s", topic.id, len(selected_evidence))
        retrieval_trace["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        retrieval_trace["fallback_used"] = bool(chat_policy.allow_fallback_generation)
        retrieval_trace["fallback_reason"] = "insufficient_evidence"
        if chat_policy.allow_fallback_generation:
            fallback_answer, fallback_model = _generate_fallback_chat_answer(
                records=records,
                message=message,
                selected_evidence=selected_evidence,
                selection_result=selection_result,
                retrieval_debug=retrieval_debug,
                fallback_reason="insufficient_evidence",
                progress_callback=progress_callback,
            )
            retrieval_trace["fallback_model"] = fallback_model
            retrieval_trace["answer_mode"] = "kb_insufficient_evidence"
            logger.info("rag_trace %s", retrieval_trace)
            draft = AssistantMessageDraft(
                content=fallback_answer,
                model=fallback_model,
                answer_mode="kb_insufficient_evidence",
                used_knowledge_base=False,
                citations_json=citations if selected_evidence else [],
                metadata_json={"retrieval": retrieval_trace},
            )
        else:
            retrieval_trace["answer_mode"] = "kb_insufficient_evidence"
            logger.info("rag_trace %s", retrieval_trace)
            draft = AssistantMessageDraft(
                content="知识库中未找到确切依据。",
                model=None,
                answer_mode="kb_insufficient_evidence",
                used_knowledge_base=False,
                citations_json=[],
                metadata_json={"retrieval": retrieval_trace},
            )

    _emit_chat_progress(
        progress_callback,
        phase="answer_ready",
        status="finished",
        message="最终答案已准备完成",
    )
    return draft


def start_topic_message(db: Session, *, user: User, topic_id: int, prompt: str) -> StartedChatTurn:
    message = (prompt or "").strip()
    if not message:
        raise RuntimeError("message is required")

    topic = get_topic(db, user_id=user.id, topic_id=topic_id)
    if topic is None:
        raise ValueError("Topic not found")

    user_message = _create_message(db, topic=topic, role="user", content=message)
    _rename_topic_from_first_message(db, topic=topic, prompt=message)
    records = list_topic_messages(db, topic_id=topic.id)
    return StartedChatTurn(
        topic=topic,
        user_message=user_message,
        records=records,
        prompt=message,
    )


def build_topic_assistant_draft(
    db: Session,
    *,
    user: User,
    started_turn: StartedChatTurn,
    progress_callback: ChatProgressCallback | None = None,
    relaxed_chat_rag: bool = False,
) -> AssistantMessageDraft:
    return _build_assistant_message_draft(
        db,
        user=user,
        topic=started_turn.topic,
        records=started_turn.records,
        message=started_turn.prompt,
        progress_callback=progress_callback,
        relaxed_chat_rag=relaxed_chat_rag,
    )


def prepare_topic_message(
    db: Session,
    *,
    user: User,
    topic_id: int,
    prompt: str,
    progress_callback: ChatProgressCallback | None = None,
    relaxed_chat_rag: bool = False,
) -> PreparedChatTurn:
    started_turn = start_topic_message(db, user=user, topic_id=topic_id, prompt=prompt)
    assistant_draft = build_topic_assistant_draft(
        db,
        user=user,
        started_turn=started_turn,
        progress_callback=progress_callback,
        relaxed_chat_rag=relaxed_chat_rag,
    )
    return PreparedChatTurn(
        topic=started_turn.topic,
        user_message=started_turn.user_message,
        assistant_draft=assistant_draft,
    )


def persist_topic_assistant_message(db: Session, *, topic: ChatTopic, assistant_draft: AssistantMessageDraft) -> ChatMessage:
    return _create_message(
        db,
        topic=topic,
        role="assistant",
        content=assistant_draft.content,
        model=assistant_draft.model,
        answer_mode=assistant_draft.answer_mode,
        used_knowledge_base=assistant_draft.used_knowledge_base,
        citations_json=assistant_draft.citations_json,
        metadata_json=assistant_draft.metadata_json,
    )


def send_topic_message(
    db: Session,
    *,
    user: User,
    topic_id: int,
    prompt: str,
    relaxed_chat_rag: bool = False,
) -> ChatTurnResult:
    prepared_turn = prepare_topic_message(
        db,
        user=user,
        topic_id=topic_id,
        prompt=prompt,
        relaxed_chat_rag=relaxed_chat_rag,
    )
    assistant_message = persist_topic_assistant_message(
        db,
        topic=prepared_turn.topic,
        assistant_draft=prepared_turn.assistant_draft,
    )
    return ChatTurnResult(
        topic=_topic_summary(db, prepared_turn.topic),
        user_message=prepared_turn.user_message,
        assistant_message=assistant_message,
    )
