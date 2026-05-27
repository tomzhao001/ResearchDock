from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import settings

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"\S+")
_SUSPICIOUS_RUN_RE = re.compile(r"[A-Za-z0-9]{1,2}(?:\s+[A-Za-z0-9]{1,2}){4,}")
_ALNUM_FRAGMENT_RE = re.compile(r"\b[A-Za-z0-9]{1,2}\b")
_PAGE_MARKERS = (
    "abstract",
    "introduction",
    "results",
    "discussion",
    "references",
    "conclusion",
    "\u6458\u8981",
    "\u5173\u952e\u8bcd",
    "\u7ed3\u679c",
    "\u8ba8\u8bba",
    "\u53c2\u8003\u6587\u732e",
)
_NUMERIC_EVIDENCE_RE = re.compile(
    r"(10\.\S+/\S+|\b\d{4}\b|\b\d+(?:\.\d+)?\s*(?:mg|kg|g|ml|mL|mmol|%)\b|\bp\s*[<=>]\s*0?\.\d+\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OcrPageQuality:
    page_number: int
    category: str
    reasons: list[str] = field(default_factory=list)
    char_count: int = 0
    cjk_ratio: float = 0.0
    ascii_printable_ratio: float = 0.0
    replacement_char_count: int = 0
    single_char_token_ratio: float = 0.0
    suspicious_run_count: int = 0
    line_break_density: float = 0.0
    alnum_fragment_ratio: float = 0.0
    contains_keywords_page_markers: bool = False
    contains_structured_numeric_evidence: bool = False

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OcrRoutingDecision:
    force_full_page_ocr: bool
    docling_ocr_engine: str
    enable_escalation: bool
    enable_postprocess: bool
    source: str
    routing_reason: str
    sampled_page_count: int
    toxic_page_count: int
    page_assessments: list[OcrPageQuality] = field(default_factory=list)

    def to_metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["page_assessments"] = [item.to_metadata() for item in self.page_assessments]
        return payload


def _fallback_docling_ocr_engine() -> str:
    engine = (settings.ocr_docling_fallback_engine or settings.docling_ocr_engine or "rapidocr").strip().lower()
    return engine or "rapidocr"


def build_default_routing_decision(*, reason: str | None = None) -> OcrRoutingDecision:
    return OcrRoutingDecision(
        force_full_page_ocr=bool(settings.docling_force_full_page_ocr),
        docling_ocr_engine=_fallback_docling_ocr_engine(),
        enable_escalation=bool(settings.ocr_escalation_enabled),
        enable_postprocess=bool(settings.ocr_postprocess_enabled),
        source="default",
        routing_reason=reason or "default_settings",
        sampled_page_count=0,
        toxic_page_count=0,
        page_assessments=[],
    )


def assess_document_text_layer(pdf_path: Path) -> OcrRoutingDecision:
    if not settings.ocr_quality_routing_enabled:
        return build_default_routing_decision(reason="routing_disabled")

    try:
        import fitz
    except Exception:
        return build_default_routing_decision(reason="pymupdf_unavailable")

    try:
        document = fitz.open(pdf_path)
    except Exception:
        return build_default_routing_decision(reason="pdf_open_failed")

    max_pages = max(1, int(settings.ocr_quality_sample_pages))
    try:
        sampled = [
            _assess_page_text(page_number=index + 1, text=document.load_page(index).get_text("text"))
            for index in range(min(document.page_count, max_pages))
        ]
    finally:
        document.close()

    if not sampled:
        return build_default_routing_decision(reason="no_pages_sampled")

    toxic_pages = [item for item in sampled if item.category == "toxic_text_layer"]
    title_pages_toxic = any(item.page_number <= 2 and item.category == "toxic_text_layer" for item in sampled)
    toxic_ratio = len(toxic_pages) / max(1, len(sampled))

    if settings.docling_force_full_page_ocr:
        return OcrRoutingDecision(
            force_full_page_ocr=True,
            docling_ocr_engine=_fallback_docling_ocr_engine(),
            enable_escalation=bool(settings.ocr_escalation_enabled),
            enable_postprocess=bool(settings.ocr_postprocess_enabled),
            source="manual_override",
            routing_reason="docling_force_full_page_ocr_enabled",
            sampled_page_count=len(sampled),
            toxic_page_count=len(toxic_pages),
            page_assessments=sampled,
        )

    force_full_page_ocr = len(toxic_pages) >= 2 or toxic_ratio >= 0.4 or title_pages_toxic
    reason_parts: list[str] = []
    if len(toxic_pages) >= 2:
        reason_parts.append("multiple_toxic_pages")
    if toxic_ratio >= 0.4:
        reason_parts.append("high_toxic_ratio")
    if title_pages_toxic:
        reason_parts.append("title_or_abstract_page_toxic")
    if not reason_parts:
        reason_parts.append("text_layer_acceptable")

    return OcrRoutingDecision(
        force_full_page_ocr=force_full_page_ocr,
        docling_ocr_engine=_fallback_docling_ocr_engine(),
        enable_escalation=bool(settings.ocr_escalation_enabled),
        enable_postprocess=bool(settings.ocr_postprocess_enabled),
        source="quality_router",
        routing_reason=",".join(reason_parts),
        sampled_page_count=len(sampled),
        toxic_page_count=len(toxic_pages),
        page_assessments=sampled,
    )


def _assess_page_text(*, page_number: int, text: str) -> OcrPageQuality:
    content = str(text or "")
    non_whitespace_chars = [char for char in content if not char.isspace()]
    char_count = len(non_whitespace_chars)
    cjk_count = sum(1 for char in non_whitespace_chars if _CJK_RE.match(char))
    ascii_printable_count = sum(1 for char in non_whitespace_chars if char.isascii() and char.isprintable())
    replacement_char_count = content.count("\ufffd")
    tokens = _TOKEN_RE.findall(content)
    single_char_token_ratio = sum(1 for token in tokens if len(token) == 1) / max(1, len(tokens))
    suspicious_run_count = len(_SUSPICIOUS_RUN_RE.findall(content))
    line_break_density = content.count("\n") / max(1, char_count)
    alnum_fragment_ratio = len(_ALNUM_FRAGMENT_RE.findall(content)) / max(1, len(tokens))
    lowered = content.casefold()
    contains_keywords_page_markers = any(marker in lowered for marker in _PAGE_MARKERS)
    contains_structured_numeric_evidence = bool(_NUMERIC_EVIDENCE_RE.search(content))

    cjk_ratio = cjk_count / max(1, char_count)
    ascii_printable_ratio = ascii_printable_count / max(1, char_count)

    reasons: list[str] = []
    category = "good_text_layer"
    if char_count < 30:
        category = "no_text_layer"
        reasons.append("char_count_lt_30")
    else:
        suspicious_flags = 0
        if cjk_ratio < 0.2:
            suspicious_flags += 1
            reasons.append("low_cjk_ratio")
        if ascii_printable_ratio > 0.55:
            suspicious_flags += 1
            reasons.append("high_ascii_printable_ratio")
        if replacement_char_count > 0:
            suspicious_flags += 1
            reasons.append("replacement_char_present")
        if single_char_token_ratio > 0.45:
            suspicious_flags += 1
            reasons.append("high_single_char_token_ratio")
        if suspicious_run_count > 0:
            suspicious_flags += 1
            reasons.append("suspicious_fragment_runs")
        if alnum_fragment_ratio > 0.35:
            suspicious_flags += 1
            reasons.append("high_alnum_fragment_ratio")
        if suspicious_flags >= 2:
            category = "toxic_text_layer"
        elif not contains_keywords_page_markers and page_number <= 2 and cjk_ratio < 0.1:
            category = "toxic_text_layer"
            reasons.append("missing_expected_page_markers")

    return OcrPageQuality(
        page_number=page_number,
        category=category,
        reasons=reasons,
        char_count=char_count,
        cjk_ratio=round(cjk_ratio, 4),
        ascii_printable_ratio=round(ascii_printable_ratio, 4),
        replacement_char_count=replacement_char_count,
        single_char_token_ratio=round(single_char_token_ratio, 4),
        suspicious_run_count=suspicious_run_count,
        line_break_density=round(line_break_density, 4),
        alnum_fragment_ratio=round(alnum_fragment_ratio, 4),
        contains_keywords_page_markers=contains_keywords_page_markers,
        contains_structured_numeric_evidence=contains_structured_numeric_evidence,
    )
