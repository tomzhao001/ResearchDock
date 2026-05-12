from __future__ import annotations

import re
from typing import Any

from app.services.pdf_extraction import DocumentExtractionResult, PageExtractionResult

_SECTION_KEYWORDS = {
    "abstract": 1,
    "introduction": 1,
    "background": 1,
    "related work": 1,
    "methods": 1,
    "method": 1,
    "materials and methods": 1,
    "results": 1,
    "discussion": 1,
    "conclusion": 1,
    "conclusions": 1,
    "references": 1,
    "摘要": 1,
    "引言": 1,
    "背景": 1,
    "相关工作": 1,
    "方法": 1,
    "研究方法": 1,
    "结果": 1,
    "讨论": 1,
    "结论": 1,
    "参考文献": 1,
}

_TABLE_PATTERN = re.compile(r"^(table|tab\.|表)\s*\d+", re.IGNORECASE)
_FIGURE_PATTERN = re.compile(r"^(figure|fig\.|图)\s*\d+", re.IGNORECASE)
_NUMBERED_HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+){0,3})[\s:：.-]+(.+)$")
_ROMAN_HEADING_PATTERN = re.compile(r"^([IVX]{1,6})[\s:：.-]+(.+)$")


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_heading_key(text: str) -> str:
    compact = _compact_whitespace(text).strip(" .:：-_")
    return compact.casefold()


def _build_page_records(document: DocumentExtractionResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    page_metadata = document.metadata.get("pages") if isinstance(document.metadata, dict) else None
    page_rows = page_metadata if isinstance(page_metadata, list) else []
    if document.pages:
        for page in document.pages:
            records.append(
                {
                    "page_number": page.page_number,
                    "char_count": page.char_count,
                    "suspected_double_column": page.suspected_double_column,
                    "used_ocr": page.used_ocr,
                    "text": page.text,
                    "blocks": page.blocks or [],
                }
            )
        return records

    for index, page in enumerate(page_rows, start=1):
        if not isinstance(page, dict):
            continue
        records.append(
            {
                "page_number": int(page.get("page_number") or index),
                "char_count": int(page.get("char_count") or 0),
                "suspected_double_column": bool(page.get("suspected_double_column")),
                "used_ocr": bool(page.get("used_ocr")),
                "text": "",
                "blocks": [],
            }
        )
    if records:
        if not any(str(page.get("text") or "").strip() for page in records) and (document.raw_text or "").strip():
            records[0]["text"] = document.raw_text
            records[0]["char_count"] = len((document.raw_text or "").strip())
        return records

    return [
        {
            "page_number": 1,
            "char_count": len((document.raw_text or "").strip()),
            "suspected_double_column": False,
            "used_ocr": False,
            "text": document.raw_text or "",
            "blocks": [],
        }
    ]


def _fallback_page_blocks(page_text: str) -> list[str]:
    text = (page_text or "").strip()
    if not text:
        return []
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n+", text) if segment.strip()]
    if paragraphs:
        return paragraphs
    return [_compact_whitespace(text)]


def _extract_block_texts(page: dict[str, Any]) -> list[str]:
    page_text = str(page.get("text") or "").strip()
    if page_text:
        lines = [_compact_whitespace(line) for line in page_text.splitlines() if _compact_whitespace(line)]
        if lines:
            grouped: list[str] = []
            buffer: list[str] = []
            for line in lines:
                is_heading, _ = _looks_like_heading(line)
                if is_heading:
                    if buffer:
                        grouped.append(" ".join(buffer).strip())
                        buffer = []
                    grouped.append(line)
                    continue
                buffer.append(line)
                if line.endswith((".", "。", "!", "！", "?", "？", ";", "；")) and len(" ".join(buffer)) >= 160:
                    grouped.append(" ".join(buffer).strip())
                    buffer = []
            if buffer:
                grouped.append(" ".join(buffer).strip())
            grouped = [item for item in grouped if item]
            if grouped:
                return grouped

    raw_blocks = page.get("blocks")
    if isinstance(raw_blocks, list) and raw_blocks:
        texts: list[str] = []
        for block in raw_blocks:
            if not isinstance(block, dict):
                continue
            text = _compact_whitespace(str(block.get("text") or ""))
            if text:
                texts.append(text)
        if texts:
            return texts
    return [_compact_whitespace(text) for text in _fallback_page_blocks(str(page.get("text") or "")) if _compact_whitespace(text)]


def _looks_like_heading(text: str) -> tuple[bool, int | None]:
    compact = _compact_whitespace(text)
    if not compact:
        return False, None

    numbered = _NUMBERED_HEADING_PATTERN.match(compact)
    if numbered:
        level = numbered.group(1).count(".") + 1
        return True, min(level, 4)

    roman = _ROMAN_HEADING_PATTERN.match(compact)
    if roman:
        return True, 1

    heading_key = _normalize_heading_key(compact)
    if heading_key in _SECTION_KEYWORDS:
        return True, _SECTION_KEYWORDS[heading_key]

    if len(compact) > 120:
        return False, None
    if compact.endswith((".", "?", "!", "。", "？", "！", ";", "；")):
        return False, None

    word_count = len(re.findall(r"[\w\u4e00-\u9fff]+", compact))
    if 0 < word_count <= 14 and compact == compact.upper() and any(char.isalpha() for char in compact):
        return True, 1

    title_like = bool(re.fullmatch(r"[\w\u4e00-\u9fff][\w\u4e00-\u9fff\s\-,:：()/%]+", compact))
    if title_like and 2 <= word_count <= 10 and (":" in compact or "：" in compact or compact.istitle()):
        return True, 2
    return False, None


def _detect_block_type(text: str, *, is_heading: bool) -> str:
    if is_heading:
        return "heading"
    compact = _compact_whitespace(text)
    if _TABLE_PATTERN.match(compact):
        return "table_caption"
    if _FIGURE_PATTERN.match(compact):
        return "figure_caption"
    digits = sum(char.isdigit() for char in compact)
    if digits >= 8 and len(re.findall(r"\s{2,}|\t", text)) >= 1:
        return "table_like"
    return "paragraph"


def preprocess_document(document: DocumentExtractionResult) -> dict[str, Any]:
    pages = _build_page_records(document)
    document_tags: set[str] = set()
    if any(bool(page.get("suspected_double_column")) for page in pages):
        document_tags.add("suspected_double_column")
    if any(bool(page.get("used_ocr")) for page in pages):
        document_tags.add("contains_ocr")
    if len(pages) > 1:
        document_tags.add("multi_page")

    sections: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    section_titles_seen: set[str] = set()
    current_titles: dict[int, str] = {}
    current_section: dict[str, Any] | None = None
    char_cursor = 0
    heading_hits = 0

    def ensure_section(page_number: int) -> dict[str, Any]:
        nonlocal current_section
        if current_section is None:
            current_section = {
                "id": f"section_{len(sections)}",
                "title": "Front Matter",
                "path": "Front Matter",
                "heading_level": 1,
                "page_from": page_number,
                "page_to": page_number,
                "block_start": len(blocks),
                "block_end": len(blocks) - 1,
                "block_count": 0,
            }
            sections.append(current_section)
        return current_section

    for page in pages:
        page_number = int(page.get("page_number") or len(sections) + 1)
        page_blocks = _extract_block_texts(page)
        if not page_blocks:
            continue
        page_text = _compact_whitespace(str(page.get("text") or " ".join(page_blocks)))
        page_has_table = False
        page_has_figure = False
        for block_index, raw_block_text in enumerate(page_blocks):
            block_text = _compact_whitespace(raw_block_text)
            if not block_text:
                continue

            is_heading, heading_level = _looks_like_heading(block_text)
            block_type = _detect_block_type(raw_block_text, is_heading=is_heading)
            if block_type.startswith("table"):
                document_tags.add("has_tables")
                page_has_table = True
            if block_type.startswith("figure"):
                document_tags.add("has_figures")
                page_has_figure = True

            if is_heading:
                heading_hits += 1
                level = heading_level or 1
                current_titles = {key: value for key, value in current_titles.items() if key < level}
                current_titles[level] = block_text
                path = " > ".join(current_titles[key] for key in sorted(current_titles))
                normalized_title = _normalize_heading_key(block_text)
                if normalized_title:
                    section_titles_seen.add(normalized_title)
                current_section = {
                    "id": f"section_{len(sections)}",
                    "title": block_text,
                    "path": path,
                    "heading_level": level,
                    "page_from": page_number,
                    "page_to": page_number,
                    "block_start": len(blocks),
                    "block_end": len(blocks) - 1,
                    "block_count": 0,
                }
                sections.append(current_section)

        if not page_text:
            continue

        active_section = ensure_section(page_number)
        active_section["page_to"] = page_number
        active_section["block_end"] = len(blocks)
        active_section["block_count"] = int(active_section["block_count"]) + 1

        char_start = char_cursor
        char_end = char_start + len(page_text)
        char_cursor = char_end + 2

        page_block_type = "paragraph"
        if page_has_table:
            page_block_type = "table_like"
        elif page_has_figure:
            page_block_type = "figure_like"

        blocks.append(
            {
                "block_index": len(blocks),
                "page_number": page_number,
                "source_block_index": 0,
                "section_id": active_section["id"],
                "section_title": active_section["title"],
                "section_path": active_section["path"],
                "heading_level": active_section["heading_level"],
                "block_type": page_block_type,
                "char_start": char_start,
                "char_end": char_end,
                "text": page_text,
            }
        )

    if heading_hits >= 2:
        document_tags.add("academic_paper")
    if section_titles_seen:
        document_tags.add("has_sections")

    return {
        "schema_version": 1,
        "document_tags": sorted(document_tags),
        "pages": [
            {
                "page_number": int(page.get("page_number") or 0),
                "char_count": int(page.get("char_count") or 0),
                "suspected_double_column": bool(page.get("suspected_double_column")),
                "used_ocr": bool(page.get("used_ocr")),
                "block_count": sum(1 for block in blocks if int(block["page_number"]) == int(page.get("page_number") or 0)),
            }
            for page in pages
        ],
        "sections": sections,
        "blocks": blocks,
        "chunking_hints": {
            "page_count": len(pages),
            "section_count": len(sections),
            "block_count": len(blocks),
            "max_block_chars": max((len(str(block["text"])) for block in blocks), default=0),
            "avg_block_chars": round(sum(len(str(block["text"])) for block in blocks) / len(blocks), 2) if blocks else 0.0,
            "fallback_windowing_required": not bool(blocks),
        },
    }
