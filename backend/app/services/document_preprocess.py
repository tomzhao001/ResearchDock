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
_TABLE_BODY_TERM_PATTERN = re.compile(
    r"(研究组|对照组|实验组|干预前|干预后|治疗前|治疗后|组别|stimulation|sham|baseline|post[- ]?(?:treatment|intervention)|pre[- ]?(?:treatment|intervention)|theta|beta|smr|frequency|score|p-?value|频率|评分)",
    re.IGNORECASE,
)
_TABLE_METRIC_PATTERN = re.compile(
    r"(?:[αβγθδμσχ]|SMR|P值|p-?value|频率|评分|均数|标准差|mean|sd|x±s|χ2|t检验)",
    re.IGNORECASE,
)


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
                    "ocr_metadata": page.ocr_metadata,
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
                "ocr_metadata": page.get("ocr_metadata") if isinstance(page.get("ocr_metadata"), dict) else None,
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
            "ocr_metadata": None,
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


def _extract_page_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    page_text = str(page.get("text") or "").strip()
    raw_blocks = page.get("blocks")
    processed_raw_blocks: list[dict[str, Any]] = []
    if isinstance(raw_blocks, list) and raw_blocks:
        for index, block in enumerate(raw_blocks):
            if not isinstance(block, dict):
                continue
            text = _compact_whitespace(str(block.get("text") or ""))
            if text:
                processed_raw_blocks.append(
                    {
                        "text": text,
                        "source_block_index": index,
                        "bbox": {
                            "x0": float(block.get("x0") or 0.0),
                            "y0": float(block.get("y0") or 0.0),
                            "x1": float(block.get("x1") or 0.0),
                            "y1": float(block.get("y1") or 0.0),
                        },
                    }
                )
    if processed_raw_blocks and _prefer_processed_raw_blocks(page_text, processed_raw_blocks):
        return processed_raw_blocks
    if page_text:
        raw_lines = [line.rstrip() for line in page_text.splitlines()]
        if raw_lines:
            grouped: list[str] = []
            buffer: list[str] = []
            for raw_line in raw_lines:
                line = _compact_whitespace(raw_line)
                if not line:
                    if buffer:
                        grouped.append(" ".join(buffer).strip())
                        buffer = []
                    continue
                is_heading, _ = _looks_like_heading(line)
                if is_heading:
                    if buffer:
                        grouped.append(" ".join(buffer).strip())
                        buffer = []
                    grouped.append(line)
                    continue
                if _TABLE_PATTERN.match(line) or _looks_like_table_body(raw_line):
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
                return [{"text": item, "source_block_index": index} for index, item in enumerate(grouped)]

    if processed_raw_blocks:
        return processed_raw_blocks

    return [
        {"text": _compact_whitespace(text), "source_block_index": index}
        for index, text in enumerate(_fallback_page_blocks(str(page.get("text") or "")))
        if _compact_whitespace(text)
    ]


def _looks_like_heading(text: str) -> tuple[bool, int | None]:
    compact = _compact_whitespace(text)
    if not compact:
        return False, None
    if _looks_like_table_body(text):
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
    if 0 < word_count <= 14 and compact == compact.upper() and any(char.isascii() and char.isalpha() for char in compact):
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
    if _looks_like_table_body(text):
        return "table_like"
    return "paragraph"


def _looks_like_table_body(text: str) -> bool:
    compact = _compact_whitespace(text)
    if not compact:
        return False
    number_count = len(re.findall(r"[-+]?\d+(?:\.\d+)?", compact))
    digit_count = sum(char.isdigit() for char in compact)
    delimiter_hits = len(re.findall(r"\s{2,}|\t|[|/]", text or ""))
    body_term_hits = len(_TABLE_BODY_TERM_PATTERN.findall(compact))
    metric_hits = len(_TABLE_METRIC_PATTERN.findall(compact))
    return bool(
        (number_count >= 3 and body_term_hits >= 1)
        or (number_count >= 4 and metric_hits >= 1 and digit_count >= 8)
        or (number_count >= 6 and digit_count >= 12)
        or (digit_count >= 10 and delimiter_hits >= 1)
    )


def _prefer_processed_raw_blocks(page_text: str, raw_blocks: list[dict[str, Any]]) -> bool:
    compact_page_text = _compact_whitespace(page_text)
    if not compact_page_text:
        return True
    page_signal = len(re.findall(r"[\w\u4e00-\u9fff]+", compact_page_text))
    raw_text = " ".join(str(block.get("text") or "") for block in raw_blocks)
    raw_signal = len(re.findall(r"[\w\u4e00-\u9fff]+", raw_text))
    if raw_signal >= max(page_signal * 2, 80):
        return True
    if any(
        _TABLE_PATTERN.match(str(block.get("text") or "")) or _looks_like_table_body(str(block.get("text") or ""))
        for block in raw_blocks
    ) and raw_signal > page_signal:
        return True
    return False


def preprocess_document(document: DocumentExtractionResult) -> dict[str, Any]:
    pages = _build_page_records(document)
    document_tags: set[str] = set()
    if any(bool(page.get("suspected_double_column")) for page in pages):
        document_tags.add("suspected_double_column")
    if any(bool(page.get("used_ocr")) for page in pages):
        document_tags.add("contains_ocr")
    if any(
        bool(
            (
                page.get("ocr_metadata", {}).get("text_quality", {})
                if isinstance(page.get("ocr_metadata"), dict)
                else {}
            ).get("normalization_applied")
        )
        for page in pages
    ):
        document_tags.add("ocr_text_normalized")
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
        page_blocks = _extract_page_blocks(page)
        if not page_blocks:
            continue
        for block_index, block_row in enumerate(page_blocks):
            raw_block_text = str(block_row.get("text") or "")
            block_text = _compact_whitespace(raw_block_text)
            if not block_text:
                continue

            is_heading, heading_level = _looks_like_heading(block_text)
            block_type = _detect_block_type(raw_block_text, is_heading=is_heading)
            if block_type.startswith("table"):
                document_tags.add("has_tables")
            if block_type.startswith("figure"):
                document_tags.add("has_figures")

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
                continue

            active_section = ensure_section(page_number)
            active_section["page_to"] = page_number

            char_start = char_cursor
            char_end = char_start + len(block_text)
            char_cursor = char_end + 2

            block_record = {
                "block_index": len(blocks),
                "page_number": page_number,
                "source_block_index": int(block_row.get("source_block_index") or block_index),
                "section_id": active_section["id"],
                "section_title": active_section["title"],
                "section_path": active_section["path"],
                "heading_level": active_section["heading_level"],
                "block_type": block_type,
                "char_start": char_start,
                "char_end": char_end,
                "text": block_text,
            }
            bbox = block_row.get("bbox")
            if isinstance(bbox, dict):
                block_record["bbox"] = bbox
            blocks.append(block_record)
            active_section["block_end"] = len(blocks) - 1
            active_section["block_count"] = int(active_section["block_count"]) + 1

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
                "ocr_text_normalized": bool(
                    (
                        page.get("ocr_metadata", {}).get("text_quality", {})
                        if isinstance(page.get("ocr_metadata"), dict)
                        else {}
                    ).get("normalization_applied")
                ),
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
