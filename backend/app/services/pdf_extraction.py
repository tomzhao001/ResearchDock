from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageExtractionResult:
    page_number: int
    char_count: int
    alpha_ratio: float
    continuous_line_ratio: float
    image_count: int
    suspected_double_column: bool
    needs_ocr: bool
    used_ocr: bool
    reasons: list[str]
    ocr_metadata: dict | None
    blocks: list[dict] | None
    text: str


@dataclass
class DocumentExtractionResult:
    raw_text: str
    metadata: dict
    pages: list[PageExtractionResult] | None = None
