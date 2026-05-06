from dataclasses import asdict, dataclass
from pathlib import Path

import fitz

from app.config import settings
from app.services.ocr import build_ocr_adapter
from app.services.ocr.base import OcrAdapter


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
    text: str


@dataclass
class DocumentExtractionResult:
    raw_text: str
    metadata: dict


class PDFTextExtractor:
    def __init__(self, ocr_backend: OcrAdapter | None = None):
        self.ocr_backend = ocr_backend or build_ocr_adapter()

    def extract(self, pdf_path: Path) -> DocumentExtractionResult:
        doc = fitz.open(pdf_path)
        pages: list[PageExtractionResult] = []
        try:
            for index, page in enumerate(doc, start=1):
                page_result = self._extract_page(page, index)
                if page_result.needs_ocr:
                    page_result = self._ocr_page(page, page_result)
                pages.append(page_result)
        finally:
            doc.close()

        combined_text = "\n\n".join(page.text.strip() for page in pages if page.text.strip()).strip()
        metadata = {
            "page_count": len(pages),
            "used_ocr_pages": [page.page_number for page in pages if page.used_ocr],
            "pages": [
                {
                    key: value
                    for key, value in asdict(page).items()
                    if key != "text"
                }
                for page in pages
            ],
        }
        return DocumentExtractionResult(raw_text=combined_text, metadata=metadata)

    def _extract_page(self, page: fitz.Page, page_number: int) -> PageExtractionResult:
        blocks = page.get_text("blocks")
        text_blocks = [block for block in blocks if len(block) >= 5 and str(block[4]).strip()]
        ordered_text = self._join_blocks(text_blocks, page.rect.width)
        page_text = ordered_text.strip()
        char_count = len(page_text)
        alpha_chars = sum(char.isalpha() for char in page_text)
        visible_chars = sum(not char.isspace() for char in page_text)
        alpha_ratio = alpha_chars / visible_chars if visible_chars else 0.0

        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        continuous_lines = sum(1 for line in lines if len(line) >= 50)
        continuous_line_ratio = continuous_lines / len(lines) if lines else 0.0

        suspected_double_column = self._has_double_column(text_blocks, page.rect.width)
        image_count = len(page.get_images(full=True))

        reasons: list[str] = []
        if char_count < settings.ocr_min_chars_per_page:
            reasons.append("low_text_volume")
        if alpha_ratio < settings.ocr_min_alpha_ratio:
            reasons.append("low_alpha_ratio")
        if continuous_line_ratio < 0.25 and char_count < settings.ocr_min_average_chars_per_page:
            reasons.append("fragmented_text")
        if image_count > 0 and char_count < settings.ocr_min_average_chars_per_page:
            reasons.append("image_heavy_page")

        return PageExtractionResult(
            page_number=page_number,
            char_count=char_count,
            alpha_ratio=round(alpha_ratio, 3),
            continuous_line_ratio=round(continuous_line_ratio, 3),
            image_count=image_count,
            suspected_double_column=suspected_double_column,
            needs_ocr=bool(reasons),
            used_ocr=False,
            reasons=reasons,
            ocr_metadata=None,
            text=page_text,
        )

    def _ocr_page(self, page: fitz.Page, result: PageExtractionResult) -> PageExtractionResult:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        recognition = self.ocr_backend.recognize_page(
            image_bytes=pixmap.tobytes("png"),
            page_number=result.page_number,
            hints={
                "reasons": result.reasons,
                "suspected_double_column": result.suspected_double_column,
                "image_count": result.image_count,
            },
        )
        ocr_text = recognition.text.strip()
        return PageExtractionResult(
            page_number=result.page_number,
            char_count=len(ocr_text),
            alpha_ratio=result.alpha_ratio,
            continuous_line_ratio=result.continuous_line_ratio,
            image_count=result.image_count,
            suspected_double_column=result.suspected_double_column,
            needs_ocr=result.needs_ocr,
            used_ocr=True,
            reasons=result.reasons,
            ocr_metadata=recognition.metadata,
            text=ocr_text,
        )

    def _join_blocks(self, blocks: list[tuple], page_width: float) -> str:
        if not blocks:
            return ""

        if self._has_double_column(blocks, page_width):
            midpoint = page_width / 2
            left = sorted((block for block in blocks if block[0] < midpoint), key=lambda item: (item[1], item[0]))
            right = sorted((block for block in blocks if block[0] >= midpoint), key=lambda item: (item[1], item[0]))
            ordered_blocks = [*left, *right]
        else:
            ordered_blocks = sorted(blocks, key=lambda item: (item[1], item[0]))

        return "\n".join(str(block[4]).strip() for block in ordered_blocks if str(block[4]).strip())

    def _has_double_column(self, blocks: list[tuple], page_width: float) -> bool:
        if len(blocks) < 4:
            return False
        left_count = sum(1 for block in blocks if block[0] < page_width * 0.45)
        right_count = sum(1 for block in blocks if block[0] > page_width * 0.55)
        return left_count >= 2 and right_count >= 2
