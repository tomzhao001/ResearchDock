from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.document_extraction import ExtractedDocument
from app.services.ocr.quality import OcrRoutingDecision
from app.services.ocr.rules import apply_custom_ocr_rules
from app.services.ocr.spell_correction import SymSpellCorrector
from app.services.ocr.text_normalization import normalize_ocr_text


@dataclass
class OcrPostprocessStats:
    postprocess_enabled: bool
    normalization_applied_count: int = 0
    symspell_correction_count: int = 0
    medical_term_correction_count: int = 0
    rule_repair_count: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


class OcrPostprocessor:
    def __init__(self, *, spell_corrector: SymSpellCorrector | None = None) -> None:
        self.spell_corrector = spell_corrector or SymSpellCorrector()

    def process(self, *, document: ExtractedDocument, pdf_path: Path, ocr_strategy: OcrRoutingDecision | None = None) -> ExtractedDocument:
        _ = pdf_path
        _ = ocr_strategy
        stats = OcrPostprocessStats(postprocess_enabled=bool(settings.ocr_postprocess_enabled))
        if not settings.ocr_postprocess_enabled:
            document.metadata["ocr_postprocess"] = stats.to_metadata()
            return document

        document.markdown_text = self._process_text(document.markdown_text, stats=stats)

        for page in document.pages:
            page.text = self._process_text(page.text, stats=stats)
            page.metadata = dict(page.metadata or {})

        for block in document.blocks:
            block.metadata = dict(block.metadata or {})
            block.text = self._process_text(block.text, stats=stats)

        for table in document.tables:
            table.metadata = dict(table.metadata or {})
            if table.caption:
                table.caption = self._process_text(table.caption, stats=stats)
            if table.markdown:
                table.markdown = self._process_text(table.markdown, stats=stats)
            if isinstance(table.data, list):
                for row in table.data:
                    if not isinstance(row, dict):
                        continue
                    for key, value in list(row.items()):
                        if isinstance(value, str):
                            row[key] = self._process_text(value, stats=stats)

        document.metadata["ocr_postprocess"] = stats.to_metadata()
        return document

    def _process_text(self, text: str, *, stats: OcrPostprocessStats) -> str:
        normalized = normalize_ocr_text(text)
        updated = normalized.text
        if normalized.normalization_applied:
            stats.normalization_applied_count += 1

        corrected, correction_count = self.spell_corrector.correct_text(updated)
        if correction_count:
            stats.symspell_correction_count += correction_count
            stats.medical_term_correction_count += correction_count
        updated = corrected

        repaired = apply_custom_ocr_rules(updated)
        if repaired.repairs:
            stats.rule_repair_count += repaired.repairs
        return repaired.text


def build_ocr_postprocessor() -> OcrPostprocessor:
    return OcrPostprocessor()
