from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.services import papers as legacy_papers
from app.services.document_extraction import ExtractedDocument
from app.services.ocr.escalation_base import (
    OcrEscalationAdapter,
    OcrEscalationRequest,
    OcrEscalationResult,
    OcrEscalationStats,
)
from app.services.ocr.factory import build_ocr_escalation_adapter
from app.services.ocr.quality import OcrRoutingDecision
from app.services.ocr.rules import looks_suspicious_ocr_text


class OcrEscalationProcessor:
    def __init__(self, *, adapter: OcrEscalationAdapter | None = None) -> None:
        self.adapter = adapter or build_ocr_escalation_adapter()

    def process(
        self,
        *,
        document: ExtractedDocument,
        pdf_path: Path,
        ocr_strategy: OcrRoutingDecision | None = None,
    ) -> ExtractedDocument:
        enabled = bool(settings.ocr_escalation_enabled)
        if ocr_strategy is not None:
            enabled = enabled and bool(ocr_strategy.enable_escalation)

        stats = OcrEscalationStats(
            escalation_enabled=enabled,
            provider=getattr(self.adapter, "provider_name", None),
            available=self.adapter.available(),
        )

        if not enabled:
            stats.disabled_reason = "ocr_escalation_disabled"
            document.metadata["ocr_escalation"] = stats.to_metadata()
            return document
        if not stats.available:
            stats.disabled_reason = "ocr_escalation_provider_unavailable"
            document.metadata["ocr_escalation"] = stats.to_metadata()
            return document

        accepted_any = False
        for block in document.blocks:
            block.metadata = dict(block.metadata or {})
            if not block.page_number or not block.bbox or not looks_suspicious_ocr_text(block.text):
                continue

            stats.attempted_block_count += 1
            result = self.adapter.recognize(
                OcrEscalationRequest(
                    pdf_path=pdf_path,
                    page_number=block.page_number,
                    bbox=block.bbox,
                    original_text=block.text,
                )
            )
            accepted, decision_reason = self._accept_candidate(block.text, result)
            metadata = {
                "provider": result.provider,
                "model_name": result.model_name,
                "original_text": block.text,
                "candidate_text": result.text,
                "accepted": accepted,
                "reason": decision_reason,
                "confidence": result.confidence,
                "usage": result.usage,
                "error": result.error,
            }
            if result.raw_response is not None:
                metadata["raw_response"] = result.raw_response
            block.metadata["ocr_escalation"] = metadata

            if accepted and result.text is not None:
                block.text = result.text.strip()
                stats.accepted_block_count += 1
                accepted_any = True
            elif result.error:
                stats.error_block_count += 1
            else:
                stats.skipped_block_count += 1

        if accepted_any:
            document.markdown_text = legacy_papers.render_document_text(document)
        document.metadata["ocr_escalation"] = stats.to_metadata()
        return document

    @staticmethod
    def _accept_candidate(original_text: str, result: OcrEscalationResult) -> tuple[bool, str]:
        original = str(original_text or "").strip()
        candidate = str(result.text or "").strip()
        if not result.should_replace:
            return False, result.reason or "provider_declined"
        if not candidate:
            return False, "empty_candidate"
        if candidate == original:
            return False, "candidate_same_as_original"
        if len(candidate) > max(len(original) * 3, len(original) + 40):
            return False, "candidate_too_long"
        if len(candidate) < max(1, len(original) // 4):
            return False, "candidate_too_short"
        if looks_suspicious_ocr_text(candidate) and not looks_suspicious_ocr_text(original):
            return False, "candidate_still_suspicious"
        return True, result.reason or "candidate_accepted"


def build_ocr_escalation_processor(*, adapter: OcrEscalationAdapter | None = None) -> OcrEscalationProcessor:
    return OcrEscalationProcessor(adapter=adapter)
