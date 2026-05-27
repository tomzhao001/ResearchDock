from __future__ import annotations

from app.config import settings
from app.services.ocr.escalation_base import OcrEscalationAdapter
from app.services.ocr.glm_escalation import GlmOcrEscalationAdapter
from app.services.ocr.noop_escalation import NoopOcrEscalationAdapter


def build_ocr_escalation_adapter() -> OcrEscalationAdapter:
    if not settings.ocr_escalation_enabled:
        return NoopOcrEscalationAdapter()
    provider = (settings.ocr_escalation_provider or "").strip().lower()
    if provider in {"glm", "glm_ocr", "glm-ocr"}:
        return GlmOcrEscalationAdapter()
    return NoopOcrEscalationAdapter()
