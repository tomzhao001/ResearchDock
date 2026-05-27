from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class OcrEscalationRequest:
    pdf_path: Path
    page_number: int
    bbox: dict[str, float] | None = None
    original_text: str = ""


@dataclass(frozen=True)
class OcrEscalationResult:
    text: str | None
    provider: str
    should_replace: bool = False
    confidence: float | None = None
    model_name: str | None = None
    reason: str | None = None
    usage: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class OcrEscalationStats:
    escalation_enabled: bool
    provider: str | None = None
    available: bool = False
    attempted_block_count: int = 0
    accepted_block_count: int = 0
    skipped_block_count: int = 0
    error_block_count: int = 0
    disabled_reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


class OcrEscalationAdapter(Protocol):
    provider_name: str

    def available(self) -> bool:
        ...

    def recognize(self, request: OcrEscalationRequest) -> OcrEscalationResult:
        ...


class NoopOcrEscalationAdapter:
    provider_name = "noop"

    def available(self) -> bool:
        return False

    def recognize(self, request: OcrEscalationRequest) -> OcrEscalationResult:
        return OcrEscalationResult(
            text=None,
            provider=self.provider_name,
            should_replace=False,
            reason="ocr_escalation_not_configured",
            error="OCR escalation is not configured",
        )
