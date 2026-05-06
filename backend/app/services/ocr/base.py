from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class OcrRecognitionResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class OcrAdapter(Protocol):
    def recognize_page(
        self,
        *,
        image_bytes: bytes,
        page_number: int,
        hints: dict[str, Any] | None = None,
    ) -> OcrRecognitionResult: ...
