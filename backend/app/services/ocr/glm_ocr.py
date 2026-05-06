import base64
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings
from app.services.ocr.base import OcrRecognitionResult

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class GlmOcrAdapter:
    api_key: str = ""
    api_url: str = settings.llm_ocr_base_url
    model: str = settings.llm_ocr_model
    timeout_seconds: int = settings.llm_ocr_timeout_seconds
    max_retries: int = settings.llm_ocr_max_retries
    retry_backoff_seconds: float = settings.llm_ocr_retry_backoff_seconds
    verify_ssl: bool = settings.llm_ocr_verify_ssl
    client: httpx.Client | None = field(default=None, repr=False)

    def recognize_page(
        self,
        *,
        image_bytes: bytes,
        page_number: int,
        hints: dict[str, Any] | None = None,
    ) -> OcrRecognitionResult:
        if not self.api_key.strip():
            raise RuntimeError("LLM_OCR_API_KEY is required for GLM-OCR fallback")

        payload = {
            "model": self.model,
            "file": self._to_data_url(image_bytes),
            "request_id": f"page-{page_number}",
        }

        response_json = self._post_with_retry(payload)
        text = self._normalize_markdown(response_json.get("md_results", ""))
        metadata = {
            "provider": "glm_ocr",
            "request_id": response_json.get("request_id"),
            "response_id": response_json.get("id"),
            "model": response_json.get("model", self.model),
            "usage": response_json.get("usage"),
            "page_number": page_number,
            "hints": hints or {},
        }
        return OcrRecognitionResult(text=text, metadata=metadata)

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        owned_client = self.client is None
        client = self.client or httpx.Client(timeout=self.timeout_seconds, verify=self.verify_ssl)
        try:
            for attempt in range(self.max_retries + 1):
                response = client.post(self.api_url, headers=headers, json=payload)
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                response.raise_for_status()
                return response.json()
        finally:
            if owned_client:
                client.close()

        raise RuntimeError("GLM-OCR request failed without a response")

    @staticmethod
    def _to_data_url(image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        normalized = text.replace("\r\n", "\n").strip()
        return normalized
