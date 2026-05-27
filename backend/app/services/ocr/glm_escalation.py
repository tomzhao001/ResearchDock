from __future__ import annotations

import time

import httpx

from app.config import settings
from app.services.http_clients import get_shared_http_client
from app.services.ocr.escalation_base import OcrEscalationRequest, OcrEscalationResult
from app.services.ocr.pdf_region import image_bytes_to_data_url, render_pdf_region_png

_DEFAULT_PROMPT = """你是一个严格忠实转写的 OCR 修复器。请只根据图片区域输出原文文本。
要求：
1. 只输出识别后的纯文本，不要解释，不要加引号。
2. 保留大小写、数字、单位、标点、换行和特殊符号。
3. 不要根据医学常识、上下文或数学推理改写原文。
4. 如果原文看起来像写错了，也不要帮忙修正成你认为正确的内容。
5. 若有不确定字符，请输出你视觉上最接近的结果，而不是猜测。
"""


class GlmOcrEscalationAdapter:
    provider_name = "glm_ocr"

    def __init__(self) -> None:
        self.base_url = self._build_chat_completions_url((settings.glm_ocr_base_url or "").strip() or settings.glm_base_url)
        self.api_key = (settings.glm_ocr_api_key or "").strip() or settings.glm_api_key.strip()
        self.model = (
            (settings.glm_ocr_model or "").strip()
            or (settings.glm_model or "").strip()
            or (settings.picture_vlm_model or "").strip()
        )
        self.timeout_seconds = max(int(settings.glm_ocr_timeout_seconds), 1)
        self.max_retries = max(int(settings.glm_ocr_max_retries), 0)
        self.verify_ssl = bool(settings.glm_ocr_verify_ssl)

    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def recognize(self, request: OcrEscalationRequest) -> OcrEscalationResult:
        if not self.available():
            return OcrEscalationResult(
                text=None,
                provider=self.provider_name,
                should_replace=False,
                model_name=self.model or None,
                reason="glm_ocr_not_configured",
                error="GLM OCR API is not configured",
            )
        image_bytes = render_pdf_region_png(request.pdf_path, page_number=request.page_number, bbox=request.bbox)
        image_url = image_bytes_to_data_url(image_bytes)
        if not image_url:
            return OcrEscalationResult(
                text=None,
                provider=self.provider_name,
                should_replace=False,
                model_name=self.model,
                reason="pdf_region_render_failed",
                error="Failed to render PDF region",
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self._build_prompt(request.original_text)},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                client = get_shared_http_client(name="glm_ocr", verify_ssl=self.verify_ssl)
                response = client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(30.0, connect=30.0, read=float(self.timeout_seconds), write=float(self.timeout_seconds), pool=60.0),
                )
                response.raise_for_status()
                data = response.json()
                text = self._extract_text(data)
                return OcrEscalationResult(
                    text=text,
                    provider=self.provider_name,
                    should_replace=bool(text and text.strip()),
                    model_name=self.model,
                    reason="glm_ocr_candidate" if text else "glm_ocr_empty",
                    usage=data.get("usage") if isinstance(data, dict) else None,
                    raw_response=data if isinstance(data, dict) else None,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4))
        return OcrEscalationResult(
            text=None,
            provider=self.provider_name,
            should_replace=False,
            model_name=self.model,
            reason="glm_ocr_request_failed",
            error=last_error,
        )

    @staticmethod
    def _build_chat_completions_url(base_url: str) -> str:
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized:
            return ""
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _build_prompt(original_text: str) -> str:
        reference = str(original_text or "").strip()
        if not reference:
            return _DEFAULT_PROMPT
        return f"{_DEFAULT_PROMPT}\n已有 OCR 结果（仅供参考，不要盲从）：\n{reference}"

    @staticmethod
    def _extract_text(data: dict) -> str | None:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            texts = [str(item.get("text") or "").strip() for item in content if isinstance(item, dict)]
            merged = "\n".join(text for text in texts if text).strip()
            return merged or None
        return None
