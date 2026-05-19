from __future__ import annotations

import base64
import time

import httpx

from app.config import settings
from app.services.vision.base import PictureDescriptionRequest, PictureDescriptionResult


_DEFAULT_PROMPT = """请面向知识库检索和论文问答，简洁描述这张论文图片或图表。
如果是图表，请尽量提取：变量、坐标轴、分组、趋势、显著结论、图中文字。
如果信息不足，请说明可见内容，不要编造。"""


class GlmVisionPictureDescriptionAdapter:
    def __init__(self) -> None:
        self.base_url = settings.picture_vlm_base_url.rstrip("/")
        self.api_key = settings.picture_vlm_api_key.strip() or settings.glm_api_key.strip()
        self.model = settings.picture_vlm_model
        self.timeout_seconds = settings.picture_vlm_timeout_seconds
        self.max_retries = max(settings.picture_vlm_max_retries, 0)
        self.prompt_version = settings.picture_vlm_prompt_version

    def describe(self, request: PictureDescriptionRequest) -> PictureDescriptionResult:
        if not self.api_key:
            return PictureDescriptionResult(description=None, model_name=self.model, prompt_version=self.prompt_version, error="missing picture VLM API key")
        image_url = request.image_url or self._data_url(request.image_bytes)
        if not image_url:
            return PictureDescriptionResult(description=None, model_name=self.model, prompt_version=self.prompt_version, error="missing image input")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self._build_prompt(request)},
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(self.base_url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                description = self._extract_text(data)
                return PictureDescriptionResult(
                    description=description,
                    model_name=self.model,
                    prompt_version=self.prompt_version,
                    usage=data.get("usage") if isinstance(data, dict) else None,
                    raw_response=data if isinstance(data, dict) else None,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4))
        return PictureDescriptionResult(description=None, model_name=self.model, prompt_version=self.prompt_version, error=last_error)

    def _build_prompt(self, request: PictureDescriptionRequest) -> str:
        parts = [_DEFAULT_PROMPT]
        if request.caption:
            parts.append(f"图片/图表 caption：{request.caption}")
        if request.page_number is not None:
            parts.append(f"页码：{request.page_number}")
        if request.context:
            parts.append(f"邻近文本上下文：{request.context}")
        if request.bbox:
            parts.append(f"页面位置 bbox：{request.bbox}")
        return "\n".join(parts)

    @staticmethod
    def _data_url(image_bytes: bytes | None) -> str | None:
        if not image_bytes:
            return None
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _extract_text(data: dict) -> str | None:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [str(item.get("text") or "").strip() for item in content if isinstance(item, dict)]
            return "\n".join(text for text in texts if text) or None
        return None
