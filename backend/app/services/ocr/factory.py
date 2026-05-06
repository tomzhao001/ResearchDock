from app.config import settings
from app.services.ocr.base import OcrAdapter
from app.services.ocr.glm_ocr import GlmOcrAdapter


def build_ocr_adapter() -> OcrAdapter:
    provider = settings.ocr_provider.strip().lower()
    if provider == "glm_ocr":
        return GlmOcrAdapter(
            api_key=settings.llm_ocr_api_key,
            api_url=settings.llm_ocr_base_url,
            model=settings.llm_ocr_model,
            timeout_seconds=settings.llm_ocr_timeout_seconds,
            max_retries=settings.llm_ocr_max_retries,
            retry_backoff_seconds=settings.llm_ocr_retry_backoff_seconds,
            verify_ssl=settings.llm_ocr_verify_ssl,
        )
    raise RuntimeError(f"Unsupported OCR provider: {settings.ocr_provider}")
