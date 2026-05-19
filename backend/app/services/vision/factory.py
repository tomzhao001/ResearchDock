from __future__ import annotations

from app.config import settings
from app.services.vision.base import NoopPictureDescriptionAdapter, PictureDescriptionAdapter
from app.services.vision.glm_vision import GlmVisionPictureDescriptionAdapter


def build_picture_description_adapter() -> PictureDescriptionAdapter:
    if not settings.docling_do_picture_description:
        return NoopPictureDescriptionAdapter()
    provider = (settings.picture_vlm_provider or "").strip().lower()
    if provider in {"glm_4_6v", "glm-4.6v", "glm"}:
        return GlmVisionPictureDescriptionAdapter()
    return NoopPictureDescriptionAdapter()
