from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class PictureDescriptionRequest:
    image_bytes: bytes | None = None
    image_url: str | None = None
    caption: str | None = None
    page_number: int | None = None
    bbox: dict | None = None
    context: str | None = None


@dataclass
class PictureDescriptionResult:
    description: str | None
    model_name: str | None = None
    prompt_version: str | None = None
    usage: dict | None = None
    raw_response: dict | None = None
    error: str | None = None


class PictureDescriptionAdapter(Protocol):
    def describe(self, request: PictureDescriptionRequest) -> PictureDescriptionResult:
        ...


class NoopPictureDescriptionAdapter:
    def describe(self, request: PictureDescriptionRequest) -> PictureDescriptionResult:
        return PictureDescriptionResult(description=None, error="picture VLM is not configured")
