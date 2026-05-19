from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ExtractedPage:
    page_number: int
    text: str = ""
    width: float | None = None
    height: float | None = None
    metadata: dict | None = None


@dataclass
class ExtractedBlock:
    block_index: int
    text: str
    block_type: str = "paragraph"
    page_number: int | None = None
    docling_label: str | None = None
    heading_level: int | None = None
    section_path: str | None = None
    bbox: dict | None = None
    provenance: dict | None = None
    metadata: dict | None = None


@dataclass
class ExtractedTable:
    table_index: int
    caption: str | None = None
    markdown: str | None = None
    data: dict | list | None = None
    page_from: int | None = None
    page_to: int | None = None
    bbox: dict | None = None
    metadata: dict | None = None


@dataclass
class ExtractedPicture:
    picture_index: int
    caption: str | None = None
    description: str | None = None
    description_model: str | None = None
    description_prompt_version: str | None = None
    page_number: int | None = None
    bbox: dict | None = None
    image_asset_path: str | None = None
    image_bytes: bytes | None = field(default=None, repr=False)
    metadata: dict | None = None


@dataclass
class ExtractedDocument:
    markdown_text: str
    metadata: dict
    pages: list[ExtractedPage] = field(default_factory=list)
    blocks: list[ExtractedBlock] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    pictures: list[ExtractedPicture] = field(default_factory=list)
    docling_json: dict | None = None

    @property
    def raw_text(self) -> str:
        return self.markdown_text

    def extraction_metadata(self) -> dict:
        payload = dict(self.metadata or {})
        payload.setdefault("engine", "docling")
        payload["page_count"] = len(self.pages)
        payload["block_count"] = len(self.blocks)
        payload["table_count"] = len(self.tables)
        payload["picture_count"] = len(self.pictures)
        return payload


class DocumentExtractor(Protocol):
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        ...


def dataclass_to_json(value) -> dict:
    return asdict(value)
