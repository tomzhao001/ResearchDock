from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.model_cache import configure_model_cache_env, model_cache_metadata
from app.services.document_extraction import (
    ExtractedBlock,
    ExtractedDocument,
    ExtractedPage,
    ExtractedPicture,
    ExtractedTable,
)


def _label_name(item: Any) -> str | None:
    label = getattr(item, "label", None)
    name = getattr(label, "name", None)
    if name:
        return str(name).lower()
    if label is not None:
        return str(label).lower()
    return None


def _page_number(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if isinstance(prov, list) and prov:
        page_no = getattr(prov[0], "page_no", None)
        if page_no is not None:
            return int(page_no)
    return None


def _bbox(item: Any) -> dict | None:
    prov = getattr(item, "prov", None)
    if not isinstance(prov, list) or not prov:
        return None
    bbox = getattr(prov[0], "bbox", None)
    if bbox is None:
        return None
    result: dict[str, float] = {}
    for source, target in (("l", "x0"), ("t", "y0"), ("r", "x1"), ("b", "y1"), ("x0", "x0"), ("y0", "y0"), ("x1", "x1"), ("y1", "y1")):
        value = getattr(bbox, source, None)
        if value is not None and target not in result:
            result[target] = float(value)
    return result or None


def _provenance(item: Any) -> dict | None:
    prov = getattr(item, "prov", None)
    if not isinstance(prov, list) or not prov:
        return None
    first = prov[0]
    payload: dict[str, Any] = {}
    page_no = getattr(first, "page_no", None)
    if page_no is not None:
        payload["page_number"] = int(page_no)
    bbox = _bbox(item)
    if bbox:
        payload["bbox"] = bbox
    return payload or None


def _block_type(label: str | None) -> str:
    normalized = (label or "").lower()
    if "section" in normalized or "heading" in normalized or "title" in normalized:
        return "heading"
    if "table" in normalized:
        return "table_like"
    if "picture" in normalized or "figure" in normalized:
        return "figure_caption"
    if "list" in normalized:
        return "list"
    return "paragraph"


def _extract_caption(item: Any, doc: Any) -> str | None:
    caption_text = getattr(item, "caption_text", None)
    if callable(caption_text):
        try:
            caption = caption_text(doc)
            if caption:
                return str(caption).strip()
        except Exception:
            return None
    caption = getattr(item, "caption", None)
    return str(caption).strip() if caption else None


def _picture_image_bytes(item: Any, doc: Any) -> bytes | None:
    get_image = getattr(item, "get_image", None)
    if not callable(get_image):
        return None
    try:
        image = get_image(doc=doc)
    except TypeError:
        image = get_image(doc)
    except Exception:
        return None
    if image is None:
        return None
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return None


class DoclingDocumentExtractor:
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        started = time.monotonic()
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, RapidOcrOptions, TesseractCliOcrOptions, TesseractOcrOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise RuntimeError("Docling is not installed. Install backend requirements before running PDF ingest.") from exc

        cache_paths = configure_model_cache_env()
        pipeline_options = PdfPipelineOptions()
        if hasattr(pipeline_options, "artifacts_path"):
            pipeline_options.artifacts_path = str(cache_paths.docling)
        pipeline_options.do_ocr = settings.docling_do_ocr
        pipeline_options.do_table_structure = settings.docling_do_table_structure
        pipeline_options.generate_picture_images = settings.docling_generate_picture_images
        pipeline_options.images_scale = float(settings.docling_images_scale)
        if settings.docling_document_timeout_seconds > 0:
            pipeline_options.document_timeout = float(settings.docling_document_timeout_seconds)

        ocr_engine = (settings.docling_ocr_engine or "").strip().lower()
        languages = [item.strip() for item in (settings.docling_ocr_languages or "").split(",") if item.strip()]
        if ocr_engine == "easyocr":
            pipeline_options.ocr_options = EasyOcrOptions(lang=languages or ["ch_sim", "en"])
        elif ocr_engine == "rapidocr":
            pipeline_options.ocr_options = RapidOcrOptions(lang=languages or ["chinese", "english"])
        elif ocr_engine == "tesserocr":
            pipeline_options.ocr_options = TesseractOcrOptions(lang=languages or ["chi_sim", "eng"])
        elif ocr_engine == "tesseract":
            pipeline_options.ocr_options = TesseractCliOcrOptions(lang=languages or ["chi_sim", "eng"])
        if pipeline_options.ocr_options is not None and hasattr(pipeline_options.ocr_options, "force_full_page_ocr"):
            pipeline_options.ocr_options.force_full_page_ocr = settings.docling_force_full_page_ocr

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        conversion = converter.convert(str(pdf_path))
        doc = conversion.document

        markdown_text = str(doc.export_to_markdown()).strip()
        try:
            docling_json = doc.export_to_dict()
        except Exception:
            docling_json = None

        pages = self._extract_pages(doc, docling_json)
        blocks = self._extract_blocks(doc)
        tables = self._extract_tables(doc)
        pictures = self._extract_pictures(doc)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        metadata = {
            "engine": "docling",
            "docling_do_ocr": settings.docling_do_ocr,
            "docling_do_table_structure": settings.docling_do_table_structure,
            "docling_ocr_engine": settings.docling_ocr_engine,
            "docling_ocr_languages": languages,
            "docling_force_full_page_ocr": settings.docling_force_full_page_ocr,
            "docling_generate_picture_images": settings.docling_generate_picture_images,
            "docling_images_scale": settings.docling_images_scale,
            "elapsed_ms": elapsed_ms,
            "model_cache": model_cache_metadata(),
        }
        return ExtractedDocument(
            markdown_text=markdown_text,
            metadata=metadata,
            pages=pages,
            blocks=blocks,
            tables=tables,
            pictures=pictures,
            docling_json=docling_json if isinstance(docling_json, dict) else None,
        )

    def _extract_pages(self, doc: Any, docling_json: Any) -> list[ExtractedPage]:
        pages_payload = docling_json.get("pages") if isinstance(docling_json, dict) else None
        pages: list[ExtractedPage] = []
        if isinstance(pages_payload, dict):
            iterable = pages_payload.items()
        elif isinstance(pages_payload, list):
            iterable = enumerate(pages_payload, start=1)
        else:
            iterable = []
        for key, page in iterable:
            if not isinstance(page, dict):
                continue
            page_number = int(page.get("page_no") or page.get("page_number") or key)
            size = page.get("size") if isinstance(page.get("size"), dict) else {}
            pages.append(
                ExtractedPage(
                    page_number=page_number,
                    width=float(size.get("width")) if size.get("width") is not None else None,
                    height=float(size.get("height")) if size.get("height") is not None else None,
                    metadata=page,
                )
            )
        if pages:
            return pages
        page_count = max((_page_number(item) or 0 for item, _level in doc.iterate_items()), default=0)
        return [ExtractedPage(page_number=index) for index in range(1, page_count + 1)]

    def _extract_blocks(self, doc: Any) -> list[ExtractedBlock]:
        blocks: list[ExtractedBlock] = []
        section_stack: dict[int, str] = {}
        for item, level in doc.iterate_items():
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            label = _label_name(item)
            block_type = _block_type(label)
            heading_level = int(level) if block_type == "heading" and str(level).isdigit() else (int(level) if block_type == "heading" else None)
            if block_type == "heading":
                heading_level = heading_level or 1
                section_stack = {key: value for key, value in section_stack.items() if key < heading_level}
                section_stack[heading_level] = text
                continue
            section_path = " > ".join(section_stack[key] for key in sorted(section_stack)) or None
            blocks.append(
                ExtractedBlock(
                    block_index=len(blocks),
                    text=text,
                    block_type=block_type,
                    page_number=_page_number(item),
                    docling_label=label,
                    heading_level=max(section_stack.keys(), default=1) if section_stack else 1,
                    section_path=section_path,
                    bbox=_bbox(item),
                    provenance=_provenance(item),
                )
            )
        return blocks

    def _extract_tables(self, doc: Any) -> list[ExtractedTable]:
        tables: list[ExtractedTable] = []
        for index, table in enumerate(getattr(doc, "tables", []) or []):
            try:
                markdown = table.export_to_markdown(doc=doc)
            except TypeError:
                markdown = table.export_to_markdown()
            except Exception:
                markdown = None
            data = None
            try:
                dataframe = table.export_to_dataframe(doc=doc)
                data = dataframe.to_dict(orient="records")
            except Exception:
                data = None
            page = _page_number(table)
            tables.append(
                ExtractedTable(
                    table_index=index,
                    caption=_extract_caption(table, doc),
                    markdown=str(markdown).strip() if markdown else None,
                    data=data,
                    page_from=page,
                    page_to=page,
                    bbox=_bbox(table),
                    metadata={"docling_label": _label_name(table)},
                )
            )
        return tables

    def _extract_pictures(self, doc: Any) -> list[ExtractedPicture]:
        pictures: list[ExtractedPicture] = []
        for index, picture in enumerate(getattr(doc, "pictures", []) or []):
            pictures.append(
                ExtractedPicture(
                    picture_index=index,
                    caption=_extract_caption(picture, doc),
                    page_number=_page_number(picture),
                    bbox=_bbox(picture),
                    image_bytes=_picture_image_bytes(picture, doc),
                    metadata={"docling_label": _label_name(picture), "provenance": _provenance(picture)},
                )
            )
        return pictures


def build_document_extractor() -> DoclingDocumentExtractor:
    if settings.document_extractor != "docling":
        raise RuntimeError(f"Unsupported document extractor: {settings.document_extractor}")
    return DoclingDocumentExtractor()
