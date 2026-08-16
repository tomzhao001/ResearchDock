from __future__ import annotations

import io
import importlib
import shutil
import time
import urllib.request
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
from app.services.ocr.quality import OcrRoutingDecision

_RAPIDOCR_MODELSCOPE_RELEASE = "v3.8.0"
_RAPIDOCR_MODELSCOPE_BASE_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve"
_RAPIDOCR_V5_COMMON_ARTIFACTS = {
    "det_model_path": "onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
    "cls_model_path": "onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
}
_RAPIDOCR_V5_LANGUAGE_ARTIFACTS = {
    "chinese": {
        "rec_model_path": "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
        "rec_keys_path": "paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt",
    },
    "english": {
        "rec_model_path": "onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx",
        "rec_keys_path": "paddle/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile/ppocrv5_en_dict.txt",
    },
}

_CONVERTER_CACHE: dict[tuple, object] = {}


def _converter_cache_key(*, ocr_engine: str, force_full_page_ocr: bool, languages: tuple[str, ...]) -> tuple:
    return (
        ocr_engine,
        force_full_page_ocr,
        languages,
        bool(settings.docling_do_ocr),
        bool(settings.docling_do_table_structure),
        bool(settings.docling_generate_picture_images),
        float(settings.docling_images_scale),
    )


def _get_or_create_converter(key: tuple, pipeline_options) -> object:
    converter = _CONVERTER_CACHE.get(key)
    if converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        _CONVERTER_CACHE[key] = converter
    return converter


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


def _bbox_distance(lhs: dict | None, rhs: dict | None) -> float:
    if not lhs or not rhs:
        return float("inf")
    try:
        left_cx = (float(lhs["x0"]) + float(lhs["x1"])) / 2
        left_cy = (float(lhs["y0"]) + float(lhs["y1"])) / 2
        right_cx = (float(rhs["x0"]) + float(rhs["x1"])) / 2
        right_cy = (float(rhs["y0"]) + float(rhs["y1"])) / 2
    except Exception:
        return float("inf")
    return abs(left_cx - right_cx) + abs(left_cy - right_cy)


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


def _resolve_rapidocr_language(languages: list[str] | None) -> str:
    normalized_languages = {str(language).strip().lower() for language in (languages or []) if str(language).strip()}
    if {"ch", "zh", "chinese", "ch_sim", "chi_sim"} & normalized_languages:
        return "chinese"
    if {"en", "english"} & normalized_languages:
        return "english"
    return "chinese"


def _rapidocr_artifact_url(relative_path: str) -> str:
    return f"{_RAPIDOCR_MODELSCOPE_BASE_URL}/{_RAPIDOCR_MODELSCOPE_RELEASE}/{relative_path}"


def _build_rapidocr_v5_manifest(artifacts_path: Path, languages: list[str] | None) -> dict[str, tuple[Path, str]]:
    model_root = artifacts_path / "RapidOcr"
    manifest: dict[str, tuple[Path, str]] = {}
    for option_name, relative_path in _RAPIDOCR_V5_COMMON_ARTIFACTS.items():
        manifest[option_name] = (model_root / relative_path, _rapidocr_artifact_url(relative_path))

    language_key = _resolve_rapidocr_language(languages)
    for option_name, relative_path in _RAPIDOCR_V5_LANGUAGE_ARTIFACTS[language_key].items():
        manifest[option_name] = (model_root / relative_path, _rapidocr_artifact_url(relative_path))
    return manifest


def _build_rapidocr_v5_params(languages: list[str] | None) -> dict[str, Any]:
    typings = importlib.import_module("rapidocr.utils.typings")
    LangCls = getattr(typings, "LangCls")
    LangDet = getattr(typings, "LangDet")
    LangRec = getattr(typings, "LangRec")
    OCRVersion = getattr(typings, "OCRVersion")

    language_key = _resolve_rapidocr_language(languages)
    rec_lang = LangRec.EN if language_key == "english" else LangRec.CH
    return {
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Det.lang_type": LangDet.CH,
        "Cls.lang_type": LangCls.CH,
        "Rec.lang_type": rec_lang,
    }


def _layout_model_repo_folder() -> str:
    from docling.datamodel.pipeline_options import LayoutOptions

    return LayoutOptions().model_spec.model_repo_folder


def _table_model_repo_folder() -> str:
    from docling.models.stages.table_structure.table_structure_model import TableStructureModel

    return TableStructureModel._model_repo_folder


def _table_model_relative_path() -> str:
    from docling.models.stages.table_structure.table_structure_model import TableStructureModel

    return TableStructureModel._model_path


def _table_model_mode_dir() -> str:
    from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions

    mode = TableStructureOptions().mode
    return "fast" if mode == TableFormerMode.FAST else "accurate"


def _docling_layout_artifact_marker(artifacts_path: Path) -> Path:
    return artifacts_path / _layout_model_repo_folder() / "model.safetensors"


def _docling_table_artifact_marker(artifacts_path: Path) -> Path:
    return (
        artifacts_path
        / _table_model_repo_folder()
        / _table_model_relative_path()
        / _table_model_mode_dir()
        / "tm_config.json"
    )


def _docling_table_artifact_markers(artifacts_path: Path) -> tuple[Path, ...]:
    mode_dir = _table_model_mode_dir()
    relative_path = Path(_table_model_relative_path())
    return (
        artifacts_path / _table_model_repo_folder() / relative_path / mode_dir / "tm_config.json",
        artifacts_path / relative_path / mode_dir / "tm_config.json",
    )


def _docling_table_artifacts_present(artifacts_path: Path) -> bool:
    return any(marker.is_file() for marker in _docling_table_artifact_markers(artifacts_path))


def _prepare_docling_standard_artifacts(artifacts_path: Path) -> None:
    need_layout = not _docling_layout_artifact_marker(artifacts_path).is_file()
    need_table = settings.docling_do_table_structure and not _docling_table_artifacts_present(artifacts_path)
    if not need_layout and not need_table:
        return

    if not settings.model_cache_auto_download:
        missing = []
        if need_layout:
            missing.append(str(_docling_layout_artifact_marker(artifacts_path)))
        if need_table:
            missing.append(str(_docling_table_artifact_marker(artifacts_path)))
        raise FileNotFoundError(
            "Missing Docling model artifacts: "
            + ", ".join(missing)
            + ". Pre-download with "
            f"'docling-tools models download layout tableformer -o {artifacts_path}' "
            "or set MODEL_CACHE_AUTO_DOWNLOAD=true."
        )

    from docling.utils.model_downloader import download_models

    download_models(
        output_dir=artifacts_path,
        with_layout=need_layout,
        with_tableformer=need_table,
        with_tableformer_v2=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
        with_easyocr=False,
    )


def _resolve_rapidocr_artifact(local_path: Path, url: str) -> Path:
    if local_path.exists():
        return local_path

    if not settings.model_cache_auto_download:
        raise FileNotFoundError(
            f"Missing RapidOCR artifact: {local_path}. "
            "Pre-download RapidOCR models into the Docling artifacts directory "
            "or set MODEL_CACHE_AUTO_DOWNLOAD=true."
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response, temp_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    temp_path.replace(local_path)
    return local_path


class DoclingDocumentExtractor:
    def __init__(self, *, ocr_strategy: OcrRoutingDecision | None = None) -> None:
        self.ocr_strategy = ocr_strategy

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        started = time.monotonic()
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, RapidOcrOptions, TesseractCliOcrOptions, TesseractOcrOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise RuntimeError("Docling is not installed. Install backend requirements before running PDF ingest.") from exc

        cache_paths = configure_model_cache_env()
        _prepare_docling_standard_artifacts(cache_paths.docling)
        pipeline_options = PdfPipelineOptions()
        if hasattr(pipeline_options, "artifacts_path"):
            pipeline_options.artifacts_path = str(cache_paths.docling)
        pipeline_options.do_ocr = settings.docling_do_ocr
        pipeline_options.do_table_structure = settings.docling_do_table_structure
        pipeline_options.generate_picture_images = settings.docling_generate_picture_images
        pipeline_options.images_scale = float(settings.docling_images_scale)
        if settings.docling_document_timeout_seconds > 0:
            pipeline_options.document_timeout = float(settings.docling_document_timeout_seconds)

        ocr_engine = self._resolve_docling_ocr_engine()
        languages = [item.strip() for item in (settings.docling_ocr_languages or "").split(",") if item.strip()]
        if ocr_engine == "easyocr":
            pipeline_options.ocr_options = EasyOcrOptions(lang=languages or ["ch_sim", "en"])
        elif ocr_engine == "rapidocr":
            manifest = _build_rapidocr_v5_manifest(cache_paths.docling, languages or ["chinese", "english"])
            pipeline_options.ocr_options = RapidOcrOptions(
                lang=languages or ["chinese", "english"],
                det_model_path=str(_resolve_rapidocr_artifact(*manifest["det_model_path"])),
                cls_model_path=str(_resolve_rapidocr_artifact(*manifest["cls_model_path"])),
                rec_model_path=str(_resolve_rapidocr_artifact(*manifest["rec_model_path"])),
                rec_keys_path=str(_resolve_rapidocr_artifact(*manifest["rec_keys_path"])),
                rapidocr_params=_build_rapidocr_v5_params(languages or ["chinese", "english"]),
            )
        elif ocr_engine == "tesserocr":
            pipeline_options.ocr_options = TesseractOcrOptions(lang=languages or ["chi_sim", "eng"])
        elif ocr_engine == "tesseract":
            pipeline_options.ocr_options = TesseractCliOcrOptions(lang=languages or ["chi_sim", "eng"])
        if pipeline_options.ocr_options is not None and hasattr(pipeline_options.ocr_options, "force_full_page_ocr"):
            pipeline_options.ocr_options.force_full_page_ocr = self._resolve_force_full_page_ocr()

        key = _converter_cache_key(
            ocr_engine=ocr_engine,
            force_full_page_ocr=self._resolve_force_full_page_ocr(),
            languages=tuple(languages),
        )
        converter = _get_or_create_converter(key, pipeline_options)
        conversion = converter.convert(str(pdf_path))
        doc = conversion.document

        markdown_text = str(doc.export_to_markdown()).strip()
        try:
            docling_json = doc.export_to_dict()
        except Exception:
            docling_json = None

        context_rows = self._collect_context_rows(doc)
        pages = self._extract_pages(doc, docling_json)
        blocks = self._extract_blocks(context_rows)
        tables = self._extract_tables(doc, context_rows)
        pictures = self._extract_pictures(doc, context_rows)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        metadata = {
            "engine": "docling",
            "docling_do_ocr": settings.docling_do_ocr,
            "docling_do_table_structure": settings.docling_do_table_structure,
            "docling_ocr_engine": ocr_engine,
            "docling_ocr_languages": languages,
            "docling_force_full_page_ocr": self._resolve_force_full_page_ocr(),
            "docling_generate_picture_images": settings.docling_generate_picture_images,
            "docling_images_scale": settings.docling_images_scale,
            "elapsed_ms": elapsed_ms,
            "model_cache": model_cache_metadata(),
        }
        if self.ocr_strategy is not None:
            metadata["ocr_strategy"] = self.ocr_strategy.to_metadata()
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

    def _collect_context_rows(self, doc: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        section_stack: dict[int, str] = {}
        for reading_order, (item, level) in enumerate(doc.iterate_items()):
            label = _label_name(item)
            block_type = _block_type(label)
            text = str(getattr(item, "text", "") or "").strip()
            if not text and block_type in {"table_like", "figure_caption"}:
                text = _extract_caption(item, doc) or ""
            if not text and block_type not in {"table_like", "figure_caption"}:
                continue
            heading_level = int(level) if block_type == "heading" and str(level).isdigit() else None
            if block_type == "heading":
                heading_level = heading_level or 1
                section_stack = {key: value for key, value in section_stack.items() if key < heading_level}
                section_stack[heading_level] = text
                section_path = " > ".join(section_stack[key] for key in sorted(section_stack)) or None
            else:
                section_path = " > ".join(section_stack[key] for key in sorted(section_stack)) or None
                heading_level = max(section_stack.keys(), default=1) if section_stack else 1
            rows.append(
                {
                    "text": text,
                    "block_type": block_type,
                    "docling_label": label,
                    "page_number": _page_number(item),
                    "heading_level": heading_level,
                    "section_path": section_path,
                    "bbox": _bbox(item),
                    "provenance": _provenance(item),
                    "reading_order": reading_order,
                }
            )
        return rows

    def _extract_blocks(self, context_rows: list[dict[str, Any]]) -> list[ExtractedBlock]:
        blocks: list[ExtractedBlock] = []
        for row in context_rows:
            block_type = str(row.get("block_type") or "paragraph")
            if block_type in {"table_like", "figure_caption"}:
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            blocks.append(
                ExtractedBlock(
                    block_index=len(blocks),
                    text=text,
                    block_type=block_type,
                    page_number=row.get("page_number"),
                    docling_label=row.get("docling_label"),
                    heading_level=row.get("heading_level"),
                    section_path=row.get("section_path"),
                    reading_order=row.get("reading_order"),
                    bbox=row.get("bbox"),
                    provenance=row.get("provenance"),
                )
            )
        return blocks

    def _match_context_row(
        self,
        context_rows: list[dict[str, Any]],
        *,
        kind: str,
        page_number: int | None,
        bbox: dict | None,
        used_indexes: set[int],
    ) -> dict[str, Any] | None:
        preferred_types = {"table": {"table_like"}, "picture": {"figure_caption"}}.get(kind, set())
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        fallback: list[tuple[float, int, dict[str, Any]]] = []
        for index, row in enumerate(context_rows):
            if index in used_indexes:
                continue
            row_page = row.get("page_number")
            if page_number is not None and row_page is not None and int(row_page) != int(page_number):
                continue
            row_bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else None
            distance = _bbox_distance(bbox, row_bbox)
            record = (distance, index, row)
            if str(row.get("block_type") or "") in preferred_types:
                candidates.append(record)
            else:
                fallback.append(record)
        ordered = sorted(candidates or fallback, key=lambda item: (item[0], item[1]))
        if not ordered:
            return None
        _, index, row = ordered[0]
        used_indexes.add(index)
        return row

    def _extract_tables(self, doc: Any, context_rows: list[dict[str, Any]]) -> list[ExtractedTable]:
        tables: list[ExtractedTable] = []
        used_indexes: set[int] = set()
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
            bbox = _bbox(table)
            provenance = _provenance(table)
            matched = self._match_context_row(
                context_rows,
                kind="table",
                page_number=page,
                bbox=bbox,
                used_indexes=used_indexes,
            )
            matched_section_path = str(matched.get("section_path") or "").strip() if matched else ""
            matched_heading_level = int(matched.get("heading_level")) if matched and str(matched.get("heading_level") or "").isdigit() else None
            matched_reading_order = int(matched.get("reading_order")) if matched and str(matched.get("reading_order") or "").isdigit() else None
            tables.append(
                ExtractedTable(
                    table_index=index,
                    caption=_extract_caption(table, doc),
                    markdown=str(markdown).strip() if markdown else None,
                    data=data,
                    page_from=page,
                    page_to=page,
                    section_path=matched_section_path or None,
                    heading_level=matched_heading_level,
                    reading_order=matched_reading_order,
                    bbox=bbox,
                    provenance=provenance or (matched.get("provenance") if matched else None),
                    metadata={"docling_label": _label_name(table)},
                )
            )
        return tables

    def _extract_pictures(self, doc: Any, context_rows: list[dict[str, Any]] | None = None) -> list[ExtractedPicture]:
        context_rows = context_rows or []
        pictures: list[ExtractedPicture] = []
        used_indexes: set[int] = set()
        for index, picture in enumerate(getattr(doc, "pictures", []) or []):
            page_number = _page_number(picture)
            bbox = _bbox(picture)
            provenance = _provenance(picture)
            matched = self._match_context_row(
                context_rows,
                kind="picture",
                page_number=page_number,
                bbox=bbox,
                used_indexes=used_indexes,
            )
            matched_section_path = str(matched.get("section_path") or "").strip() if matched else ""
            matched_heading_level = int(matched.get("heading_level")) if matched and str(matched.get("heading_level") or "").isdigit() else None
            matched_reading_order = int(matched.get("reading_order")) if matched and str(matched.get("reading_order") or "").isdigit() else None
            pictures.append(
                ExtractedPicture(
                    picture_index=index,
                    caption=_extract_caption(picture, doc),
                    page_number=page_number,
                    section_path=matched_section_path or None,
                    heading_level=matched_heading_level,
                    reading_order=matched_reading_order,
                    bbox=bbox,
                    provenance=provenance or (matched.get("provenance") if matched else None),
                    image_bytes=_picture_image_bytes(picture, doc),
                    metadata={"docling_label": _label_name(picture), "provenance": _provenance(picture)},
                )
            )
        return pictures

    def _resolve_docling_ocr_engine(self) -> str:
        if self.ocr_strategy is not None:
            return str(self.ocr_strategy.docling_ocr_engine or "").strip().lower() or "rapidocr"
        return (settings.docling_ocr_engine or "").strip().lower()

    def _resolve_force_full_page_ocr(self) -> bool:
        if self.ocr_strategy is not None:
            return bool(self.ocr_strategy.force_full_page_ocr)
        return bool(settings.docling_force_full_page_ocr)


def build_document_extractor(*, ocr_strategy: OcrRoutingDecision | None = None) -> DoclingDocumentExtractor:
    if settings.document_extractor != "docling":
        raise RuntimeError(f"Unsupported document extractor: {settings.document_extractor}")
    return DoclingDocumentExtractor(ocr_strategy=ocr_strategy)
