from __future__ import annotations

import sys
import types
from pathlib import Path

from PIL import Image

import pytest

from app.services.docling_extraction import (
    DoclingDocumentExtractor,
    _build_rapidocr_v5_manifest,
    _build_rapidocr_v5_params,
    _docling_layout_artifact_marker,
    _docling_table_artifact_marker,
    _docling_table_artifacts_present,
    _prepare_docling_standard_artifacts,
    _resolve_rapidocr_artifact,
)


class _FakeLabel:
    name = "PICTURE"


class _FakeBbox:
    x0 = 10
    y0 = 20
    x1 = 110
    y1 = 220


class _FakeProv:
    page_no = 2
    bbox = _FakeBbox()


class _FakePicture:
    label = _FakeLabel()
    caption = "Figure 1. Trend"
    prov = [_FakeProv()]

    def get_image(self, doc=None):
        _ = doc
        return Image.new("RGB", (4, 4), color="red")


class _FakeDoc:
    pictures = [_FakePicture()]


def test_extract_pictures_includes_png_bytes() -> None:
    extractor = DoclingDocumentExtractor()

    pictures = extractor._extract_pictures(_FakeDoc())

    assert len(pictures) == 1
    assert pictures[0].caption == "Figure 1. Trend"
    assert pictures[0].page_number == 2
    assert pictures[0].bbox == {"x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 220.0}
    assert pictures[0].image_bytes is not None
    assert pictures[0].image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_docling_standard_artifact_markers_use_repo_folders(tmp_path: Path) -> None:
    docling_root = tmp_path / "docling"

    assert _docling_layout_artifact_marker(docling_root) == (
        docling_root / "docling-project--docling-layout-heron" / "model.safetensors"
    )
    assert _docling_table_artifact_marker(docling_root) == (
        docling_root
        / "docling-project--docling-models"
        / "model_artifacts"
        / "tableformer"
        / "accurate"
        / "tm_config.json"
    )


def test_docling_table_artifacts_present_accepts_legacy_model_artifacts_path(tmp_path: Path) -> None:
    legacy_marker = tmp_path / "model_artifacts" / "tableformer" / "accurate" / "tm_config.json"
    legacy_marker.parent.mkdir(parents=True, exist_ok=True)
    legacy_marker.write_text("{}", encoding="utf-8")

    assert _docling_table_artifacts_present(tmp_path) is True


def test_prepare_docling_standard_artifacts_raises_when_auto_download_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.docling_extraction.settings.model_cache_auto_download", False)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_table_structure", True)

    with pytest.raises(FileNotFoundError, match="Missing Docling model artifacts"):
        _prepare_docling_standard_artifacts(tmp_path)


def test_resolve_rapidocr_artifact_raises_when_auto_download_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.docling_extraction.settings.model_cache_auto_download", False)
    missing_path = tmp_path / "RapidOcr" / "missing.onnx"

    with pytest.raises(FileNotFoundError, match="Missing RapidOCR artifact"):
        _resolve_rapidocr_artifact(missing_path, "https://example.com/missing.onnx")


def test_rapidocr_v5_manifest_prefers_chinese_recognition_for_bilingual_docs(tmp_path: Path) -> None:
    manifest = _build_rapidocr_v5_manifest(tmp_path, ["ch_sim", "en"])

    assert manifest["det_model_path"][0] == tmp_path / "RapidOcr" / "onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx"
    assert manifest["cls_model_path"][0] == tmp_path / "RapidOcr" / "onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"
    assert manifest["rec_model_path"][0] == tmp_path / "RapidOcr" / "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx"
    assert manifest["rec_keys_path"][0] == tmp_path / "RapidOcr" / "paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt"


def test_rapidocr_v5_manifest_uses_chinese_recognition_without_english(tmp_path: Path) -> None:
    manifest = _build_rapidocr_v5_manifest(tmp_path, ["ch_sim"])

    assert manifest["rec_model_path"][0] == tmp_path / "RapidOcr" / "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx"
    assert manifest["rec_keys_path"][0] == tmp_path / "RapidOcr" / "paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt"


def test_rapidocr_v5_params_switch_internal_config_to_v5() -> None:
    params = _build_rapidocr_v5_params(["ch_sim", "en"])

    assert params["Det.ocr_version"].value == "PP-OCRv5"
    assert params["Cls.ocr_version"].value == "PP-OCRv5"
    assert params["Rec.ocr_version"].value == "PP-OCRv5"
    assert params["Det.lang_type"].value == "ch"
    assert params["Cls.lang_type"].value == "ch"
    assert params["Rec.lang_type"].value == "ch"


def _install_fake_docling_converter(monkeypatch: pytest.MonkeyPatch, *, convert_side_effect: BaseException | None = None):
    class InputFormat:
        PDF = "pdf"

    class PdfFormatOption:
        def __init__(self, *, pipeline_options=None):
            self.pipeline_options = pipeline_options

    class DocumentConverter:
        call_count = 0

        def __init__(self, *, format_options=None):
            DocumentConverter.call_count += 1
            self.format_options = format_options

        def convert(self, source):
            _ = source
            if convert_side_effect is not None:
                raise convert_side_effect

            class _Doc:
                tables = []
                pictures = []

                def export_to_markdown(self):
                    return "ok"

                def export_to_dict(self):
                    return {"pages": {}}

                def iterate_items(self):
                    return iter([])

            return types.SimpleNamespace(document=_Doc())

    docling = types.ModuleType("docling")
    docling.__path__ = []
    datamodel = types.ModuleType("docling.datamodel")
    datamodel.__path__ = []
    base_models = types.ModuleType("docling.datamodel.base_models")
    document_converter = types.ModuleType("docling.document_converter")
    base_models.InputFormat = InputFormat
    document_converter.DocumentConverter = DocumentConverter
    document_converter.PdfFormatOption = PdfFormatOption

    monkeypatch.setitem(sys.modules, "docling", docling)
    monkeypatch.setitem(sys.modules, "docling.datamodel", datamodel)
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", base_models)
    monkeypatch.setitem(sys.modules, "docling.document_converter", document_converter)
    return DocumentConverter


class _FakePdfPipelineOptions:
    def __init__(self) -> None:
        self.artifacts_path = None
        self.do_ocr = True
        self.do_table_structure = True
        self.generate_picture_images = True
        self.images_scale = 2.0
        self.document_timeout = None
        self.ocr_options = None


class _FakeOcrOptions:
    def __init__(self, **kwargs) -> None:
        self.force_full_page_ocr = False
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_fake_pdf_pipeline_options(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")
    pipeline_options.PdfPipelineOptions = _FakePdfPipelineOptions
    pipeline_options.EasyOcrOptions = _FakeOcrOptions
    pipeline_options.RapidOcrOptions = _FakeOcrOptions
    pipeline_options.TesseractOcrOptions = _FakeOcrOptions
    pipeline_options.TesseractCliOcrOptions = _FakeOcrOptions
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", pipeline_options)


def _stub_extract_model_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_paths = types.SimpleNamespace(docling=tmp_path / "docling")
    cache_paths.docling.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.services.docling_extraction.configure_model_cache_env", lambda: cache_paths)
    monkeypatch.setattr("app.services.docling_extraction.resolve_model_cache_paths", lambda: cache_paths)
    monkeypatch.setattr("app.services.docling_extraction._prepare_docling_standard_artifacts", lambda _path: None)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_ocr_engine", "easyocr")
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_artifacts_path", str(cache_paths.docling))


# Hardcoded control lists: a new pipeline_options mutation without a matching cache-key
# field (or the reverse) must fail this file. force_full_page_ocr stays in the cache key
# because DocumentConverter hashes the full options dump and would otherwise spawn a
# second in-process pipeline.
_CACHE_KEY_FIELDS = frozenset(
    {
        "ocr_engine",
        "force_full_page_ocr",
        "languages",
        "do_ocr",
        "do_table_structure",
        "generate_picture_images",
        "images_scale",
        "artifacts_path",
        "document_timeout",
    }
)
_PIPELINE_OPTIONS_MUTATED_FIELDS = frozenset(
    {
        "artifacts_path",
        "do_ocr",
        "do_table_structure",
        "generate_picture_images",
        "images_scale",
        "document_timeout",
        "ocr_options",
    }
)
_SNAPSHOT_FIELDS_VIA_OCR_OPTIONS = frozenset({"ocr_engine", "languages", "force_full_page_ocr"})
_CACHE_KEY_KWARGS = {
    "ocr_engine": "rapidocr",
    "force_full_page_ocr": False,
    "languages": ("ch_sim", "en"),
}


def _converter_cache_api():
    from app.services.docling_extraction import (
        _CONVERTER_CACHE,
        _converter_cache_key,
        _get_or_create_converter,
    )

    return _CONVERTER_CACHE, _converter_cache_key, _get_or_create_converter


def test_converter_cache_key_is_stable_for_same_inputs() -> None:
    _, _converter_cache_key, _ = _converter_cache_api()
    kwargs = {
        "ocr_engine": "rapidocr",
        "force_full_page_ocr": False,
        "languages": ("ch_sim", "en"),
    }

    assert _converter_cache_key(**kwargs) == _converter_cache_key(**kwargs)


def test_converter_cache_key_differs_by_ocr_engine() -> None:
    _, _converter_cache_key, _ = _converter_cache_api()
    shared = {"force_full_page_ocr": False, "languages": ("ch_sim", "en")}

    assert _converter_cache_key(ocr_engine="rapidocr", **shared) != _converter_cache_key(
        ocr_engine="easyocr", **shared
    )


def test_converter_cache_key_differs_by_force_full_page_ocr() -> None:
    _, _converter_cache_key, _ = _converter_cache_api()
    shared = {"ocr_engine": "rapidocr", "languages": ("ch_sim", "en")}

    assert _converter_cache_key(force_full_page_ocr=False, **shared) != _converter_cache_key(
        force_full_page_ocr=True, **shared
    )


def test_converter_cache_key_differs_by_languages() -> None:
    _, _converter_cache_key, _ = _converter_cache_api()
    shared = {"ocr_engine": "rapidocr", "force_full_page_ocr": False}

    assert _converter_cache_key(languages=("ch_sim", "en"), **shared) != _converter_cache_key(
        languages=("en",), **shared
    )


def test_converter_cache_key_includes_pipeline_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _converter_cache_key, _ = _converter_cache_api()
    kwargs = {
        "ocr_engine": "rapidocr",
        "force_full_page_ocr": False,
        "languages": ("ch_sim", "en"),
    }

    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_ocr", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_table_structure", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_generate_picture_images", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_images_scale", 2.0)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_document_timeout_seconds", 240)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_artifacts_path", "/tmp/docling-artifacts-a")
    baseline = _converter_cache_key(**kwargs)

    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_ocr", False)
    assert baseline != _converter_cache_key(**kwargs)

    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_ocr", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_table_structure", False)
    assert baseline != _converter_cache_key(**kwargs)

    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_table_structure", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_generate_picture_images", False)
    assert baseline != _converter_cache_key(**kwargs)

    monkeypatch.setattr("app.services.docling_extraction.settings.docling_generate_picture_images", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_images_scale", 1.0)
    assert baseline != _converter_cache_key(**kwargs)

    monkeypatch.setattr("app.services.docling_extraction.settings.docling_images_scale", 2.0)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_document_timeout_seconds", 60)
    assert baseline != _converter_cache_key(**kwargs)

    monkeypatch.setattr("app.services.docling_extraction.settings.docling_document_timeout_seconds", 240)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_artifacts_path", "/tmp/docling-artifacts-b")
    assert baseline != _converter_cache_key(**kwargs)


def test_get_or_create_converter_reuses_instance_for_same_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_converter_cls = _install_fake_docling_converter(monkeypatch)
    cache, _, _get_or_create_converter = _converter_cache_api()
    cache.clear()
    try:
        key = ("rapidocr", False, ("ch_sim", "en"), True, True, True, 2.0)
        first = _get_or_create_converter(key, object())
        second = _get_or_create_converter(key, object())

        assert first is second
        assert fake_converter_cls.call_count == 1
    finally:
        cache.clear()


def test_get_or_create_converter_creates_new_instance_for_different_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_converter_cls = _install_fake_docling_converter(monkeypatch)
    cache, _, _get_or_create_converter = _converter_cache_api()
    cache.clear()
    try:
        first_key = ("rapidocr", False, ("ch_sim", "en"), True, True, True, 2.0)
        second_key = ("rapidocr", True, ("ch_sim", "en"), True, True, True, 2.0)
        first = _get_or_create_converter(first_key, object())
        second = _get_or_create_converter(second_key, object())

        assert first is not second
        assert fake_converter_cls.call_count == 2
    finally:
        cache.clear()


def test_converter_cache_evicts_oldest_when_over_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_converter_cls = _install_fake_docling_converter(monkeypatch)
    cache, _, _get_or_create_converter = _converter_cache_api()
    cache.clear()
    try:
        first_key = ("engine-a",)
        second_key = ("engine-b",)
        third_key = ("engine-c",)
        _get_or_create_converter(first_key, object())
        _get_or_create_converter(second_key, object())
        _get_or_create_converter(third_key, object())

        assert len(cache) <= 2
        assert first_key not in cache
        assert second_key in cache
        assert third_key in cache
        assert fake_converter_cls.call_count == 3
    finally:
        cache.clear()


def test_converter_cache_key_normalizes_non_positive_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _converter_cache_key, _ = _converter_cache_api()
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_document_timeout_seconds", 0)
    zero_key = _converter_cache_key(**_CACHE_KEY_KWARGS)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_document_timeout_seconds", -5)
    negative_key = _converter_cache_key(**_CACHE_KEY_KWARGS)

    assert zero_key == negative_key


def test_converter_cache_key_normalizes_non_positive_images_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _converter_cache_key, _ = _converter_cache_api()
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_images_scale", 0)
    zero_key = _converter_cache_key(**_CACHE_KEY_KWARGS)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_images_scale", -1)
    negative_key = _converter_cache_key(**_CACHE_KEY_KWARGS)

    assert zero_key == negative_key


def test_cache_key_fields_match_pipeline_options_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.docling_extraction import _apply_pdf_pipeline_options, _converter_config_snapshot

    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_ocr", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_do_table_structure", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_generate_picture_images", True)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_images_scale", 2.0)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_document_timeout_seconds", 240)
    monkeypatch.setattr("app.services.docling_extraction.settings.docling_artifacts_path", "/tmp/docling-artifacts-a")

    snapshot = _converter_config_snapshot(**_CACHE_KEY_KWARGS)
    assert set(snapshot) == _CACHE_KEY_FIELDS
    assert _CACHE_KEY_FIELDS - _SNAPSHOT_FIELDS_VIA_OCR_OPTIONS == _PIPELINE_OPTIONS_MUTATED_FIELDS - {"ocr_options"}

    pipeline_options = _FakePdfPipelineOptions()
    ocr_options = _FakeOcrOptions(lang=["ch_sim", "en"])
    _apply_pdf_pipeline_options(pipeline_options, snapshot, ocr_options=ocr_options)
    mutated_fields = {name for name in vars(pipeline_options) if getattr(pipeline_options, name) is not None or name == "ocr_options"}
    assert mutated_fields == _PIPELINE_OPTIONS_MUTATED_FIELDS
    assert pipeline_options.ocr_options.force_full_page_ocr is False

    _, _converter_cache_key, _ = _converter_cache_api()
    baseline_key = _converter_cache_key(**_CACHE_KEY_KWARGS)
    baseline_settings = {
        "docling_do_ocr": True,
        "docling_do_table_structure": True,
        "docling_generate_picture_images": True,
        "docling_images_scale": 2.0,
        "docling_document_timeout_seconds": 240,
        "docling_artifacts_path": "/tmp/docling-artifacts-a",
    }
    setting_to_field = {
        "docling_do_ocr": ("do_ocr", False),
        "docling_do_table_structure": ("do_table_structure", False),
        "docling_generate_picture_images": ("generate_picture_images", False),
        "docling_images_scale": ("images_scale", 1.0),
        "docling_document_timeout_seconds": ("document_timeout", 60),
        "docling_artifacts_path": ("artifacts_path", "/tmp/docling-artifacts-b"),
    }
    changed_fields: set[str] = set()
    for setting_name, (field_name, alt_value) in setting_to_field.items():
        monkeypatch.setattr(f"app.services.docling_extraction.settings.{setting_name}", alt_value)
        if _converter_cache_key(**_CACHE_KEY_KWARGS) != baseline_key:
            changed_fields.add(field_name)
        monkeypatch.setattr(
            f"app.services.docling_extraction.settings.{setting_name}",
            baseline_settings[setting_name],
        )

    assert changed_fields == _CACHE_KEY_FIELDS - _SNAPSHOT_FIELDS_VIA_OCR_OPTIONS
    assert _converter_cache_key(ocr_engine="easyocr", force_full_page_ocr=False, languages=("ch_sim", "en")) != baseline_key
    assert _converter_cache_key(ocr_engine="rapidocr", force_full_page_ocr=True, languages=("ch_sim", "en")) != baseline_key
    assert _converter_cache_key(ocr_engine="rapidocr", force_full_page_ocr=False, languages=("en",)) != baseline_key


def test_extract_evicts_converter_when_convert_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert_error = RuntimeError("docling convert failed")
    fake_converter_cls = _install_fake_docling_converter(monkeypatch, convert_side_effect=convert_error)
    _install_fake_pdf_pipeline_options(monkeypatch)
    _stub_extract_model_cache(monkeypatch, tmp_path)
    cache, _, _ = _converter_cache_api()
    cache.clear()
    try:
        extractor = DoclingDocumentExtractor()
        with pytest.raises(RuntimeError, match="docling convert failed"):
            extractor.extract(tmp_path / "doc.pdf")
        assert fake_converter_cls.call_count == 1
        assert len(cache) == 0
    finally:
        cache.clear()
