from __future__ import annotations

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
