from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.services.docling_extraction import (
    DoclingDocumentExtractor,
    _build_rapidocr_v5_manifest,
    _build_rapidocr_v5_params,
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
