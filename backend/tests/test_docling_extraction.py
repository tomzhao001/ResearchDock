from __future__ import annotations

from PIL import Image

from app.services.docling_extraction import DoclingDocumentExtractor


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
