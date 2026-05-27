from __future__ import annotations

import base64
from pathlib import Path


def render_pdf_region_png(pdf_path: Path, *, page_number: int, bbox: dict | None = None) -> bytes | None:
    try:
        import fitz
    except Exception:
        return None

    try:
        document = fitz.open(pdf_path)
        page = document.load_page(max(0, page_number - 1))
        clip = None
        if isinstance(bbox, dict):
            rect = fitz.Rect(
                float(bbox.get("x0") or 0.0),
                float(bbox.get("y0") or 0.0),
                float(bbox.get("x1") or page.rect.width),
                float(bbox.get("y1") or page.rect.height),
            )
            clip = rect & page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False)
        image_bytes = pix.tobytes("png")
        document.close()
        return image_bytes
    except Exception:
        return None


def image_bytes_to_data_url(image_bytes: bytes | None, *, mime_type: str = "image/png") -> str | None:
    if not image_bytes:
        return None
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
