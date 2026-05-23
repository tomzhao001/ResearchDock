from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import PaperDocumentBlock, PaperDocumentPage, PaperDocumentPicture, PaperDocumentTable
from app.services.document_extraction import ExtractedDocument


def structure_sort_key(
    *,
    reading_order: int | None,
    page_number: int | None,
    fallback_group: int,
    fallback_index: int,
) -> tuple[int, int, int, int]:
    if reading_order is not None:
        return (0, int(reading_order), fallback_group, fallback_index)
    return (1, int(page_number or 0), fallback_group, fallback_index)


def serialize_table_rows(data: object, *, max_rows: int = 6) -> str:
    if not isinstance(data, list):
        return ""
    lines: list[str] = []
    for row in data[:max_rows]:
        if not isinstance(row, dict):
            continue
        cells = [f"{str(key).strip()}: {str(value).strip()}" for key, value in row.items() if str(value).strip()]
        if cells:
            lines.append("; ".join(cells))
    return "\n".join(lines)


def normalize_rendered_lines(lines: list[str]) -> str:
    compacted: list[str] = []
    for line in lines:
        normalized = str(line or "").strip()
        if not normalized:
            if compacted and compacted[-1] != "":
                compacted.append("")
            continue
        compacted.append(normalized)
    while compacted and compacted[-1] == "":
        compacted.pop()
    return "\n".join(compacted).strip()


def render_document_text(document: ExtractedDocument) -> str:
    rows: list[tuple[tuple[int, int, int, int], str]] = []
    for fallback_index, block in enumerate(document.blocks):
        text = str(block.text or "").strip()
        if not text:
            continue
        if block.block_type == "heading":
            heading_level = max(int(block.heading_level or 1), 1)
            text = f"{'#' * min(heading_level, 6)} {text}"
        rows.append(
            (
                structure_sort_key(
                    reading_order=block.reading_order,
                    page_number=block.page_number,
                    fallback_group=0,
                    fallback_index=fallback_index,
                ),
                text,
            )
        )
    for fallback_index, table in enumerate(document.tables):
        parts = [str(table.caption or "").strip()]
        table_rows = serialize_table_rows(table.data)
        if table_rows:
            parts.append(table_rows)
        elif str(table.markdown or "").strip():
            parts.append(str(table.markdown or "").strip())
        rendered = "\n".join(part for part in parts if part)
        if not rendered:
            continue
        rows.append(
            (
                structure_sort_key(
                    reading_order=table.reading_order,
                    page_number=table.page_from,
                    fallback_group=1,
                    fallback_index=fallback_index,
                ),
                rendered,
            )
        )
    for fallback_index, picture in enumerate(document.pictures):
        rendered = "\n".join(
            part
            for part in (str(picture.caption or "").strip(), str(picture.description or "").strip())
            if part
        )
        if not rendered:
            continue
        rows.append(
            (
                structure_sort_key(
                    reading_order=picture.reading_order,
                    page_number=picture.page_number,
                    fallback_group=2,
                    fallback_index=fallback_index,
                ),
                rendered,
            )
        )
    return normalize_rendered_lines([text for sort_key, text in sorted(rows, key=lambda item: item[0])])


def render_paper_text_from_structure(db: Session, paper_id: int) -> str:
    rows: list[tuple[tuple[int, int, int, int], str]] = []
    blocks = db.scalars(
        select(PaperDocumentBlock)
        .where(PaperDocumentBlock.paper_id == paper_id)
        .order_by(PaperDocumentBlock.block_index.asc(), PaperDocumentBlock.id.asc())
    ).all()
    for fallback_index, block in enumerate(blocks):
        text = str(block.text or "").strip()
        if not text:
            continue
        if (block.block_type or "") == "heading":
            heading_level = max(int(block.heading_level or 1), 1)
            text = f"{'#' * min(heading_level, 6)} {text}"
        page_number = None
        page = getattr(block, "page_id", None)
        if page is not None:
            linked_page = db.get(PaperDocumentPage, block.page_id)
            page_number = linked_page.page_number if linked_page is not None else None
        rows.append(
            (
                structure_sort_key(
                    reading_order=block.reading_order,
                    page_number=page_number,
                    fallback_group=0,
                    fallback_index=fallback_index,
                ),
                text,
            )
        )
    tables = db.scalars(
        select(PaperDocumentTable)
        .where(PaperDocumentTable.paper_id == paper_id)
        .order_by(PaperDocumentTable.table_index.asc(), PaperDocumentTable.id.asc())
    ).all()
    for fallback_index, table in enumerate(tables):
        parts = [str(table.caption or "").strip()]
        table_rows = serialize_table_rows(table.data_json)
        if table_rows:
            parts.append(table_rows)
        elif str(table.markdown or "").strip():
            parts.append(str(table.markdown or "").strip())
        rendered = "\n".join(part for part in parts if part)
        if not rendered:
            continue
        rows.append(
            (
                structure_sort_key(
                    reading_order=table.reading_order,
                    page_number=table.page_from,
                    fallback_group=1,
                    fallback_index=fallback_index,
                ),
                rendered,
            )
        )
    pictures = db.scalars(
        select(PaperDocumentPicture)
        .where(PaperDocumentPicture.paper_id == paper_id)
        .order_by(PaperDocumentPicture.picture_index.asc(), PaperDocumentPicture.id.asc())
    ).all()
    for fallback_index, picture in enumerate(pictures):
        rendered = "\n".join(
            part
            for part in (str(picture.caption or "").strip(), str(picture.description or "").strip())
            if part
        )
        if not rendered:
            continue
        rows.append(
            (
                structure_sort_key(
                    reading_order=picture.reading_order,
                    page_number=picture.page_number,
                    fallback_group=2,
                    fallback_index=fallback_index,
                ),
                rendered,
            )
        )
    return normalize_rendered_lines([text for sort_key, text in sorted(rows, key=lambda item: item[0])])
