from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job, PaperAsset, PaperChunk, PaperDocumentBlock, PaperDocumentPage, PaperDocumentPicture, PaperDocumentTable
from app.services.document_extraction import ExtractedBlock, ExtractedDocument, ExtractedPage, ExtractedPicture, ExtractedTable
from app.services.papers import create_upload_artifacts, run_pdf_ingest_job
from app.services.vision.base import PictureDescriptionRequest, PictureDescriptionResult


class FakeDoclingExtractor:
    def __init__(self, *, title: str = "Abstract"):
        self.title = title

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        return ExtractedDocument(
            markdown_text=f"# {self.title}\n\nThis is Docling markdown text.",
            metadata={
                "engine": "docling",
                "docling_do_ocr": True,
                "docling_do_table_structure": True,
            },
            pages=[ExtractedPage(page_number=1, text="This is page text.", width=612, height=792)],
            blocks=[
                ExtractedBlock(
                    block_index=0,
                    text="This is Docling markdown text.",
                    block_type="paragraph",
                    page_number=1,
                    docling_label="text",
                    heading_level=1,
                    section_path=self.title,
                )
            ],
            tables=[
                ExtractedTable(
                    table_index=0,
                    caption="Table 1. Accuracy",
                    markdown="| Group | Score |\n| --- | --- |\n| A | 0.92 |",
                    page_from=1,
                    page_to=1,
                )
            ],
            pictures=[
                ExtractedPicture(
                    picture_index=0,
                    caption="Figure 1. Trend",
                    page_number=1,
                    image_bytes=b"fake-png",
                )
            ],
        )


class SecondFakeDoclingExtractor(FakeDoclingExtractor):
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        document = super().extract(pdf_path)
        document.markdown_text = "# Results\n\nReplacement parse."
        document.blocks[0].text = "Replacement parse."
        document.blocks[0].section_path = "Results"
        document.tables = []
        document.pictures = []
        return document


class FailingDoclingExtractor:
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        raise RuntimeError("Docling conversion failed")


class FakePictureAdapter:
    def describe(self, request: PictureDescriptionRequest) -> PictureDescriptionResult:
        assert request.caption == "Figure 1. Trend"
        return PictureDescriptionResult(
            description="图表显示 A 组分数随时间上升。",
            model_name="glm-4.6v",
            prompt_version="test-picture-prompt",
            usage={"total_tokens": 10},
        )


def make_pdf_payload() -> bytes:
    return b"%PDF-1.7\n% fake test payload\n"


def test_worker_persists_docling_structure_and_picture_descriptions(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="docling.pdf",
        content_type="application/pdf",
        payload=make_pdf_payload(),
    )

    run_pdf_ingest_job(
        artifacts.job_id,
        session_factory=session_factory,
        extractor=FakeDoclingExtractor(),
        picture_adapter=FakePictureAdapter(),
    )

    job = db_session.get(Job, artifacts.job_id)
    asset = db_session.scalar(select(PaperAsset).where(PaperAsset.paper_id == artifacts.paper_id))
    chunks = db_session.scalars(select(PaperChunk).where(PaperChunk.paper_id == artifacts.paper_id)).all()
    pages = db_session.scalars(select(PaperDocumentPage).where(PaperDocumentPage.paper_id == artifacts.paper_id)).all()
    blocks = db_session.scalars(select(PaperDocumentBlock).where(PaperDocumentBlock.paper_id == artifacts.paper_id)).all()
    tables = db_session.scalars(select(PaperDocumentTable).where(PaperDocumentTable.paper_id == artifacts.paper_id)).all()
    pictures = db_session.scalars(select(PaperDocumentPicture).where(PaperDocumentPicture.paper_id == artifacts.paper_id)).all()

    assert job is not None
    assert job.status == "completed"
    assert asset is not None
    assert asset.raw_text == "# Abstract\n\nThis is Docling markdown text."
    assert asset.metadata_json is not None
    assert asset.metadata_json["extraction"]["engine"] == "docling"
    assert asset.metadata_json["extraction"]["page_count"] == 1
    assert asset.metadata_json["extraction"]["block_count"] == 1
    assert asset.metadata_json["extraction"]["table_count"] == 1
    assert asset.metadata_json["extraction"]["picture_count"] == 1
    assert len(pages) == 1
    assert len(blocks) == 1
    assert len(tables) == 1
    assert len(pictures) == 1
    assert pictures[0].description == "图表显示 A 组分数随时间上升。"
    assert pictures[0].description_model == "glm-4.6v"
    assert chunks
    assert any("Table 1. Accuracy" in chunk.content for chunk in chunks)
    assert any("图表显示 A 组分数随时间上升" in chunk.content for chunk in chunks)


def test_worker_reparse_replaces_existing_structure(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="docling.pdf",
        content_type="application/pdf",
        payload=make_pdf_payload(),
    )

    run_pdf_ingest_job(
        artifacts.job_id,
        session_factory=session_factory,
        extractor=FakeDoclingExtractor(),
        picture_adapter=FakePictureAdapter(),
    )
    reparse_job = Job(job_type="pdf_ingest", paper_id=artifacts.paper_id, status="queued")
    db_session.add(reparse_job)
    db_session.commit()
    db_session.refresh(reparse_job)

    run_pdf_ingest_job(
        reparse_job.id,
        session_factory=session_factory,
        extractor=SecondFakeDoclingExtractor(),
        picture_adapter=FakePictureAdapter(),
    )

    asset = db_session.scalar(select(PaperAsset).where(PaperAsset.paper_id == artifacts.paper_id))
    chunks = db_session.scalars(select(PaperChunk).where(PaperChunk.paper_id == artifacts.paper_id)).all()
    tables = db_session.scalars(select(PaperDocumentTable).where(PaperDocumentTable.paper_id == artifacts.paper_id)).all()
    pictures = db_session.scalars(select(PaperDocumentPicture).where(PaperDocumentPicture.paper_id == artifacts.paper_id)).all()

    assert asset is not None
    assert asset.raw_text == "# Results\n\nReplacement parse."
    assert tables == []
    assert pictures == []
    assert chunks
    assert all("Table 1. Accuracy" not in chunk.content for chunk in chunks)


def test_worker_marks_job_failed_when_docling_errors(
    db_session: Session,
    session_factory: sessionmaker,
    organization,
) -> None:
    artifacts = create_upload_artifacts(
        db_session,
        organization_id=organization.id,
        filename="broken.pdf",
        content_type="application/pdf",
        payload=make_pdf_payload(),
    )

    with pytest.raises(RuntimeError, match="Docling conversion failed"):
        run_pdf_ingest_job(
            artifacts.job_id,
            session_factory=session_factory,
            extractor=FailingDoclingExtractor(),
            picture_adapter=FakePictureAdapter(),
        )

    job = db_session.get(Job, artifacts.job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "Docling conversion failed"
