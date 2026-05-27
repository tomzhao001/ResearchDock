from __future__ import annotations

from pathlib import Path

import pytest

from app.services.document_extraction import ExtractedBlock, ExtractedDocument, ExtractedPage, ExtractedTable
from app.services.ocr import medical_terms
from app.services.ocr.escalation import OcrEscalationProcessor
from app.services.ocr.escalation_base import OcrEscalationRequest, OcrEscalationResult
from app.services.ocr.postprocess import OcrPostprocessor
from app.services.ocr.quality import OcrRoutingDecision


class FakeOcrEscalationAdapter:
    provider_name = "fake_glm_ocr"

    def __init__(self, *, text: str = "") -> None:
        self.text = text
        self.calls: list[OcrEscalationRequest] = []

    def available(self) -> bool:
        return True

    def recognize(self, request: OcrEscalationRequest) -> OcrEscalationResult:
        self.calls.append(request)
        return OcrEscalationResult(
            text=self.text,
            provider=self.provider_name,
            should_replace=bool(self.text),
            confidence=0.99,
            model_name="fake-glm-ocr",
            reason="unit_test_candidate",
        )


def test_postprocess_repairs_medical_abbreviation_units_and_plus_minus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(medical_terms.settings, "ocr_medical_wordlist_path", "")
    medical_terms.load_medical_terms.cache_clear()
    medical_terms.load_default_medical_terms.cache_clear()

    document = ExtractedDocument(
        markdown_text="ADIID Iog 3.0土4.0",
        metadata={},
        pages=[ExtractedPage(page_number=1, text="ADIID Iog 3.0土4.0")],
        blocks=[
            ExtractedBlock(
                block_index=0,
                text="ADIID Iog 3.0土4.0",
                page_number=1,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            )
        ],
        tables=[
            ExtractedTable(
                table_index=0,
                caption="ADIID Iog",
                markdown="| Dose |\n| --- |\n| Iog |",
                data=[{"Dose": "Iog"}],
            )
        ],
    )
    postprocessor = OcrPostprocessor(spell_corrector=None)
    routing = OcrRoutingDecision(
        force_full_page_ocr=False,
        docling_ocr_engine="rapidocr",
        enable_escalation=False,
        enable_postprocess=True,
        source="test",
        routing_reason="unit_test",
        sampled_page_count=1,
        toxic_page_count=0,
        page_assessments=[],
    )

    result = postprocessor.process(document=document, pdf_path=Path("fake.pdf"), ocr_strategy=routing)

    assert result.markdown_text == "ADHD 10g 3.0±4.0"
    assert result.blocks[0].text == "ADHD 10g 3.0±4.0"
    assert result.tables[0].caption == "ADHD 10g"
    assert result.tables[0].markdown == "| Dose |\n| --- |\n| 10g |"
    assert result.tables[0].data == [{"Dose": "10g"}]
    assert result.metadata["ocr_postprocess"]["symspell_correction_count"] >= 1
    assert result.metadata["ocr_postprocess"]["rule_repair_count"] >= 2


def test_escalation_uses_glm_rerun_for_suspicious_blocks() -> None:
    document = ExtractedDocument(
        markdown_text="A D I I D",
        metadata={},
        blocks=[
            ExtractedBlock(
                block_index=0,
                text="A D I I D",
                page_number=1,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
                metadata={},
            )
        ],
    )
    fake_adapter = FakeOcrEscalationAdapter(text="ADHD 10g")
    processor = OcrEscalationProcessor(adapter=fake_adapter)
    routing = OcrRoutingDecision(
        force_full_page_ocr=True,
        docling_ocr_engine="rapidocr",
        enable_escalation=True,
        enable_postprocess=True,
        source="test",
        routing_reason="unit_test",
        sampled_page_count=1,
        toxic_page_count=1,
        page_assessments=[],
    )

    result = processor.process(document=document, pdf_path=Path("fake.pdf"), ocr_strategy=routing)

    assert len(fake_adapter.calls) == 1
    assert fake_adapter.calls[0].page_number == 1
    assert fake_adapter.calls[0].bbox == {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    assert result.blocks[0].text == "ADHD 10g"
    assert result.markdown_text == "ADHD 10g"
    assert result.blocks[0].metadata["ocr_escalation"]["confidence"] == 0.99
    assert result.blocks[0].metadata["ocr_escalation"]["accepted"] is True
    assert result.metadata["ocr_escalation"]["accepted_block_count"] == 1
