from __future__ import annotations

import pytest

from app.models import Paper, PaperChunk
from app.services.document_preprocess import preprocess_document
from app.services.pdf_extraction import DocumentExtractionResult, PageExtractionResult
from app.services.rag import (
    _boost_exact_match_candidates,
    _build_crosslingual_query_plan,
    _extract_exact_match_terms,
    _is_exact_match_heavy_query,
    _split_text,
)


def test_preprocess_document_keeps_page_level_blocks() -> None:
    page = PageExtractionResult(
        page_number=1,
        char_count=240,
        alpha_ratio=0.9,
        continuous_line_ratio=0.8,
        image_count=0,
        suspected_double_column=False,
        needs_ocr=False,
        used_ocr=False,
        reasons=[],
        ocr_metadata=None,
        blocks=None,
        text=(
            "Abstract\n"
            "This is the first paragraph with enough text to stand alone.\n\n"
            "Table 1 Stimulation p-value 0.028\n\n"
            "This is the second paragraph."
        ),
    )
    document = DocumentExtractionResult(raw_text=page.text, metadata={"pages": []}, pages=[page])

    preanalysis = preprocess_document(document)

    assert preanalysis["chunking_hints"]["block_count"] >= 3
    assert [block["block_type"] for block in preanalysis["blocks"]] == [
        "paragraph",
        "table_caption",
        "paragraph",
    ]
    assert all(block["section_title"] == "Abstract" for block in preanalysis["blocks"])


def test_preprocess_document_marks_table_body_like_blocks() -> None:
    page = PageExtractionResult(
        page_number=1,
        char_count=320,
        alpha_ratio=0.9,
        continuous_line_ratio=0.7,
        image_count=0,
        suspected_double_column=False,
        needs_ocr=False,
        used_ocr=False,
        reasons=[],
        ocr_metadata=None,
        blocks=None,
        text=(
            "Results\n"
            "Table 3 两组患儿干预前后脑电波频率比较（x±s）\n"
            "研究组 干预后 θ波频率 20.09 2.04 β波频率 14.55 1.26\n"
            "结论段落。"
        ),
    )
    document = DocumentExtractionResult(raw_text=page.text, metadata={"pages": []}, pages=[page])

    preanalysis = preprocess_document(document)

    assert [block["block_type"] for block in preanalysis["blocks"]] == [
        "table_caption",
        "table_like",
        "paragraph",
    ]


def test_preprocess_document_prefers_raw_blocks_when_page_text_is_low_quality() -> None:
    page = PageExtractionResult(
        page_number=1,
        char_count=120,
        alpha_ratio=0.1,
        continuous_line_ratio=0.1,
        image_count=0,
        suspected_double_column=False,
        needs_ocr=False,
        used_ocr=False,
        reasons=[],
        ocr_metadata=None,
        blocks=[
            {"text": "表3 两组患儿干预前后脑电波频率比较（x±s）", "x0": 0, "y0": 0, "x1": 100, "y1": 20},
            {"text": "研究组 55 6.93±0.67 7.96±0.91 26.21±2.35 20.09±2.04 5.86±0.86 6.78±1.03", "x0": 0, "y0": 21, "x1": 100, "y1": 40},
        ],
        text=". . .",
    )
    document = DocumentExtractionResult(raw_text=page.text, metadata={"pages": []}, pages=[page])

    preanalysis = preprocess_document(document)

    assert [block["block_type"] for block in preanalysis["blocks"]] == [
        "table_caption",
        "table_like",
    ]


def test_preprocess_document_keeps_table_row_labels_as_blocks() -> None:
    page = PageExtractionResult(
        page_number=1,
        char_count=220,
        alpha_ratio=0.3,
        continuous_line_ratio=0.4,
        image_count=0,
        suspected_double_column=False,
        needs_ocr=False,
        used_ocr=False,
        reasons=[],
        ocr_metadata=None,
        blocks=None,
        text=(
            "Results\n"
            "表3 两组患儿干预前后脑电波频率比较（x±s）\n"
            "研究组\n"
            "55 6.93±0.67 7.96±0.91 26.21±2.35 20.09±2.04 5.86±0.86 6.78±1.03\n"
            "对照组\n"
        ),
    )
    document = DocumentExtractionResult(raw_text=page.text, metadata={"pages": []}, pages=[page])

    preanalysis = preprocess_document(document)

    assert [block["block_type"] for block in preanalysis["blocks"]] == [
        "table_caption",
        "paragraph",
        "table_like",
        "paragraph",
    ]


def test_split_text_adds_supporting_context_for_small_to_big(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.rag_chunk_size", 260)
    monkeypatch.setattr("app.config.settings.rag_chunk_overlap", 40)
    preanalysis = {
        "blocks": [
            {
                "block_index": 0,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "Results",
                "section_path": "Results",
                "heading_level": 1,
                "block_type": "paragraph",
                "char_start": 0,
                "char_end": 90,
                "text": "Alpha findings sentence. " * 4,
            },
            {
                "block_index": 1,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "Results",
                "section_path": "Results",
                "heading_level": 1,
                "block_type": "paragraph",
                "char_start": 91,
                "char_end": 180,
                "text": "Beta findings sentence. " * 4,
            },
            {
                "block_index": 2,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "Results",
                "section_path": "Results",
                "heading_level": 1,
                "block_type": "paragraph",
                "char_start": 181,
                "char_end": 270,
                "text": "Gamma findings sentence. " * 4,
            },
        ]
    }

    chunks = _split_text("unused", preanalysis=preanalysis, paper_title="Demo Paper")

    assert len(chunks) >= 2
    assert chunks[0]["metadata_json"]["body_text"]
    assert chunks[0]["metadata_json"]["supporting_context"]
    assert "上下文补充" not in chunks[0]["content"]


def test_extract_exact_match_terms_prefers_table_and_hyphenated_terms() -> None:
    terms = _extract_exact_match_terms("Table 1 中 Stimulation 这一项的 p-value 是多少？")

    assert "Table 1" in terms
    assert "Stimulation" in terms
    assert "p-value" in terms


def test_build_crosslingual_query_plan_creates_bilingual_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.rag_crosslingual_query_rewrite_enabled", True)
    monkeypatch.setattr("app.config.settings.glm_api_key", "test-key")

    def fake_chat(messages: list[dict[str, str]], *, temperature: float = 0.3) -> tuple[str, str | None]:
        return (
            """
            {
              "detected_language": "zh",
              "retrieval_query_en": "What does tES stand for in the paper?",
              "exact_terms": ["tES", "transcranial electrical stimulation"],
              "subqueries_en": [],
              "generation_instruction": "请用中文回答，保留关键英文术语原文，并引用英文证据。"
            }
            """.strip(),
            "test-model",
        )

    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    plan = _build_crosslingual_query_plan("文中的 tES 指什么？")

    assert plan.detected_language == "zh"
    assert plan.retrieval_query_en == "What does tES stand for in the paper?"
    assert "tES" in plan.exact_terms
    assert "transcranial electrical stimulation" in plan.exact_terms
    assert [variant.name for variant in plan.variants] == ["zh_original", "en_rewrite"]
    assert plan.rerank_query == "What does tES stand for in the paper?"


def test_exact_match_heavy_query_stays_conservative_for_cjk_single_fact() -> None:
    query = "本文纳入研究对象总共多少例？"

    assert _extract_exact_match_terms(query) == []
    assert _is_exact_match_heavy_query(query) is False


def test_split_text_is_more_conservative_for_cjk_paragraphs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.rag_chunk_size", 400)
    monkeypatch.setattr("app.config.settings.rag_chunk_overlap", 40)
    paragraph = "这是一个中文段落，用来验证中文正文在新的回调式修复后不会被切得过碎。"
    preanalysis = {
        "blocks": [
            {
                "block_index": 0,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "方法",
                "section_path": "方法",
                "heading_level": 1,
                "block_type": "paragraph",
                "char_start": 0,
                "char_end": 80,
                "text": paragraph * 2,
            },
            {
                "block_index": 1,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "方法",
                "section_path": "方法",
                "heading_level": 1,
                "block_type": "paragraph",
                "char_start": 81,
                "char_end": 160,
                "text": paragraph * 2,
            },
            {
                "block_index": 2,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "方法",
                "section_path": "方法",
                "heading_level": 1,
                "block_type": "paragraph",
                "char_start": 161,
                "char_end": 240,
                "text": paragraph * 2,
            },
        ]
    }

    chunks = _split_text("unused", preanalysis=preanalysis, paper_title="中文论文")

    assert len(chunks) <= 2
    assert chunks[0]["metadata_json"]["supporting_context"] is None


def test_split_text_pairs_table_caption_with_following_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.rag_chunk_size", 260)
    monkeypatch.setattr("app.config.settings.rag_chunk_overlap", 40)
    preanalysis = {
        "blocks": [
            {
                "block_index": 0,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "结果",
                "section_path": "结果",
                "heading_level": 1,
                "block_type": "table_caption",
                "char_start": 0,
                "char_end": 30,
                "text": "表3 两组患儿干预前后脑电波频率比较（x±s）",
            },
            {
                "block_index": 1,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "结果",
                "section_path": "结果",
                "heading_level": 1,
                "block_type": "table_like",
                "char_start": 31,
                "char_end": 120,
                "text": "研究组 干预后 θ波频率 20.09±2.04 β波频率 14.55±1.26",
            },
            {
                "block_index": 2,
                "page_number": 1,
                "section_id": "section_0",
                "section_title": "结果",
                "section_path": "结果",
                "heading_level": 1,
                "block_type": "paragraph",
                "char_start": 121,
                "char_end": 180,
                "text": "结论段落说明研究组在干预后表现更好。",
            },
        ]
    }

    chunks = _split_text("unused", preanalysis=preanalysis, paper_title="中文论文")

    assert chunks[0]["metadata_json"]["block_types"] == ["table_caption", "table_like"]
    assert "表3" in chunks[0]["metadata_json"]["body_text"]
    assert "20.09" in chunks[0]["metadata_json"]["body_text"]


def test_boost_exact_match_candidates_demotes_caption_only_for_table_queries() -> None:
    caption_chunk = PaperChunk(
        id=1,
        paper_id=9,
        chunk_index=0,
        content="表3 两组患儿干预前后脑电波频率比较（x±s）",
        embedding=None,
        token_count=12,
        page_from=2,
        page_to=2,
        metadata_json={
            "body_text": "表3 两组患儿干预前后脑电波频率比较（x±s）",
            "block_types": ["table_caption"],
        },
    )
    body_chunk = PaperChunk(
        id=2,
        paper_id=9,
        chunk_index=1,
        content="研究组 干预后 θ波频率 20.09±2.04 β波频率 14.55±1.26",
        embedding=None,
        token_count=18,
        page_from=2,
        page_to=2,
        metadata_json={
            "body_text": "研究组 干预后 θ波频率 20.09±2.04 β波频率 14.55±1.26",
            "block_types": ["table_like"],
        },
    )
    paper = Paper(id=9, title="中文论文", status="completed")

    boosted = _boost_exact_match_candidates(
        "表3中，研究组干预后的 θ 波频率是多少？",
        candidates=[
            {"chunk_id": 1, "paper_id": 9, "score": 0.2, "source_scores": {"dense": 0.2}},
            {"chunk_id": 2, "paper_id": 9, "score": 0.2, "source_scores": {"dense": 0.2}},
        ],
        record_map={
            1: (caption_chunk, paper),
            2: (body_chunk, paper),
        },
    )

    assert boosted[0]["chunk_id"] == 2
    assert boosted[0]["exact_match_bonus"] > boosted[1]["exact_match_bonus"]
