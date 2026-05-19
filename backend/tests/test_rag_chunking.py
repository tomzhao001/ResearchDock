from __future__ import annotations

import pytest

from app.models import Paper, PaperChunk
from app.services.ocr.text_normalization import normalize_ocr_text
from app.services.rag import (
    RetrievalResult,
    _boost_exact_match_candidates,
    _build_evidence_candidates,
    _build_crosslingual_query_plan,
    _extract_exact_match_terms,
    _is_exact_match_heavy_query,
    _select_claim_supporting_evidence,
    _split_text,
    _verify_grounded_answer,
)


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

    chunks = _split_text(preanalysis=preanalysis, paper_title="Demo Paper")

    assert len(chunks) >= 2
    assert chunks[0]["metadata_json"]["body_text"]
    assert chunks[0]["metadata_json"]["supporting_context"]
    assert "上下文补充" not in chunks[0]["content"]


def test_split_text_emits_child_and_parent_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.rag_chunk_size", 260)
    monkeypatch.setattr("app.config.settings.rag_chunk_overlap", 40)
    preanalysis = {
        "blocks": [
            {
                "block_index": 0,
                "page_number": 1,
                "section_id": "results",
                "section_title": "Results",
                "section_path": "Results",
                "heading_level": 1,
                "block_type": "heading",
                "reading_order": 0,
                "char_start": 0,
                "char_end": 7,
                "text": "Results",
            },
            {
                "block_index": 1,
                "page_number": 1,
                "section_id": "results",
                "section_title": "Results",
                "section_path": "Results",
                "heading_level": 1,
                "block_type": "paragraph",
                "reading_order": 1,
                "char_start": 8,
                "char_end": 120,
                "text": "Alpha findings sentence. " * 4,
            },
            {
                "block_index": 2,
                "page_number": 1,
                "section_id": "results",
                "section_title": "Results",
                "section_path": "Results",
                "heading_level": 1,
                "block_type": "table_like",
                "reading_order": 2,
                "char_start": 121,
                "char_end": 220,
                "text": "Table 1\nGroup A: 0.92\nGroup B: 0.88",
                "table_data_json": [{"Group": "A", "Score": "0.92"}],
            },
        ]
    }

    chunks = _split_text(preanalysis=preanalysis, paper_title="Demo Paper")

    child_chunks = [chunk for chunk in chunks if chunk["chunk_role"] == "child"]
    parent_chunks = [chunk for chunk in chunks if chunk["chunk_role"] == "parent"]

    assert child_chunks
    assert len(parent_chunks) == 1
    assert all(chunk["metadata_json"]["parent_text"] for chunk in child_chunks)
    assert parent_chunks[0]["metadata_json"]["chunk_role"] == "parent"
    assert "Results" in parent_chunks[0]["metadata_json"]["body_text"]


def test_extract_exact_match_terms_prefers_table_and_hyphenated_terms() -> None:
    terms = _extract_exact_match_terms("Table 1 中 Stimulation 这一项的 p-value 是多少？")

    assert "Table 1" in terms
    assert "Stimulation" in terms
    assert "p-value" in terms


def test_extract_exact_match_terms_works_after_ocr_normalization() -> None:
    normalized_text = normalize_ocr_text(
        "\uff34\uff41\uff42\uff4c\uff45\u3000\uff11\u3000\u4e2d\u3000\uff21\uff24\uff28\uff24\uff0d\uff32\uff53\u3000\u7684\u3000\uff30\uff24\uff26\u3000\u662f\u4ec0\u4e48\uff1f"
    )

    terms = _extract_exact_match_terms(normalized_text.text)

    assert "Table 1" in terms
    assert "ADHD-Rs" in terms
    assert "PDF" in terms


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

    chunks = _split_text(preanalysis=preanalysis, paper_title="中文论文")

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

    chunks = _split_text(preanalysis=preanalysis, paper_title="中文论文")

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
    paper = Paper(id=9, organization_id=1, title="中文论文", status="completed")

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


def test_claim_level_evidence_selection_supports_crosslingual_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    question = "文中的 tES 指什么？"
    paper = Paper(id=7, organization_id=1, title="ADHD Study", status="completed")
    chunk = PaperChunk(
        id=101,
        paper_id=7,
        chunk_index=0,
        content="Table 1 reports that tES stands for transcranial electrical stimulation.",
        embedding=None,
        token_count=10,
        page_from=3,
        page_to=3,
        metadata_json={
            "section_path": "Results > Table 1",
            "body_text": "Table 1 reports that tES stands for transcranial electrical stimulation.",
        },
    )
    candidates = _build_evidence_candidates(
        [RetrievalResult(chunk=chunk, paper=paper, score=0.88)],
        query=question,
    )

    def fake_chat(messages: list[dict[str, str]], *, temperature: float = 0.3) -> tuple[str, str | None]:
        system = messages[0]["content"]
        if "归因证据选择器" in system:
            return (
                """
                {
                  "claims": [
                    {
                      "claim_text": "tES 指 transcranial electrical stimulation",
                      "supporting_evidence_ids": ["chunk-101"],
                      "support_score": 0.93,
                      "selection_reason": "英文证据给出了缩写全称"
                    }
                  ],
                  "selected_evidence": [
                    {
                      "evidence_id": "chunk-101",
                      "support_score": 0.93,
                      "selection_reason": "直接术语定义"
                    }
                  ],
                  "overall_support_score": 0.93,
                  "is_sufficient": true,
                  "missing_information": ""
                }
                """.strip(),
                "selector-model",
            )
        return (
            """
            {
              "supported": true,
              "support_score": 0.91,
              "unsupported_claims": [],
              "notes": "crosslingual support ok"
            }
            """.strip(),
            "verifier-model",
        )

    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    selection = _select_claim_supporting_evidence(
        question=question,
        query_plan={"detected_language": "zh", "generation_instruction": "请用中文回答，保留英文术语原文。"},
        evidence_candidates=candidates,
    )

    assert selection.method == "llm"
    assert selection.sufficiency_decision["is_sufficient"] is True
    assert selection.selected_evidence[0]["evidence_id"] == "chunk-101"
    assert selection.selected_evidence[0]["claim_texts"] == ["tES 指 transcranial electrical stimulation"]

    verifier = _verify_grounded_answer(
        question=question,
        answer="tES 指 transcranial electrical stimulation。",
        query_plan={"detected_language": "zh"},
        selected_evidence=[dict(item) for item in selection.selected_evidence],
        selection_result=selection,
    )

    assert verifier["supported"] is True
    assert verifier["support_score"] >= 0.9
