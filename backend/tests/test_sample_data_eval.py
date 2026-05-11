from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.sample_data import (
    ingest_sample_data_papers,
    load_sample_data_paper_specs,
    load_sample_data_questions,
    load_sample_data_sessions,
    load_sample_data_smoke_question_ids,
    resolve_question_gold,
    run_sample_data_evaluation,
)


def test_sample_data_benchmark_files_have_expected_shape() -> None:
    papers = load_sample_data_paper_specs()
    questions = load_sample_data_questions()
    sessions = load_sample_data_sessions()
    smoke_ids = load_sample_data_smoke_question_ids()

    assert len(papers) == 2
    assert len(questions) == 96
    assert len(sessions) == 4
    assert len(smoke_ids) == 12
    assert papers[0].pdf_path.exists()
    assert papers[1].pdf_path.exists()


def test_sample_data_gold_evidence_resolves_against_ingested_sample_papers(
    db_session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.settings.openai_api_key", "")
    papers_by_key = ingest_sample_data_papers(db_session, session_factory=session_factory)
    questions = load_sample_data_questions()

    from app.models import PaperChunk
    from sqlalchemy import select

    chunk_cache = {
        paper_key: db_session.scalars(
            select(PaperChunk).where(PaperChunk.paper_id == paper.id).order_by(PaperChunk.chunk_index.asc())
        ).all()
        for paper_key, paper in papers_by_key.items()
    }

    resolved = resolve_question_gold(questions, papers_by_key=papers_by_key, chunk_cache=chunk_cache)
    eligible = [item for item in resolved if item.question.gold_evidence]

    assert len(eligible) == 92
    assert all(item.gold_chunk_ids for item in eligible)


def test_sample_data_smoke_eval_runs_and_returns_reports(
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("app.config.settings.file_storage_path", tmp_path / "files")
    monkeypatch.setattr("app.config.settings.openai_api_key", "")

    answer_map = {
        "这项研究采用的研究设计是什么？": "It was a randomized double-blind active-controlled crossover study.",
        "这项研究的主要结局指标是什么？": "The primary outcome was the ADHD-RS diagnostic questionnaire completed by the parents.",
        "文中的 tES 指什么？": "tES means transcranial electrical stimulation.",
        "Table 1 中 Stimulation 这一项的 p-value 是多少？": "Table 1 reports a p-value of 0.028 for Stimulation.",
        "作者为什么选择 within-subject design，并给出了怎样的样本量对比？": "They used a within-subject design to control individual differences; a between-subject design would need 72 participants, more than 3.5 times more.",
        "请概括论文摘要中的核心结果：tRNS 相对 tDCS 带来了哪些临床和认知变化？": "tRNS produced clinical improvement on the ADHD rating-scale and improved working memory compared with tDCS.",
        "这里的主要终点具体是哪份量表？": "The primary outcome was the ADHD-RS diagnostic questionnaire completed by the parents.",
        "那个重新校准后的基线，在图 2 里叫什么？": "The recalibrated baseline was called New T0 (N-T0).",
        "本文纳入研究对象总共多少例？": "本文纳入109例患儿。",
        "本文用什么量表评估抽动障碍程度？": "本文采用YGTSS评估抽动障碍程度，包括运动抽动和发声抽动评分。",
        "表3中，研究组干预后的 θ 波频率是多少？": "研究组干预后的θ波频率为20.09±2.04。",
        "拒答题：本文是否报告了按性别分层后的疗效差异？": "知识库中未找到确切依据。",
    }

    def fake_chat(messages: list[dict[str, str]], *, temperature: float = 0.3) -> tuple[str, str | None]:
        content = messages[-1]["content"]
        if content.startswith("用户问题："):
            question = content.split("\n", 1)[0].replace("用户问题：", "").strip()
            return answer_map.get(question, "知识库中未找到确切依据。"), "sample-data-test-model"
        return answer_map.get(content.strip(), "知识库中未找到确切依据。"), "sample-data-test-model"

    def fake_embeddings(texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(len(text.split()))] for text in texts]

    def fake_rerank(query: str, documents: list[str], *, top_n: int | None = None):
        limit = min(top_n or len(documents), len(documents))
        return [
            type("RerankResult", (), {"index": index, "relevance_score": float(limit - index), "document": None})()
            for index in range(limit)
        ]

    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)
    monkeypatch.setattr("app.services.rag.embed_texts", fake_embeddings)
    monkeypatch.setattr("app.services.rag.rerank_documents", fake_rerank)

    report = run_sample_data_evaluation(session_factory, mode="both", subset="smoke", judge_mode="heuristic")

    assert report["retrieval"]["summary"]["count"] == 11
    assert report["retrieval"]["summary"]["skipped_without_gold"] == 1
    assert report["end_to_end"]["summary"]["count"] == 12
    assert "groundedness" in report["end_to_end"]["summary"]
    assert report["end_to_end"]["questions"][0]["metadata"]["retrieval"]["retrieval_backend"] == "legacy"
