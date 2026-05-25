from sqlalchemy import select

from app.models import ChatMessage, Paper, PaperChunk
from app.services.rag import EvidenceSelectionResult


def login(client) -> None:
    response = client.post("/api/auth/login", json={"username": "admin", "password": "123456"})
    assert response.status_code == 200


def test_chat_routes_metadata_queries_without_calling_llm(client, user, db_session, monkeypatch) -> None:
    login(client)

    db_session.add_all(
        [
            Paper(organization_id=user.organization_id, title="中文研究一", status="completed"),
            Paper(organization_id=user.organization_id, title="中文研究二", status="completed"),
            Paper(organization_id=user.organization_id, title="English Study", status="completed"),
        ]
    )
    db_session.commit()

    def fail_if_llm_called(*_args, **_kwargs):
        raise AssertionError("metadata route should not call chat_with_messages")

    monkeypatch.setattr("app.services.rag.chat_with_messages", fail_if_llm_called)

    topic_response = client.post("/api/chat/topics", json={})
    assert topic_response.status_code == 201
    topic_id = topic_response.json()["id"]

    response = client.post(
        f"/api/chat/topics/{topic_id}/messages",
        json={"message": "有几个中文文档？"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["answer_mode"] == "metadata_query"
    assert body["assistant_message"]["used_knowledge_base"] is True
    assert "2 篇" in body["assistant_message"]["content"]

    assistant_message = db_session.scalar(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
    )
    assert assistant_message is not None
    retrieval = assistant_message.metadata_json["retrieval"]
    assert retrieval["engine_name"] == "metadata_engine"
    assert retrieval["route_decision"]["engine_name"] == "metadata_engine"
    assert retrieval["engine_execution"]["total_count"] == 2


def test_chat_routes_hybrid_queries_to_metadata_then_scoped_rag(client, user, db_session, monkeypatch) -> None:
    login(client)

    missing_doi_paper = Paper(
        organization_id=user.organization_id,
        title="ADHD Missing DOI",
        status="completed",
        doi=None,
    )
    other_paper = Paper(
        organization_id=user.organization_id,
        title="Paper With DOI",
        status="completed",
        doi="10.1000/example",
    )
    db_session.add_all([missing_doi_paper, other_paper])
    db_session.flush()
    missing_doi_chunk = PaperChunk(
        paper_id=missing_doi_paper.id,
        chunk_index=0,
        chunk_role="child",
        content="The paper studies ADHD intervention outcomes and symptom changes.",
        embedding=None,
        token_count=9,
        page_from=1,
        page_to=1,
        metadata_json={"body_text": "The paper studies ADHD intervention outcomes and symptom changes."},
    )
    other_chunk = PaperChunk(
        paper_id=other_paper.id,
        chunk_index=0,
        chunk_role="child",
        content="This paper focuses on sleep quality and circadian rhythm.",
        embedding=None,
        token_count=9,
        page_from=1,
        page_to=1,
        metadata_json={"body_text": "This paper focuses on sleep quality and circadian rhythm."},
    )
    db_session.add_all([missing_doi_chunk, other_chunk])
    db_session.commit()

    monkeypatch.setattr(
        "app.services.rag.chat_with_messages",
        lambda *_args, **_kwargs: ("The paper missing a DOI studies ADHD intervention outcomes.", "hybrid-rag-model"),
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        lambda **_kwargs: EvidenceSelectionResult(
            selected_evidence=(
                {
                    "evidence_id": f"chunk-{missing_doi_chunk.id}",
                    "chunk_id": missing_doi_chunk.id,
                    "paper_id": missing_doi_paper.id,
                    "paper_title": missing_doi_paper.title,
                    "source_url": missing_doi_paper.source_url,
                    "snippet": "The paper studies ADHD intervention outcomes and symptom changes.",
                    "full_text": "The paper studies ADHD intervention outcomes and symptom changes.",
                    "score": 0.9,
                    "support_score": 0.9,
                    "page_from": 1,
                    "page_to": 1,
                    "section_path": None,
                    "selection_reason": "direct match",
                    "claim_texts": ["The paper studies ADHD intervention outcomes."],
                    "rank": 1,
                },
            ),
            claims=({"claim_text": "The paper studies ADHD intervention outcomes."},),
            overall_support_score=0.9,
            sufficiency_decision={
                "is_sufficient": True,
                "llm_sufficient": None,
                "evidence_count": 1,
                "top_support_score": 0.9,
                "total_support_score": 0.9,
                "overall_support_score": 0.9,
                "min_support_score_threshold": 0.0,
                "min_total_support_score_threshold": 0.0,
                "policy_name": "test",
                "reason_codes": [],
            },
            missing_information="",
            method="test",
        ),
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.verify_grounded_answer",
        lambda **_kwargs: {
            "method": "test",
            "supported": True,
            "support_score": 0.95,
            "unsupported_claims": [],
            "notes": "scoped hybrid route",
        },
    )

    topic_response = client.post("/api/chat/topics", json={})
    assert topic_response.status_code == 201
    topic_id = topic_response.json()["id"]

    response = client.post(
        f"/api/chat/topics/{topic_id}/messages",
        json={"message": "Which papers are missing DOI and what do they study?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["answer_mode"] == "knowledge_base"
    assert body["assistant_message"]["used_knowledge_base"] is True
    assert body["assistant_message"]["model"] == "hybrid-rag-model"
    assert len(body["assistant_message"]["citations"]) == 1
    assert body["assistant_message"]["citations"][0]["paper_id"] == missing_doi_paper.id

    assistant_message = db_session.scalar(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
    )
    assert assistant_message is not None
    retrieval = assistant_message.metadata_json["retrieval"]
    assert retrieval["engine_name"] == "hybrid_sql_rag_engine"
    assert retrieval["route_decision"]["engine_name"] == "hybrid_sql_rag_engine"
    assert retrieval["paper_scope_ids"] == [missing_doi_paper.id]
    assert retrieval["engine_execution"]["paper_ids"] == [missing_doi_paper.id]


def test_chat_defaults_to_rag_route_for_paper_content_question(client, user, db_session, monkeypatch) -> None:
    login(client)

    paper = Paper(
        organization_id=user.organization_id,
        title="Transformer Paper",
        source_url="https://example.com/paper",
        status="completed",
    )
    db_session.add(paper)
    db_session.flush()
    db_session.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            chunk_role="child",
            content="transformer attention improves translation accuracy",
            embedding=None,
            token_count=5,
            page_from=2,
            page_to=2,
            metadata_json={"body_text": "transformer attention improves translation accuracy"},
        )
    )
    db_session.commit()

    monkeypatch.setattr(
        "app.services.rag.chat_with_messages",
        lambda *_args, **_kwargs: ("Transformer attention improves translation accuracy.", "rag-model"),
    )

    topic_response = client.post("/api/chat/topics", json={})
    assert topic_response.status_code == 201
    topic_id = topic_response.json()["id"]

    response = client.post(
        f"/api/chat/topics/{topic_id}/messages",
        json={"message": "What does transformer attention improve?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["answer_mode"] == "knowledge_base"

    assistant_message = db_session.scalar(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
    )
    assert assistant_message is not None
    retrieval = assistant_message.metadata_json["retrieval"]
    assert retrieval["engine_name"] == "rag_engine"
    assert retrieval["route_decision"]["engine_name"] == "rag_engine"
