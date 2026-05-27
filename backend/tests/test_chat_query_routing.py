from sqlalchemy import select

from app.models import ChatMessage, Paper, PaperChunk
from app.services.chat_rag import routing
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
    assert body["assistant_message"]["response_kind"] == "metadata_answer"
    assert body["assistant_message"]["attribution_status"] == "metadata_only"
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
    assert body["assistant_message"]["response_kind"] == "grounded_rag"
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
    assert retrieval["engine_execution"]["structured_phase"]["paper_ids"] == [missing_doi_paper.id]


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


def test_chat_uses_selector_for_ambiguous_filter_scope_query(client, user, db_session, monkeypatch) -> None:
    login(client)

    db_session.add(
        Paper(
            organization_id=user.organization_id,
            title="English Study 2024",
            status="completed",
        )
    )
    db_session.commit()

    def fake_chat_with_messages(messages, **_kwargs):
        system_prompt = messages[0]["content"]
        if "query router selector" in system_prompt:
            return (
                '{"engine_name":"metadata_engine","confidence":0.88,"reason":"filter_only_listing","intent_family":"list","answer_shape":"list"}',
                "selector-model",
            )
        raise AssertionError("ambiguous metadata route should not call generation llm")

    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat_with_messages)

    topic_response = client.post("/api/chat/topics", json={})
    assert topic_response.status_code == 201
    topic_id = topic_response.json()["id"]

    response = client.post(
        f"/api/chat/topics/{topic_id}/messages",
        json={"message": "english papers"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["answer_mode"] == "metadata_query"

    assistant_message = db_session.scalar(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
    )
    assert assistant_message is not None
    retrieval = assistant_message.metadata_json["retrieval"]
    assert retrieval["route_decision"]["decision_source"] == "selector"
    assert retrieval["selector_result"]["status"] == "ok"


def test_selector_low_confidence_falls_back_to_rag_unit(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.rag.chat_with_messages",
        lambda *_args, **_kwargs: (
            '{"engine_name":"metadata_engine","confidence":0.31,"reason":"too_uncertain","intent_family":"list","answer_shape":"list"}',
            "selector-model",
        ),
    )
    monkeypatch.setattr(
        "app.services.rag.get_chat_llm_configuration",
        lambda: {"configured": True},
    )

    route_plan = routing.build_route_plan("english papers")
    base_decision = routing.rule_route("english papers", route_plan=route_plan)
    assert base_decision.low_confidence is True
    decision, selector_result = routing.selector_route("english papers", route_plan=route_plan, base_decision=base_decision)
    assert selector_result is not None
    assert decision.engine_name == "rag_engine"
    assert decision.decision_source == "fallback"
    assert decision.reason == "selector_low_confidence_fallback"


def test_route_uses_hybrid_for_chinese_conversation_scope_summary_followup() -> None:
    route_plan = {
        "intent_family": "rag",
        "answer_shape": "paragraph",
        "conversation_context": {
            "paper_scope_ids": [11, 12],
            "last_engine_name": "metadata_engine",
        },
        "metadata_query_plan": None,
    }
    decision = routing.rule_route("这些论文主要研究什么？", route_plan=route_plan)
    assert decision.engine_name == "hybrid_sql_rag_engine"
    assert decision.reason == "conversation_scope_content_followup"
    assert decision.conversation_scope_used is True


def test_chat_uses_conversation_scoped_hybrid_followup(client, user, db_session, monkeypatch) -> None:
    login(client)

    missing_doi_paper = Paper(
        organization_id=user.organization_id,
        title="Scoped ADHD Paper",
        status="completed",
        doi=None,
    )
    other_paper = Paper(
        organization_id=user.organization_id,
        title="Unrelated Paper",
        status="completed",
        doi="10.1000/ok",
    )
    db_session.add_all([missing_doi_paper, other_paper])
    db_session.flush()
    scoped_chunk = PaperChunk(
        paper_id=missing_doi_paper.id,
        chunk_index=0,
        chunk_role="child",
        content="The paper focuses on ADHD outcome variables and symptom improvement.",
        embedding=None,
        token_count=10,
        page_from=1,
        page_to=1,
        metadata_json={"body_text": "The paper focuses on ADHD outcome variables and symptom improvement."},
    )
    other_chunk = PaperChunk(
        paper_id=other_paper.id,
        chunk_index=0,
        chunk_role="child",
        content="This paper studies sleep quality.",
        embedding=None,
        token_count=5,
        page_from=1,
        page_to=1,
        metadata_json={"body_text": "This paper studies sleep quality."},
    )
    db_session.add_all([scoped_chunk, other_chunk])
    db_session.commit()

    monkeypatch.setattr(
        "app.services.rag.chat_with_messages",
        lambda messages, **_kwargs: (
            '{"engine_name":"metadata_engine","confidence":0.95,"reason":"filter_only_listing","intent_family":"list","answer_shape":"list"}',
            "selector-model",
        )
        if "query router selector" in messages[0]["content"]
        else ("The scoped paper reports ADHD outcome variables and symptom improvement.", "hybrid-rag-model")
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        lambda **_kwargs: EvidenceSelectionResult(
            selected_evidence=(
                {
                    "evidence_id": "chunk-scoped",
                    "chunk_id": scoped_chunk.id,
                    "paper_id": missing_doi_paper.id,
                    "paper_title": missing_doi_paper.title,
                    "source_url": missing_doi_paper.source_url,
                    "snippet": "The paper focuses on ADHD outcome variables and symptom improvement.",
                    "full_text": "The paper focuses on ADHD outcome variables and symptom improvement.",
                    "score": 0.92,
                    "support_score": 0.92,
                    "page_from": 1,
                    "page_to": 1,
                    "section_path": None,
                    "selection_reason": "scoped followup",
                    "claim_texts": ["The scoped paper reports ADHD outcome variables."],
                    "rank": 1,
                },
            ),
            claims=({"claim_text": "The scoped paper reports ADHD outcome variables."},),
            overall_support_score=0.92,
            sufficiency_decision={
                "is_sufficient": True,
                "llm_sufficient": None,
                "evidence_count": 1,
                "top_support_score": 0.92,
                "total_support_score": 0.92,
                "overall_support_score": 0.92,
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
            "support_score": 0.98,
            "unsupported_claims": [],
            "notes": "conversation scoped",
        },
    )

    topic_response = client.post("/api/chat/topics", json={})
    assert topic_response.status_code == 201
    topic_id = topic_response.json()["id"]

    first_response = client.post(
        f"/api/chat/topics/{topic_id}/messages",
        json={"message": "english papers without doi"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["assistant_message"]["answer_mode"] == "metadata_query"

    second_response = client.post(
        f"/api/chat/topics/{topic_id}/messages",
        json={"message": "What do these papers study?"},
    )
    assert second_response.status_code == 200
    assert second_response.json()["assistant_message"]["answer_mode"] == "knowledge_base"

    assistant_message = db_session.scalar(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
    )
    assert assistant_message is not None
    retrieval = assistant_message.metadata_json["retrieval"]
    assert retrieval["engine_name"] == "hybrid_sql_rag_engine"
    assert retrieval["route_decision"]["reason"] == "conversation_scope_content_followup"
    assert retrieval["route_decision"]["conversation_scope_used"] is True
    assert retrieval["paper_scope_ids"] == [missing_doi_paper.id]


def test_answer_shape_guard_reroutes_forced_metadata_to_hybrid(client, user, db_session, monkeypatch) -> None:
    login(client)

    missing_doi_paper = Paper(
        organization_id=user.organization_id,
        title="Forced Hybrid Paper",
        status="completed",
        doi=None,
    )
    db_session.add(missing_doi_paper)
    db_session.flush()
    reroute_chunk = PaperChunk(
        paper_id=missing_doi_paper.id,
        chunk_index=0,
        chunk_role="child",
        content="The paper studies primary outcome variables and symptom reduction.",
        embedding=None,
        token_count=9,
        page_from=1,
        page_to=1,
        metadata_json={"body_text": "The paper studies primary outcome variables and symptom reduction."},
    )
    db_session.add(reroute_chunk)
    db_session.commit()

    monkeypatch.setattr(
        "app.services.chat_rag.routing.rule_route",
        lambda *_args, **_kwargs: routing.RouteDecision(
            engine_name="metadata_engine",
            confidence=0.99,
            reason="forced_metadata_for_test",
            decision_source="rules",
            intent_family="content_extraction",
            answer_shape="paragraph",
            filters={"doi_missing": True},
            aggregation="filter_scope",
        ),
    )
    monkeypatch.setattr(
        "app.services.rag.chat_with_messages",
        lambda *_args, **_kwargs: ("The paper studies primary outcome variables and symptom reduction.", "hybrid-rag-model"),
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        lambda **_kwargs: EvidenceSelectionResult(
            selected_evidence=(
                {
                    "evidence_id": "chunk-hybrid",
                    "chunk_id": reroute_chunk.id,
                    "paper_id": missing_doi_paper.id,
                    "paper_title": missing_doi_paper.title,
                    "source_url": missing_doi_paper.source_url,
                    "snippet": "The paper studies primary outcome variables and symptom reduction.",
                    "full_text": "The paper studies primary outcome variables and symptom reduction.",
                    "score": 0.91,
                    "support_score": 0.91,
                    "page_from": 1,
                    "page_to": 1,
                    "section_path": None,
                    "selection_reason": "forced reroute",
                    "claim_texts": ["The paper studies primary outcome variables."],
                    "rank": 1,
                },
            ),
            claims=({"claim_text": "The paper studies primary outcome variables."},),
            overall_support_score=0.91,
            sufficiency_decision={
                "is_sufficient": True,
                "llm_sufficient": None,
                "evidence_count": 1,
                "top_support_score": 0.91,
                "total_support_score": 0.91,
                "overall_support_score": 0.91,
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
            "notes": "answer shape reroute",
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
    assert response.json()["assistant_message"]["answer_mode"] == "knowledge_base"

    assistant_message = db_session.scalar(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
    )
    assert assistant_message is not None
    retrieval = assistant_message.metadata_json["retrieval"]
    assert retrieval["engine_name"] == "hybrid_sql_rag_engine"
    assert retrieval["router_debug"]["answer_shape_guard"]["triggered"] is True
    assert retrieval["router_debug"]["answer_shape_guard"]["reroute_engine"] == "hybrid_sql_rag_engine"
