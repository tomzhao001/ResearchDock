import pytest
from sqlalchemy import select

from app.models import ChatMessage, Paper, PaperChunk


def login(client) -> None:
    response = client.post("/api/auth/login", json={"username": "admin", "password": "123456"})
    assert response.status_code == 200


def test_chat_requires_auth(client, user) -> None:
    response = client.get("/api/chat/topics")
    assert response.status_code == 401


def test_chat_topic_roundtrip_with_knowledge_base_citations(client, user, db_session, monkeypatch: pytest.MonkeyPatch) -> None:
    login(client)

    paper = Paper(title="Transformer Paper", source_url="https://example.com/paper", status="completed")
    db_session.add(paper)
    db_session.flush()
    db_session.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            content="transformer attention method improves translation accuracy and training stability",
            embedding=None,
            token_count=9,
            page_from=None,
            page_to=None,
            metadata_json={"char_start": 0, "char_end": 78},
        )
    )
    db_session.commit()

    def fake_chat(_: list[dict[str, str]], *, temperature: float = 0.3) -> tuple[str, str | None]:
        return "基于知识库，论文指出该方法提升了翻译准确率与训练稳定性。", "rag-model"

    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    topic_response = client.post("/api/chat/topics", json={})
    assert topic_response.status_code == 201
    topic_id = topic_response.json()["id"]

    response = client.post(
        f"/api/chat/topics/{topic_id}/messages",
        json={"message": "What does the transformer method improve?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["content"] == "基于知识库，论文指出该方法提升了翻译准确率与训练稳定性。"
    assert body["assistant_message"]["model"] == "rag-model"
    assert body["assistant_message"]["answer_mode"] == "knowledge_base"
    assert body["assistant_message"]["used_knowledge_base"] is True
    assert len(body["assistant_message"]["citations"]) == 1
    assert body["assistant_message"]["citations"][0]["paper_title"] == "Transformer Paper"
    assert body["topic"]["title"] != "新话题"

    messages_response = client.get(f"/api/chat/topics/{topic_id}/messages")
    assert messages_response.status_code == 200
    items = messages_response.json()["items"]
    assert len(items) == 2
    assert items[0]["role"] == "user"
    assert items[1]["role"] == "assistant"

    assistant_message = db_session.scalar(
        select(ChatMessage)
        .where(ChatMessage.topic_id == topic_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
    )
    assert assistant_message is not None
    assert assistant_message.metadata_json is not None
    assert assistant_message.metadata_json["retrieval"]["retrieval_query"] == "What does the transformer method improve?"
    assert assistant_message.metadata_json["retrieval"]["answer_mode"] == "knowledge_base"


def test_chat_falls_back_to_general_answer_when_no_kb_match(client, user, monkeypatch: pytest.MonkeyPatch) -> None:
    login(client)

    def fake_chat(_: list[dict[str, str]], *, temperature: float = 0.3) -> tuple[str, str | None]:
        return "知识库中未找到确切依据。基于通用知识，CRISPR 是一种基因编辑技术。", "fallback-model"

    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    topic_response = client.post("/api/chat/topics", json={"title": "基因编辑"})
    assert topic_response.status_code == 201
    topic_id = topic_response.json()["id"]

    response = client.post(
        f"/api/chat/topics/{topic_id}/messages",
        json={"message": "What is CRISPR?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["answer_mode"] == "fallback_general"
    assert body["assistant_message"]["used_knowledge_base"] is False
    assert body["assistant_message"]["citations"] == []
    assert "知识库中未找到确切依据" in body["assistant_message"]["content"]
