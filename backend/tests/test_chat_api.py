import pytest

from app.services import llm


def login(client) -> None:
    response = client.post("/api/auth/login", json={"username": "admin", "password": "123456"})
    assert response.status_code == 200


def test_chat_requires_auth(client, user) -> None:
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 401


def test_chat_returns_model_response(client, user, monkeypatch: pytest.MonkeyPatch) -> None:
    login(client)

    def fake_chat(_: str) -> tuple[str, str | None]:
        return "这是模型回复。", "test-model"

    monkeypatch.setattr(llm, "chat_with_model", fake_chat)
    monkeypatch.setattr("app.routers.chat.chat_with_model", fake_chat)

    response = client.post("/api/chat", json={"message": "帮我总结一下"})
    assert response.status_code == 200
    assert response.json() == {"answer": "这是模型回复。", "model": "test-model"}
