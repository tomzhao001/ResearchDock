from __future__ import annotations

import pytest

from app.services.llm import (
    embed_texts,
    is_embedding_configured,
    is_glm_embedding_configured,
    is_glm_rerank_configured,
    is_rerank_configured,
    rerank_documents,
)


def test_embed_texts_batches_requests_by_configured_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.embedding_provider", "glm")
    monkeypatch.setattr("app.config.settings.glm_embedding_api_key", "test-embedding-key")
    monkeypatch.setattr("app.config.settings.embedding_batch_size", 32)

    calls: list[list[str]] = []

    def fake_glm_embeddings(inputs: list[str]) -> list[list[float]]:
        calls.append(list(inputs))
        return [[float(index)] for index, _ in enumerate(inputs, start=1)]

    monkeypatch.setattr("app.services.llm._request_glm_embeddings", fake_glm_embeddings)

    payload = [f"text-{index}" for index in range(65)]
    embeddings = embed_texts(payload)

    assert [len(batch) for batch in calls] == [32, 32, 1]
    assert len(embeddings) == 65


def test_embed_texts_skips_blank_inputs_before_batching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.embedding_provider", "glm")
    monkeypatch.setattr("app.config.settings.glm_embedding_api_key", "test-embedding-key")
    monkeypatch.setattr("app.config.settings.embedding_batch_size", 2)

    calls: list[list[str]] = []

    def fake_glm_embeddings(inputs: list[str]) -> list[list[float]]:
        calls.append(list(inputs))
        return [[float(len(text))] for text in inputs]

    monkeypatch.setattr("app.services.llm._request_glm_embeddings", fake_glm_embeddings)

    embeddings = embed_texts(["alpha", " ", "", "beta", " gamma "])

    assert calls == [["alpha", "beta"], ["gamma"]]
    assert embeddings == [[5.0], [4.0], [5.0]]


def test_glm_embedding_uses_embedding_key_not_chat_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.embedding_provider", "glm")
    monkeypatch.setattr("app.config.settings.glm_api_key", "")
    monkeypatch.setattr("app.config.settings.glm_embedding_api_key", "embedding-only-key")
    monkeypatch.setattr("app.config.settings.glm_embedding_base_url", "https://glm.example/v4")

    captured: dict[str, str] = {}

    def fake_glm_embeddings(inputs: list[str]) -> list[list[float]]:
        from app.config import settings

        captured["api_key"] = settings.glm_embedding_api_key
        captured["base_url"] = settings.glm_embedding_base_url
        return [[1.0] for _ in inputs]

    monkeypatch.setattr("app.services.llm._request_glm_embeddings", fake_glm_embeddings)

    assert is_glm_embedding_configured() is True
    assert is_embedding_configured() is True
    embed_texts(["hello"])
    assert captured["api_key"] == "embedding-only-key"
    assert captured["base_url"] == "https://glm.example/v4"


def test_chat_and_embedding_credentials_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.llm_provider", "glm")
    monkeypatch.setattr("app.config.settings.glm_api_key", "chat-key")
    monkeypatch.setattr("app.config.settings.glm_base_url", "https://chat.example/v4")
    monkeypatch.setattr("app.config.settings.glm_embedding_api_key", "embedding-key")
    monkeypatch.setattr("app.config.settings.glm_rerank_api_key", "rerank-key")
    monkeypatch.setattr("app.config.settings.embedding_provider", "glm")

    assert is_glm_embedding_configured() is True
    assert is_glm_rerank_configured() is True


def test_rerank_skips_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.glm_rerank_api_key", "")
    monkeypatch.setattr("app.config.settings.glm_rerank_base_url", "https://glm.example/v4")

    calls: list[str] = []

    def fake_glm_rerank(query: str, documents: list[str], *, top_n: int | None = None) -> list:
        calls.append(query)
        return []

    monkeypatch.setattr("app.services.llm._request_glm_rerank", fake_glm_rerank)

    assert is_rerank_configured() is False
    assert rerank_documents("query", ["doc"]) == []
    assert calls == []
