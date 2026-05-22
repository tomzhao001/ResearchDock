from __future__ import annotations

import pytest

from app.services.llm import embed_texts


def test_embed_texts_batches_requests_by_configured_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.embedding_provider", "glm")
    monkeypatch.setattr("app.config.settings.glm_api_key", "test-key")
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
    monkeypatch.setattr("app.config.settings.glm_api_key", "test-key")
    monkeypatch.setattr("app.config.settings.embedding_batch_size", 2)

    calls: list[list[str]] = []

    def fake_glm_embeddings(inputs: list[str]) -> list[list[float]]:
        calls.append(list(inputs))
        return [[float(len(text))] for text in inputs]

    monkeypatch.setattr("app.services.llm._request_glm_embeddings", fake_glm_embeddings)

    embeddings = embed_texts(["alpha", " ", "", "beta", " gamma "])

    assert calls == [["alpha", "beta"], ["gamma"]]
    assert embeddings == [[5.0], [4.0], [5.0]]
