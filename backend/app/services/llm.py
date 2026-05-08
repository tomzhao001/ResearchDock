from __future__ import annotations

import json
from collections.abc import Sequence

import httpx

from app.config import settings

DEFAULT_SYSTEM_PROMPT = (
    "你是 ResearchDock 的研究助理。"
    "当用户是在自由提问时，可以基于通用知识作答；"
    "如果用户问题缺少上下文，请直接说明限制，不要假装引用系统中并不存在的论文内容。"
)

SUMMARY_SYSTEM_PROMPT = (
    "你是一名学术论文分析助手。"
    "请只基于提供的论文内容生成结果，不要补充文本中没有出现的事实。"
)

SUMMARY_USER_TEMPLATE = """
请阅读下面的论文文本，并输出一个 JSON 对象，字段必须严格包含：
- abstract_cn: 中文摘要，1 到 2 段
- key_points: 3 到 5 条字符串数组
- research_question: 研究问题
- method: 方法
- findings: 主要发现
- limitations: 局限性

如果信息不足，请返回空字符串或空数组，不要编造。

论文文本：
{paper_text}
""".strip()


def _build_chat_completions_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise RuntimeError("OPENAI_BASE_URL is not configured")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _build_embeddings_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise RuntimeError("OPENAI_BASE_URL is not configured")
    if normalized.endswith("/embeddings"):
        return normalized
    return f"{normalized}/embeddings"


def _extract_json_object(content: str) -> dict:
    text = (content or "").strip()
    if not text:
        raise RuntimeError("模型未返回内容")

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise RuntimeError("模型返回的摘要结果不是合法 JSON") from None
        return json.loads(text[start : end + 1])


def _request_chat_completion(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float = 0.3,
) -> tuple[str, str | None]:
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is not configured")

    response = httpx.post(
        _build_chat_completions_url(settings.openai_base_url),
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.openai_model,
            "messages": list(messages),
            "temperature": temperature,
        },
        timeout=settings.openai_timeout_seconds,
        verify=settings.openai_verify_ssl,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    message = choices[0].get("message") if choices else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型未返回有效内容")
    model_name = payload.get("model")
    return content.strip(), model_name if isinstance(model_name, str) else settings.openai_model


def _request_embeddings(inputs: Sequence[str]) -> list[list[float]]:
    cleaned_inputs = [item.strip() for item in inputs if item and item.strip()]
    if not cleaned_inputs:
        return []
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is not configured")

    response = httpx.post(
        _build_embeddings_url(settings.openai_base_url),
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.openai_embedding_model,
            "input": cleaned_inputs,
        },
        timeout=settings.openai_timeout_seconds,
        verify=settings.openai_verify_ssl,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or []
    embeddings: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise RuntimeError("Embedding 响应格式无效")
        vector = item.get("embedding")
        if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
            raise RuntimeError("Embedding 响应缺少向量数据")
        embeddings.append([float(value) for value in vector])
    if len(embeddings) != len(cleaned_inputs):
        raise RuntimeError("Embedding 数量与输入不匹配")
    return embeddings


def chat_with_messages(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float = 0.3,
) -> tuple[str, str | None]:
    if not list(messages):
        raise RuntimeError("messages are required")
    return _request_chat_completion(messages, temperature=temperature)


def chat_with_model(user_message: str) -> tuple[str, str | None]:
    prompt = (user_message or "").strip()
    if not prompt:
        raise RuntimeError("message is required")
    return chat_with_messages(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    return _request_embeddings(texts)


def summarize_paper_text(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        raise RuntimeError("论文文本为空，无法生成摘要")

    content, _ = _request_chat_completion(
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUMMARY_USER_TEMPLATE.format(paper_text=text[:12000]),
            },
        ]
    )
    parsed = _extract_json_object(content)
    return {
        "abstract_cn": parsed.get("abstract_cn") or "",
        "key_points": parsed.get("key_points") or [],
        "research_question": parsed.get("research_question") or "",
        "method": parsed.get("method") or "",
        "findings": parsed.get("findings") or "",
        "limitations": parsed.get("limitations") or "",
    }
