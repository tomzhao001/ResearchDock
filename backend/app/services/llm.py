from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.services.http_clients import get_shared_http_client

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
- authors: 作者，单个字符串；未知时返回空字符串
- doi: DOI 字符串；未知时返回空字符串
- source_url: 论文来源链接；未知时返回空字符串
- published_at: 发布时间，使用 ISO 8601 日期或日期时间字符串；未知时返回空字符串

如果信息不足，请返回空字符串或空数组，不要编造。
只能使用论文文本中明确出现的信息，不要根据常识补全作者、DOI、链接或发布时间。

论文文本：
{paper_text}
""".strip()

QUESTION_SET_USER_TEMPLATE = """
请基于给定论文摘要和论文文本，回答问题集，并输出一个 JSON 对象，字段必须严格包含：
- answers: 数组，每个元素都是一个对象，且必须包含 id、question、answer

要求：
- answers 必须与输入问题逐一对应，保持相同顺序
- answer 只能使用论文文本或摘要中明确出现的信息
- 如果信息不足，请返回空字符串，不要编造

问题集：
{question_set}

论文摘要（JSON）：
{summary_json}

论文文本：
{paper_text}
""".strip()


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float
    document: str | None = None


def _get_llm_provider() -> str:
    provider = (settings.llm_provider or "").strip().lower()
    if not provider or provider == "auto":
        return "glm" if settings.glm_api_key.strip() else "openai"
    if provider in {"openai", "glm"}:
        return provider
    raise RuntimeError("LLM_PROVIDER must be one of: auto, openai, glm")


def _get_embedding_provider() -> str:
    provider = (settings.embedding_provider or "").strip().lower()
    if not provider or provider == "auto":
        return "glm" if settings.glm_api_key.strip() else "openai"
    if provider in {"openai", "glm"}:
        return provider
    raise RuntimeError("EMBEDDING_PROVIDER must be one of: auto, openai, glm")


def is_chat_llm_configured() -> bool:
    """True when :func:`_request_chat_completion` can run for the resolved ``LLM_PROVIDER``."""
    try:
        provider = _get_llm_provider()
    except RuntimeError:
        return False
    if provider == "glm":
        return bool(settings.glm_api_key.strip() and (settings.glm_base_url or "").strip())
    return bool(settings.openai_api_key.strip() and (settings.openai_base_url or "").strip())


def get_chat_llm_configuration() -> dict[str, str | bool | None]:
    """Describe the resolved chat provider and whether it is actually callable."""
    try:
        provider = _get_llm_provider()
    except RuntimeError as exc:
        return {
            "provider": None,
            "configured": False,
            "model": None,
            "base_url": None,
            "reason": str(exc),
        }

    if provider == "glm":
        base_url = (settings.glm_base_url or "").strip()
        model = (settings.glm_model or "").strip() or settings.openai_model
        configured = bool(settings.glm_api_key.strip() and base_url)
        reason = None if configured else "glm_chat_config_incomplete"
    else:
        base_url = (settings.openai_base_url or "").strip()
        model = settings.openai_model
        configured = bool(settings.openai_api_key.strip() and base_url)
        reason = None if configured else "openai_chat_config_incomplete"

    return {
        "provider": provider,
        "configured": configured,
        "model": model,
        "base_url": base_url or None,
        "reason": reason,
    }


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


def _build_rerank_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise RuntimeError("GLM_BASE_URL is not configured")
    if normalized.endswith("/rerank"):
        return normalized
    return f"{normalized}/rerank"


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


def _httpx_timeout(read_write_seconds: int) -> httpx.Timeout:
    """读/写阶段可较长（生成摘要慢），连接与连接池不宜拖到与读相同。"""
    s = max(float(read_write_seconds), 1.0)
    return httpx.Timeout(30.0, connect=30.0, read=s, write=s, pool=60.0)


def _post_json(
    url: str,
    *,
    client_name: str,
    api_key: str,
    payload: dict,
    timeout: int | httpx.Timeout,
    verify_ssl: bool,
) -> dict:
    timeout_val = timeout if isinstance(timeout, httpx.Timeout) else _httpx_timeout(timeout)
    client = get_shared_http_client(name=client_name, verify_ssl=verify_ssl)
    response = client.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_val,
    )
    response.raise_for_status()
    return response.json()


def _request_chat_completion(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float = 0.3,
) -> tuple[str, str | None]:
    provider = _get_llm_provider()
    chat_timeout = max(int(settings.llm_chat_timeout_seconds), 1)
    retries = max(int(settings.llm_chat_max_retries), 0)
    if provider == "glm":
        if not settings.glm_api_key.strip():
            raise RuntimeError("GLM_API_KEY is not configured")
        model_name = (settings.glm_model or "").strip() or settings.openai_model
        for attempt in range(retries + 1):
            try:
                payload = _post_json(
                    _build_chat_completions_url(settings.glm_base_url),
                    client_name="glm",
                    api_key=settings.glm_api_key,
                    payload={
                        "model": model_name,
                        "messages": list(messages),
                        "temperature": temperature,
                    },
                    timeout=chat_timeout,
                    verify_ssl=settings.glm_verify_ssl,
                )
                break
            except httpx.TimeoutException:
                if attempt >= retries:
                    raise
                time.sleep(1.0 * (attempt + 1))
    else:
        if not settings.openai_api_key.strip():
            raise RuntimeError("OPENAI_API_KEY is not configured")
        model_name = settings.openai_model
        for attempt in range(retries + 1):
            try:
                payload = _post_json(
                    _build_chat_completions_url(settings.openai_base_url),
                    client_name="openai",
                    api_key=settings.openai_api_key,
                    payload={
                        "model": model_name,
                        "messages": list(messages),
                        "temperature": temperature,
                    },
                    timeout=chat_timeout,
                    verify_ssl=settings.openai_verify_ssl,
                )
                break
            except httpx.TimeoutException:
                if attempt >= retries:
                    raise
                time.sleep(1.0 * (attempt + 1))
    choices = payload.get("choices") or []
    message = choices[0].get("message") if choices else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型未返回有效内容")
    response_model = payload.get("model")
    return content.strip(), response_model if isinstance(response_model, str) else model_name


def _request_openai_embeddings(inputs: Sequence[str]) -> list[list[float]]:
    cleaned_inputs = [item.strip() for item in inputs if item and item.strip()]
    if not cleaned_inputs:
        return []
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is not configured")

    payload = _post_json(
        _build_embeddings_url(settings.openai_base_url),
        client_name="openai",
        api_key=settings.openai_api_key,
        payload={
            "model": settings.openai_embedding_model,
            "input": cleaned_inputs,
        },
        timeout=settings.openai_timeout_seconds,
        verify_ssl=settings.openai_verify_ssl,
    )
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


def _request_glm_embeddings(inputs: Sequence[str]) -> list[list[float]]:
    cleaned_inputs = [item.strip() for item in inputs if item and item.strip()]
    if not cleaned_inputs:
        return []
    if not settings.glm_api_key.strip():
        raise RuntimeError("GLM_API_KEY is not configured")

    payload_data: dict[str, object] = {
        "model": settings.glm_embedding_model,
        "input": cleaned_inputs,
    }
    if settings.glm_embedding_dimensions > 0:
        payload_data["dimensions"] = settings.glm_embedding_dimensions

    payload = _post_json(
        _build_embeddings_url(settings.glm_base_url),
        client_name="glm",
        api_key=settings.glm_api_key,
        payload=payload_data,
        timeout=settings.glm_timeout_seconds,
        verify_ssl=settings.glm_verify_ssl,
    )
    data = payload.get("data") or []
    embeddings: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise RuntimeError("GLM Embedding 响应格式无效")
        vector = item.get("embedding")
        if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
            raise RuntimeError("GLM Embedding 响应缺少向量数据")
        embeddings.append([float(value) for value in vector])
    if len(embeddings) != len(cleaned_inputs):
        raise RuntimeError("GLM Embedding 数量与输入不匹配")
    return embeddings


def _request_glm_rerank(
    query: str,
    documents: Sequence[str],
    *,
    top_n: int | None = None,
) -> list[RerankResult]:
    cleaned_query = (query or "").strip()
    cleaned_documents = [item.strip() for item in documents if item and item.strip()]
    if not cleaned_query or not cleaned_documents:
        return []
    if not settings.glm_api_key.strip():
        raise RuntimeError("GLM_API_KEY is not configured")

    payload_data: dict[str, object] = {
        "model": settings.glm_rerank_model,
        "query": cleaned_query,
        "documents": cleaned_documents[:128],
        "return_documents": False,
    }
    if top_n is not None:
        payload_data["top_n"] = max(min(int(top_n), len(cleaned_documents), 128), 1)

    payload = _post_json(
        _build_rerank_url(settings.glm_base_url),
        client_name="glm",
        api_key=settings.glm_api_key,
        payload=payload_data,
        timeout=settings.glm_timeout_seconds,
        verify_ssl=settings.glm_verify_ssl,
    )
    results = payload.get("results") or []
    reranked: list[RerankResult] = []
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError("GLM Rerank 响应格式无效")
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if not isinstance(index, int) or not isinstance(score, (int, float)):
            raise RuntimeError("GLM Rerank 响应缺少排序结果")
        document = item.get("document")
        reranked.append(
            RerankResult(
                index=index,
                relevance_score=float(score),
                document=document if isinstance(document, str) else None,
            )
        )
    return reranked


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
    if _get_embedding_provider() == "glm":
        return _request_glm_embeddings(texts)
    return _request_openai_embeddings(texts)


def rerank_documents(
    query: str,
    documents: Sequence[str],
    *,
    top_n: int | None = None,
) -> list[RerankResult]:
    return _request_glm_rerank(query, documents, top_n=top_n)


def summarize_paper_text(paper_text: str) -> dict:
    text = (paper_text or "").strip()
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
    key_points = parsed.get("key_points")
    return {
        "abstract_cn": parsed.get("abstract_cn") or "",
        "key_points": [str(item).strip() for item in key_points if str(item).strip()] if isinstance(key_points, list) else [],
        "research_question": parsed.get("research_question") or "",
        "method": parsed.get("method") or "",
        "findings": parsed.get("findings") or "",
        "limitations": parsed.get("limitations") or "",
        "authors": parsed.get("authors") or "",
        "doi": parsed.get("doi") or "",
        "source_url": parsed.get("source_url") or "",
        "published_at": parsed.get("published_at") or "",
    }


def answer_question_set_questions(
    paper_text: str,
    *,
    structured_summary: dict | None,
    questions: Sequence[dict[str, str]],
) -> dict:
    text = (paper_text or "").strip()
    if not text:
        raise RuntimeError("论文文本为空，无法生成问题集结果")
    if not questions:
        raise RuntimeError("问题集为空，无法生成问题集结果")

    normalized_questions = [
        {
            "id": str(item.get("id") or "").strip(),
            "question": str(item.get("question") or "").strip(),
        }
        for item in questions
        if str(item.get("id") or "").strip() and str(item.get("question") or "").strip()
    ]
    if not normalized_questions:
        raise RuntimeError("问题集为空，无法生成问题集结果")

    content, model_name = _request_chat_completion(
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QUESTION_SET_USER_TEMPLATE.format(
                    question_set=json.dumps(normalized_questions, ensure_ascii=False),
                    summary_json=json.dumps(structured_summary or {}, ensure_ascii=False),
                    paper_text=text[:12000],
                ),
            },
        ],
        temperature=0.1,
    )
    parsed = _extract_json_object(content)
    raw_answers = parsed.get("answers")
    answer_by_id: dict[str, str] = {}
    if isinstance(raw_answers, list):
        for item in raw_answers:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                continue
            answer_by_id[item_id] = str(item.get("answer") or "").strip()

    answers = [
        {
            "id": item["id"],
            "question": item["question"],
            "answer": answer_by_id.get(item["id"], ""),
        }
        for item in normalized_questions
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name or "",
        "questions": answers,
    }
