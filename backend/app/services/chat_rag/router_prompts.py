from __future__ import annotations

import json
from typing import Any


SELECTOR_SYSTEM_PROMPT = (
    "你是 ResearchDock 的 query router selector。"
    "你的任务是在 rag_engine、metadata_engine、hybrid_sql_rag_engine 三个候选中选择最合适的目标。"
    "必须返回 JSON 对象，不要输出额外解释。"
)


def build_selector_messages(*, query: str, route_plan: dict[str, Any], base_decision: dict[str, Any]) -> list[dict[str, str]]:
    user_payload = {
        "query": query,
        "intent_family": route_plan.get("intent_family"),
        "answer_shape": route_plan.get("answer_shape"),
        "conversation_context": route_plan.get("conversation_context"),
        "metadata_query_plan": route_plan.get("metadata_query_plan"),
        "engine_candidates": route_plan.get("candidates"),
        "base_rule_decision": base_decision,
        "instructions": {
            "prefer_metadata_engine_for": ["count", "exists", "explicit_list", "single_paper_stats"],
            "prefer_rag_engine_for": ["paper facts", "table content", "content extraction without structured prefilter"],
            "prefer_hybrid_sql_rag_engine_for": ["structured filters plus content followup", "conversation-scoped content followup"],
            "avoid_metadata_engine_when": ["the answer must explain paper content, outcomes, findings, variables, conclusions"],
        },
        "response_schema": {
            "engine_name": "rag_engine | metadata_engine | hybrid_sql_rag_engine",
            "confidence": "0-1 float",
            "reason": "short snake_case reason",
            "intent_family": "count | list | exists | single_paper_stats | content_extraction | summary | comparison | rag",
            "answer_shape": "scalar | list | paragraph",
        },
    }
    return [
        {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]
