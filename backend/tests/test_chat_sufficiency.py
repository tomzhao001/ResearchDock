import pytest

from app.models import Paper, PaperChunk
from app.services.chat_rag.evidence import (
    build_sufficiency_decision,
    heuristic_select_claim_supporting_evidence,
    heuristic_verify_answer,
)
from app.services.rag import EvidenceSelectionResult, RetrievalResult, create_topic, get_chat_attribution_policy, send_topic_message


def test_get_chat_attribution_policy_maps_scene_variants() -> None:
    factual_policy = get_chat_attribution_policy(
        relaxed_chat_rag=False,
        intent_family="content_extraction",
        has_history=False,
    )
    synthesis_policy = get_chat_attribution_policy(
        relaxed_chat_rag=False,
        intent_family="summary",
        has_history=False,
    )
    followup_policy = get_chat_attribution_policy(
        relaxed_chat_rag=False,
        intent_family="content_extraction",
        has_history=True,
    )

    assert factual_policy.name == "strict_factual_strict"
    assert factual_policy.llm_insufficient_hard_gate is True
    assert factual_policy.allow_partial_answer is False
    assert factual_policy.verifier_requires_claim_coverage is True
    assert factual_policy.verifier_negative_answer_guard == "strict"

    assert synthesis_policy.name == "strict_synthesis_balanced"
    assert synthesis_policy.llm_insufficient_hard_gate is False
    assert synthesis_policy.allow_partial_answer is True
    assert synthesis_policy.verifier_requires_claim_coverage is False
    assert synthesis_policy.verifier_partial_answer_strictness == "balanced"

    assert followup_policy.name == "strict_followup_balanced"
    assert followup_policy.llm_insufficient_hard_gate is False
    assert followup_policy.allow_partial_answer is True
    assert followup_policy.verifier_requires_claim_coverage is True
    assert followup_policy.verifier_negative_answer_guard == "strict"


def test_build_sufficiency_decision_allows_partial_answer_for_synthesis_policy() -> None:
    policy = get_chat_attribution_policy(
        relaxed_chat_rag=False,
        intent_family="summary",
        has_history=False,
    )

    decision = build_sufficiency_decision(
        [{"evidence_id": "chunk-1", "support_score": 0.72}],
        overall_support_score=0.72,
        llm_sufficient=False,
        policy=policy,
    )

    assert decision["is_sufficient"] is False
    assert decision["is_partially_sufficient"] is True
    assert decision["should_generate_answer"] is True
    assert "llm_marked_insufficient_advisory" in decision["reason_codes"]
    assert "partial_answer_allowed" in decision["reason_codes"]


def test_build_sufficiency_decision_keeps_hard_gate_for_factual_policy() -> None:
    policy = get_chat_attribution_policy(
        relaxed_chat_rag=False,
        intent_family="content_extraction",
        has_history=False,
    )

    decision = build_sufficiency_decision(
        [{"evidence_id": "chunk-1", "support_score": 0.92}],
        overall_support_score=0.92,
        llm_sufficient=False,
        policy=policy,
    )

    assert decision["is_sufficient"] is False
    assert decision["is_partially_sufficient"] is False
    assert decision["should_generate_answer"] is False
    assert "llm_marked_insufficient" in decision["reason_codes"]
    assert "llm_marked_insufficient_advisory" not in decision["reason_codes"]


def test_exact_heavy_query_requires_exact_term_coverage() -> None:
    policy = get_chat_attribution_policy(
        relaxed_chat_rag=False,
        intent_family="summary",
        has_history=False,
    )

    result = heuristic_select_claim_supporting_evidence(
        question="论文是否报告了 sex-stratified efficacy difference？",
        query_plan={
            "query_type": "exact_heavy_short",
            "exact_terms": ["sex-stratified", "gender-stratified"],
        },
        evidence_candidates=[
            {
                "evidence_id": "chunk-1",
                "chunk_id": 1,
                "paper_id": 1,
                "paper_title": "Guardrail Paper",
                "source_url": "https://example.com/guardrail",
                "snippet": "The study population was randomized into two treatment groups.",
                "score": 0.98,
                "support_score": 0.95,
                "section_path": "Study population",
                "page_from": 3,
                "page_to": 3,
            }
        ],
        policy=policy,
    )

    assert result.sufficiency_decision["is_sufficient"] is False
    assert result.sufficiency_decision["should_generate_answer"] is False
    assert "exact_terms_not_covered" in result.sufficiency_decision["reason_codes"]


def test_heuristic_verifier_does_not_trust_sufficiency_for_negative_abstention_like_answer() -> None:
    policy = get_chat_attribution_policy(
        relaxed_chat_rag=False,
        intent_family="summary",
        has_history=False,
    )
    selection_result = EvidenceSelectionResult(
        selected_evidence=(
            {
                "evidence_id": "chunk-1",
                "support_score": 0.95,
                "snippet": "The study population was randomized into two groups.",
            },
        ),
        claims=(
            {
                "claim_text": "论文是否报告 sex-stratified 差异",
                "supporting_evidence_ids": ["chunk-1"],
                "support_score": 0.95,
                "selection_reason": "模拟误放行",
            },
        ),
        overall_support_score=0.95,
        sufficiency_decision={
            "is_sufficient": True,
            "is_partially_sufficient": False,
            "should_generate_answer": True,
            "reason_codes": ["sufficient"],
            "policy_name": "strict_synthesis_balanced",
        },
        missing_information="",
        method="mock",
    )

    result = heuristic_verify_answer(
        question="论文是否报告了 sex-stratified efficacy difference？",
        answer="现有证据未提及 sex-stratified efficacy difference，因此当前材料还不能确认。",
        selection_result=selection_result,
        policy=policy,
        verifier_context={"expected_abstention_like_query": True, "partial_answer_mode": False},
    )

    assert result["supported"] is False
    assert result["abstention_recommended"] is True
    assert result["failure_mode"] == "abstention_like_negative_answer"


def test_send_topic_message_uses_partial_sufficiency_answer_path(
    user,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.rag._llm_available_for_query_rewrite", lambda: False)

    paper = Paper(
        organization_id=user.organization_id,
        title="Summary Paper",
        source_url="https://example.com/summary",
        status="completed",
    )
    db_session.add(paper)
    db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        content="The paper reports that tES improved attention scores, while long-term durability remains unclear.",
        embedding=None,
        token_count=14,
        page_from=1,
        page_to=1,
        metadata_json={
            "section_path": "Results",
            "body_text": "The paper reports that tES improved attention scores, while long-term durability remains unclear.",
        },
    )
    db_session.add(chunk)
    db_session.commit()

    policy = get_chat_attribution_policy(
        relaxed_chat_rag=False,
        intent_family="summary",
        has_history=False,
    )
    decision = build_sufficiency_decision(
        [
            {
                "evidence_id": f"chunk-{chunk.id}",
                "chunk_id": chunk.id,
                "paper_id": paper.id,
                "paper_title": paper.title,
                "source_url": paper.source_url,
                "snippet": chunk.content,
                "support_score": 0.72,
                "section_path": "Results",
                "page_from": 1,
                "page_to": 1,
            }
        ],
        overall_support_score=0.72,
        llm_sufficient=False,
        policy=policy,
    )

    def fake_select_claim_supporting_evidence(**kwargs):
        selected = list(kwargs["evidence_candidates"][:1])
        selected[0]["support_score"] = 0.72
        selected[0]["selection_reason"] = "可支持主要发现，但并不能覆盖全部结论"
        selected[0]["claim_texts"] = ["论文主要发现"]
        return EvidenceSelectionResult(
            selected_evidence=tuple(selected),
            claims=(
                {
                    "claim_text": "论文报告了主要发现",
                    "supporting_evidence_ids": [str(selected[0]["evidence_id"])],
                    "support_score": 0.72,
                    "selection_reason": "结果段可支持主要发现",
                },
            ),
            overall_support_score=0.72,
            sufficiency_decision=decision,
            missing_information="长期效果缺少足够证据。",
            method="mock",
        )

    def fake_verify_grounded_answer(**_kwargs):
        return {
            "method": "mock",
            "supported": True,
            "support_score": 0.72,
            "unsupported_claims": [],
            "notes": "partial answer verified",
        }

    def fake_chat(messages, *, temperature: float = 0.3):
        user_content = messages[-1]["content"]
        assert "partial_answer_mode: True" in user_content
        return "现有证据显示，论文报告 tES 改善了注意力表现；至于长期持续性，当前材料还不能确认。", "rag-model"

    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        fake_select_claim_supporting_evidence,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.verify_grounded_answer",
        fake_verify_grounded_answer,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.retrieval.search_chunks",
        lambda *args, **kwargs: [RetrievalResult(chunk=chunk, paper=paper, score=0.91)],
    )
    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    topic = create_topic(db_session, user=user, title="Sufficiency Partial")
    result = send_topic_message(
        db_session,
        user=user,
        topic_id=topic.topic.id,
        prompt="请总结这篇论文的主要发现。",
    )

    assert result.assistant_message.answer_mode == "knowledge_base"
    assert result.assistant_message.used_knowledge_base is True
    assert result.assistant_message.model == "rag-model"
    assert len(result.assistant_message.citations_json) == 1
    retrieval = result.assistant_message.metadata_json["retrieval"]
    assert retrieval["sufficiency_decision"]["is_partially_sufficient"] is True
    assert retrieval["chat_response_policy"]["name"] == "strict_synthesis_balanced"
    assert retrieval["answer_mode"] == "knowledge_base"


def test_send_topic_message_keeps_abstention_like_negative_answer_on_safe_path(
    user,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.rag._llm_available_for_query_rewrite", lambda: False)

    paper = Paper(
        organization_id=user.organization_id,
        title="Abstention Paper",
        source_url="https://example.com/abstention",
        status="completed",
    )
    db_session.add(paper)
    db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        content="The study population was randomized into two treatment groups.",
        embedding=None,
        token_count=10,
        page_from=3,
        page_to=3,
        metadata_json={
            "section_path": "Study population",
            "body_text": "The study population was randomized into two treatment groups.",
        },
    )
    db_session.add(chunk)
    db_session.commit()

    def fake_select_claim_supporting_evidence(**kwargs):
        selected = list(kwargs["evidence_candidates"][:1])
        selected[0]["support_score"] = 0.95
        selected[0]["selection_reason"] = "高分但并未覆盖用户关心的分层结果"
        selected[0]["claim_texts"] = ["论文是否报告 sex-stratified 差异"]
        return EvidenceSelectionResult(
            selected_evidence=tuple(selected),
            claims=(
                {
                    "claim_text": "论文是否报告 sex-stratified 差异",
                    "supporting_evidence_ids": [str(selected[0]["evidence_id"])],
                    "support_score": 0.95,
                    "selection_reason": "先模拟一条被误判为充分的证据",
                },
            ),
            overall_support_score=0.95,
            sufficiency_decision={
                "is_sufficient": True,
                "is_partially_sufficient": False,
                "should_generate_answer": True,
                "reason_codes": ["sufficient"],
                "policy_name": "strict_synthesis_balanced",
            },
            missing_information="",
            method="mock",
        )

    def fake_verify_grounded_answer(**_kwargs):
        return {
            "method": "mock",
            "supported": True,
            "support_score": 0.93,
            "unsupported_claims": [],
            "notes": "negative answer is grounded but should stay abstentive",
        }

    def fake_chat(messages, *, temperature: float = 0.3):
        return "现有证据未提及 sex-stratified efficacy difference，因此当前材料还不能确认。", "rag-model"

    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        fake_select_claim_supporting_evidence,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.verify_grounded_answer",
        fake_verify_grounded_answer,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.retrieval.search_chunks",
        lambda *args, **kwargs: [RetrievalResult(chunk=chunk, paper=paper, score=0.97)],
    )
    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    topic = create_topic(db_session, user=user, title="Abstention Guardrail")
    result = send_topic_message(
        db_session,
        user=user,
        topic_id=topic.topic.id,
        prompt="论文是否报告了 sex-stratified efficacy difference？",
    )

    assert result.assistant_message.answer_mode == "kb_insufficient_evidence"
    assert result.assistant_message.used_knowledge_base is False
    assert result.assistant_message.content == "知识库中未找到确切依据。"


def test_send_topic_message_abstains_when_verifier_claim_coverage_is_low(
    user,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.rag._llm_available_for_query_rewrite", lambda: False)

    paper = Paper(
        organization_id=user.organization_id,
        title="Coverage Paper",
        source_url="https://example.com/coverage",
        status="completed",
    )
    db_session.add(paper)
    db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        content="tES is short for transcranial electrical stimulation.",
        embedding=None,
        token_count=7,
        page_from=1,
        page_to=1,
        metadata_json={"section_path": "Intro", "body_text": "tES is short for transcranial electrical stimulation."},
    )
    db_session.add(chunk)
    db_session.commit()

    def fake_select_claim_supporting_evidence(**kwargs):
        selected = list(kwargs["evidence_candidates"][:1])
        selected[0]["support_score"] = 0.92
        selected[0]["selection_reason"] = "只覆盖其中一个 claim"
        selected[0]["claim_texts"] = ["tES 的含义", "是否按性别分层报告结果"]
        return EvidenceSelectionResult(
            selected_evidence=tuple(selected),
            claims=(
                {
                    "claim_text": "tES 的含义",
                    "supporting_evidence_ids": [str(selected[0]["evidence_id"])],
                    "support_score": 0.92,
                    "selection_reason": "术语可确认",
                },
                {
                    "claim_text": "是否按性别分层报告结果",
                    "supporting_evidence_ids": [],
                    "support_score": 0.0,
                    "selection_reason": "当前证据未覆盖",
                },
            ),
            overall_support_score=0.92,
            sufficiency_decision={
                "is_sufficient": True,
                "is_partially_sufficient": False,
                "should_generate_answer": True,
                "reason_codes": ["sufficient"],
                "policy_name": "strict_factual_strict",
            },
            missing_information="",
            method="mock",
        )

    def fake_verify_grounded_answer(**_kwargs):
        return {
            "method": "mock",
            "supported": True,
            "support_score": 0.93,
            "claim_coverage": 0.4,
            "unsupported_claims": [],
            "missing_claims": ["是否按性别分层报告结果"],
            "abstention_recommended": False,
            "failure_mode": "verifier_claim_coverage_low",
            "notes": "claim coverage too low",
        }

    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        fake_select_claim_supporting_evidence,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.verify_grounded_answer",
        fake_verify_grounded_answer,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.retrieval.search_chunks",
        lambda *args, **kwargs: [RetrievalResult(chunk=chunk, paper=paper, score=0.97)],
    )
    monkeypatch.setattr(
        "app.services.rag.chat_with_messages",
        lambda *args, **kwargs: ("tES 指 transcranial electrical stimulation，而且论文还按性别分层报告了主要终点。", "rag-model"),
    )

    topic = create_topic(db_session, user=user, title="Coverage Guard")
    result = send_topic_message(
        db_session,
        user=user,
        topic_id=topic.topic.id,
        prompt="文中的 tES 指什么？",
    )

    assert result.assistant_message.answer_mode == "kb_insufficient_evidence"
    assert result.assistant_message.used_knowledge_base is False
    assert result.assistant_message.metadata_json["retrieval"]["verifier_failure_mode"] == "verifier_claim_coverage_low"


def test_multi_turn_negative_followup_passes_history_to_verifier_and_abstains(
    user,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.rag._llm_available_for_query_rewrite", lambda: False)

    paper = Paper(
        organization_id=user.organization_id,
        title="Followup Paper",
        source_url="https://example.com/followup",
        status="completed",
    )
    db_session.add(paper)
    db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        content="The primary outcome measure is the total score of the ADHD-RS questionnaire completed by parents.",
        embedding=None,
        token_count=15,
        page_from=1,
        page_to=1,
        metadata_json={"section_path": "Methods", "body_text": "The primary outcome measure is the total score of the ADHD-RS questionnaire completed by parents."},
    )
    db_session.add(chunk)
    db_session.commit()

    def fake_select_claim_supporting_evidence(**kwargs):
        selected = list(kwargs["evidence_candidates"][:1])
        selected[0]["support_score"] = 0.9
        selected[0]["selection_reason"] = "只覆盖主要终点定义"
        selected[0]["claim_texts"] = [kwargs["question"]]
        return EvidenceSelectionResult(
            selected_evidence=tuple(selected),
            claims=(
                {
                    "claim_text": kwargs["question"],
                    "supporting_evidence_ids": [str(selected[0]["evidence_id"])],
                    "support_score": 0.9,
                    "selection_reason": "当前仅能确认主要终点定义",
                },
            ),
            overall_support_score=0.9,
            sufficiency_decision={
                "is_sufficient": True,
                "is_partially_sufficient": False,
                "should_generate_answer": True,
                "reason_codes": ["sufficient"],
                "policy_name": "strict_followup_balanced" if "性别分层" in kwargs["question"] else "strict_factual_strict",
            },
            missing_information="",
            method="mock",
        )

    def fake_verify_grounded_answer(**kwargs):
        if "性别分层" in kwargs["question"]:
            assert kwargs["verifier_context"]["expected_abstention_like_query"] is True
            assert any("主要终点" in item["content"] for item in kwargs["verifier_context"]["conversation_history_excerpt"])
            return {
                "method": "mock",
                "supported": True,
                "support_score": 0.9,
                "claim_coverage": 1.0,
                "unsupported_claims": [],
                "missing_claims": [],
                "abstention_recommended": True,
                "failure_mode": "abstention_like_negative_answer",
                "notes": "followup negative should abstain",
            }
        return {
            "method": "mock",
            "supported": True,
            "support_score": 0.95,
            "claim_coverage": 1.0,
            "unsupported_claims": [],
            "missing_claims": [],
            "abstention_recommended": False,
            "failure_mode": None,
            "notes": "baseline supported",
        }

    def fake_chat(messages, *, temperature: float = 0.3):
        user_content = messages[-1]["content"]
        if "性别分层" in user_content:
            return "现有证据未提及按性别分层后的主要终点差异，因此当前材料还不能确认。", "rag-model"
        return "本文的主要终点是由家长填写的 ADHD-RS 总分。", "rag-model"

    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        fake_select_claim_supporting_evidence,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.verify_grounded_answer",
        fake_verify_grounded_answer,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.retrieval.search_chunks",
        lambda *args, **kwargs: [RetrievalResult(chunk=chunk, paper=paper, score=0.96)],
    )
    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    topic = create_topic(db_session, user=user, title="Followup Negative")
    first = send_topic_message(
        db_session,
        user=user,
        topic_id=topic.topic.id,
        prompt="这里的主要终点具体是哪份量表？",
    )
    assert first.assistant_message.answer_mode == "knowledge_base"

    second = send_topic_message(
        db_session,
        user=user,
        topic_id=topic.topic.id,
        prompt="那按性别分层后，这个主要终点有没有显著差异？",
    )
    assert second.assistant_message.answer_mode == "kb_insufficient_evidence"
    assert second.assistant_message.metadata_json["retrieval"]["verifier_failure_mode"] == "abstention_like_negative_answer"


def test_send_topic_message_recovers_from_hard_abstain_with_verifier_guided_partial_answer(
    user,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.rag._llm_available_for_query_rewrite", lambda: False)

    paper = Paper(
        organization_id=user.organization_id,
        title="Recovery Paper",
        source_url="https://example.com/recovery",
        status="completed",
    )
    db_session.add(paper)
    db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        content="tRNS yielded clinical improvement and improved working memory compared with tDCS.",
        embedding=None,
        token_count=12,
        page_from=1,
        page_to=1,
        metadata_json={"section_path": "Abstract", "body_text": "tRNS yielded clinical improvement and improved working memory compared with tDCS."},
    )
    db_session.add(chunk)
    db_session.commit()

    def fake_select_claim_supporting_evidence(**kwargs):
        selected = list(kwargs["evidence_candidates"][:1])
        selected[0]["support_score"] = 0.84
        selected[0]["selection_reason"] = "可支持摘要级总结"
        selected[0]["claim_texts"] = ["tRNS 相比 tDCS 的临床和认知改善"]
        return EvidenceSelectionResult(
            selected_evidence=tuple(selected),
            claims=(
                {
                    "claim_text": "tRNS 相比 tDCS 的临床和认知改善",
                    "supporting_evidence_ids": [str(selected[0]["evidence_id"])],
                    "support_score": 0.84,
                    "selection_reason": "摘要段支持核心结果",
                },
            ),
            overall_support_score=0.84,
            sufficiency_decision={
                "is_sufficient": True,
                "is_partially_sufficient": False,
                "should_generate_answer": True,
                "reason_codes": ["sufficient", "relaxed_chat_policy"],
                "policy_name": "strict_synthesis_balanced",
            },
            missing_information="",
            method="mock",
        )

    def fake_verify_grounded_answer(**_kwargs):
        return {
            "method": "mock",
            "supported": True,
            "support_score": 0.84,
            "claim_coverage": 1.0,
            "unsupported_claims": [],
            "missing_claims": [],
            "abstention_recommended": False,
            "failure_mode": None,
            "notes": "supported after hard abstain draft",
        }

    def fake_chat(messages, *, temperature: float = 0.3):
        user_content = messages[-1]["content"]
        if "fallback_reason: verifier_guided_partial_answer" in user_content:
            return "现有证据表明，tRNS 相比 tDCS 带来了临床症状改善，并提升了工作记忆表现。", "fallback-model"
        return "知识库中未找到确切依据。", "rag-model"

    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        fake_select_claim_supporting_evidence,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.verify_grounded_answer",
        fake_verify_grounded_answer,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.retrieval.search_chunks",
        lambda *args, **kwargs: [RetrievalResult(chunk=chunk, paper=paper, score=0.95)],
    )
    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    topic = create_topic(db_session, user=user, title="Verifier Guided Recovery")
    result = send_topic_message(
        db_session,
        user=user,
        topic_id=topic.topic.id,
        prompt="请概括论文摘要中的核心结果：tRNS 相对 tDCS 带来了哪些临床和认知变化？",
    )

    assert result.assistant_message.answer_mode == "knowledge_base"
    assert result.assistant_message.used_knowledge_base is True
    assert result.assistant_message.model == "fallback-model"
    assert "临床症状改善" in result.assistant_message.content
    assert result.assistant_message.metadata_json["retrieval"]["fallback_reason"] == "verifier_guided_partial_answer"


def test_send_topic_message_recovers_from_verifier_guided_partial_answer(
    user,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.rag._llm_available_for_query_rewrite", lambda: False)

    paper = Paper(
        organization_id=user.organization_id,
        title="Verifier Partial Paper",
        source_url="https://example.com/verifier-partial",
        status="completed",
    )
    db_session.add(paper)
    db_session.flush()
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_index=0,
        content="The within-subject design controlled for individual differences.",
        embedding=None,
        token_count=8,
        page_from=2,
        page_to=2,
        metadata_json={"section_path": "Study design", "body_text": "The within-subject design controlled for individual differences."},
    )
    db_session.add(chunk)
    db_session.commit()

    def fake_select_claim_supporting_evidence(**kwargs):
        selected = list(kwargs["evidence_candidates"][:1])
        selected[0]["support_score"] = 0.8
        selected[0]["selection_reason"] = "只覆盖问题的一部分"
        selected[0]["claim_texts"] = ["控制个体差异"]
        return EvidenceSelectionResult(
            selected_evidence=tuple(selected),
            claims=(
                {
                    "claim_text": "为什么选择 within-subject design",
                    "supporting_evidence_ids": [str(selected[0]["evidence_id"])],
                    "support_score": 0.8,
                    "selection_reason": "设计理由可支持",
                },
                {
                    "claim_text": "样本量对比",
                    "supporting_evidence_ids": [],
                    "support_score": 0.0,
                    "selection_reason": "当前缺少样本量对比证据",
                },
            ),
            overall_support_score=0.8,
            sufficiency_decision={
                "is_sufficient": True,
                "is_partially_sufficient": False,
                "should_generate_answer": True,
                "reason_codes": ["sufficient", "relaxed_chat_policy"],
                "policy_name": "strict_synthesis_balanced",
            },
            missing_information="",
            method="mock",
        )

    def fake_verify_grounded_answer(**_kwargs):
        return {
            "method": "mock",
            "supported": False,
            "support_score": 0.45,
            "claim_coverage": 0.5,
            "unsupported_claims": [],
            "missing_claims": ["样本量对比"],
            "abstention_recommended": True,
            "failure_mode": "verifier_claim_coverage_low",
            "notes": "missing one key claim but partial answer is possible",
        }

    def fake_chat(messages, *, temperature: float = 0.3):
        user_content = messages[-1]["content"]
        if "fallback_reason: verifier_guided_partial_answer" in user_content:
            return "现有证据表明，作者选择 within-subject design 是为了更好地控制个体差异；至于样本量对比，当前材料还不能完整确认。", "fallback-model"
        return "知识库中未找到确切依据。", "rag-model"

    monkeypatch.setattr(
        "app.services.chat_rag.evidence.select_claim_supporting_evidence",
        fake_select_claim_supporting_evidence,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.evidence.verify_grounded_answer",
        fake_verify_grounded_answer,
    )
    monkeypatch.setattr(
        "app.services.chat_rag.retrieval.search_chunks",
        lambda *args, **kwargs: [RetrievalResult(chunk=chunk, paper=paper, score=0.94)],
    )
    monkeypatch.setattr("app.services.rag.chat_with_messages", fake_chat)

    topic = create_topic(db_session, user=user, title="Verifier Partial Recovery")
    result = send_topic_message(
        db_session,
        user=user,
        topic_id=topic.topic.id,
        prompt="作者为什么选择 within-subject design，并给出了怎样的样本量对比？",
    )

    assert result.assistant_message.answer_mode == "knowledge_base"
    assert result.assistant_message.used_knowledge_base is True
    assert result.assistant_message.model == "fallback-model"
    assert "控制个体差异" in result.assistant_message.content
