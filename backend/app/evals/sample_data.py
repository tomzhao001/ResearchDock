from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.auth import pwd_context
from app.models import Paper, PaperChunk, User
from app.services.llm import chat_with_messages
from app.services.papers import create_upload_artifacts, run_pdf_ingest_job
from app.services.pdf_extraction import PDFTextExtractor
from app.services.rag import _search_chunks, create_topic, send_topic_message

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE_DATA_DIR = _REPO_ROOT / "benchmarks" / "sample-data"
_ABSTAIN_PHRASE = "知识库中未找到确切依据"


@dataclass(frozen=True)
class SamplePaperSpec:
    key: str
    title: str
    language: str
    relative_pdf_path: str
    upload_filename: str
    source_url: str | None = None
    doi: str | None = None

    @property
    def pdf_path(self) -> Path:
        return _REPO_ROOT / self.relative_pdf_path


@dataclass(frozen=True)
class GoldEvidence:
    paper_key: str
    snippet: str


@dataclass(frozen=True)
class SampleQuestion:
    q_id: str
    question: str
    category: str
    difficulty: str
    language: str
    requires_multi_span: bool
    needs_table_or_figure: bool
    allow_fallback_general: bool
    expected_abstention: bool
    gold_evidence: tuple[GoldEvidence, ...]
    expected_keywords: tuple[str, ...]
    keyword_hit_threshold: int
    session_id: str | None = None
    turn_index: int | None = None


@dataclass(frozen=True)
class SampleSession:
    session_id: str
    question_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedQuestion:
    question: SampleQuestion
    gold_chunk_ids: tuple[int, ...]
    gold_chunk_indices: tuple[int, ...]
    gold_paper_ids: tuple[int, ...]
    gold_evidence_texts: tuple[str, ...]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text or ""))
    lowered = "".join(char for char in normalized.lower() if not unicodedata.combining(char))
    lowered = lowered.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", lowered)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ndcg_at(relevant_ids: set[int], retrieved_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    truncated = retrieved_ids[:k]
    dcg = 0.0
    for index, chunk_id in enumerate(truncated, start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _snippet_matches_chunk(normalized_snippet: str, normalized_chunk: str) -> bool:
    if not normalized_snippet:
        return False
    if normalized_snippet in normalized_chunk:
        return True
    for window_size in (24, 18, 12, 8):
        if len(normalized_snippet) < window_size:
            continue
        for start in range(0, len(normalized_snippet) - window_size + 1):
            if normalized_snippet[start : start + window_size] in normalized_chunk:
                return True
    return False


def _token_sequence_matches(snippet: str, chunk_text: str) -> bool:
    snippet_tokens = re.findall(r"[A-Za-z]+|[-+]?\d+(?:\.\d+)?", snippet or "")
    chunk_tokens = re.findall(r"[A-Za-z]+|[-+]?\d+(?:\.\d+)?", chunk_text or "")
    if len(snippet_tokens) < 3 or len(chunk_tokens) < len(snippet_tokens):
        return False
    target = [token.lower() for token in snippet_tokens]
    source = [token.lower() for token in chunk_tokens]
    for start in range(0, len(source) - len(target) + 1):
        if source[start : start + len(target)] == target:
            return True
    return False


def _evidence_matches_chunk(snippet: str, chunk_text: str) -> bool:
    normalized_snippet = _normalize_text(snippet)
    normalized_chunk = _normalize_text(chunk_text)
    return _snippet_matches_chunk(normalized_snippet, normalized_chunk) or _token_sequence_matches(snippet, chunk_text)


def _chunk_match_text(chunk: PaperChunk) -> str:
    metadata = chunk.metadata_json if isinstance(chunk.metadata_json, dict) else {}
    body_text = metadata.get("body_text") if isinstance(metadata, dict) else None
    return str(body_text or chunk.content or "")


def _summarize_groups(
    items: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
    score_fns: dict[str, Callable[[dict[str, Any]], float]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = key_fn(item)
        grouped.setdefault(str(key), []).append(item)
    summary: dict[str, dict[str, float]] = {}
    for key, values in grouped.items():
        summary[key] = {score_name: round(_mean([score_fn(value) for value in values]), 4) for score_name, score_fn in score_fns.items()}
        summary[key]["count"] = len(values)
    return summary


def load_sample_data_paper_specs(data_dir: Path | None = None) -> list[SamplePaperSpec]:
    payload = _read_json((data_dir or _SAMPLE_DATA_DIR) / "papers.json")
    specs = [
        SamplePaperSpec(
            key=item["key"],
            title=item["title"],
            language=item["language"],
            relative_pdf_path=item["relative_pdf_path"],
            upload_filename=item["upload_filename"],
            source_url=item.get("source_url"),
            doi=item.get("doi"),
        )
        for item in payload
    ]
    keys = [item.key for item in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate paper keys found in sample-data papers.json")
    return specs


def load_sample_data_questions(data_dir: Path | None = None) -> list[SampleQuestion]:
    rows = _read_jsonl((data_dir or _SAMPLE_DATA_DIR) / "questions.jsonl")
    questions: list[SampleQuestion] = []
    for item in rows:
        gold_evidence = tuple(GoldEvidence(paper_key=evidence["paper_key"], snippet=evidence["snippet"]) for evidence in item.get("gold_evidence", []))
        questions.append(
            SampleQuestion(
                q_id=item["q_id"],
                question=item["question"],
                category=item["category"],
                difficulty=item["difficulty"],
                language=item["language"],
                requires_multi_span=bool(item["requires_multi_span"]),
                needs_table_or_figure=bool(item["needs_table_or_figure"]),
                allow_fallback_general=bool(item["allow_fallback_general"]),
                expected_abstention=bool(item["expected_abstention"]),
                gold_evidence=gold_evidence,
                expected_keywords=tuple(item.get("expected_keywords", [])),
                keyword_hit_threshold=int(item.get("keyword_hit_threshold", 1)),
                session_id=item.get("session_id"),
                turn_index=item.get("turn_index"),
            )
        )
    qids = [item.q_id for item in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate q_id found in sample-data questions.jsonl")
    return questions


def load_sample_data_sessions(data_dir: Path | None = None) -> list[SampleSession]:
    rows = _read_jsonl((data_dir or _SAMPLE_DATA_DIR) / "sessions.jsonl")
    return [SampleSession(session_id=item["session_id"], question_ids=tuple(item["question_ids"])) for item in rows]


def load_sample_data_smoke_question_ids(data_dir: Path | None = None) -> set[str]:
    payload = _read_json((data_dir or _SAMPLE_DATA_DIR) / "smoke_question_ids.json")
    return {str(item) for item in payload}


def ensure_sample_data_eval_user(db: Session) -> User:
    user = db.scalar(select(User).where(User.username == "sample_data_eval"))
    if user is not None:
        return user
    now = datetime.now(timezone.utc)
    user = User(
        username="sample_data_eval",
        password_hash=pwd_context.hash("sample_data_eval"),
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def ingest_sample_data_papers(
    db: Session,
    *,
    session_factory: sessionmaker,
    extractor: PDFTextExtractor | None = None,
    data_dir: Path | None = None,
) -> dict[str, Paper]:
    papers_by_key: dict[str, Paper] = {}
    for spec in load_sample_data_paper_specs(data_dir):
        payload = spec.pdf_path.read_bytes()
        upload = create_upload_artifacts(
            db,
            filename=spec.upload_filename,
            content_type="application/pdf",
            payload=payload,
            overwrite=True,
        )
        run_pdf_ingest_job(upload.job_id, session_factory=session_factory, extractor=extractor)
        paper = db.get(Paper, upload.paper_id)
        if paper is None:
            raise RuntimeError(f"Failed to ingest sample paper: {spec.key}")
        paper.title = spec.title
        paper.source_url = spec.source_url
        paper.doi = spec.doi
        paper.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(paper)
        papers_by_key[spec.key] = paper
    return papers_by_key


def _load_chunk_cache(db: Session, papers_by_key: dict[str, Paper]) -> dict[str, list[PaperChunk]]:
    cache: dict[str, list[PaperChunk]] = {}
    for paper_key, paper in papers_by_key.items():
        chunks = db.scalars(
            select(PaperChunk)
            .where(PaperChunk.paper_id == paper.id)
            .order_by(PaperChunk.chunk_index.asc(), PaperChunk.id.asc())
        ).all()
        cache[paper_key] = chunks
    return cache


def resolve_question_gold(
    questions: list[SampleQuestion],
    *,
    papers_by_key: dict[str, Paper],
    chunk_cache: dict[str, list[PaperChunk]],
) -> list[ResolvedQuestion]:
    resolved: list[ResolvedQuestion] = []
    for question in questions:
        gold_chunk_ids: list[int] = []
        gold_chunk_indices: list[int] = []
        gold_paper_ids: list[int] = []
        gold_evidence_texts: list[str] = []
        for evidence in question.gold_evidence:
            chunks = chunk_cache.get(evidence.paper_key)
            if chunks is None:
                raise ValueError(f"Unknown paper key in evidence: {evidence.paper_key}")
            match = next(
                (
                    chunk
                    for chunk in chunks
                    if _evidence_matches_chunk(evidence.snippet, _chunk_match_text(chunk))
                ),
                None,
            )
            matched_chunks: list[PaperChunk] = [match] if match is not None else []
            if match is None:
                for window_size in (2, 3, 4, 5, 6, 8):
                    for start in range(0, max(len(chunks) - window_size + 1, 0)):
                        window = chunks[start : start + window_size]
                        combined_text = " ".join(_chunk_match_text(chunk) for chunk in window)
                        if _evidence_matches_chunk(evidence.snippet, combined_text):
                            matched_chunks = window
                            break
                    if matched_chunks:
                        break
            if not matched_chunks:
                raise ValueError(f"Unable to resolve gold evidence for {question.q_id}: {evidence.snippet}")
            for matched_chunk in matched_chunks:
                if matched_chunk.id in gold_chunk_ids:
                    continue
                gold_chunk_ids.append(matched_chunk.id)
                gold_chunk_indices.append(matched_chunk.chunk_index)
                gold_paper_ids.append(matched_chunk.paper_id)
            gold_evidence_texts.append(evidence.snippet)
        resolved.append(
            ResolvedQuestion(
                question=question,
                gold_chunk_ids=tuple(gold_chunk_ids),
                gold_chunk_indices=tuple(gold_chunk_indices),
                gold_paper_ids=tuple(gold_paper_ids),
                gold_evidence_texts=tuple(gold_evidence_texts),
            )
        )
    return resolved


def evaluate_retrieval(
    db: Session,
    questions: list[ResolvedQuestion],
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    max_k = max(k_values)
    rows: list[dict[str, Any]] = []
    for resolved in questions:
        if not resolved.gold_chunk_ids:
            continue
        results = _search_chunks(db, query=resolved.question.question, top_k=max_k)
        retrieved_ids = [item.chunk.id for item in results]
        gold_set = set(resolved.gold_chunk_ids)
        first_hit_rank = next((index for index, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in gold_set), None)
        row = {
            "q_id": resolved.question.q_id,
            "question": resolved.question.question,
            "category": resolved.question.category,
            "language": resolved.question.language,
            "difficulty": resolved.question.difficulty,
            "expected_abstention": resolved.question.expected_abstention,
            "gold_chunk_ids": list(resolved.gold_chunk_ids),
            "retrieved": [
                {
                    "chunk_id": item.chunk.id,
                    "chunk_index": item.chunk.chunk_index,
                    "paper_id": item.paper.id,
                    "paper_title": item.paper.title,
                    "score": round(item.score, 4),
                }
                for item in results
            ],
            "mrr": round(1.0 / first_hit_rank, 4) if first_hit_rank else 0.0,
        }
        for k in k_values:
            row[f"hit@{k}"] = any(chunk_id in gold_set for chunk_id in retrieved_ids[:k])
            row[f"ndcg@{k}"] = round(_ndcg_at(gold_set, retrieved_ids, k), 4)
        rows.append(row)

    summary = {
        "count": len(rows),
        "skipped_without_gold": len(questions) - len(rows),
        **{f"hit@{k}": round(_mean([1.0 if row[f"hit@{k}"] else 0.0 for row in rows]), 4) for k in k_values},
        **{f"ndcg@{k}": round(_mean([float(row[f"ndcg@{k}"]) for row in rows]), 4) for k in k_values},
        "mrr": round(_mean([float(row["mrr"]) for row in rows]), 4),
    }
    breakdown = {
        "by_category": _summarize_groups(
            rows,
            key_fn=lambda row: row["category"],
            score_fns={"hit@10": lambda row: 1.0 if row.get("hit@10") else 0.0, "mrr": lambda row: float(row["mrr"])},
        ),
        "by_language": _summarize_groups(
            rows,
            key_fn=lambda row: row["language"],
            score_fns={"hit@10": lambda row: 1.0 if row.get("hit@10") else 0.0, "mrr": lambda row: float(row["mrr"])},
        ),
    }
    return {"summary": summary, "breakdown": breakdown, "questions": rows}


def _extract_json(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return json.loads(text[text.find("{") : text.rfind("}") + 1])


def _grade_heuristic(resolved: ResolvedQuestion, answer: str, citations: list[dict], answer_mode: str | None) -> dict[str, Any]:
    normalized_answer = _normalize_text(answer)
    keyword_hits = sum(1 for keyword in resolved.question.expected_keywords if _normalize_text(keyword) in normalized_answer)
    cited_gold = sum(1 for citation in citations if citation.get("chunk_id") in resolved.gold_chunk_ids)
    citation_precision = cited_gold / len(citations) if citations else (1.0 if resolved.question.expected_abstention else 0.0)
    abstained = _ABSTAIN_PHRASE in (answer or "")
    abstention_accuracy = abstained == resolved.question.expected_abstention
    grounded = abstention_accuracy if resolved.question.expected_abstention else (
        cited_gold > 0 and keyword_hits >= resolved.question.keyword_hit_threshold and answer_mode == "knowledge_base"
    )
    return {
        "keyword_hits": keyword_hits,
        "cited_gold": cited_gold,
        "citation_precision": round(citation_precision, 4),
        "abstained": abstained,
        "abstention_accuracy": abstention_accuracy,
        "grounded": grounded,
    }


def _grade_with_llm(resolved: ResolvedQuestion, answer: str, citations: list[dict], heuristic: dict[str, Any]) -> dict[str, Any]:
    if resolved.question.expected_abstention:
        return {"grounded": heuristic["grounded"], "citation_precision": heuristic["citation_precision"], "notes": "heuristic-abstention"}
    evidence_lines = "\n\n".join(f"- {item}" for item in resolved.gold_evidence_texts)
    citation_lines = "\n".join(f"- {citation.get('snippet', '')}" for citation in citations)
    prompt = (
        "请你作为论文问答评测器，只返回 JSON。\n"
        "字段必须包含 grounded(boolean), citation_precision(number), notes(string)。\n"
        "判断 grounded 时，只看答案是否被 gold evidence 支持；判断 citation_precision 时，给 0 到 1 的分数。\n\n"
        f"问题：{resolved.question.question}\n\n"
        f"答案：{answer}\n\n"
        f"Gold evidence:\n{evidence_lines}\n\n"
        f"Returned citations:\n{citation_lines or '- none'}\n"
    )
    try:
        content, _ = chat_with_messages(
            [
                {"role": "system", "content": "你是严格的 RAG 评测器。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        parsed = _extract_json(content)
        return {
            "grounded": bool(parsed.get("grounded")),
            "citation_precision": float(parsed.get("citation_precision", heuristic["citation_precision"])),
            "notes": str(parsed.get("notes", "")),
        }
    except Exception:
        return {"grounded": heuristic["grounded"], "citation_precision": heuristic["citation_precision"], "notes": "heuristic-fallback"}


def evaluate_end_to_end(
    db: Session,
    resolved_questions: list[ResolvedQuestion],
    *,
    judge_mode: str = "heuristic",
    sessions: list[SampleSession] | None = None,
) -> dict[str, Any]:
    user = ensure_sample_data_eval_user(db)
    question_by_id = {item.question.q_id: item for item in resolved_questions}
    rows: list[dict[str, Any]] = []
    session_qids = {q_id for session in sessions or [] for q_id in session.question_ids}

    def run_question(resolved: ResolvedQuestion, *, topic_id: int) -> None:
        result = send_topic_message(db, user=user, topic_id=topic_id, prompt=resolved.question.question)
        assistant = result.assistant_message
        citations = assistant.citations_json if isinstance(assistant.citations_json, list) else []
        heuristic = _grade_heuristic(resolved, assistant.content, citations, assistant.answer_mode)
        llm_grade = _grade_with_llm(resolved, assistant.content, citations, heuristic) if judge_mode == "llm" else None
        grounded = bool(llm_grade["grounded"]) if llm_grade else bool(heuristic["grounded"])
        citation_precision = float(llm_grade["citation_precision"]) if llm_grade else float(heuristic["citation_precision"])
        rows.append(
            {
                "q_id": resolved.question.q_id,
                "question": resolved.question.question,
                "category": resolved.question.category,
                "language": resolved.question.language,
                "answer_mode": assistant.answer_mode,
                "used_knowledge_base": assistant.used_knowledge_base,
                "expected_abstention": resolved.question.expected_abstention,
                "answer": assistant.content,
                "citations": citations,
                "metadata": assistant.metadata_json if isinstance(assistant.metadata_json, dict) else {},
                "grounded": grounded,
                "citation_precision": round(citation_precision, 4),
                "abstention_accuracy": bool(heuristic["abstention_accuracy"]),
                "keyword_hits": int(heuristic["keyword_hits"]),
                "cited_gold": int(heuristic["cited_gold"]),
                "judge_notes": llm_grade["notes"] if llm_grade else "heuristic",
            }
        )

    for session in sessions or []:
        topic = create_topic(db, user=user, title=f"SampleData {session.session_id}")
        for q_id in session.question_ids:
            run_question(question_by_id[q_id], topic_id=topic.topic.id)

    for resolved in resolved_questions:
        if resolved.question.q_id in session_qids:
            continue
        topic = create_topic(db, user=user, title=f"SampleData {resolved.question.q_id}")
        run_question(resolved, topic_id=topic.topic.id)

    summary = {
        "count": len(rows),
        "groundedness": round(_mean([1.0 if row["grounded"] else 0.0 for row in rows]), 4),
        "citation_precision": round(_mean([float(row["citation_precision"]) for row in rows]), 4),
        "abstention_accuracy": round(_mean([1.0 if row["abstention_accuracy"] else 0.0 for row in rows]), 4),
    }
    breakdown = {
        "by_category": _summarize_groups(
            rows,
            key_fn=lambda row: row["category"],
            score_fns={
                "groundedness": lambda row: 1.0 if row["grounded"] else 0.0,
                "abstention_accuracy": lambda row: 1.0 if row["abstention_accuracy"] else 0.0,
            },
        ),
        "by_language": _summarize_groups(
            rows,
            key_fn=lambda row: row["language"],
            score_fns={
                "groundedness": lambda row: 1.0 if row["grounded"] else 0.0,
                "abstention_accuracy": lambda row: 1.0 if row["abstention_accuracy"] else 0.0,
            },
        ),
    }
    return {"summary": summary, "breakdown": breakdown, "questions": rows}


def run_sample_data_evaluation(
    session_factory: sessionmaker,
    *,
    mode: str = "both",
    subset: str = "full",
    judge_mode: str = "heuristic",
    extractor: PDFTextExtractor | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    db = session_factory()
    try:
        papers_by_key = ingest_sample_data_papers(db, session_factory=session_factory, extractor=extractor, data_dir=data_dir)
        questions = load_sample_data_questions(data_dir)
        sessions = load_sample_data_sessions(data_dir)
        if subset == "smoke":
            smoke_ids = load_sample_data_smoke_question_ids(data_dir)
            questions = [item for item in questions if item.q_id in smoke_ids]
            sessions = [
                SampleSession(session_id=item.session_id, question_ids=tuple(q_id for q_id in item.question_ids if q_id in smoke_ids))
                for item in sessions
                if any(q_id in smoke_ids for q_id in item.question_ids)
            ]
            sessions = [item for item in sessions if item.question_ids]
        chunk_cache = _load_chunk_cache(db, papers_by_key)
        resolved = resolve_question_gold(questions, papers_by_key=papers_by_key, chunk_cache=chunk_cache)

        report: dict[str, Any] = {
            "mode": mode,
            "subset": subset,
            "judge_mode": judge_mode,
            "paper_ids": {key: paper.id for key, paper in papers_by_key.items()},
        }
        if mode in {"retrieval", "both"}:
            report["retrieval"] = evaluate_retrieval(db, resolved)
        if mode in {"e2e", "both"}:
            report["end_to_end"] = evaluate_end_to_end(db, resolved, judge_mode=judge_mode, sessions=sessions)
        return report
    finally:
        db.close()
