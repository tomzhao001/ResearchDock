from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import PaperChunk
from app.services import papers as legacy_papers
from app.services.chat_rag.indexing import get_paper_indexing_service
from app.services.docling_extraction import build_document_extractor
from app.services.document_extraction import DocumentExtractor
from app.services.ocr.escalation import OcrEscalationProcessor, build_ocr_escalation_processor
from app.services.ocr.postprocess import OcrPostprocessor, build_ocr_postprocessor
from app.services.ocr.quality import build_default_routing_decision, assess_document_text_layer
from app.services.paper_pipeline import helpers as paper_helpers
from app.services.paper_pipeline.rendering import render_paper_text_from_structure
from app.services.paper_pipeline.repository import PaperJobRepository
from app.services.paper_pipeline.state import PaperJobGraphState
from app.services.vision import PictureDescriptionAdapter, build_picture_description_adapter


class PdfIngestGraphRunner:
    def __init__(self, *, repository: PaperJobRepository | None = None) -> None:
        self.repository = repository or PaperJobRepository()
        self.indexing_service = get_paper_indexing_service()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PaperJobGraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("mark_processing", self._mark_processing)
        graph.add_node("assess_ocr_strategy", self._assess_ocr_strategy)
        graph.add_node("extract_document", self._extract_document)
        graph.add_node("postprocess_ocr_text", self._postprocess_ocr_text)
        graph.add_node("glm_ocr_escalation", self._glm_ocr_escalation)
        graph.add_node("describe_pictures", self._describe_pictures)
        graph.add_node("persist_document_structure", self._persist_document_structure)
        graph.add_node("rebuild_index", self._rebuild_index)
        graph.add_node("complete_job", self._complete_job)
        graph.add_edge(START, "load_context")
        graph.add_conditional_edges(
            "load_context",
            self._route_if_loaded,
            {
                "mark_processing": "mark_processing",
                "end": END,
            },
        )
        graph.add_edge("mark_processing", "assess_ocr_strategy")
        graph.add_edge("assess_ocr_strategy", "extract_document")
        graph.add_edge("extract_document", "postprocess_ocr_text")
        graph.add_edge("postprocess_ocr_text", "glm_ocr_escalation")
        graph.add_edge("glm_ocr_escalation", "describe_pictures")
        graph.add_edge("describe_pictures", "persist_document_structure")
        graph.add_edge("persist_document_structure", "rebuild_index")
        graph.add_edge("rebuild_index", "complete_job")
        graph.add_edge("complete_job", END)
        return graph.compile()

    def _load_context(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        job = self.repository.get_job(db, state["job_id"])
        if job is None:
            return {}
        paper = self.repository.get_paper(db, job.paper_id)
        asset = self.repository.get_original_pdf_asset(db, job.paper_id)
        if paper is None or asset is None or not asset.storage_path:
            raise RuntimeError("Missing upload asset for job")
        paper_helpers.raise_if_cancel_requested(db, job, paper)
        return {
            "job": job,
            "paper": paper,
            "asset": asset,
            "file_path": Path(asset.storage_path),
        }

    @staticmethod
    def _route_if_loaded(state: PaperJobGraphState) -> str:
        if state.get("job") is None:
            return "end"
        return "mark_processing"

    def _mark_processing(self, state: PaperJobGraphState) -> dict[str, Any]:
        if not state.get("job") or not state.get("paper"):
            return {}
        self.repository.mark_processing(state["db"], job=state["job"], paper=state["paper"])
        return {}

    def _assess_ocr_strategy(self, state: PaperJobGraphState) -> dict[str, Any]:
        file_path = state.get("file_path")
        if file_path is None:
            return {"ocr_strategy": build_default_routing_decision(reason="missing_file_path")}
        return {"ocr_strategy": assess_document_text_layer(file_path)}

    def _extract_document(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        paper_helpers.raise_if_cancel_requested(db, state["job"], state["paper"])
        extractor = state.get("extractor") or build_document_extractor(ocr_strategy=state.get("ocr_strategy"))
        document = extractor.extract(state["file_path"])
        paper_helpers.raise_if_cancel_requested(db, state["job"], state["paper"])
        return {"document": document}

    def _postprocess_ocr_text(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        paper_helpers.raise_if_cancel_requested(db, state["job"], state["paper"])
        postprocessor = state.get("ocr_postprocessor") or build_ocr_postprocessor()
        document = postprocessor.process(
            document=state["document"],
            pdf_path=state["file_path"],
            ocr_strategy=state.get("ocr_strategy"),
        )
        paper_helpers.raise_if_cancel_requested(db, state["job"], state["paper"])
        return {
            "document": document,
            "ocr_postprocess_stats": dict(document.metadata.get("ocr_postprocess") or {}),
        }

    def _glm_ocr_escalation(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        paper_helpers.raise_if_cancel_requested(db, state["job"], state["paper"])
        processor = state.get("ocr_escalation_processor") or build_ocr_escalation_processor()
        document = processor.process(
            document=state["document"],
            pdf_path=state["file_path"],
            ocr_strategy=state.get("ocr_strategy"),
        )
        paper_helpers.raise_if_cancel_requested(db, state["job"], state["paper"])
        return {
            "document": document,
            "ocr_escalation_stats": dict(document.metadata.get("ocr_escalation") or {}),
        }

    def _describe_pictures(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        paper_helpers.raise_if_cancel_requested(db, state["job"], state["paper"])
        paper_helpers.describe_pictures(state["document"], state.get("picture_adapter") or build_picture_description_adapter())
        paper_helpers.raise_if_cancel_requested(db, state["job"], state["paper"])
        return {}

    def _persist_document_structure(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        paper = state["paper"]
        asset = state["asset"]
        document = state["document"]
        paper_helpers.clear_document_structure(db, paper.id)
        db.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper.id))
        paper_helpers.persist_document_structure(db, paper_id=paper.id, asset_id=asset.id, document=document)
        db.flush()
        metadata = dict(asset.metadata_json or {})
        metadata["extraction"] = {
            **document.extraction_metadata(),
            "ocr_strategy": state["ocr_strategy"].to_metadata() if state.get("ocr_strategy") is not None else None,
            "ocr_postprocess": state.get("ocr_postprocess_stats") or dict(document.metadata.get("ocr_postprocess") or {}),
            "ocr_escalation": state.get("ocr_escalation_stats") or dict(document.metadata.get("ocr_escalation") or {}),
            "picture_vlm": {
                "provider": legacy_papers.settings.picture_vlm_provider,
                "model": legacy_papers.settings.picture_vlm_model,
                "prompt_version": legacy_papers.settings.picture_vlm_prompt_version,
            },
        }
        asset.metadata_json = metadata
        asset.raw_text = None
        return {}

    def _rebuild_index(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        paper = state["paper"]
        asset = state["asset"]
        self.indexing_service.rebuild_paper_index_from_document_structure(
            db,
            paper_id=paper.id,
            paper_title=paper.title,
            structured_summary=paper_helpers.extract_structured_summary_from_asset(asset),
        )
        paper_helpers.raise_if_cancel_requested(db, state["job"], paper)
        return {}

    def _complete_job(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        self.repository.mark_completed(db, job=state["job"], paper=state["paper"])
        next_job_id = paper_helpers.queue_summary_job_if_needed(db, state["paper"], state["asset"])
        return {"next_job_id": next_job_id}

    def run(
        self,
        job_id: int,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        extractor: DocumentExtractor | None = None,
        picture_adapter: PictureDescriptionAdapter | None = None,
        ocr_postprocessor: OcrPostprocessor | None = None,
        ocr_escalation_processor: OcrEscalationProcessor | None = None,
    ) -> int | None:
        db = session_factory()
        state: PaperJobGraphState = {
            "db": db,
            "job_id": job_id,
            "extractor": extractor,
            "picture_adapter": picture_adapter,
            "ocr_postprocessor": ocr_postprocessor,
            "ocr_escalation_processor": ocr_escalation_processor,
        }
        try:
            result = self._graph.invoke(state)
            return result.get("next_job_id")
        except legacy_papers.JobCancellationRequested:
            return None
        except Exception as exc:
            job = self.repository.get_job(db, job_id)
            paper = self.repository.get_paper(db, job.paper_id) if job is not None else None
            self.repository.mark_failed(db, job=job, paper=paper, error=exc)
            raise
        finally:
            db.close()


class PaperSummaryGraphRunner:
    def __init__(self, *, repository: PaperJobRepository | None = None) -> None:
        self.repository = repository or PaperJobRepository()
        self.indexing_service = get_paper_indexing_service()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PaperJobGraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("mark_processing", self._mark_processing)
        graph.add_node("generate_summary", self._generate_summary)
        graph.add_node("apply_summary", self._apply_summary)
        graph.add_node("complete_job", self._complete_job)
        graph.add_edge(START, "load_context")
        graph.add_conditional_edges(
            "load_context",
            self._route_if_loaded,
            {
                "mark_processing": "mark_processing",
                "end": END,
            },
        )
        graph.add_edge("mark_processing", "generate_summary")
        graph.add_edge("generate_summary", "apply_summary")
        graph.add_edge("apply_summary", "complete_job")
        graph.add_edge("complete_job", END)
        return graph.compile()

    def _load_context(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        job = self.repository.get_job(db, state["job_id"])
        if job is None:
            return {}
        paper = self.repository.get_paper(db, job.paper_id)
        asset = self.repository.get_original_pdf_asset(db, job.paper_id)
        if paper is None or asset is None:
            raise RuntimeError("Missing paper or upload asset for summary job")
        if not legacy_papers.is_chat_llm_configured():
            raise RuntimeError("LLM is not configured for summarization")
        paper_text = render_paper_text_from_structure(db, paper.id)
        if not paper_text:
            raise RuntimeError("No parsed text available")
        paper_helpers.raise_if_cancel_requested(db, job, paper)
        return {"job": job, "paper": paper, "asset": asset, "paper_text": paper_text}

    @staticmethod
    def _route_if_loaded(state: PaperJobGraphState) -> str:
        if state.get("job") is None:
            return "end"
        return "mark_processing"

    def _mark_processing(self, state: PaperJobGraphState) -> dict[str, Any]:
        self.repository.mark_processing(state["db"], job=state["job"], paper=state["paper"])
        return {}

    def _generate_summary(self, state: PaperJobGraphState) -> dict[str, Any]:
        paper_helpers.raise_if_cancel_requested(state["db"], state["job"], state["paper"])
        summary = legacy_papers.summarize_paper_text(state["paper_text"])
        paper_helpers.raise_if_cancel_requested(state["db"], state["job"], state["paper"])
        summary["authors"] = legacy_papers._normalize_summary_text(summary.get("authors"))
        summary["doi"] = legacy_papers._normalize_summary_doi(summary.get("doi"))
        summary["source_url"] = legacy_papers._normalize_summary_url(summary.get("source_url"))
        published_at = legacy_papers._parse_summary_published_at(summary.get("published_at"))
        summary["published_at"] = legacy_papers._serialize_summary_published_at(published_at)
        return {"summary": summary}

    def _apply_summary(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        paper = state["paper"]
        asset = state["asset"]
        summary = state["summary"]
        metadata = dict(asset.metadata_json or {})
        metadata["structured_summary"] = summary
        asset.metadata_json = metadata
        paper.abstract_raw = summary.get("abstract_cn") or paper.abstract_raw
        if legacy_papers._paper_text_field_is_empty(paper.authors) and summary["authors"]:
            paper.authors = summary["authors"]
        if legacy_papers._paper_text_field_is_empty(paper.doi) and summary["doi"]:
            paper.doi = summary["doi"]
        if legacy_papers._paper_text_field_is_empty(paper.source_url) and summary["source_url"]:
            paper.source_url = summary["source_url"]
        published_at_value = legacy_papers._parse_summary_published_at(summary.get("published_at"))
        if paper.published_at is None and published_at_value is not None:
            paper.published_at = published_at_value
        self.indexing_service.rebuild_paper_index_from_document_structure(
            db,
            paper_id=paper.id,
            paper_title=paper.title,
            structured_summary=summary,
        )
        return {}

    def _complete_job(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        self.repository.mark_completed(db, job=state["job"], paper=state["paper"])
        next_job_id = paper_helpers.queue_question_set_job_if_needed(db, state["paper"], state["asset"])
        return {"next_job_id": next_job_id}

    def run(
        self,
        job_id: int,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> int | None:
        db = session_factory()
        state: PaperJobGraphState = {"db": db, "job_id": job_id}
        try:
            result = self._graph.invoke(state)
            return result.get("next_job_id")
        except legacy_papers.JobCancellationRequested:
            return None
        except Exception as exc:
            job = self.repository.get_job(db, job_id)
            paper = self.repository.get_paper(db, job.paper_id) if job is not None else None
            self.repository.mark_failed(db, job=job, paper=paper, error=exc)
            raise
        finally:
            db.close()


class PaperQuestionSetGraphRunner:
    def __init__(self, *, repository: PaperJobRepository | None = None) -> None:
        self.repository = repository or PaperJobRepository()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PaperJobGraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("mark_processing", self._mark_processing)
        graph.add_node("generate_question_set", self._generate_question_set)
        graph.add_node("apply_question_result", self._apply_question_result)
        graph.add_node("complete_job", self._complete_job)
        graph.add_edge(START, "load_context")
        graph.add_conditional_edges(
            "load_context",
            self._route_if_loaded,
            {
                "mark_processing": "mark_processing",
                "end": END,
            },
        )
        graph.add_edge("mark_processing", "generate_question_set")
        graph.add_edge("generate_question_set", "apply_question_result")
        graph.add_edge("apply_question_result", "complete_job")
        graph.add_edge("complete_job", END)
        return graph.compile()

    def _load_context(self, state: PaperJobGraphState) -> dict[str, Any]:
        db = state["db"]
        job = self.repository.get_job(db, state["job_id"])
        if job is None:
            return {}
        paper = self.repository.get_paper(db, job.paper_id)
        asset = self.repository.get_original_pdf_asset(db, job.paper_id)
        if paper is None or asset is None:
            raise RuntimeError("Missing paper or upload asset for question set job")
        if not legacy_papers.is_chat_llm_configured():
            raise RuntimeError("LLM is not configured for question set extraction")
        paper_text = render_paper_text_from_structure(db, paper.id)
        if not paper_text:
            raise RuntimeError("No parsed text available")
        structured_summary = paper_helpers.extract_structured_summary_from_asset(asset)
        if structured_summary is None:
            raise RuntimeError("No structured summary available")
        questions = legacy_papers.get_organization_question_items(db, organization_id=paper.organization_id)
        if not questions:
            raise RuntimeError("No organization question set configured")
        paper_helpers.raise_if_cancel_requested(db, job, paper)
        return {
            "job": job,
            "paper": paper,
            "asset": asset,
            "paper_text": paper_text,
            "structured_summary": structured_summary,
            "questions": questions,
        }

    @staticmethod
    def _route_if_loaded(state: PaperJobGraphState) -> str:
        if state.get("job") is None:
            return "end"
        return "mark_processing"

    def _mark_processing(self, state: PaperJobGraphState) -> dict[str, Any]:
        self.repository.mark_processing(state["db"], job=state["job"], paper=state["paper"])
        return {}

    def _generate_question_set(self, state: PaperJobGraphState) -> dict[str, Any]:
        paper_helpers.raise_if_cancel_requested(state["db"], state["job"], state["paper"])
        result = legacy_papers.answer_question_set_questions(
            state["paper_text"],
            structured_summary=state["structured_summary"],
            questions=state["questions"],
        )
        paper_helpers.raise_if_cancel_requested(state["db"], state["job"], state["paper"])
        return {"question_result": result}

    def _apply_question_result(self, state: PaperJobGraphState) -> dict[str, Any]:
        metadata = dict(state["asset"].metadata_json or {})
        metadata["question_set_extraction"] = state["question_result"]
        state["asset"].metadata_json = metadata
        return {}

    def _complete_job(self, state: PaperJobGraphState) -> dict[str, Any]:
        self.repository.mark_completed(state["db"], job=state["job"], paper=state["paper"])
        return {}

    def run(
        self,
        job_id: int,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        db = session_factory()
        state: PaperJobGraphState = {"db": db, "job_id": job_id}
        try:
            self._graph.invoke(state)
        except legacy_papers.JobCancellationRequested:
            return None
        except Exception as exc:
            job = self.repository.get_job(db, job_id)
            paper = self.repository.get_paper(db, job.paper_id) if job is not None else None
            self.repository.mark_failed(db, job=job, paper=paper, error=exc)
            raise
        finally:
            db.close()


class PaperWorkflowService:
    def __init__(self) -> None:
        repository = PaperJobRepository()
        self.pdf_ingest = PdfIngestGraphRunner(repository=repository)
        self.paper_summary = PaperSummaryGraphRunner(repository=repository)
        self.paper_question_set = PaperQuestionSetGraphRunner(repository=repository)

    def run_pdf_ingest_job(
        self,
        job_id: int,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        extractor: DocumentExtractor | None = None,
        picture_adapter: PictureDescriptionAdapter | None = None,
        ocr_postprocessor: OcrPostprocessor | None = None,
        ocr_escalation_processor: OcrEscalationProcessor | None = None,
    ) -> int | None:
        return self.pdf_ingest.run(
            job_id,
            session_factory=session_factory,
            extractor=extractor,
            picture_adapter=picture_adapter,
            ocr_postprocessor=ocr_postprocessor,
            ocr_escalation_processor=ocr_escalation_processor,
        )

    def run_paper_summary_job(
        self,
        job_id: int,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> int | None:
        return self.paper_summary.run(job_id, session_factory=session_factory)

    def run_paper_question_set_job(
        self,
        job_id: int,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        return self.paper_question_set.run(job_id, session_factory=session_factory)


_PAPER_WORKFLOW_SERVICE: PaperWorkflowService | None = None


def get_paper_workflow_service() -> PaperWorkflowService:
    global _PAPER_WORKFLOW_SERVICE
    if _PAPER_WORKFLOW_SERVICE is None:
        _PAPER_WORKFLOW_SERVICE = PaperWorkflowService()
    return _PAPER_WORKFLOW_SERVICE
