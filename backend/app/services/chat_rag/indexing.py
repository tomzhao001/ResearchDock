from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models import PaperChunk
from app.services import rag as legacy_rag
from app.services.llm import embed_texts


class PaperIndexingService:
    def rebuild_paper_index(
        self,
        db: Session,
        *,
        paper_id: int,
        preanalysis: dict[str, Any] | None = None,
        paper_title: str | None = None,
        structured_summary: dict[str, Any] | None = None,
    ) -> int:
        chunks = legacy_rag._split_text(
            preanalysis=preanalysis,
            paper_title=paper_title,
            structured_summary=structured_summary,
        )
        embeddings: list[list[float]] = []
        if chunks and (settings.glm_api_key.strip() or settings.openai_api_key.strip()):
            try:
                embeddings = embed_texts([str(chunk.get("embedding_input") or chunk["content"]) for chunk in chunks])
            except Exception:
                embeddings = []

        db.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
        parent_chunk_ids_by_section: dict[str, int] = {}
        child_chunk_ids_by_section: dict[str, list[int]] = {}
        child_rows: list[tuple[PaperChunk, str]] = []
        section_summary_rows: list[tuple[PaperChunk, str]] = []
        paper_summary_rows: list[PaperChunk] = []
        for index, chunk in enumerate(chunks):
            record = PaperChunk(
                paper_id=paper_id,
                parent_chunk_id=None,
                chunk_index=chunk["chunk_index"],
                chunk_role=str(chunk.get("chunk_role") or "child"),
                content=chunk["content"],
                embedding=embeddings[index] if index < len(embeddings) else None,
                token_count=chunk["token_count"],
                page_from=chunk["page_from"],
                page_to=chunk["page_to"],
                metadata_json=chunk["metadata_json"],
            )
            db.add(record)
            db.flush()
            if record.chunk_role == "parent":
                section_key = str(chunk.get("section_key") or chunk["metadata_json"].get("section_id") or "")
                if section_key:
                    parent_chunk_ids_by_section[section_key] = int(record.id)
            elif record.chunk_role == "child":
                child_rows.append((record, str(chunk.get("parent_key") or "")))
                section_key = str(chunk.get("parent_key") or chunk["metadata_json"].get("section_id") or "")
                if section_key:
                    child_chunk_ids_by_section.setdefault(section_key, []).append(int(record.id))
            elif record.chunk_role == "section_summary":
                section_summary_rows.append((record, str(chunk.get("section_key") or chunk["metadata_json"].get("section_id") or "")))
            elif record.chunk_role == "paper_summary":
                paper_summary_rows.append(record)
        for record, section_key in child_rows:
            parent_chunk_id = parent_chunk_ids_by_section.get(section_key)
            if parent_chunk_id is not None:
                record.parent_chunk_id = parent_chunk_id
        anchor_limit = max(settings.rag_summary_anchor_limit, 1)
        for record, section_key in section_summary_rows:
            metadata = dict(record.metadata_json or {})
            parent_chunk_id = parent_chunk_ids_by_section.get(section_key)
            anchor_chunk_ids: list[int] = []
            if parent_chunk_id is not None:
                anchor_chunk_ids.append(parent_chunk_id)
                metadata["resolved_chunk_id"] = parent_chunk_id
            anchor_chunk_ids.extend(child_chunk_ids_by_section.get(section_key, [])[:anchor_limit])
            metadata["anchor_chunk_ids"] = anchor_chunk_ids
            metadata["anchor_section_id"] = section_key or None
            record.metadata_json = metadata
        ordered_parent_ids = [
            chunk_id
            for section_key, chunk_id in parent_chunk_ids_by_section.items()
            if section_key
        ]
        for record in paper_summary_rows:
            metadata = dict(record.metadata_json or {})
            metadata["anchor_chunk_ids"] = ordered_parent_ids[:anchor_limit]
            metadata["resolved_chunk_id"] = ordered_parent_ids[0] if ordered_parent_ids else None
            metadata["anchor_section_ids"] = list(parent_chunk_ids_by_section.keys())[:anchor_limit]
            record.metadata_json = metadata
        db.commit()
        if legacy_rag._is_postgres_session(db):
            db.execute(
                update(PaperChunk)
                .where(PaperChunk.paper_id == paper_id)
                .values(
                    search_vector=func.to_tsvector(
                        settings.rag_text_search_config,
                        PaperChunk.content,
                    )
                )
            )
            db.commit()
        return len(chunks)

    def rebuild_paper_index_from_document_structure(
        self,
        db: Session,
        *,
        paper_id: int,
        paper_title: str | None = None,
        structured_summary: dict[str, Any] | None = None,
    ) -> int:
        preanalysis = legacy_rag._build_preanalysis_from_document_structure(db, paper_id=paper_id)
        return self.rebuild_paper_index(
            db,
            paper_id=paper_id,
            preanalysis=preanalysis,
            paper_title=paper_title,
            structured_summary=structured_summary,
        )


_PAPER_INDEXING_SERVICE: PaperIndexingService | None = None


def get_paper_indexing_service() -> PaperIndexingService:
    global _PAPER_INDEXING_SERVICE
    if _PAPER_INDEXING_SERVICE is None:
        _PAPER_INDEXING_SERVICE = PaperIndexingService()
    return _PAPER_INDEXING_SERVICE
