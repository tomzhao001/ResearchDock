from datetime import datetime

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, String, Text, desc, false, func, true
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.config import settings
from app.database import Base

json_field = JSON().with_variant(JSONB, "postgresql")
bigint_sqlite = BigInteger().with_variant(Integer, "sqlite")
vector_field = Vector(max(settings.glm_embedding_dimensions, 1)).with_variant(JSON(), "sqlite")
search_vector_field = Text().with_variant(TSVECTOR(), "postgresql")


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class OrganizationSettings(Base):
    __tablename__ = "organization_settings"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("organizations.id"), nullable=False, unique=True)
    auto_extraction_questions_json: Mapped[list[dict] | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class AppSettings(Base):
    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    openai_base_url: Mapped[str | None] = mapped_column(Text)
    openai_api_key_encrypted: Mapped[str | None] = mapped_column(Text)
    chat_model: Mapped[str | None] = mapped_column(String(255))
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    default_summary_language: Mapped[str | None] = mapped_column(String(32))
    default_chunk_size: Mapped[int | None] = mapped_column(Integer)
    default_chunk_overlap: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("organizations.id"), nullable=False)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="org_member", server_default="org_member")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())
    schedule_cron: Mapped[str | None] = mapped_column(String(128))
    max_items_per_run: Mapped[int | None] = mapped_column(Integer)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("organizations.id"), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[str | None] = mapped_column(Text)
    abstract_raw: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str | None] = mapped_column(String(128))
    ingest_type: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PaperAsset(Base):
    __tablename__ = "paper_assets"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    asset_type: Mapped[str | None] = mapped_column(String(64))
    storage_path: Mapped[str | None] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(String(128))
    raw_text: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class PaperDocumentPage(Base):
    __tablename__ = "paper_document_pages"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    asset_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("paper_assets.id", ondelete="CASCADE"), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str | None] = mapped_column(Text)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class PaperDocumentBlock(Base):
    __tablename__ = "paper_document_blocks"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    page_id: Mapped[int | None] = mapped_column(bigint_sqlite, ForeignKey("paper_document_pages.id", ondelete="CASCADE"))
    block_index: Mapped[int] = mapped_column(Integer, nullable=False)
    block_type: Mapped[str] = mapped_column(String(64), nullable=False, default="paragraph", server_default="paragraph")
    docling_label: Mapped[str | None] = mapped_column(String(128))
    heading_level: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    bbox_json: Mapped[dict | None] = mapped_column(json_field)
    provenance_json: Mapped[dict | None] = mapped_column(json_field)
    metadata_json: Mapped[dict | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class PaperDocumentTable(Base):
    __tablename__ = "paper_document_tables"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    page_from: Mapped[int | None] = mapped_column(Integer)
    page_to: Mapped[int | None] = mapped_column(Integer)
    table_index: Mapped[int] = mapped_column(Integer, nullable=False)
    caption: Mapped[str | None] = mapped_column(Text)
    markdown: Mapped[str | None] = mapped_column(Text)
    data_json: Mapped[dict | list | None] = mapped_column(json_field)
    bbox_json: Mapped[dict | None] = mapped_column(json_field)
    metadata_json: Mapped[dict | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class PaperDocumentPicture(Base):
    __tablename__ = "paper_document_pictures"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    picture_index: Mapped[int] = mapped_column(Integer, nullable=False)
    caption: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    description_model: Mapped[str | None] = mapped_column(String(255))
    description_prompt_version: Mapped[str | None] = mapped_column(String(64))
    bbox_json: Mapped[dict | None] = mapped_column(json_field)
    image_asset_path: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class PaperSummary(Base):
    __tablename__ = "paper_summaries"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    summary_language: Mapped[str | None] = mapped_column(String(32))
    abstract_zh: Mapped[str | None] = mapped_column(Text)
    summary_points: Mapped[list[dict] | None] = mapped_column(json_field)
    research_problem: Mapped[str | None] = mapped_column(Text)
    method: Mapped[str | None] = mapped_column(Text)
    findings: Mapped[str | None] = mapped_column(Text)
    limitations: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str | None] = mapped_column(String(255))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    job_type: Mapped[str | None] = mapped_column(String(64))
    source_id: Mapped[int | None] = mapped_column(bigint_sqlite, ForeignKey("sources.id", ondelete="SET NULL"))
    paper_id: Mapped[int | None] = mapped_column(bigint_sqlite, ForeignKey("papers.id", ondelete="SET NULL"))
    status: Mapped[str | None] = mapped_column(String(32))
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(vector_field)
    search_vector: Mapped[str | None] = mapped_column(search_vector_field)
    token_count: Mapped[int | None] = mapped_column(Integer)
    page_from: Mapped[int | None] = mapped_column(Integer)
    page_to: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ChatTopic(Base):
    __tablename__ = "chat_topics"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="新话题", server_default="新话题")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    topic_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("chat_topics.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(255))
    answer_mode: Mapped[str | None] = mapped_column(String(32))
    used_knowledge_base: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=false())
    citations_json: Mapped[list[dict] | None] = mapped_column(json_field)
    metadata_json: Mapped[dict | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


Index("idx_paper_chunks_paper_id", PaperChunk.paper_id)
Index("idx_paper_chunks_paper_chunk", PaperChunk.paper_id, PaperChunk.chunk_index)
Index("idx_paper_document_pages_paper_page", PaperDocumentPage.paper_id, PaperDocumentPage.page_number)
Index("idx_paper_document_blocks_paper_block", PaperDocumentBlock.paper_id, PaperDocumentBlock.block_index)
Index("idx_paper_document_blocks_paper_type", PaperDocumentBlock.paper_id, PaperDocumentBlock.block_type)
Index("idx_paper_document_tables_paper_table", PaperDocumentTable.paper_id, PaperDocumentTable.table_index)
Index("idx_paper_document_pictures_paper_picture", PaperDocumentPicture.paper_id, PaperDocumentPicture.picture_index)
Index("idx_paper_chunks_search_vector", PaperChunk.search_vector, postgresql_using="gin")
Index(
    "idx_paper_chunks_embedding_cosine",
    PaperChunk.embedding,
    postgresql_using="ivfflat",
    postgresql_ops={"embedding": "vector_cosine_ops"},
    postgresql_with={"lists": 100},
)
Index("idx_chat_topics_user_updated_at", ChatTopic.user_id, desc(ChatTopic.updated_at))
Index("idx_chat_messages_topic_created_at", ChatMessage.topic_id, ChatMessage.created_at)
