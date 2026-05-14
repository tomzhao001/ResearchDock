from datetime import datetime

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, func
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
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
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


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(bigint_sqlite, ForeignKey("organizations.id"), nullable=False)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="org_member")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


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
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, nullable=False)
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


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(bigint_sqlite, primary_key=True, autoincrement=True)
    job_type: Mapped[str | None] = mapped_column(String(64))
    source_id: Mapped[int | None] = mapped_column(bigint_sqlite)
    paper_id: Mapped[int | None] = mapped_column(bigint_sqlite)
    status: Mapped[str | None] = mapped_column(String(32))
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
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
    paper_id: Mapped[int] = mapped_column(bigint_sqlite, nullable=False)
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
    user_id: Mapped[int] = mapped_column(bigint_sqlite, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="新话题")
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
    topic_id: Mapped[int] = mapped_column(bigint_sqlite, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(255))
    answer_mode: Mapped[str | None] = mapped_column(String(32))
    used_knowledge_base: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    citations_json: Mapped[list[dict] | None] = mapped_column(json_field)
    metadata_json: Mapped[dict | None] = mapped_column(json_field)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
