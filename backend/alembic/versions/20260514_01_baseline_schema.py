"""baseline schema

Revision ID: 20260514_01
Revises:
Create Date: 2026-05-14 17:20:00
"""

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260514_01"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "organizations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=128), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("slug"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("organization_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), server_default="org_member", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )
    op.create_table(
        "app_settings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("openai_base_url", sa.Text(), nullable=True),
        sa.Column("openai_api_key_encrypted", sa.Text(), nullable=True),
        sa.Column("chat_model", sa.String(length=255), nullable=True),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("default_summary_language", sa.String(length=32), nullable=True),
        sa.Column("default_chunk_size", sa.Integer(), nullable=True),
        sa.Column("default_chunk_overlap", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "organization_settings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("organization_id", sa.BigInteger(), nullable=False),
        sa.Column("auto_extraction_questions_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id"),
    )
    op.create_table(
        "sources",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("schedule_cron", sa.String(length=128), nullable=True),
        sa.Column("max_items_per_run", sa.Integer(), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "papers",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("organization_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("abstract_raw", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("doi", sa.String(length=255), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=True),
        sa.Column("ingest_type", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "paper_assets",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.BigInteger(), nullable=False),
        sa.Column("asset_type", sa.String(length=64), nullable=True),
        sa.Column("storage_path", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "paper_summaries",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.BigInteger(), nullable=False),
        sa.Column("summary_language", sa.String(length=32), nullable=True),
        sa.Column("abstract_zh", sa.Text(), nullable=True),
        sa.Column("summary_points", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("research_problem", sa.Text(), nullable=True),
        sa.Column("method", sa.Text(), nullable=True),
        sa.Column("findings", sa.Text(), nullable=True),
        sa.Column("limitations", sa.Text(), nullable=True),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "paper_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("page_from", sa.Integer(), nullable=True),
        sa.Column("page_to", sa.Integer(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "chat_topics",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.String(length=255), server_default="新话题", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("topic_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("answer_mode", sa.String(length=32), nullable=True),
        sa.Column("used_knowledge_base", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("citations_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["topic_id"], ["chat_topics.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=True),
        sa.Column("source_id", sa.BigInteger(), nullable=True),
        sa.Column("paper_id", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("idx_paper_chunks_paper_id", "paper_chunks", ["paper_id"], unique=False)
    op.create_index("idx_paper_chunks_paper_chunk", "paper_chunks", ["paper_id", "chunk_index"], unique=False)
    op.execute("CREATE INDEX idx_paper_chunks_search_vector ON paper_chunks USING GIN (search_vector)")
    op.execute(
        "CREATE INDEX idx_paper_chunks_embedding_cosine ON paper_chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.execute("CREATE INDEX idx_chat_topics_user_updated_at ON chat_topics (user_id, updated_at DESC)")
    op.execute("CREATE INDEX idx_chat_messages_topic_created_at ON chat_messages (topic_id, created_at ASC)")

    op.execute(
        """
        INSERT INTO organizations (name, slug, is_active)
        VALUES ('Default Organization', 'default', TRUE)
        ON CONFLICT (slug) DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO users (organization_id, username, password_hash, role, is_active)
        VALUES (
            (SELECT id FROM organizations WHERE slug = 'default'),
            'admin',
            '$2b$12$FWCFMmz/kramxYvmhhW8e.Icx3D/TOEeoknZAffydgnEai/G6OEny',
            'org_owner',
            TRUE
        )
        ON CONFLICT (username) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("idx_chat_messages_topic_created_at", table_name="chat_messages")
    op.drop_index("idx_chat_topics_user_updated_at", table_name="chat_topics")
    op.drop_index("idx_paper_chunks_embedding_cosine", table_name="paper_chunks")
    op.drop_index("idx_paper_chunks_search_vector", table_name="paper_chunks")
    op.drop_index("idx_paper_chunks_paper_chunk", table_name="paper_chunks")
    op.drop_index("idx_paper_chunks_paper_id", table_name="paper_chunks")

    op.drop_table("jobs")
    op.drop_table("chat_messages")
    op.drop_table("chat_topics")
    op.drop_table("paper_chunks")
    op.drop_table("paper_summaries")
    op.drop_table("paper_assets")
    op.drop_table("papers")
    op.drop_table("sources")
    op.drop_table("organization_settings")
    op.drop_table("app_settings")
    op.drop_table("users")
    op.drop_table("organizations")
