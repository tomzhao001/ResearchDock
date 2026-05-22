"""phase2 structure-first schema

Revision ID: 20260519_04
Revises: 20260519_03
Create Date: 2026-05-19 16:10:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql


revision: str = "20260519_04"
down_revision: Union[str, None] = "20260519_03"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    columns = inspect(bind).get_columns(table_name)
    return column_name in {column["name"] for column in columns}


def _has_index(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    indexes = inspect(bind).get_indexes(table_name)
    return index_name in {index["name"] for index in indexes}


def _has_fk(constraint_name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text("SELECT 1 FROM pg_constraint WHERE conname = :name"),
        {"name": constraint_name},
    ).scalar()
    return result is not None


def upgrade() -> None:
    if not _has_column("paper_document_blocks", "reading_order"):
        op.add_column("paper_document_blocks", sa.Column("reading_order", sa.Integer(), nullable=True))
    if not _has_column("paper_document_tables", "reading_order"):
        op.add_column("paper_document_tables", sa.Column("reading_order", sa.Integer(), nullable=True))
    if not _has_column("paper_document_tables", "heading_level"):
        op.add_column("paper_document_tables", sa.Column("heading_level", sa.Integer(), nullable=True))
    if not _has_column("paper_document_tables", "section_path"):
        op.add_column("paper_document_tables", sa.Column("section_path", sa.Text(), nullable=True))
    if not _has_column("paper_document_tables", "provenance_json"):
        op.add_column(
            "paper_document_tables",
            sa.Column("provenance_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        )
    if not _has_column("paper_document_pictures", "reading_order"):
        op.add_column("paper_document_pictures", sa.Column("reading_order", sa.Integer(), nullable=True))
    if not _has_column("paper_document_pictures", "heading_level"):
        op.add_column("paper_document_pictures", sa.Column("heading_level", sa.Integer(), nullable=True))
    if not _has_column("paper_document_pictures", "section_path"):
        op.add_column("paper_document_pictures", sa.Column("section_path", sa.Text(), nullable=True))
    if not _has_column("paper_document_pictures", "provenance_json"):
        op.add_column(
            "paper_document_pictures",
            sa.Column("provenance_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        )

    if not _has_column("paper_chunks", "parent_chunk_id"):
        op.add_column("paper_chunks", sa.Column("parent_chunk_id", sa.BigInteger(), nullable=True))
    if not _has_column("paper_chunks", "chunk_role"):
        op.add_column(
            "paper_chunks",
            sa.Column("chunk_role", sa.String(length=32), server_default="child", nullable=False),
        )
    if not _has_fk("fk_paper_chunks_parent_chunk_id"):
        op.create_foreign_key(
            "fk_paper_chunks_parent_chunk_id",
            "paper_chunks",
            "paper_chunks",
            ["parent_chunk_id"],
            ["id"],
            ondelete="SET NULL",
        )
    if not _has_index("paper_chunks", "idx_paper_chunks_parent_chunk"):
        op.create_index("idx_paper_chunks_parent_chunk", "paper_chunks", ["parent_chunk_id"], unique=False)
    if not _has_index("paper_chunks", "idx_paper_chunks_chunk_role"):
        op.create_index("idx_paper_chunks_chunk_role", "paper_chunks", ["chunk_role"], unique=False)
    if not _has_index("paper_document_blocks", "idx_paper_document_blocks_reading_order"):
        op.create_index(
            "idx_paper_document_blocks_reading_order",
            "paper_document_blocks",
            ["paper_id", "reading_order"],
            unique=False,
        )
    if not _has_index("paper_document_tables", "idx_paper_document_tables_reading_order"):
        op.create_index(
            "idx_paper_document_tables_reading_order",
            "paper_document_tables",
            ["paper_id", "reading_order"],
            unique=False,
        )
    if not _has_index("paper_document_pictures", "idx_paper_document_pictures_reading_order"):
        op.create_index(
            "idx_paper_document_pictures_reading_order",
            "paper_document_pictures",
            ["paper_id", "reading_order"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_index("idx_paper_document_pictures_reading_order", table_name="paper_document_pictures")
    op.drop_index("idx_paper_document_tables_reading_order", table_name="paper_document_tables")
    op.drop_index("idx_paper_document_blocks_reading_order", table_name="paper_document_blocks")
    op.drop_index("idx_paper_chunks_chunk_role", table_name="paper_chunks")
    op.drop_index("idx_paper_chunks_parent_chunk", table_name="paper_chunks")
    op.drop_constraint("fk_paper_chunks_parent_chunk_id", "paper_chunks", type_="foreignkey")
    op.drop_column("paper_chunks", "chunk_role")
    op.drop_column("paper_chunks", "parent_chunk_id")

    op.drop_column("paper_document_pictures", "provenance_json")
    op.drop_column("paper_document_pictures", "section_path")
    op.drop_column("paper_document_pictures", "heading_level")
    op.drop_column("paper_document_pictures", "reading_order")
    op.drop_column("paper_document_tables", "provenance_json")
    op.drop_column("paper_document_tables", "section_path")
    op.drop_column("paper_document_tables", "heading_level")
    op.drop_column("paper_document_tables", "reading_order")
    op.drop_column("paper_document_blocks", "reading_order")
