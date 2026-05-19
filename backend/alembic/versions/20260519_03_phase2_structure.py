"""phase2 structure-first schema

Revision ID: 20260519_03
Revises: 20260519_02
Create Date: 2026-05-19 16:10:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260519_03"
down_revision: Union[str, None] = "20260519_02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("paper_document_blocks", sa.Column("reading_order", sa.Integer(), nullable=True))
    op.add_column("paper_document_tables", sa.Column("reading_order", sa.Integer(), nullable=True))
    op.add_column("paper_document_tables", sa.Column("heading_level", sa.Integer(), nullable=True))
    op.add_column("paper_document_tables", sa.Column("section_path", sa.Text(), nullable=True))
    op.add_column("paper_document_tables", sa.Column("provenance_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("paper_document_pictures", sa.Column("reading_order", sa.Integer(), nullable=True))
    op.add_column("paper_document_pictures", sa.Column("heading_level", sa.Integer(), nullable=True))
    op.add_column("paper_document_pictures", sa.Column("section_path", sa.Text(), nullable=True))
    op.add_column("paper_document_pictures", sa.Column("provenance_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    op.add_column("paper_chunks", sa.Column("parent_chunk_id", sa.BigInteger(), nullable=True))
    op.add_column("paper_chunks", sa.Column("chunk_role", sa.String(length=32), server_default="child", nullable=False))
    op.create_foreign_key(
        "fk_paper_chunks_parent_chunk_id",
        "paper_chunks",
        "paper_chunks",
        ["parent_chunk_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("idx_paper_chunks_parent_chunk", "paper_chunks", ["parent_chunk_id"], unique=False)
    op.create_index("idx_paper_chunks_chunk_role", "paper_chunks", ["chunk_role"], unique=False)
    op.create_index("idx_paper_document_blocks_reading_order", "paper_document_blocks", ["paper_id", "reading_order"], unique=False)
    op.create_index("idx_paper_document_tables_reading_order", "paper_document_tables", ["paper_id", "reading_order"], unique=False)
    op.create_index("idx_paper_document_pictures_reading_order", "paper_document_pictures", ["paper_id", "reading_order"], unique=False)


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
