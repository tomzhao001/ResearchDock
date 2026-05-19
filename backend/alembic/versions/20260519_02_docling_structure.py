"""docling structure tables

Revision ID: 20260519_02
Revises: 20260514_01
Create Date: 2026-05-19 11:40:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260519_02"
down_revision: Union[str, None] = "20260514_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "paper_document_pages",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.BigInteger(), nullable=False),
        sa.Column("asset_id", sa.BigInteger(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["asset_id"], ["paper_assets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "paper_document_blocks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.BigInteger(), nullable=False),
        sa.Column("page_id", sa.BigInteger(), nullable=True),
        sa.Column("block_index", sa.Integer(), nullable=False),
        sa.Column("block_type", sa.String(length=64), server_default="paragraph", nullable=False),
        sa.Column("docling_label", sa.String(length=128), nullable=True),
        sa.Column("heading_level", sa.Integer(), nullable=True),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("bbox_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("provenance_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["page_id"], ["paper_document_pages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "paper_document_tables",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.BigInteger(), nullable=False),
        sa.Column("page_from", sa.Integer(), nullable=True),
        sa.Column("page_to", sa.Integer(), nullable=True),
        sa.Column("table_index", sa.Integer(), nullable=False),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("markdown", sa.Text(), nullable=True),
        sa.Column("data_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("bbox_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "paper_document_pictures",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.BigInteger(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("picture_index", sa.Integer(), nullable=False),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("description_model", sa.String(length=255), nullable=True),
        sa.Column("description_prompt_version", sa.String(length=64), nullable=True),
        sa.Column("bbox_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("image_asset_path", sa.Text(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_paper_document_pages_paper_page", "paper_document_pages", ["paper_id", "page_number"], unique=False)
    op.create_index("idx_paper_document_blocks_paper_block", "paper_document_blocks", ["paper_id", "block_index"], unique=False)
    op.create_index("idx_paper_document_blocks_paper_type", "paper_document_blocks", ["paper_id", "block_type"], unique=False)
    op.create_index("idx_paper_document_tables_paper_table", "paper_document_tables", ["paper_id", "table_index"], unique=False)
    op.create_index("idx_paper_document_pictures_paper_picture", "paper_document_pictures", ["paper_id", "picture_index"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_paper_document_pictures_paper_picture", table_name="paper_document_pictures")
    op.drop_index("idx_paper_document_tables_paper_table", table_name="paper_document_tables")
    op.drop_index("idx_paper_document_blocks_paper_type", table_name="paper_document_blocks")
    op.drop_index("idx_paper_document_blocks_paper_block", table_name="paper_document_blocks")
    op.drop_index("idx_paper_document_pages_paper_page", table_name="paper_document_pages")
    op.drop_table("paper_document_pictures")
    op.drop_table("paper_document_tables")
    op.drop_table("paper_document_blocks")
    op.drop_table("paper_document_pages")
