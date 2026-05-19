"""job cancellation and soft delete fields

Revision ID: 20260519_03
Revises: 20260519_02
Create Date: 2026-05-19 14:58:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260519_03"
down_revision: Union[str, None] = "20260519_02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("celery_task_id", sa.String(length=255), nullable=True))
    op.add_column("jobs", sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("jobs", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "deleted_at")
    op.drop_column("jobs", "cancel_requested_at")
    op.drop_column("jobs", "celery_task_id")
