"""Add model_capabilities table for benchmark-derived routing scores.

Revision ID: 007
Revises: 006
Create Date: 2026-08-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_capabilities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model", sa.String(), nullable=False, index=True),
        sa.Column("task_type", sa.String(), nullable=False, index=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="livebench"),
        sa.Column("benchmark_category", sa.String(), nullable=True),
        sa.Column("raw_score", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("model_capabilities")
