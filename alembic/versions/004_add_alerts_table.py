"""Add alerts table for persisted alert history.

Revision ID: 004
Revises: 003
Create Date: 2026-08-06
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("dedup_key", sa.String(), nullable=False, index=True),
        sa.Column("rule", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), server_default="firing"),
        sa.Column("acknowledged", sa.Integer(), server_default="0"),
        sa.Column("acknowledged_at", sa.String(), nullable=True),
        sa.Column("resolved_at", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("alerts")
