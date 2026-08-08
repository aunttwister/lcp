"""Add failover_events table for chain fallback tracking.

Revision ID: 006
Revises: 005
Create Date: 2026-08-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "failover_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.String(), nullable=False),
        sa.Column("profile", sa.String(), nullable=False),
        sa.Column("from_provider", sa.String(), nullable=False),
        sa.Column("to_provider", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("request_id", sa.Integer(), sa.ForeignKey("requests.id"), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("failover_events")
