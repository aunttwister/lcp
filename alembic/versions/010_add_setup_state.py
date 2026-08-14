"""Add setup_state table for the first-run setup wizard.

Revision ID: 010
Revises: 009
Create Date: 2026-08-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "setup_state",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(), nullable=False, unique=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.String(), nullable=False),
    )
    op.create_index("ix_setup_state_key", "setup_state", ["key"])


def downgrade() -> None:
    op.drop_index("ix_setup_state_key", table_name="setup_state")
    op.drop_table("setup_state")
