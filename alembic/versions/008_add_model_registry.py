"""Add model_registry table for explicit model alias → benchmark mapping.

Revision ID: 008
Revises: 007
Create Date: 2026-08-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("logical_name", sa.String(), nullable=False, unique=True),
        sa.Column("benchmark_key", sa.String(), nullable=False),
        sa.Column("aliases_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("updated_at", sa.String(), nullable=False),
    )
    op.create_index("ix_model_registry_logical_name", "model_registry", ["logical_name"])


def downgrade() -> None:
    op.drop_index("ix_model_registry_logical_name", table_name="model_registry")
    op.drop_table("model_registry")
