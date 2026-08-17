"""Add capability_metrics table.

Revision ID: 017
Revises: 016
Create Date: 2026-08-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "capability_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("schema_id", sa.String(), nullable=False),
        sa.Column("release_label", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("task", sa.String(), nullable=True),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="livebench"),
        sa.Column("updated_at", sa.String(), nullable=False),
    )
    op.create_index("ix_capability_metrics_schema", "capability_metrics", ["schema_id"])
    op.create_index("ix_capability_metrics_release", "capability_metrics", ["release_label"])
    op.create_index("ix_capability_metrics_model", "capability_metrics", ["model"])
    op.create_index("ix_capability_metrics_category", "capability_metrics", ["category"])
    op.create_index("ix_capability_metrics_task", "capability_metrics", ["task"])


def downgrade() -> None:
    op.drop_index("ix_capability_metrics_task", table_name="capability_metrics")
    op.drop_index("ix_capability_metrics_category", table_name="capability_metrics")
    op.drop_index("ix_capability_metrics_model", table_name="capability_metrics")
    op.drop_index("ix_capability_metrics_release", table_name="capability_metrics")
    op.drop_index("ix_capability_metrics_schema", table_name="capability_metrics")
    op.drop_table("capability_metrics")
