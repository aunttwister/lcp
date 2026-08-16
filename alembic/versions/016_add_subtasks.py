"""Add model_capability_subtasks table.

Revision ID: 016
Revises: 015
Create Date: 2026-08-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_capability_subtasks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("task", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="livebench"),
        sa.Column("raw_score", sa.Float(), nullable=True),
        sa.Column("release_label", sa.String(), nullable=True),
        sa.Column("updated_at", sa.String(), nullable=False),
    )
    op.create_index("ix_subtasks_model", "model_capability_subtasks", ["model"])
    op.create_index("ix_subtasks_category", "model_capability_subtasks", ["category"])
    op.create_index("ix_subtasks_task", "model_capability_subtasks", ["task"])
    op.create_index("ix_subtasks_release", "model_capability_subtasks", ["release_label"])


def downgrade() -> None:
    op.drop_index("ix_subtasks_release", table_name="model_capability_subtasks")
    op.drop_index("ix_subtasks_task", table_name="model_capability_subtasks")
    op.drop_index("ix_subtasks_category", table_name="model_capability_subtasks")
    op.drop_index("ix_subtasks_model", table_name="model_capability_subtasks")
    op.drop_table("model_capability_subtasks")
