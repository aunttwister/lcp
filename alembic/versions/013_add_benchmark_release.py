"""Add benchmark_release to model_registry.

Revision ID: 013
Revises: 012
Create Date: 2026-08-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_registry",
        sa.Column("benchmark_release", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_registry", "benchmark_release")
