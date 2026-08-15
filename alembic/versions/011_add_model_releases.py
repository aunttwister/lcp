"""Add release/versioning columns to model capabilities + registry.

Revision ID: 011
Revises: 010
Create Date: 2026-08-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_capabilities",
        sa.Column("release_label", sa.String(), nullable=True),
    )
    op.create_index(
        "ix_model_capabilities_release_label",
        "model_capabilities",
        ["release_label"],
    )
    op.add_column(
        "model_registry",
        sa.Column("active_release", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_model_capabilities_release_label",
        table_name="model_capabilities",
    )
    op.drop_column("model_capabilities", "release_label")
    op.drop_column("model_registry", "active_release")
