"""Drop aliases_json from model_registry.

Revision ID: 014
Revises: 013
Create Date: 2026-08-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("model_registry", "aliases_json")


def downgrade() -> None:
    op.add_column(
        "model_registry",
        sa.Column("aliases_json", sa.Text(), nullable=False, server_default="[]"),
    )
