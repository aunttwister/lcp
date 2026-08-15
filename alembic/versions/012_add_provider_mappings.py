"""Add provider_mappings_json to model_registry.

Revision ID: 012
Revises: 011
Create Date: 2026-08-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_registry",
        sa.Column(
            "provider_mappings_json",
            sa.Text(),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("model_registry", "provider_mappings_json")
