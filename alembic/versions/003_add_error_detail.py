"""Add error_detail column to requests table.

Revision ID: 003
Revises: 002
Create Date: 2026-08-04
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE requests ADD COLUMN error_detail TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE requests DROP COLUMN error_detail")
