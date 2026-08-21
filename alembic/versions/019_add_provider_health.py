"""Add provider_health table (persisted circuit-breaker state).

Revision ID: 019
Revises: 018
Create Date: 2026-08-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    """Return True when the table already exists in the connected database.

    ``src.main`` runs ``Base.metadata.create_all(engine)`` at boot, which can
    create the new models' tables before Alembic reaches this migration (e.g.
    a container started with new code before ``alembic upgrade head`` ran, or
    a locally-seeded DB). Guarding on existence keeps ``upgrade head``
    idempotent against that state instead of failing with
    "table provider_health already exists".
    """
    bind = op.get_bind()
    return sa.inspect(bind).has_table(name)


def upgrade() -> None:
    # Persisted circuit-breaker health per (provider, base_url, profile)
    if not _table_exists("provider_health"):
        op.create_table(
            "provider_health",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("base_url", sa.String(), nullable=False),
            sa.Column("profile", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("consecutive_failures", sa.Integer(), nullable=False),
            sa.Column("last_failure", sa.String(), nullable=True),
            sa.Column("last_failure_reason", sa.Text(), nullable=True),
            sa.Column("last_success", sa.String(), nullable=True),
            sa.Column("tripped_until", sa.Float(), nullable=True),
            sa.Column("manual_override", sa.String(), nullable=True),
            sa.Column("updated_at", sa.String(), nullable=False),
            sa.UniqueConstraint("provider", "base_url", "profile", name="uq_provider_health_key"),
        )
        op.create_index("ix_provider_health_provider", "provider_health", ["provider"])


def downgrade() -> None:
    if _table_exists("provider_health"):
        op.drop_index("ix_provider_health_provider", table_name="provider_health")
        op.drop_table("provider_health")
