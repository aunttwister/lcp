"""Add settings and cost_plugin_cache tables.

Revision ID: 018
Revises: 017
Create Date: 2026-08-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    """Return True when the table already exists in the connected database.

    ``src.main`` runs ``Base.metadata.create_all(engine)`` at boot, which can
    create the new models' tables before Alembic reaches this migration (e.g.
    a container started with new code before ``alembic upgrade head`` ran, or
    a locally-seeded DB). Guarding on existence keeps ``upgrade head``
    idempotent against that state instead of failing with
    "table settings already exists".
    """
    bind = op.get_bind()
    return sa.inspect(bind).has_table(name)


def upgrade() -> None:
    # Admin key/value settings (e.g. cost_cache_ttl_minutes)
    if not _table_exists("settings"):
        op.create_table(
            "settings",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("key", sa.String(), nullable=False),
            sa.Column("value", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.String(), nullable=False),
        )
        op.create_index("ix_settings_key", "settings", ["key"], unique=True)

    # DB cache of scraped cost-plugin data (subscriptions, balances)
    if not _table_exists("cost_plugin_cache"):
        op.create_table(
            "cost_plugin_cache",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("kind", sa.String(), nullable=False),
            sa.Column("payload_json", sa.Text(), nullable=False),
            sa.Column("fetched_at", sa.String(), nullable=False),
            sa.Column("stale_error", sa.Text(), nullable=True),
            sa.UniqueConstraint("provider", "kind", name="uq_cost_plugin_cache_provider_kind"),
        )
        op.create_index("ix_cost_plugin_cache_provider", "cost_plugin_cache", ["provider"])
        op.create_index("ix_cost_plugin_cache_kind", "cost_plugin_cache", ["kind"])


def downgrade() -> None:
    if _table_exists("cost_plugin_cache"):
        op.drop_index("ix_cost_plugin_cache_kind", table_name="cost_plugin_cache")
        op.drop_index("ix_cost_plugin_cache_provider", table_name="cost_plugin_cache")
        op.drop_table("cost_plugin_cache")
    if _table_exists("settings"):
        op.drop_index("ix_settings_key", table_name="settings")
        op.drop_table("settings")
