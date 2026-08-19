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


def upgrade() -> None:
    # Admin key/value settings (e.g. cost_cache_ttl_minutes)
    op.create_table(
        "settings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
    )
    op.create_index("ix_settings_key", "settings", ["key"], unique=True)

    # DB cache of scraped cost-plugin data (subscriptions, balances)
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
    op.drop_index("ix_cost_plugin_cache_kind", table_name="cost_plugin_cache")
    op.drop_index("ix_cost_plugin_cache_provider", table_name="cost_plugin_cache")
    op.drop_table("cost_plugin_cache")
    op.drop_index("ix_settings_key", table_name="settings")
    op.drop_table("settings")
