"""Add the §9 L4 replayable-decision column to routing_decisions.

M4: every routing decision carries ``l4_json`` — {layer, rule_id, intent,
margin, scores, balances, from, to, reason} — so a recorded row replays to
the same winner (invariant 2).

Revision ID: 025
Revises: 024
Create Date: 2026-10-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    """True when the table exists (``create_all`` may have beaten us here)."""
    bind = op.get_bind()
    return sa.inspect(bind).has_table(name)


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    if _table_exists("routing_decisions") and not _column_exists(
            "routing_decisions", "l4_json"):
        op.add_column("routing_decisions",
                      sa.Column("l4_json", sa.Text(), nullable=True))


def downgrade() -> None:
    if _table_exists("routing_decisions") and _column_exists(
            "routing_decisions", "l4_json"):
        op.drop_column("routing_decisions", "l4_json")
