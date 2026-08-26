"""Add routing_decisions.conversation_json + routing_judgments table.

``conversation_json`` captures a shape-preserving summary of the classified
conversation so a decision can be replayed and judged later. ``routing_judgments``
stores human verdicts on decisions (the accumulating labeled dataset).

Revision ID: 021
Revises: 020
Create Date: 2026-08-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    return sa.inspect(bind).has_table(name)


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    # Input capture: shape-preserving summary of the classified conversation.
    if _table_exists("routing_decisions") and not _column_exists("routing_decisions", "conversation_json"):
        op.add_column("routing_decisions", sa.Column("conversation_json", sa.Text(), nullable=True))
    # Human judgment on decisions (accumulated labeled real-traffic dataset).
    if not _table_exists("routing_judgments"):
        op.create_table(
            "routing_judgments",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("decision_id", sa.Integer(), nullable=True),
            sa.Column("profile", sa.String(), nullable=False, server_default=""),
            sa.Column("task", sa.String(), nullable=False, server_default=""),
            sa.Column("path", sa.String(), nullable=True),
            sa.Column("verdict", sa.String(), nullable=False),
            sa.Column("expected_task", sa.String(), nullable=True),
            sa.Column("note", sa.Text(), nullable=True),
            sa.Column("judged_at", sa.String(), nullable=False, server_default=""),
        )
        op.create_index("ix_routing_judgments_decision_id", "routing_judgments", ["decision_id"])
        op.create_index("ix_routing_judgments_judged_at", "routing_judgments", ["judged_at"])


def downgrade() -> None:
    if _table_exists("routing_judgments"):
        op.drop_index("ix_routing_judgments_judged_at", table_name="routing_judgments")
        op.drop_index("ix_routing_judgments_decision_id", table_name="routing_judgments")
        op.drop_table("routing_judgments")
    if _table_exists("routing_decisions") and _column_exists("routing_decisions", "conversation_json"):
        op.drop_column("routing_decisions", "conversation_json")
