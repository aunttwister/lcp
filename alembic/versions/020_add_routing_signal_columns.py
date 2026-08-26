"""Add routing-decision signal columns (classification rationale).

Adds the observability columns to ``routing_decisions``: which stage won
(``path``), the matched keyword, the extracted intent text, the semantic
top-N scores, the semantic gate, and whether the embedder was up.

Revision ID: 020
Revises: 019
Create Date: 2026-08-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    """Return True when the table already exists in the connected database.

    ``src.main`` runs ``Base.metadata.create_all(engine)`` at boot, which can
    create the new models' columns before Alembic reaches this migration (e.g.
    a container started with new code before ``alembic upgrade head`` ran, or
    a locally-seeded DB). Guarding on existence keeps ``upgrade head``
    idempotent against that state.
    """
    bind = op.get_bind()
    return sa.inspect(bind).has_table(name)


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in {c["name"] for c in sa.inspect(bind).get_columns(table)}


_NEW_COLUMNS = (
    ("path", sa.String()),
    ("keyword", sa.String()),
    ("intent_text", sa.Text()),
    ("semantic_json", sa.Text()),
    ("min_score", sa.Float()),
    ("sem_available", sa.Boolean()),
)


def upgrade() -> None:
    if _table_exists("routing_decisions"):
        for name, type_ in _NEW_COLUMNS:
            if not _column_exists("routing_decisions", name):
                op.add_column("routing_decisions", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    if _table_exists("routing_decisions"):
        for name, _ in _NEW_COLUMNS:
            if _column_exists("routing_decisions", name):
                op.drop_column("routing_decisions", name)
