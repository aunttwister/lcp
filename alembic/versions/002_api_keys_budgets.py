"""Add api_keys and budgets tables.

Revision ID: 002
Revises: 001
Create Date: 2026-07-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT UNIQUE NOT NULL,
            key_prefix TEXT NOT NULL,
            name TEXT NOT NULL,
            allowed_profiles TEXT,
            spend_limit REAL DEFAULT 0.0,
            total_spend REAL DEFAULT 0.0,
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            expires_at TEXT,
            revoked_at TEXT,
            metadata_tags TEXT
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            key_id INTEGER REFERENCES api_keys(id),
            profile TEXT,
            amount REAL NOT NULL,
            current_spend REAL DEFAULT 0.0,
            period TEXT DEFAULT 'monthly',
            threshold_pct TEXT DEFAULT '80',
            action TEXT DEFAULT 'log',
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL,
            last_alert_at TEXT
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS budgets")
    op.execute("DROP TABLE IF EXISTS api_keys")
