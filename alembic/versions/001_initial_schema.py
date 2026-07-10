"""Initial schema — create all tables.

Revision ID: 001
Revises: None
Create Date: 2026-06-18
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Only create tables that don't exist (safe for existing databases)
    op.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT 'unknown',
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            cache_hit_tokens INTEGER DEFAULT 0,
            cache_miss_tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            latency_ms INTEGER DEFAULT 0,
            success INTEGER DEFAULT 1,
            error_type TEXT,
            tools_blocked TEXT
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            monthly_budget REAL DEFAULT 0.0,
            created_at TEXT NOT NULL
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER REFERENCES teams(id),
            username TEXT UNIQUE NOT NULL,
            api_key_hash TEXT NOT NULL,
            credit_limit REAL DEFAULT 100.0,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id INTEGER REFERENCES users(id),
            team_id INTEGER REFERENCES teams(id),
            action TEXT NOT NULL,
            detail TEXT,
            ip_address TEXT
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit_logs")
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TABLE IF EXISTS teams")
    op.execute("DROP TABLE IF EXISTS requests")
