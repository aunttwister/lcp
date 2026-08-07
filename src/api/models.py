"""SQLAlchemy models for the LCP gateway."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .logging_config import get_logger

logger = get_logger("lcp.models")


class Base(DeclarativeBase):
    pass


# ── Phase 1-3 tables (already live in production) ──────────────────────────

class Request(Base):
    """Per-request cost tracking."""
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False)
    profile = Column(String, nullable=False, default="unknown")
    model = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cache_hit_tokens = Column(Integer, default=0)
    cache_miss_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    latency_ms = Column(Integer, default=0)
    success = Column(Integer, default=1)  # 1=success, 0=failure
    error_type = Column(String, nullable=True)
    error_detail = Column(Text, nullable=True)  # full traceback or error message
    tools_blocked = Column(String, nullable=True)  # comma-separated list


# ── API Key Management ────────────────────────────────────────────────────

class ApiKey(Base):
    """Virtual API keys for authentication and spend tracking."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_hash = Column(String, unique=True, nullable=False)
    key_prefix = Column(String, nullable=False)  # first 8 chars for display
    name = Column(String, nullable=False)
    allowed_profiles = Column(String, nullable=True)  # comma-separated or null=all
    spend_limit = Column(Float, default=0.0)  # 0 = unlimited
    total_spend = Column(Float, default=0.0)
    status = Column(String, default="active")  # active, revoked
    created_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())
    last_used_at = Column(String, nullable=True)
    expires_at = Column(String, nullable=True)
    revoked_at = Column(String, nullable=True)
    metadata_tags = Column(String, nullable=True)  # JSON string


class ProviderCredential(Base):
    """Encrypted API key for an upstream provider, managed via the UI.

    The raw key is encrypted with Fernet (see src.api.crypto) using the master
    key from ``LCP_SECRET_KEY`` (or the on-disk fallback). Only ciphertext is
    stored here — the gateway.yaml only ever references the env var name.
    """
    __tablename__ = "provider_credentials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, unique=True, nullable=False, index=True)
    encrypted_key = Column(Text, nullable=False)  # Fernet token (ciphertext)
    created_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())
    updated_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())


class Budget(Base):
    """Spending budgets per key and/or profile."""
    __tablename__ = "budgets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=True)  # null = global/profile budget
    profile = Column(String, nullable=True)  # null = all profiles
    amount = Column(Float, nullable=False)  # budget cap in USD
    current_spend = Column(Float, default=0.0)
    period = Column(String, default="monthly")  # monthly, total
    threshold_pct = Column(String, default="80")  # comma-separated alert thresholds
    action = Column(String, default="log")  # log, block (hard stop)
    status = Column(String, default="active")  # active, paused, exceeded
    created_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())
    last_alert_at = Column(String, nullable=True)


# ── Alerting ────────────────────────────────────────────────────────────────

class Alert(Base):
    """Persisted alert for webhook notifications and UI history."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())
    dedup_key = Column(String, nullable=False, index=True)
    rule = Column(String, nullable=False)
    severity = Column(String, nullable=False)  # info, warning, critical
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON blob
    status = Column(String, default="firing")  # firing, resolved
    acknowledged = Column(Integer, default=0)
    acknowledged_at = Column(String, nullable=True)
    resolved_at = Column(String, nullable=True)


# ── Engine + session factory ───────────────────────────────────────────────

def get_engine(db_path: str):
    """Create SQLAlchemy engine with WAL mode for SQLite."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    # Enable WAL mode on connect
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def set_wal(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA journal_mode=WAL")

    return engine


def init_db(db_path: str, create_all: bool = True):
    """Initialize database — create engine, optionally create all tables."""
    engine = get_engine(db_path)
    if create_all:
        Base.metadata.create_all(engine)
    return engine


def get_session(engine) -> Session:
    """Create a new session."""
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
