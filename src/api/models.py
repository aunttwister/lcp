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


# ── Provider Health / Failover Tracking ────────────────────────────────────

class FailoverEvent(Base):
    """Records a chain fallback: provider A failed → provider B took over."""
    __tablename__ = "failover_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())
    profile = Column(String, nullable=False)
    from_provider = Column(String, nullable=False)
    to_provider = Column(String, nullable=False)
    reason = Column(String, nullable=False)  # error_type from the failing provider
    error_message = Column(Text, nullable=True)
    request_id = Column(Integer, ForeignKey("requests.id"), nullable=True)


class ModelCapability(Base):
    """Per-task capability scores for model routing.

    Populated from public benchmarks (LiveBench, Arena), runtime benchmark
    runs (``lcp_benchmark``), and manual user entry (``manual``).

    ``release_label`` separates scores for different releases of the SAME
    logical model (e.g. ``deepseek-v4-pro`` 2026-06-25 vs 2026-08-13). The
    registry's ``active_release`` picks which release feeds the router.
    """
    __tablename__ = "model_capabilities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String, nullable=False, index=True)
    task_type = Column(String, nullable=False, index=True)
    score = Column(Float, nullable=False)  # 0.0–1.0 normalized
    source = Column(String, nullable=False, default="livebench")  # livebench, arena, gateway_yaml, lcp_benchmark, manual
    benchmark_category = Column(String, nullable=True)  # raw LiveBench category (coding, math, etc.)
    raw_score = Column(Float, nullable=True)  # original score before normalization (e.g. 70.0 out of 100)
    release_label = Column(String, nullable=True, index=True)  # e.g. "2026-08-13"; None = unversioned/legacy
    updated_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())


class ModelCapabilitySubtask(Base):
    """Per-subtask LiveBench scores (e.g. theory_of_mind, zebra_puzzle).

    LiveBench's ``all_tasks.csv`` / ``table_<release>.csv`` grades each model
    down to individual tasks (23 tasks across 7 categories). These rows back
    the "Subtask breakdown" panel on the Models page, keyed by the model's
    ``benchmark_key`` with the same ``source`` + ``release_label`` semantics
    as ``ModelCapability``.
    """
    __tablename__ = "model_capability_subtasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String, nullable=False, index=True)  # benchmark_key (logical)
    category = Column(String, nullable=False, index=True)  # reasoning, coding, math, …
    task = Column(String, nullable=False, index=True)  # theory_of_mind, zebra_puzzle, …
    score = Column(Float, nullable=False)  # 0.0–1.0 normalized
    source = Column(String, nullable=False, default="livebench")  # livebench | lcp_benchmark
    raw_score = Column(Float, nullable=True)  # original 0–100
    release_label = Column(String, nullable=True, index=True)
    updated_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())


class CapabilityMetric(Base):
    """Imported benchmark metrics — the source of truth for capability scores.

    One row per (schema, release, model, category, task) datum. Top-level
    category scores have ``category`` set and ``task`` NULL; per-subtask
    scores (e.g. theory_of_mind) have both set. Values are 0–100.

    The typed query tables (``model_capabilities``,
    ``model_capability_subtasks``) are MATERIALIZED from these rows on import
    so the router and Models page keep their existing fast query paths.
    """
    __tablename__ = "capability_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    schema_id = Column(String, nullable=False, index=True)  # dataset id, e.g. "livebench"
    release_label = Column(String, nullable=False, index=True)  # snapshot date, e.g. "2026-06-25"
    model = Column(String, nullable=False, index=True)  # logical / benchmark key
    category = Column(String, nullable=True, index=True)  # NULL = top-level rollup
    task = Column(String, nullable=True, index=True)  # NULL = category-level datum
    value = Column(Float, nullable=False)  # 0–100
    source = Column(String, nullable=False, default="livebench")
    updated_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())


class ModelRegistryEntry(Base):
    """Explicit model registry: canonical logical model ↔ benchmark key ↔ providers.

    Providers use different model-ID conventions (Command Code prefixes
    ``deepseek/...``, OpenCode uses bare names). This table pins those
    relationships explicitly so the router never has to guess from string
    patterns.

    Each logical model has exactly one row. ``benchmark_key`` is the STABLE,
    release-independent key used inside ``model_capabilities`` (e.g.
    ``deepseek-v4-flash``). ``active_release`` names the CURRENT model version
    (``2026-08-13`` for DeepSeek V4 Pro 0813) whose scores feed the router,
    while ``benchmark_release`` names the LiveBench leaderboard snapshot those
    scores were taken from (``2026-06-25``) — the benchmark date is separate
    from the model version.

    ``provider_mappings_json`` pins the exact provider-side model ID for each
    provider, e.g. ``{"opencode": "deepseek-v4-pro", "commandcode":
    "deepseek/deepseek-v4-pro", "deepseek": "deepseek-v4-pro"}`` — so the same
    logical model exposed by multiple providers resolves to ONE identity and
    ONE scoring regardless of provider naming. The provider keys themselves
    are also the canonical "providers" list shown in the UI.
    """
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    logical_name = Column(String, unique=True, nullable=False, index=True)
    benchmark_key = Column(String, nullable=False)  # stable key in model_capabilities
    provider_mappings_json = Column(Text, nullable=False, default="{}")  # {provider: provider-side model ID}
    active_release = Column(String, nullable=True)  # CURRENT model version (e.g. 2026-08-13); None = newest
    benchmark_release = Column(String, nullable=True)  # leaderboard snapshot date (e.g. 2026-06-25)
    quantization = Column(String, nullable=True)  # e.g. "Q4_K_M"; None = unquantized
    updated_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())


class BenchmarkRun(Base):
    """A LiveBench benchmark execution, tracked as a background job.

    ``target_kind`` is ``provider`` (benchmark the raw model directly against
    its provider) or ``profile`` (future: route the benchmark through an LCP
    profile to measure council / dynamic-routed profiles end-to-end).

    ``target_json`` holds the target spec — ``{"provider": ..., "model": ...}``
    for provider-kind, ``{"profile": ...}`` for profile-kind.
    """
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_kind = Column(String, nullable=False, default="provider")  # provider | profile
    target_json = Column(Text, nullable=False)  # JSON object
    categories_json = Column(Text, nullable=True)  # JSON array, or null = all categories
    status = Column(String, nullable=False, default="queued")  # queued | running | done | failed
    started_at = Column(String, nullable=True)
    finished_at = Column(String, nullable=True)
    result_json = Column(Text, nullable=True)  # per-category scores + raw output
    error = Column(Text, nullable=True)
    created_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())


class SetupState(Base):
    """First-run setup wizard progress.

    One row per installable module step plus a ``wizard`` marker row that
    records when the user skipped the wizard (status=skipped).
    """
    __tablename__ = "setup_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, unique=True, nullable=False, index=True)  # e.g. provider:deepseek, module:livebench, wizard
    status = Column(String, nullable=False)  # done | skipped | failed | running
    detail = Column(Text, nullable=True)
    updated_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())


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
