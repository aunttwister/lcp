# Feature: Provider Health Dashboard

**Created:** 2026-08-06
**Status:** open
**Phase:** post-Phase 7

## North Star

The Providers page today is a static CRUD table (name, base URL, key env, models). Turn it into a live health dashboard showing per-provider uptime, failure breakdowns, failover history, and circuit breaker state — with the ability to manually toggle a provider degraded/dead for maintenance.

---

## Current State (What Exists)

### CircuitBreaker (`src/api/circuit_breaker.py`)
- Tracks per-(provider, base_url, profile) with 3 states: `healthy` → `degraded` → `dead`
- Half-open ladder: cooldown expiry promotes one step (dead→degraded→healthy)
- Error weight system: auth failures = 3x, others = 1x
- `stats` property returns `{total, healthy, degraded, dead}` counts
- `get_all_health()` returns dict keyed by `(provider, url, profile)` with: status, consecutive_failures, last_success, last_failure, tripped_until

### `/health` Endpoint
- Returns JSON: `{status: "ok", profiles: [...], providers: { "provider/profile": {status, failures, last_success, last_failure, base_url, tripped_until} }}`
- Uses CircuitBreaker's in-memory health map — lost on restart

### Providers Page (`pages/providers.html`)
- CRUD table: Name, Base URL, Key Env, Models, Actions (Edit/Del)
- Add provider form with test connection + model discovery
- Quick-add presets for known providers
- Profile chains managed separately
- **No health indicators, no uptime, no failure breakdown, no failover log**

### Sidebar
- Plugin status dots only (cost plugins), **not provider health**

---

## Plan

### Phase 1: Persist Provider Health & Events

| Step | What |
|---|---|
| 1.1 | Add `ProviderHealth` table: provider, profile, status, consecutive_failures, total_requests, total_failures, last_success, last_failure, tripped_until, uptime_24h, uptime_7d, uptime_30d |
| 1.2 | Add `ProviderEvent` table: timestamp, provider, profile, event_type (success/failure/degraded/dead/recovered/probe), error_type, error_message, latency_ms |
| 1.3 | Alembic migration for both tables |
| 1.4 | Refactor `CircuitBreaker.record_success()` and `record_failure()` to write events to DB |
| 1.5 | Compute rolling uptime: `successes / (successes + failures)` over 24h / 7d / 30d windows |

### Phase 2: Failover Event Log

| Step | What |
|---|---|
| 2.1 | In `request_pipeline.py` → when `try_chain` falls back to next provider, log a `FailoverEvent` |
| 2.2 | Columns: timestamp, profile, from_provider, to_provider, reason (error_type), request_id |
| 2.3 | Expose via `GET /api/providers/failovers?profile=&from=&to=` |
| 2.4 | Dashboard widget: "Last 20 failovers" table with reason and timestamp |

### Phase 3: Circuit Breaker Toggle

| Step | What |
|---|---|
| 3.1 | `POST /api/providers/{name}/toggle` — body: `{profile: "l2", action: "degrade"|"resume"|"kill"}` |
| 3.2 | `degrade`: force provider into `degraded` state with N-second cooldown |
| 3.3 | `resume`: reset to `healthy`, clear failure count |
| 3.4 | `kill`: force `dead` state (indefinite, requires manual `resume`) |
| 3.5 | Audit log: who toggled what and when |
| 3.6 | UI button per provider: dropdown with "Force Degrade (30s)", "Force Kill", "Resume" |

### Phase 4: Provider Health UI

| Step | What |
|---|---|
| 4.1 | Redesign providers page: add health columns |
| 4.2 | Per-provider row: status dot (green=healthy, amber=degraded, red=dead), uptime % (24h/7d/30d), failure count, last failure, circuit breaker toggle button |
| 4.3 | Click provider row → expand health detail panel: failure breakdown pie chart (timeout / 5xx / rate_limit / auth / other), latency p50/p95/p99, failover events from this provider |
| 4.4 | Provider health card at top of page: summary of all providers (X healthy, Y degraded, Z dead) |
| 4.5 | Sidebar update: replace plugin-only status dots with full provider health dots per profile |
| 4.6 | Real-time polling (every 10s via JS `setInterval`) to keep health dots live |

### Phase 5: Failure Breakdown

| Step | What |
|---|---|
| 5.1 | `GET /api/providers/{name}/failures?window=24h` — returns breakdown by error_type |
| 5.2 | Response: `{timeout: 12, internal_error: 3, rate_limit: 1, auth: 0, bad_request: 0, total: 16}` |
| 5.3 | Chart.js donut chart in provider detail panel |
| 5.4 | Time series: failure count over time (sparkline per provider) |

---

## Data Model Additions

```python
class ProviderHealth(Base):
    __tablename__ = "provider_health"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, nullable=False)
    base_url = Column(String, nullable=False)
    profile = Column(String, nullable=False)
    status = Column(String, default="healthy")
    consecutive_failures = Column(Integer, default=0)
    total_requests = Column(Integer, default=0)
    total_failures = Column(Integer, default=0)
    last_success = Column(String, nullable=True)
    last_failure = Column(String, nullable=True)
    tripped_until = Column(Float, nullable=True)
    uptime_24h = Column(Float, default=100.0)
    uptime_7d = Column(Float, default=100.0)
    uptime_30d = Column(Float, default=100.0)

    __table_args__ = (
        UniqueConstraint("provider", "profile", name="uq_provider_profile"),
    )

class ProviderEvent(Base):
    __tablename__ = "provider_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    profile = Column(String, nullable=False)
    event_type = Column(String, nullable=False)  # success, failure, degraded, dead, recovered, probe
    error_type = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=True)

class FailoverEvent(Base):
    __tablename__ = "failover_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False)
    profile = Column(String, nullable=False)
    from_provider = Column(String, nullable=False)
    to_provider = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    request_id = Column(Integer, ForeignKey("requests.id"), nullable=True)
```

---

## API Endpoints

| Method | Path | Status | Notes |
|---|---|---|---|
| GET | `/api/providers/health` | **NEW** | All provider health with uptime |
| GET | `/api/providers/{name}/health` | **NEW** | Single provider detail |
| GET | `/api/providers/{name}/failures` | **NEW** | Failure breakdown by error_type |
| POST | `/api/providers/{name}/toggle` | **NEW** | degrade/resume/kill |
| GET | `/api/providers/failovers` | **NEW** | Failover event log (paginated) |
| GET | `/health` | exists | Extend with uptime %, failover count |

---

## Files Changed

- `src/api/models.py` — add `ProviderHealth`, `ProviderEvent`, `FailoverEvent`
- `src/api/circuit_breaker.py` — persist to DB, compute uptime, manual toggle
- `src/server/endpoints.py` — new health/failure/failover/toggle endpoints
- `src/server/handler.py` — route new endpoints
- `src/api/request_pipeline.py` — log failover events
- `src/ui/templates/jinja/pages/providers.html` — redesign with health columns
- `src/ui/templates/jinja/_sidebar.html` — provider health dots
- `src/ui/dashboard.js` — health polling
- `alembic/versions/005_add_provider_health.py` — **new**
