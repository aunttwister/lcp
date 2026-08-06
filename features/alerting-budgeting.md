# Feature: Alerting & Budgeting Enhancement

**Created:** 2026-08-06
**Status:** open
**Phase:** post-Phase 7

## North Star

Move from the current in-memory `AlertManager` singleton (volatile, no persistence, no UI) to a full alerting + budgeting system backed by the database, with a dedicated UI page. Budgets should enforce spend limits before requests hit the LLM. Alerts should persist across restarts and be visible/acknowledgeable in the dashboard.

---

## Current State (What Exists)

### AlertManager (`src/api/alert_manager.py`)
- In-memory singleton, **all alert history lost on restart**
- 6 rule types: `budget_breach`, `provider_dead`, `provider_degraded`, `error_spike`, `circuit_breaker_trip`, `circuit_breaker_recovery`
- Webhook dispatch in background thread (POST with `X-LCP-Webhook-Secret`)
- 5-minute cooldown per dedup key
- Error spike tracking: rolling window, configurable threshold
- Config via in-memory dict (also lost on restart)
- `fire_budget_breach()` and `fire_provider_status()` convenience methods
- API endpoints exist: `GET /api/alerts`, `GET /api/alerts/active`, `GET /api/alerts/config`, `PUT /api/alerts/config`, `POST /api/alerts/{id}/acknowledge`, `POST /api/alerts/webhook/test`

### Budgeting (`models.py` + `endpoints.py`)
- `Budget` table exists in schema (key_id, profile, amount, current_spend, period, threshold_pct, action, status)
- `ApiKey` table has `spend_limit` and `total_spend` fields
- Handler checks spend limit on key auth, returns 403
- Budget endpoint and UI are **stubs/empty** — no CRUD, no enforcement, no display

### What's Missing
- No persistent alert storage (SQLite)
- No alert UI page in dashboard
- No sidebar alert badge
- Budget CRUD is defined in schema but not wired up
- Budget enforcement only checks key-level spend limit, not profile-level or global budgets
- No budget status card on dashboard
- No 429 response when budget exceeded (only 403 on key spend limit)
- Alert config not persisted (vanishes on restart)

---

## Plan

### Phase 1: Persist Alerts to SQLite

| Step | What |
|---|---|
| 1.1 | Add `Alert` table to `models.py`: id, timestamp, dedup_key, rule, severity, title, message, metadata (JSON), status (firing/resolved), acknowledged, acknowledged_at, resolved_at |
| 1.2 | Alembic migration for `alerts` table |
| 1.3 | Refactor `AlertManager` to write alerts to DB instead of in-memory list. Keep in-memory `_active_alerts` cache for fast lookup but back it with DB |
| 1.4 | Load alert config from DB on startup (fall back to defaults) |
| 1.5 | Load alert cooldowns from DB |

### Phase 2: Budget Enforcement

| Step | What |
|---|---|
| 2.1 | Wire up `POST/PUT/GET/DELETE /api/budgets` CRUD endpoints (schema exists, endpoints are stubs) |
| 2.2 | Add `check_budget(profile, key_id, estimated_cost) → bool` to handler pipeline — enforces BEFORE sending to LLM |
| 2.3 | Return 429 with `{"error": "budget_exceeded", "budget_name": "..."}` when blocked |
| 2.4 | After each successful request, increment `budgets.current_spend` and `api_keys.total_spend` |
| 2.5 | After spend update, check thresholds and fire alerts via `fire_budget_breach()` |
| 2.6 | Budget `action` field: `log` (alert only) vs `block` (429) |

### Phase 3: Alert UI Page

| Step | What |
|---|---|
| 3.1 | New `pages/alerts.html` Jinja2 template — full-page alert viewer |
| 3.2 | Table columns: timestamp, severity badge (info/warning/critical), rule, title, status (firing/resolved), acknowledged |
| 3.3 | Filter by: severity, status, rule type, acknowledged state |
| 3.4 | Click row to expand: full message, metadata JSON, webhook payload |
| 3.5 | Acknowledge button per alert (POST to existing endpoint) |
| 3.6 | Navbar link "Alerts" with red badge count of active (unresolved) alerts |
| 3.7 | Alert config panel: enable/disable rules, set min_severity, cooldown, webhook URL + test button |
| 3.8 | Budget table on same page or separate budget card: name, profile, amount, current spend, %, status |

### Phase 4: Budget UI

| Step | What |
|---|---|
| 4.1 | Budget management modal/page: create/edit/delete budgets |
| 4.2 | Fields: name, key (dropdown), profile (dropdown), amount, period (monthly/total), thresholds (comma-separated pcts), action (log/block) |
| 4.3 | Budget status widget on main dashboard: current month spend vs total budget, per-budget progress bars |
| 4.4 | Budget history chart: monthly spend vs budget over time |

---

## Data Model Additions

```python
class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False)
    dedup_key = Column(String, nullable=False, index=True)
    rule = Column(String, nullable=False)
    severity = Column(String, nullable=False)  # info, warning, critical
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON blob
    status = Column(String, default="firing")     # firing, resolved
    acknowledged = Column(Integer, default=0)
    acknowledged_at = Column(String, nullable=True)
    resolved_at = Column(String, nullable=True)
```

---

## Config (`gateway.yaml` additions)

```yaml
alerts:
  enabled: true
  cooldown_seconds: 300
  webhook_url: ""
  webhook_secret: ""
  rules:
    budget_breach:        { enabled: true,  min_severity: warning }
    provider_dead:        { enabled: true,  min_severity: critical }
    provider_degraded:    { enabled: true,  min_severity: warning }
    error_spike:          { enabled: false, min_severity: warning, threshold: 10, window_minutes: 5 }
    circuit_breaker_trip:    { enabled: true,  min_severity: warning }
    circuit_breaker_recovery:{ enabled: true,  min_severity: info }
```

---

## API Endpoints (existing + new)

| Method | Path | Status | Notes |
|---|---|---|---|
| GET | `/api/alerts` | exists | Add DB-backed pagination |
| GET | `/api/alerts/active` | exists | fine as-is |
| GET | `/api/alerts/config` | exists | Load from DB + gateway.yaml |
| PUT | `/api/alerts/config` | exists | Persist to DB |
| POST | `/api/alerts/{id}/acknowledge` | exists | Write to DB |
| POST | `/api/alerts/webhook/test` | exists | fine as-is |
| GET | `/api/budgets` | **NEW** | List budgets |
| POST | `/api/budgets` | **NEW** | Create budget |
| PUT | `/api/budgets/{id}` | **NEW** | Update budget |
| DELETE | `/api/budgets/{id}` | **NEW** | Delete budget |
| GET | `/api/budgets/status` | **NEW** | Current spend vs budget for all |

---

## Files Changed

- `src/api/models.py` — add `Alert` table ✅
- `src/api/alert_manager.py` — DB-backed persistence, engine-bound singleton ✅
- `src/server/endpoints.py` — budget CRUD ✅
- `src/server/handler.py` — budget enforcement in pipeline (check_budget → 429, spend tracking → alerts) ✅
- `src/ui/pages.py` — render alerts page (pending)
- `src/ui/templates/jinja/pages/alerts.html` — **new** (pending)
- `src/ui/templates/jinja/_sidebar.html` — alert badge (pending)
- `config/gateway.yaml` — alerts section (pending)
- `alembic/versions/004_add_alerts_table.py` — **new** ✅

---

## Phase 5: Extensive unit tests ✅ DONE

| Step | File | What |
|---|---|---|
| 5.1 | `tests/test_budget_endpoints.py` | **NEW** — 30 tests: budget CRUD, budget status, `_check_budget_block` (exceeded/log/profile scoping/global/exceeded-status), `_increment_budget_spend` (increment/threshold crossing/multiple thresholds/exceeded/key budgets), `_track_budget_spend` (alert firing/critical) |
| 5.2 | `tests/test_alert_manager_db.py` | **NEW** — 16 tests: DB persistence, metadata JSON, restart survival, status filtering, resolve/acknowledge updates, `_alert_to_dict`, singleton engine init, graceful degradation |
| 5.3 | `tests/test_alert_manager.py` | Updated for engine-less fallback (in-memory `list_alerts`) |

Full suite: **567 passed, 15 deselected** (integration tests deselected by default).
