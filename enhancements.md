# LCP Dashboard — Enhancement Plan

> Based on LiteLLM UI comparison (2026-07-21).

---

## Priority 1: API Key Management + Budgeting + Alerting

### API Key Management
A dedicated page to manage virtual keys for API access.

- [ ] **Key listing table** — list all keys with columns: name, key prefix (masked), created date, last used, spend to date, status (active/revoked)
- [ ] **Create key** — modal with fields: name, allowed profiles, spend limit, expiry (optional), metadata tags
- [ ] **Rotate key** — generate new key value, soft-delete old one after grace period
- [ ] **Revoke key** — soft-delete with confirmation dialog, show revocation timestamp
- [ ] **Key detail view** — click a key row to see: full usage history, per-profile spend breakdown, recent requests made with this key
- [ ] **Copy-to-clipboard** for key value (shown once at creation, with warning)
- [ ] **Quick filter/search** — by name, profile, status
- [ ] **Backend endpoints:**
  - `POST /api/keys` — create key
  - `GET /api/keys` — list keys
  - `GET /api/keys/{id}` — key detail + stats
  - `POST /api/keys/{id}/rotate` — rotate
  - `DELETE /api/keys/{id}` — revoke

### Budgeting
Spend limits and alerts per key and per profile.

- [ ] **Per-key budget** — set monthly/total spend cap (hard stop or soft alert)
- [ ] **Per-profile budget** — cap spend across all keys on a given profile
- [ ] **Budget alert thresholds** — warn at 50%, 80%, 90% of budget
- [ ] **Budget status card** on main dashboard — current month spend vs budget
- [ ] **Budget enforcement** — return 429 when budget exceeded (configurable: hard stop vs. log-and-allow)
- [ ] **Budget history table** — monthly spend per key/profile with overage flags
- [ ] **Backend endpoints:**
  - `POST /api/budgets` — create budget
  - `GET /api/budgets` — list budgets
  - `PUT /api/budgets/{id}` — update budget
  - `GET /api/budgets/status` — current month status for all budgets

### Alerting
Real-time notifications on critical events.

- [ ] **Alert rules engine** — configurable triggers:
  - Budget breach (50%, 80%, 90%, 100% thresholds)
  - Provider status change (healthy → degraded → dead)
  - Error rate spike (X errors in Y minutes)
  - Circuit breaker trip / recovery
- [ ] **Webhook notifications** — POST JSON payload to configurable URL on alert fire/resolve
- [ ] **Alert history table** — log all fired alerts with: timestamp, rule, severity, status (firing/resolved), acknowledged by
- [ ] **Alert config page in UI** — enable/disable rules, set thresholds, configure webhook URL + secret
- [ ] **Alert badge in sidebar** — red dot / count of active (unresolved) alerts
- [ ] **Severity levels:** info, warning, critical
- [ ] **Cooldown** — don't re-fire the same alert within N minutes
- [ ] **Backend endpoints:**
  - `GET /api/alerts/config` — current alert rules config
  - `PUT /api/alerts/config` — update alert rules
  - `GET /api/alerts` — alert history (paginated, filterable)
  - `POST /api/alerts/{id}/acknowledge` — mark alert as acknowledged
  - `POST /api/alerts/webhook/test` — send test webhook

---

## Priority 2: More Dashboards

### Usage Analytics Dashboard
Deeper metrics beyond the current summary cards.

- [ ] **Date range picker** — custom from/to with presets (7d, 14d, 30d, 90d, all)
- [ ] **Top keys by spend** — bar chart / ranked list
- [ ] **Top models by usage** — token volume per model
- [ ] **Per-profile breakdown** — side-by-side spend comparison across profiles
- [ ] **Hourly heatmap** — request volume by hour of day (last 7 days)
- [ ] **Latency percentiles** — p50/p95/p99 per provider per profile
- [ ] **Export to CSV** — download filtered usage data

### Request Log Viewer
Deep inspection of individual requests.

- [ ] **Full request/response body viewer** — click any request row to expand inline JSON viewer
- [ ] **Filter by:** date range, profile, model, provider, status (success/error/fallback), key
- [ ] **Search by:** request body text, error message
- [ ] **Pagination** — current limit of 50 rows is too small for debugging
- [ ] **Timing waterfall** — show DNS, connect, TTFB, streaming breakdown per request

### Provider Health Dashboard
Dedicated view beyond the sidebar dots.

- [ ] **Provider uptime %** — rolling 24h / 7d / 30d
- [ ] **Failure reason breakdown** — timeout vs. 5xx vs. rate limit vs. auth error
- [ ] **Failover event log** — when a provider was skipped, why, and which fallback was used
- [ ] **Manual circuit breaker toggle** — force a provider into degraded/dead for maintenance

### Real-time Streaming View
- [ ] **Live tail** of incoming requests (SSE or polling every 2s)
- [ ] **Live cost ticker** — running total for current day
- [ ] **Active request count** — gauge of in-flight requests

---

## Priority 3: Authentication (Low)

Deferred — project is single-user / localhost for now.

- [ ] Login page (simple token or username/password)
- [ ] Session management
- [ ] Optional: SSO (OAuth/OIDC)
- [ ] Role-based access (admin vs. viewer)

---

## Nice-to-Have (Backlog)

- [ ] **LLM Playground** — interactive chat testing UI per profile
- [ ] **Swagger / API doc embed** — interactive OpenAPI explorer at `/docs`
- [ ] **Config hot-reload indicator** — show when config was last reloaded, diff of changes
- [ ] **Tag management** — tag keys/requests for cost allocation
- [ ] **Dark/light theme toggle** (currently dark-only)
- [ ] **Multi-select & bulk actions** in tables (e.g., revoke multiple keys)
- [ ] **Audit log viewer** — who did what when (create/revoke key, change budget, etc.)

---

## Implementation Notes

- Keep server-rendered HTML approach (no SPA framework needed) — fast, simple, no build step
- Add new pages as URL routes served by the existing `LCPHandler` in `server.py`
- New database tables (via Alembic migration) for: `api_keys`, `budgets`
- All new UI in `src/ui/` — templates, CSS, JS modules
- Use existing Chart.js for any new charts
- Reuse existing shadcn-inspired CSS design tokens
