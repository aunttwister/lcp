# Task: 5mall-gw

Created: 2026-06-18
Status: in_progress (Phases 1-4 ✅ resolved, 5-7 in_progress, 8 open)
Merged from: llm-gateway-router (completed Phases 1-3), llm-cost-tracking, os-sandboxing
Name chosen: 5mall-gw — "control plane" as in the networking architecture term: routes traffic, enforces policy, manages state.
Active consideration: Replace with https://github.com/theopenco/llmgateway (TypeScript, 1.3K stars) — see replacement task.

## North Star

A self-hosted LLM management gateway that a homelab operator or small team deploys to:
- **Control costs** — per-user credit limits, per-team budgets, cost breakdown
- **Enforce permissions** — tool stripping per profile/team (no open-source gateway has this)
- **Track utilization** — who's using what, time-series analytics
- **Route intelligently** — provider chains with circuit breaker, automatic fallback

**What's already live** (Phases 1-3, deployed on production host):
- Python stdlib HTTP proxy, single Docker container, port :8734, SQLite
- URL-path profile routing (`/l2`, `/l1`, `/career`, `/cron`)
- Tool stripping per profile (L2: write_file/patch/cronjob blocked; L1: everything except read; Cron: all tools blocked)
- Provider chains with automatic fallback (opencode → deepseek)
- Circuit breaker with provider health tracking
- Cost tracking per profile with cache hit/miss breakdown
- Server-rendered HTML dashboard at :8734/
- /health and /v1/models endpoints

## Legacy Completed Phases (from llm-gateway-router)

### Phase 1: URL-path gateway + tool stripping + fallback ✅
**Status: resolved** (completed June 15-16)
**Resolution:** Deployed on production host (`/opt/lcp/`). Docker container on :8734 with 4 profiles:
- `/l2` → strip write_file/patch/cronjob → chain: opencode(v4-pro) → deepseek(v4-pro)
- `/l1` → strip terminal/write_file/patch/execute_code/cronjob/memory → chain: opencode(v4-flash) → deepseek(v4-flash)
- `/career` → strip all except web_search/session_search → deepseek(v4-flash)
- `/cron` → strip ALL tools → deepseek(v4-flash)

### Phase 2: Provider health tracking ✅
**Status: resolved** (completed June 16)
**Resolution:** Circuit breaker with track failures, skip unhealthy providers. /health endpoint shows provider status.

### Phase 3: Cost dashboard ✅
**Status: resolved** (completed June 16-17)
**Resolution:** Server-rendered HTML page at :8734/ showing daily costs per profile. SQLite database with cost data including cache hit/miss breakdown.

---

## Remaining Phases (Productization)

### Phase 4: Config-Driven Chains & Code Quality ✅
**Problem:** Providers hardcoded in Python dict. No YAML config. Code is a single script.
**Status:** resolved (2026-06-19 — previously marked "open" but all items were already deployed)
**Resolution:** All 4 goals achieved:
1. ✅ YAML config (`config/gateway.yaml`) — profiles, providers, chains, forbidden_tools, pricing, circuit_breaker
2. ✅ Separate modules: `config.py`, `router.py`, `cost_estimator.py`, `models.py`, `prompt_cache.py`, `token_verifier.py`, `logging_config.py`, `exceptions.py`
3. ✅ Structured logging (`logging_config.py` → structlog), LCPError hierarchy (`exceptions.py`)
4. ✅ Config hot-reload (5-second polling, no restart needed)

### Phase 5: Multi-Tenant — Users, Teams, Credit Limits
**Problem:** No user/team model. No API key auth. No credit enforcement.
**Status:** in_progress — schema defined, NOT wired to runtime
**Features this addresses:** ① Credit limits per user (admin-set), ② Teams
**What exists:** SQLite schema deployed in dev DB (`teams`, `users` tables with FK, `audit_logs` table). Dev DB seeded with 3 teams + 6 users.
**What's missing:**
1. ✅ SQLite schema: `teams` (id, name, monthly_budget), `users` (id, team_id, credit_limit, api_key_hash, is_admin)
2. ❌ API key authentication (profiles still use URL paths — no `Authorization: Bearer lcp-...` parsing)
3. ❌ Admin API to set/change credit limits per user
4. ❌ Credit enforcement: reject request when user over their limit
5. ❌ Team budget enforcement: reject when team over monthly budget
6. ❌ Admin can create/manage teams and assign users

### Phase 6: Dashboard Upgrade — Time Series & Utilization
**Problem:** Current dashboard is server-rendered HTML showing daily totals. No time series, no per-user breakdown.
**Status:** in_progress — 6a complete, 6b+6c in progress
**Features this addresses:** ⑤ Nice dashboards with time series data

**6a — Sidebar Drawer + Fixed Charts ✅ (2026-07-01)**
- Collapsible 240px sidebar, push-mode desktop, overlay mobile
- Profile nav links with active state
- localStorage persistence for collapsed state
- Chart fix: removed `height="80"`, added `.chart-container { min-height: 280px; }`

**6b — Per-Profile View Flipping (next)**
- "Per Profile" toggle flips charts from single stacked bar → grouped bars per profile
- Summary cards switch from totals → per-profile cards
- URL state preserved (`?view=per-profile`)

**6c — Provider Configuration UI ✅ (2026-07-02)**
- Sidebar tree: Profiles → Providers → Models with health dots
- Provider edit modal: click edit icon → edit URL/key env/models + test connection + save
- Provider CRUD modal (from sidebar "Providers" link): add/delete providers, drag-and-drop fallback chains per profile, "Save All Chains"
- Dashboard cleanup: Provider Health section removed, sidebar tree replaces it

**What's missing (blocked on Phase 5):**
4. ❌ Per-user utilization: request counts, token consumption, cost breakdown
5. ❌ Budget/credit status: progress bars showing usage vs limit
6. ❌ Single-page admin view with all teams and users
7. ❌ Latency distribution / error-rate-over-time charts (can be done independently)

### Phase 7: Permission Matrix — Fails-Closed Rule Engine
**Problem:** Current tool stripping uses flat `forbidden_tools` lists per profile. Fails open: any new tool added to Hermes passes through silently. No cross-cutting blocks. Maintainer must update every profile when a new dangerous tool is added.
**Status:** in_progress — design approved, implementation pending
**Inspiration:** [`@gotgenes/pi-permission-system`](https://www.npmjs.com/package/@gotgenes/pi-permission-system) (agent-side permission enforcement for Pi coding agent). Adopted: fails-closed default-deny, cross-cutting layers, last-match-wins evaluation. Rejected: `ask` state (no UI in gateway), bash command parsing (gateway doesn't see bash), file-path matching (gateway only sees tool names in API body).

**Design (2026-06-27):**

Three architectural changes from current `forbidden_tools`:

1. **Fails-closed** — default `deny` for all profiles. A tool passes ONLY if explicitly listed in `allow`. New Hermes tools are blocked by default until admin whitelists them.
2. **Cross-cutting `blocked_globally` layer** — deny rules that no profile can override. `cronjob: deny` here means even an admin profile can't schedule through the gateway.
3. **Last-match-wins per profile** — broad catch-all first (`"*": deny`), specific overrides after. Deterministic, predictable.

Binary model only (`allow` / `deny`). No `ask` — gateway is headless, no UI to prompt.

**Config structure (proposed `gateway.yaml` section):**

```yaml
permission_matrix:
  # Layer 1: Cross-cutting — cannot be overridden by any profile
  blocked_globally:
    - cronjob
    - skill_manage        # No profile modifies skills through gateway

  # Layer 2: Per-profile rules — last-match-wins within block
  profiles:
    l1:
      default: deny
      allow:
        - read_file
        - search_files
        - session_search
    l2:
      default: deny
      allow:
        - read_file
        - search_files
        - session_search
        - write_file
        - patch
        - terminal
        - memory
        - execute_code
        - delegate_task
        - skill_view
        - clarify
        - todo
    career:
      default: deny
      allow:
        - web_search
        - session_search
    cron:
      default: deny
      allow: []           # Cron calls zero tools through gateway
```

**vs current `forbidden_tools` approach:**

| Aspect | Current (forbidden_tools) | New (permission_matrix) |
|---|---|---|
| Default | Pass-through (fails open) | Deny (fails closed) |
| New tool added to Hermes | Silently passes through | Blocked until whitelisted |
| Cross-cutting blocks | Repeat `cronjob` in every profile | `blocked_globally` once |
| Profile maintenance | Maintain growing deny list per profile | Maintain explicit allow list |
| Audit surface | "What tools are blocked?" | "What can this profile do?" — easier to reason about |

**To build:**
1. **Config parser:** Extend `config.py` to load `permission_matrix` from `gateway.yaml`, merge global + profile layers, produce final `allow` set per profile
2. **Tool gate:** Replace `strip_forbidden_tools()` with `apply_permission_matrix()` — fails-closed, checks global deny first, then profile allow list
3. **Config migration:** Convert existing `forbidden_tools` to allow-list format for all 4 profiles (l2, l1, career, cron)
4. **Admin dashboard section:** Read-only view showing per-profile permission matrix (Phase 6 dashboard upgrade)
5. **Admin API (future):** PUT endpoint to edit allow/deny rules per profile (requires Phase 5 auth first)

**Remaining Phase 7 items (not permission matrix):**
6. **Rate limiting:** Token bucket per user + global rate cap
7. **Audit log:** Wire up runtime logging — every request logged with user, model, tokens, cost, tools blocked, latency (schema exists: `audit_logs` table)
8. **Notifications:** Webhook/email alerts on budget exhaustion, provider failure
9. **Access control:** Admin vs viewer roles on dashboard

### Phase 8: Agent Permission Infrastructure — Daemon Socket
**Problem:** Daemon socket permissions are an adjacent service to the gateway. The gateway provides tool policy; the OS enforces sandboxing.
**Status:** open
**Features this addresses:** ③ Daemon socket permission limiting
**Approach:**
1. Gateway provides the "what tools can this agent use" enforcement. The OS-level sandboxing (systemd service with restricted capabilities, cgroups) is a separate task (from `os-sandboxing` — completed).

## Feature → Phase Mapping

| Feature | Phase | What gets built |
|---|---|---|
| ① Credit limits per user (admin-set) | Phase 5 | User model + credit_limit column + enforcement |
| ② Teams/groups | Phase 5 | Team model + team budgets + user assignment |
| ③ Daemon socket permission limiting | Phase 8 | Integration point — gateway provides tool policy, OS enforces sandbox |
| ⑤ Nice dashboards / time series | Phase 6 | Chart.js embedded charts + per-user utilization views |

## Current Architecture (Phases 1-4)

```
                          ┌─────────────────────────────────┐
Hermes profiles ────────► │  LLM Gateway (:8734)             │
                          │                                  │
L2 → /l2/chat/completions│  URL Router     Tool Stripper     │
L1 → /l1/chat/completions│  (profile→chain) (blocked tools)  │
Career → /career/...     │                                  │
Cron → /cron/...         │  Circuit Breaker   Cost Tracker   │
                          │  (provider health) (SQLite)      │
                          │                                  │
                          │  Dashboard (:8734/)              │
                          │  (daily costs per profile)       │
                          └─────────────────────────────────┘
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                      DeepSeek    OpenCode     (more)
```

## Target Architecture (Phases 4-8)

```
                          ┌──────────────────────────────────────┐
Agent authenticates ────► │  LLM Management Gateway (:8734)       │
with API key              │                                      │
                          │  ┌────────────────────────────────┐  │
                          │  │ Auth Layer (API key → user map) │  │
                          │  ├────────────────────────────────┤  │
                          │  │ Credit Enforcer                 │  │
                          │  │ (user credit + team budget)     │  │
                          │  ├────────────────────────────────┤  │
                          │  │ Permission Engine (Phase 1)     │  │
                          │  │ (tool allow/deny per team)      │  │
                          │  ├────────────────────────────────┤  │
                          │  │ Router + Circuit Breaker (Ph1)  │  │
                          │  │ (YAML config, provider health)  │  │
                          │  ├────────────────────────────────┤  │
                          │  │ Cost Tracker (Phases 1-2)       │  │
                          │  │ (SQLite: tokens, $, cache)      │  │
                          │  ├────────────────────────────────┤  │
                          │  │ Audit Log (Phase 7)             │  │
                          │  │ (every request + tools blocked) │  │
                          │  └────────────────────────────────┘  │
                          │                                      │
                          │  Dashboard (:8734/)                  │
                          │  ┌────────────────────────────────┐  │
                          │  │ Time-series costs (Phase 6)     │  │
                          │  │ Per-user utilization (Phase 6)  │  │
                          │  │ Provider health (Phase 6)       │  │
                          │  │ Budget/credit status (Phase 5)  │  │
                          │  │ Admin panel (Phase 5)           │  │
                          │  └────────────────────────────────┘  │
                          └──────────────────────────────────────┘
```

## Stack (unchanged from Phases 1-3)

| Concern | Choice | Why |
|---|---|---|
| Language | Python (stdlib + http.server) | Already working, fast enough for proxy |
| Database | SQLite | Single-node, zero infra, millions of rows fine |
| Dashboard | Chart.js (CDN) + server-rendered HTML | No build step, single binary |
| Auth | API key hashing (hashlib) | Simple, effective, no JWT complexity |
| Config | YAML | Human-readable, hot-reloadable |
| Deploy | Single Docker container, port :8734 | Already deployed on bridge |

**We do NOT add** PostgreSQL, Redis, pnpm, Next.js, or anything from the TypeScript ecosystem. SQLite handles our scale. One binary, one port.

## Open Source Strategy (if applicable)

- **License:** MIT or AGPLv3 — prevents enterprise upsell like llmgateway
- **Differentiator vs llmgateway & LiteLLM:** "The only LLM gateway that understands agent tool permissions"
- **Positioning:** Lean (no PG/Redis), agent-native (tool stripping), truly open source (no enterprise tears)

## Related Tasks

| Task | Status | Relationship |
|---|---|---|
| llm-cost-tracking | ✅ completed | Absorbed into llm-gateway-router Phase 1 |
| os-sandboxing | ✅ completed | Parallel task — OS-level sandboxing, complements Phase 8 |
| llm-gateway-router | ✅ completed | Phases 1-3 absorbed here |