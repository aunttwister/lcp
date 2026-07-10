# LLM Control Plane — Project Plan

> Self-hosted LLM gateway proxy for Hermes Agent. Provider routing, cost tracking, tool permission enforcement, circuit breaking.

**Repo:** `github.com/aunttwister/smallm-gw`
**Stack:** Python 3.11+, SQLite + Alembic, stdlib `http.server`, Docker, Chart.js
**Dev instance:** `http://localhost:8735/` (container `llm-control-plane-dev`)
**Prod instance:** `http://localhost:8734/` (container `llm-cost-tracker`, single-file proxy — DO NOT TOUCH)

---

## Architecture

```
Hermes Agent → LCP Gateway (:8735) → [OpenCode Zen/Go → DeepSeek fallback]
                     │
                     ├── Tool stripping (per-profile forbidden_tools)
                     ├── Circuit breaker (degraded/dead cooldowns)
                     ├── Prompt cache (hash-based, in-memory)
                     ├── Token verification (provider vs local estimate)
                     ├── Cost tracking (SQLite, per-request)
                     └── Dashboard (server-rendered HTML + Chart.js)
```

**Key design decisions:**
- **No external deps** except Python stdlib + SQLite + Chart.js CDN
- **Single Docker container**, no Redis, no PostgreSQL, no TypeScript
- **Fails-open tool blocking** (legacy `strip_forbidden_tools`) → migrating to **fails-closed** (Phase 7)
- **Config hot-reload**: edit `config/gateway.yaml` + `touch` → live in seconds
- **Dashboard**: server-rendered HTML f-strings, no frontend build step

---

## Phases

### Phase 1-2: Scaffold ✅
- Project layout, Dockerfile, `docker-compose.yml`
- `config/gateway.yaml` with profiles, providers, pricing, circuit breaker
- Alembic migrations, SQLAlchemy models

### Phase 3: Models ✅
- `models.py`: `Request` table (14 columns: id, timestamp, profile, model, provider, prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens, cost, latency_ms, success, error_type, tools_blocked)
- `cost_estimator.py`: tiktoken-based token counting
- `prompt_cache.py`: hash-based semantic cache with TTL
- `token_verifier.py`: provider usage vs local estimate comparison
- `router.py`: dynamic flash↔pro routing by token count

### Phase 4: Server ✅
- `main.py` (~2,985 lines): `ThreadingHTTPServer` + `LCPHandler`
- Routes: `/health`, `/v1/models`, `/errors`, `/cache/stats`, `/metrics`, `/export`, `/api/daily-costs`, `/api/recent-requests`, `/` (dashboard)
- Chat proxy: `/{profile}/chat/completions` → `try_chain()` → provider forwarding
- `strip_forbidden_tools()`: per-profile deny-list, fails open
- `calculate_cost()`: token usage → USD using pricing table
- `record_cost()`: SQLite INSERT with full cost breakdown
- Circuit breaker state machine: healthy → degraded → dead → cooldown → probe
- `proxy.py` (1,377 lines): standalone single-file version (legacy prod)

### Phase 5: Intelligence ❌ NOT STARTED
**Multi-tenant auth, API keys, credit enforcement.**

- Alembic schema exists (users, teams, API keys tables)
- No code yet
- Scope: per-user API keys, team credit limits, usage quotas, key revocation
- This is the biggest remaining piece

### Phase 6: Observability ✅
**Dashboard, charts, provider management.**

- **6a**: Sidebar tree (Profiles → Providers → Models), Chart.js time-series (daily costs), collapsible sidebar with mobile overlay
- **6b**: Per-profile chart toggles, aggregate/per-profile/per-model view flipping, server-side data pivot
- **6c**: Provider Configuration CRUD (modal dialog):
  - Add/edit/delete providers
  - 6 built-in presets (DeepSeek, OpenCode, OpenAI, Anthropic, Groq, xAI)
  - Test Connection button (real API call)
  - Per-profile fallback chain management (drag-and-drop reorder via SortableJS)
  - Profile API key management (generate/revoke per profile)
  - All persisted to `gateway.yaml` with atomic write + hot-reload

### Phase 7: Permission Matrix 🔶 DESIGN APPROVED — NOT IMPLEMENTED
**Replace `strip_forbidden_tools()` with fails-closed rule engine.**

Design reference: `references/pi-permission-system-analysis.md` (based on `@gotgenes/pi-permission-system` v16.2.1)

**Approved design:**
```
permission_matrix:
  blocked_globally:           # Cross-cutting — no profile can override
    - cronjob
    - skill_manage
  profiles:
    l2:
      rules:
        - "*": deny            # Fails-closed default
        - terminal: allow
        - execute_code: allow
        - read_file: allow
        - search_files: allow
        - web: allow
        - ...
      tools:
        - terminal
        - execute_code
        ...
```

**Key principles:**
- **Fails-closed**: default `deny` per profile. New Hermes tools blocked until whitelisted.
- **Binary model**: `allow` / `deny` only. No `ask` — gateway is headless.
- **`blocked_globally`**: deny rules no profile can override.
- **Last-match-wins** per profile: broad catch-all first (`"*": deny`), specific overrides after.

**Implementation order:**
1. Config parser: extend `config.py` → `apply_permission_matrix()`
2. Tool gate: replace `strip_forbidden_tools()` call site in `main.py`
3. Config migration: convert existing `forbidden_tools` → allow-list for all 4 profiles
4. Dashboard: read-only permission matrix view per profile
5. Admin API (future): PUT endpoint to edit rules per profile (requires Phase 5 auth)

### Phase 8: Agent Infrastructure ❌ NOT STARTED
**OS-level sandboxing + git token broker.**

- Gateway provides "what tools can this agent use" policy enforcement
- OS sandboxing via systemd cgroups handled by separate `os-sandboxing` task
- GitHub token broker: gateway issues scoped, time-limited repo tokens per agent profile
- Heavy effort, separate from gateway core

---

## Project Layout

```
/opt/lcp/
├── Dockerfile                  # Python 3.11-alpine, pip + apk build deps
├── docker-compose.yml          # Single service: llm-control-plane-dev (:8735→:8734)
├── pyproject.toml              # pytest config
├── alembic.ini                 # DB migration config
├── config/
│   └── gateway.yaml            # Profiles, providers, pricing, circuit breaker (hot-reload)
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 001_initial_schema.py
├── src/
│   ├── main.py                 # HTTP server (~2,985 lines) — all routes + dashboard
│   ├── config.py               # YAML loading, hot-reload, pricing table
│   ├── models.py               # SQLAlchemy models, get_engine(db_path), get_session(engine)
│   ├── cost_estimator.py       # tiktoken-based token counting
│   ├── prompt_cache.py         # Hash-based semantic cache with TTL
│   ├── router.py               # Dynamic routing: flash↔pro by token count
│   ├── token_verifier.py       # Provider usage vs local estimate comparison
│   ├── exceptions.py           # LCPError hierarchy
│   └── logging_config.py       # structlog setup
├── proxy.py                    # Standalone single-file version (legacy prod on :8734)
├── aggregate.py                # Daily cost aggregation script
└── tests/                      # 13 files, 118 tests (100% passing as of 2026-07-10)
    ├── test_config.py          # 11 tests
    ├── test_main.py            # 12 tests (tool stripping, cost calc, health keys)
    ├── test_main_core.py       # 8 tests (record_cost, forward_request, try_chain)
    ├── test_main_health.py     # 9 tests (circuit breaker state machine)
    ├── test_handler_inprocess.py  # 13 tests (do_GET/do_POST in-process)
    ├── test_integration.py     # 17 tests (HTTP against live :8735)
    ├── test_models.py          # 7 tests
    ├── test_cost_estimator.py  # 7 tests
    ├── test_exceptions.py      # 11 tests
    ├── test_prompt_cache.py    # 6 tests
    ├── test_router.py          # 6 tests
    ├── test_token_verifier.py  # 6 tests
    └── test_logging_config.py  # 4 tests
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | HTML dashboard (all profiles) |
| GET | `/{profile}/dashboard` | Per-profile dashboard |
| GET | `/health` | JSON health + per-provider circuit breaker state |
| GET | `/v1/models` | Available models (OpenAI-compat) |
| GET | `/errors` | Recent errors |
| GET | `/cache/stats` | Prompt cache hit/miss stats |
| GET | `/metrics` | Prometheus-compatible |
| GET | `/export?limit=N` | CSV cost data export |
| GET | `/api/daily-costs` | JSON daily costs |
| GET | `/api/recent-requests` | JSON recent requests |
| GET | `/api/providers` | List providers + chains |
| POST | `/api/providers` | Create/update provider |
| GET | `/api/providers/presets` | 6 built-in provider presets |
| POST | `/api/providers/test` | Test provider connection |
| PUT | `/api/providers/<name>` | Update provider |
| DELETE | `/api/providers/<name>` | Remove provider |
| PUT | `/api/chains/<profile>` | Reorder fallback chain |
| GET | `/api/profiles` | List profiles |
| POST | `/api/profiles` | Create new profile |
| GET | `/api/keys` | List API keys |
| POST | `/api/keys` | Generate new key |
| DELETE | `/api/keys/<id>` | Revoke key |
| POST | `/{profile}/chat/completions` | Chat proxy (profile-scoped) |
| POST | `/{profile}/v1/chat/completions` | Chat proxy (v1 compat) |

---

## Test Conventions

```bash
cd /opt/lcp

# Full suite (118 tests)
python3 -m pytest tests/ -v --tb=short

# With coverage
python3 -m pytest tests/ -v --cov=src --cov-report=term-missing

# Integration only (requires live dev container on :8735)
LCP_TEST_URL=http://localhost:8735 python3 -m pytest tests/test_integration.py -v

# In-process only (no container needed)
python3 -m pytest tests/test_handler_inprocess.py tests/test_main.py tests/test_main_core.py tests/test_main_health.py -v
```

**Rebuild after source changes:**
```bash
cd /opt/lcp
docker compose build --no-cache && docker compose up -d --force-recreate
```

---

## Known Pitfalls (learned the hard way)

See `skill: llm-control-plane` for the full 31-pitfall catalog. Highlights:

1. **Code is baked into image** — must rebuild after any source change (exception: `gateway.yaml` is bind-mounted, hot-reload via `touch`)
2. **`HTTPServer` is single-threaded** — dashboard deadlocks. Always use `ThreadingHTTPServer`
3. **Cloudflare WAF blocks default Python User-Agent** — `forward_request()` MUST set `User-Agent: LLMControlPlane/1.0`
4. **`strip_forbidden_tools(body, None)` = block ALL tools** — `None` is not "allow all"
5. **`.filter()` after `.limit()` destroys dashboards** — `.limit()` must be the final method before `.all()`
6. **Query params break API routing → 404** — strip query string before path matching
7. **OpenCode provider is perma-dead** (Cloudflare 403) — all L2 traffic falls through to DeepSeek

---

## Next Steps (priority order)

1. **Phase 7 — Permission Matrix** (medium effort, highest impact)
   - Replace fails-open `strip_forbidden_tools()` with fails-closed `apply_permission_matrix()`
   - Config migration: convert existing 4 profiles to allow-list format
   - Dashboard view only (admin API requires Phase 5)

2. **Phase 5 — Multi-tenant Auth** (heavy effort)
   - Users, teams, API keys, credit limits
   - Schema exists, code not started

3. **Phase 8 — Agent Infrastructure** (heavy effort, separate system)
   - OS sandboxing, git token broker
   - Depends on Phase 7 for policy enforcement

4. **Test gap coverage**
   - Dashboard HTML rendering tests (0 tests for 14.5KB f-string template)
   - Provider CRUD endpoint tests (`/api/providers`, presets, test-connection, chain reorder)
   - Profile config modal tests (`/api/profiles`, `/api/keys`)
