# LLM Control Plane (LCP)

**A self-hosted LLM gateway. Route, meter, control — one container, one port, no cloud dependency.**

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com/)
[![CI](https://github.com/aunttwister/lcp/actions/workflows/ci.yml/badge.svg)](https://github.com/aunttwister/lcp/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-804%20passed-brightgreen.svg)](https://github.com/aunttwister/lcp/actions/workflows/ci.yml)

---

## Contents

- [What is LCP?](#what-is-lcp)
- [Features](#features)
- [Quick Start](#quick-start)
- [VS Code Integration](#vs-code-integration)
- [Configuration](#configuration)
- [Why LCP over alternatives?](#why-lcp-over-alternatives)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Test Coverage](#test-coverage)
- [API](#api)
- [Roadmap](#roadmap)
- [Status](#status)

---

## What is LCP?

LCP (LLM Control Plane) is a self-hosted LLM gateway that sits between your clients and LLM
providers. It routes requests, tracks costs, enforces spending limits, and manages API keys —
all from a single Docker container backed by SQLite. No PostgreSQL. No Redis. No external
services.

**Two primary use cases:**

- **AI agents** — Route Hermes, Claude Code, or custom agents through LCP to control which
  tools they can use, how much they can spend, and which providers they hit
- **VS Code / GitHub Copilot** — Point the [GitHub Copilot LLM Gateway extension](#vs-code-integration)
  at LCP to make all your profiles appear as model providers in Copilot Chat with full context
  windows and capabilities

It was built for a production multi-agent homelab setup managing 6 Hermes profiles with 15+
custom skills, where knowing exactly what is being spent, by whom, and with what tools is
critical. The same instance also serves as the LLM backend for daily VS Code Copilot usage.

```
Clients (agents, VS Code, scripts, curl)
          |
          v
+--------------------------------------------+
|             LCP (:8734)                   |
|                                            |
|  Auth -> Estimate Cost -> Route            |
|       -> Circuit Breaker -> Track Cost     |
|                                            |
|  Dashboard     API Keys     Budgets        |
|  (:8734/)      Alerts      Export          |
+-------------------+------------------------+
                    |
        +-----------+-----------+
        v           v           v
    DeepSeek    OpenCode    llama.cpp
```

## Features

### Intelligent routing
- Provider chains with automatic fallback — if one provider fails, the next in chain takes over
- Circuit breaker with configurable thresholds — degraded providers are probed, dead providers are skipped
- Profile-based routing by URL path: `/l2`, `/l1`, `/career`, `/coder`, `/cron`
- SSE streaming passthrough — real-time token delivery, no buffering
- *(Planned)* Harness-style dynamic routing — auto-route to flash (cheap) or pro (capable) based on request complexity

### Tool permission control
- Strip dangerous tools per profile — an L1 triage agent cannot `terminal`, a cron profile strips everything
- *(Being replaced)* Tool stripping will be decommissioned in favor of a fails-closed permission matrix
- *(Planned)* Permission matrix — declarative `allow` / `block` / `blocked_globally` rules per profile, with audit trail

### Cost tracking
- Per-request cost with DeepSeek cache hit/miss breakdown
- Daily cost summaries per profile, per model, per provider
- Prompt prefix caching — normalizes messages so repeated system prompts hit the provider cache
- Cache hits are 120x cheaper than cache misses; LCP tracks both accurately

### API key management
- Create, rotate, and revoke virtual API keys through the dashboard
- Per-key spend limits with hard-stop enforcement
- Per-profile access control — each key can be scoped to specific profiles
- Usage breakdown per key

### Provider health & credentials
- **Encrypted provider keys** — paste an upstream API key (DeepSeek, OpenCode, etc.) directly in the Providers → Configuration tab; it is encrypted with Fernet using the `LCP_SECRET_KEY` master key and stored in SQLite, never in the git-tracked `gateway.yaml`
- No env vars needed — keys come exclusively from the encrypted credential store (UI-managed)
- **Circuit breaker health** — Providers → Health tab shows live status per provider/profile with failure counts, last error reason, and cooldown timers
- **Reset cooldown** — force a provider back to healthy with one click instead of waiting out the cooldown
- Dashboard shows provider health mini-badges (healthy / degraded / dead counts)

### Dashboard
- Server-rendered HTML — no SPA, no build step, loads instantly
- Daily cost charts with Chart.js — stacked bars, per-profile views
- Provider health monitoring with live status
- Drag-and-drop chain editing — reorder fallback providers in the UI
- Request log with error inspection
- Budget alerts with configurable thresholds

### Plugin architecture
- Provider cost extraction plugins — DeepSeek, OpenCode, OpenAI
- Memory plugin with embedded [LanceDB](https://github.com/lancedb/lancedb) backend — columnar vector storage, ANN indexing, no separate service
- See [features/memory.md](features/memory.md) for the unified memory specification

## Quick Start

```bash
# Clone
git clone https://github.com/aunttwister/lcp.git
cd lcp

# Configure providers — add your API keys
cp config/.env.example config/.env
# Edit config/.env: set LCP_SECRET_KEY (used to encrypt provider keys)
# Provider keys are managed via the dashboard (Providers → Configuration tab)
# Only LCP_SECRET_KEY needs to be set (used to encrypt stored keys).

# Run
docker compose up -d
```

LCP is now running at `http://localhost:8734`. Open the dashboard, send a request:

```bash
curl http://localhost:8734/l2/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## VS Code Integration

Use the [**GitHub Copilot LLM Gateway**](https://marketplace.visualstudio.com/items?itemName=arbs-io.github-copilot-llm-gateway)
extension by Andrew Butson to make your LCP profiles appear directly in GitHub Copilot Chat as
model providers. **All your profiles (coder, l2, career, etc.) show up in the Copilot model
picker with their full context windows and capabilities.**

### Setup

1. Install the extension: `arbs-io.github-copilot-llm-gateway`
2. In VS Code settings, search for **"Copilot Llm Gateway"** and set:
   - **Server URL**: `https://lcp.example.com/v1` (or your LCP instance)
   - **API Key**: your LCP API key (from the Keys dashboard)
3. (Optional) **Model Context Windows**: if you need to override server-reported context sizes
4. **Disable "Enable Image Input"** — your LCP profiles use text-only models (deepseek-v4 etc.).
   The gateway already blocks image requests with a clear error as a safety net.

> **Why this extension?** Unlike OAI Copilot (which requires manually configuring each model
> and doesn't support automatic discovery), the LLM Gateway extension fetches `/v1/models`
> from your LCP instance and populates the model picker automatically. Profile entries like
> `coder`, `l2`, and `career` appear with their correct 1M context windows, tool-calling
> support, and the `supports_vision: false` flag that the gateway reports.

### Troubleshooting

- **Models don't appear?** Run "GitHub Copilot LLM Gateway: Refresh Models" from the command palette.
- **128k context instead of 1M?** Ensure your LCP instance is running the latest version that
  serves `max_model_len` and `context_length` in `/v1/models`.
- **"model does not support vision" errors?** Make sure "Enable Image Input" is turned **off**
  in the extension settings — your DeepSeek models are text-only.
- **Chat errors with "received 0 chars / 0 text parts / 0 tool calls"?** This happens on long
  agentic tasks when a reasoning model (`deepseek-v4-*`) spends its entire output budget on
  thinking and produces no answer before hitting the extension's output-token cap. The gateway
  caps output at **Default Max Output Tokens** (default `4096`) even though LCP reports the full
  1M context. Raise `github.copilot.llm-gateway.defaultMaxOutputTokens` in VS Code settings
  (e.g. `8192`–`16384`) so the model has room to finish reasoning and emit a real response.
- **DeepSeek 400 "reasoning_content must be passed back"?** This happens when an agent / Copilot
  strips the `reasoning_content` field from thinking-mode assistant turns in multi-turn history.
  LCP automatically recovers the real reasoning content it saw in earlier responses and
  re-attaches it. See [Thinking-Mode Reasoning Recovery](features/thinking-mode-recovery.md)
  for the architecture.

## Configuration

Everything lives in `config/gateway.yaml`. Hot-reloaded — edit while the server is running, no
restart needed.

```yaml
profiles:
  l2:
    forbidden_tools: [write_file, patch, cronjob]
    chain:
      - provider: opencode
        model: deepseek-v4-pro
      - provider: deepseek                       # fallback
        model: deepseek-v4-pro

  cron:
    forbidden_tools: null                        # null = strip ALL tools
    chain:
      - provider: deepseek
        model: deepseek-v4-flash

providers:
  deepseek:
    # API key is entered via the dashboard (Providers → Configuration tab),
    # encrypted with LCP_SECRET_KEY and stored in SQLite — no env vars here.
    cache:
      strategy: prefix
      savings: cost
      hit_field: prompt_cache_hit_tokens

pricing:
  - provider: deepseek
    model: deepseek-v4-pro
    cache_hit: 0.003625
    cache_miss: 0.435
    output: 0.87
```

## Why LCP over alternatives?

| | LCP | LiteLLM | llmgateway |
|---|---|---|---|
| **Tool permission enforcement** | Yes, per profile | No | No |
| **Deployment** | Single container, SQLite | Container + PostgreSQL + Redis | Container |
| **Dashboard** | Server-rendered HTML, no build | React SPA | Next.js SPA |
| **License** | AGPL-3.0, all features included | MIT core, enterprise tier | Source-available |
| **Memory footprint** | ~100 MB baseline | 500 MB+ service stack | Node.js baseline |
| **Agent-native** | Built for Hermes agents | Generic API proxy | Generic API proxy |
| **API key management** | Virtual keys, spend limits | Virtual keys | Limited |

LCP is the only open-source LLM gateway with agent tool permission enforcement. No other
gateway strips tools based on who is calling. If you run AI agents in production, this matters.

## Architecture

```
LCP (:8734) — single process, single port
|
+-- Python stdlib (http.server) — no framework overhead
+-- SQLite (costs.db) — zero-infrastructure persistence
+-- YAML config — human-readable, hot-reloadable
+-- Server-rendered dashboard — Chart.js, no SPA
+-- Plugin architecture — provider costs, memory backends
+-- Docker — python:3.11-slim
```

LCP deliberately avoids PostgreSQL, Redis, Next.js, pnpm, Kubernetes, and the broader
TypeScript ecosystem. SQLite handles millions of cost rows at homelab scale. One container,
one port, one `docker compose up`.

## Dependencies

Every runtime dependency and what it does in LCP:

| Package | Role in LCP |
|---|---|
| **`structlog`** | Structured JSON logging to stdout. Every request, error, budget breach, and startup step is a machine-readable log line — `docker logs lcp` is grep-friendly. |
| **`sqlalchemy`** | SQLite ORM for the `requests`, `budgets`, `alerts`, and `api_keys` tables. All cost history, spend limits, and alert state lives here. No external database. |
| **`alembic`** | Database schema migrations. Every schema change (alerts table, API keys, error_detail column) gets a numbered migration in `alembic/versions/`. Run automatically on container startup via `alembic upgrade head`. |
| **`pyyaml`** | Reads and writes `config/gateway.yaml`. The config is hot-reloaded — edit providers, profiles, or pricing while the server is running and changes take effect on the next request. |
| **`tiktoken`** | Exact BPE token counts using the `cl100k_base` encoding (same tokenizer used by DeepSeek and OpenAI models). Powers the pre-request `X-Estimated-Cost` header and the dynamic flash/pro router. The ~1 MB vocabulary file is pre-downloaded at Docker build time and persisted to a volume — zero CDN dependency at runtime. |
| **`jinja2`** | Server-rendered HTML templates for the dashboard, profiles page, providers, API keys, alerts, and logs. No SPA, no build step, no npm — pages load instantly from the server. Shared partials for sidebar, modals, and JS utilities. |

Dev-only dependencies (`pip install .[dev]`):

| Package | Role |
|---|---|
| `pytest` | Test runner — 804 unit tests covering routing, budgets, alerts, cost estimation, auth enforcement, circuit breaker, encrypted credentials, and the plugin system |
| `pytest-cov` | Coverage reports — `pytest --cov=src --cov-report=term-missing` |
| `pytest-mock` | Mocking utilities for the `unittest.mock` patch system |

## Test Coverage

**94% overall** — 3,433 of 3,657 statements covered (804 tests, 0 integration tests).

Run: `.venv/bin/python -m pytest --cov=src --cov-report=term-missing -q`

| Module | Coverage |
|---|---|
| `src/api/alert_manager.py` | 99% |
| `src/api/circuit_breaker.py` | 100% |
| `src/api/cost_estimator.py` | 100% |
| `src/api/cost_plugins/opencode_api.py` | 100% |
| `src/api/cost_plugins/base.py` | 100% |
| `src/api/credential_store.py` | 97% |
| `src/api/crypto.py` | 93% |
| `src/api/exceptions.py` | 100% |
| `src/api/key_manager.py` | 98% |
| `src/api/logging_config.py` | 100% |
| `src/api/models.py` | 100% |
| `src/api/prompt_cache.py` | 100% |
| `src/api/request_pipeline.py` | 99% |
| `src/api/router.py` | 100% |
| `src/server/server.py` | 100% |
| `src/server/sse_helpers.py` | 100% |
| `src/ui/dashboard.py` | 100% |
| `src/ui/pages.py` | 100% |
| `src/api/config.py` | 98% |
| `src/api/cost_plugins/deepseek.py` | 94% |
| `src/api/cost_plugins/llamacpp.py` | 96% |
| `src/api/cost_plugins/opencode.py` | 94% |
| `src/api/token_verifier.py` | 97% |
| `src/main.py` | 98% |
| `src/ui/render.py` | 96% |
| `src/server/handler.py` | 90% |
| `src/server/endpoints.py` | 85% |

## API

| Endpoint | Description |
|---|---|
| `POST /{profile}/v1/chat/completions` | OpenAI-compatible chat completions |
| `GET /health` | Provider health and circuit breaker status |
| `GET /v1/models` | Available models across all providers |
| `GET /` | Dashboard |
| `GET /api/daily-costs` | JSON cost data |
| `POST /api/keys` | Create API key |
| `GET /api/keys` | List API keys |

See [PLAN.md](PLAN.md) for the full API reference.

## Roadmap

### Shipped
- OpenAI-compatible chat completions API (`/{profile}/v1/chat/completions`)
- Profile-based routing with provider chains and automatic fallback
- Circuit breaker — degraded (probed) and dead (skipped) provider states
- Per-request cost tracking with DeepSeek cache hit/miss breakdown
- Prompt prefix caching — deterministic message ordering for maximum cache-hit rate
- Server-rendered dashboard — Chart.js, budget cards, provider health, alert badge
- Budget system — per-profile and per-key budgets, unified spend tracking
- Budget enforcement — pre-LLM block (HTTP 429), post-request spend increment + threshold alerts
- Alerting — DB-persisted alerts, webhook dispatch, acknowledge/resolve, alert history page
- API key management — create, rotate, revoke; per-key spend limits; profile access scoping
- SSE streaming passthrough — real-time token delivery, no buffering
- Provider plugins — DeepSeek (balance API), OpenCode (web API), llama.cpp (local /models)
- Provider model discovery — auto-detect models from `/v1/models` with metadata
- tiktoken integration — exact BPE token counts, pre-downloaded at build time
- Startup observability — per-step timing logs in `docker logs`

### Scope boundary
- **Credit limits** — already covered by per-key budgets (spend caps with hard-stop)
- **Alerting** — already covered by the alert system (threshold breaches, webhooks)
- **Multi-tenant teams/users** — out of scope. LCP stays a single-instance, single-org
  gateway; we will not build org/team membership, per-user billing, or role management.

### Planned
- Permission matrix — declarative `allow` / `block` / `blocked_globally` rules,
  replaces tool stripping ([spec](features/permission-plugin.md))
- Harness-style dynamic routing — auto-route flash vs pro based on request complexity
- Logging enhancements — structured request telemetry, token-usage logging
  ([spec](features/logging-enhancements.md))
- Provider health dashboard — richer uptime/health surfacing
  ([spec](features/provider-health.md))
- Memory plugin — embedded LanceDB, unified `/v1/memories` endpoint
  ([spec](features/memory.md))
- Dashboard enhancements — backlog from the original dashboard plan
  ([spec](features/enhancements.md))
- Rate limiting — *only if needed*: budgets already cap spend, but they do not
  cap request frequency. A per-key/per-profile requests-per-minute limiter is a
  separate feature, only worth building if throughput (not cost) becomes a concern.

Full details in [PLAN.md](PLAN.md).

## Status

LCP runs continuously in production, routing all LLM traffic for a multi-agent homelab
(6 Hermes profiles, 15+ custom skills, daily cron jobs) and serving as the LLM backend
for daily VS Code Copilot usage. It handles approximately 200 requests per day across 5
profiles with real cost tracking, budget enforcement, and alerting.

## AI Attribution

This project was built with significant assistance from AI coding agents (Hermes Agent with
DeepSeek V4). Architecture decisions, code, tests, and documentation were produced in
collaboration between a human operator and AI. The bugs are mine; the architecture is ours.

## License

[GNU Affero General Public License v3.0](LICENSE) — you can use, modify, and
distribute LCP freely. If you modify it and offer it as a network service
(e.g. a hosted LLM gateway SaaS), you must make your modified source code
available to users under the same license. This protects the project from
being wrapped into proprietary services without contributing improvements
back to the community.
