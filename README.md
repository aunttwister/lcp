# LLM Control Plane (LCP)

**A self-hosted LLM gateway. Route, meter, control — one container, one port, no cloud dependency.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-758%20passed-brightgreen.svg)]()

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

LCP (LLM Control Plane) is a self-hosted LLM gateway that sits between your agents and LLM providers. It routes
requests, enforces tool permissions, tracks costs, and manages API keys — all from a single Docker
container backed by SQLite. No PostgreSQL. No Redis. No external services.

It was built for a production [Hermes Agent](https://github.com/NousResearch/hermes-agent)
deployment managing multiple AI agents across a homelab infrastructure, where knowing exactly
what is being spent, by whom, and with what tools is critical.

```
Your agents (Hermes, scripts, external tools)
          |
          v
+--------------------------------------------+
|             LCP (:8734)                   |
|                                            |
|  Auth -> Strip Tools -> Route              |
|       -> Circuit Breaker -> Track Cost     |
|                                            |
|  Dashboard     API Keys     Budgets        |
|  (:8734/)      Alerts      Export          |
+-------------------+------------------------+
                    |
        +-----------+-----------+
        v           v           v
    DeepSeek    OpenCode    (more)
```

## Features

### Intelligent routing
- Provider chains with automatic fallback — if one provider fails, the next in chain takes over
- Circuit breaker with configurable thresholds — degraded providers are probed, dead providers are skipped
- Profile-based routing by URL path: `/l2`, `/l1`, `/career`, `/coder`
- SSE streaming passthrough — real-time token delivery, no buffering

### Tool permission enforcement
- Strip dangerous tools per profile — an L1 triage agent cannot `terminal`, a cron profile strips everything
- Configurable blocklists per profile via YAML
- Roadmap: fails-closed permission matrix with cross-cutting `blocked_globally` rules

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
# Edit config/.env: DEEPSEEK_API_KEY=sk-..., OPENCODE_API_KEY=...

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
    api_key_env: DEEPSEEK_API_KEY
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
| **License** | MIT, all features included | MIT core, enterprise tier | Source-available |
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
| **`sqlalchemy`** | SQLite ORM for the `requests`, `budgets`, `alerts`, `api_keys`, `teams`, `users`, and `audit_logs` tables. All cost history, spend limits, and alert state lives here. No external database. |
| **`alembic`** | Database schema migrations. Every schema change (alerts table, API keys, error_detail column) gets a numbered migration in `alembic/versions/`. Run automatically on container startup via `alembic upgrade head`. |
| **`pyyaml`** | Reads and writes `config/gateway.yaml`. The config is hot-reloaded — edit providers, profiles, or pricing while the server is running and changes take effect on the next request. |
| **`tiktoken`** | Exact BPE token counts using the `cl100k_base` encoding (same tokenizer used by DeepSeek and OpenAI models). Powers the pre-request `X-Estimated-Cost` header and the dynamic flash/pro router. The ~1 MB vocabulary file is pre-downloaded at Docker build time and persisted to a volume — zero CDN dependency at runtime. |
| **`jinja2`** | Server-rendered HTML templates for the dashboard, profiles page, providers, API keys, alerts, and logs. No SPA, no build step, no npm — pages load instantly from the server. Shared partials for sidebar, modals, and JS utilities. |

Dev-only dependencies (`pip install .[dev]`):

| Package | Role |
|---|---|
| `pytest` | Test runner — 758 unit tests covering routing, budgets, alerts, cost estimation, auth enforcement, and the plugin system |
| `pytest-cov` | Coverage reports — `pytest --cov=src --cov-report=term-missing` |
| `pytest-mock` | Mocking utilities for the `unittest.mock` patch system |

## Test Coverage

**95% overall** — 3,228 of 3,410 statements covered (758 tests, 0 integration tests).

Run: `.venv/bin/python -m pytest --cov=src --cov-report=term-missing -q`

| Module | Coverage |
|---|---|
| `src/api/models.py` | 100% |
| `src/api/circuit_breaker.py` | 100% |
| `src/api/cost_estimator.py` | 100% |
| `src/api/cost_plugins/opencode_api.py` | 100% |
| `src/api/cost_plugins/base.py` | 100% |
| `src/api/exceptions.py` | 100% |
| `src/api/logging_config.py` | 100% |
| `src/api/prompt_cache.py` | 100% |
| `src/api/router.py` | 100% |
| `src/server/server.py` | 100% |
| `src/server/sse_helpers.py` | 100% |
| `src/ui/dashboard.py` | 100% |
| `src/ui/pages.py` | 100% |
| `src/ui/render.py` | 100% |
| `src/api/request_pipeline.py` | 99% |
| `src/api/alert_manager.py` | 99% |
| `src/api/config.py` | 98% |
| `src/api/key_manager.py` | 98% |
| `src/main.py` | 98% |
| `src/api/token_verifier.py` | 97% |
| `src/api/cost_plugins/deepseek.py` | 96% |
| `src/api/cost_plugins/llamacpp.py` | 96% |
| `src/api/cost_plugins/opencode.py` | 96% |
| `src/server/handler.py` | 91% |
| `src/server/endpoints.py` | 87% |

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

| Phase | Status |
|---|---|
| 1 through 4 | Shipped — routing, tool stripping, cost tracking, YAML config |
| 5 | In progress — multi-tenant users, teams, credit limits (schema complete) |
| 6 | In progress — dashboard upgrade, time series, per-user utilization |
| 7 | Planned — fails-closed permission matrix, rate limiting, audit log |
| Memory plugin | [Specification complete](features/memory.md) — embedded LanceDB, unified memory endpoint |

Full details in [PLAN.md](PLAN.md).

## Status

LCP runs continuously in production, routing all LLM traffic for a multi-agent homelab setup
(6 Hermes profiles, 15+ custom skills, daily cron jobs). It handles approximately 200 requests
per day across 4 profiles with real cost tracking. Phase 5 (multi-tenant authentication and
authorization) is the next milestone toward a 1.0 release.

## AI Attribution

This project was built with significant assistance from AI coding agents (Hermes Agent with
DeepSeek V4). Architecture decisions, code, tests, and documentation were produced in
collaboration between a human operator and AI. The bugs are mine; the architecture is ours.

## License

MIT — use it however you like. If you build something with it, I would enjoy hearing about it.
