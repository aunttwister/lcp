# smallm

**The LLM gateway for people who run their own shit.** Route, meter, control — one binary, one port, no cloud dependency.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com/)

---

## What is smallm?

smallm is a **self-hosted LLM gateway** that sits between your agents and LLM providers. It
routes requests, enforces tool permissions, tracks costs, and manages API keys — all from a
single Docker container backed by SQLite. No PostgreSQL. No Redis. No SaaS dashboard phoning home.

It was built for a real homelab running [Hermes Agent](https://github.com/NousResearch/hermes-agent)
in production, where one person manages multiple AI agents and needs to know exactly what's being
spent, by who, and with what tools.

```
Your agents (Hermes, scripts, tools)
          │
          ▼
┌─────────────────────────────────────┐
│           smallm (:8734)            │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  Auth → Strip Tools → Route  │   │
│  │  → Circuit Breaker → Track $ │   │
│  └──────────────────────────────┘   │
│                                     │
│  Dashboard    API Keys    Budgets   │
│  (:8734/)     Alerts     Export     │
└─────────────┬───────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 DeepSeek  OpenCode   (more)
```

## Features

### 🔀 Intelligent routing
- **Provider chains** with automatic fallback — opencode dies, DeepSeek takes over
- **Circuit breaker** — failing providers are degraded, dead providers are skipped
- **Profile-based routing** by URL path: `/l2`, `/l1`, `/career`, `/coder`
- **SSE streaming passthrough** — real-time token delivery, no buffering

### 🛡️ Tool permission enforcement
- **Strip dangerous tools per profile** — L1 can't `terminal`, cron can't anything
- Configurable blocklists per profile via YAML
- [Phase 7 roadmap]: Fails-closed permission matrix with cross-cutting `blocked_globally` rules

### 💰 Cost tracking
- **Per-request cost** with DeepSeek cache hit/miss breakdown
- **Daily cost summaries** per profile, per model, per provider
- **Prompt prefix caching** — normalizes messages so repeated system prompts hit cache
- Realistic pricing: cache hits are **120x cheaper** and smallm tracks both

### 🔑 API key management
- **Create/rotate/revoke** virtual API keys through the dashboard
- **Per-key spend limits** with hard stops
- **Per-profile access control** — a key can be scoped to specific profiles
- **Usage per key** — see exactly which key is costing money

### 📊 Dashboard
- **Server-rendered HTML** — no SPA, no build step, loads instantly
- **Daily cost charts** with Chart.js — stacked bars, per-profile views
- **Provider health** with live status dots
- **Drag-and-drop chain editing** — reorder fallback providers in UI
- **Request log** with error inspection
- **Budget alerts** with configurable thresholds (50%, 80%, 90%, 100%)

### 🧩 Plugin architecture (in progress)
- Provider cost extraction plugins — DeepSeek, OpenCode, OpenAI
- Memory plugin with [LanceDB](https://github.com/lancedb/lancedb) backend (embedded vector DB)
- [features/memory.md](features/memory.md) — unifying memory across devices

## Quick Start

```bash
# Clone
git clone https://github.com/aunttwister/smallm-gw.git
cd smallm-gw

# Configure providers — add your API keys
cp config/.env.example config/.env
# Edit config/.env: DEEPSEEK_API_KEY=sk-..., OPENCODE_API_KEY=...

# Run
docker compose up -d
```

smallm is now running at `http://localhost:8734`. Open the dashboard, send a request:

```bash
curl http://localhost:8734/l2/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Configuration

Everything lives in `config/gateway.yaml`. Hot-reloaded — edit while running, no restart needed.

```yaml
profiles:
  l2:
    forbidden_tools: [write_file, patch, cronjob]   # strip these tools
    chain:
      - provider: opencode
        model: deepseek-v4-pro
      - provider: deepseek                          # fallback
        model: deepseek-v4-pro

  cron:
    forbidden_tools: null                           # null = strip ALL tools
    chain:
      - provider: deepseek
        model: deepseek-v4-flash

providers:
  deepseek:
    api_key_env: DEEPSEEK_API_KEY                   # reads from env
    cache:
      strategy: prefix
      savings: cost
      hit_field: prompt_cache_hit_tokens

pricing:
  - provider: deepseek
    model: deepseek-v4-pro
    cache_hit: 0.003625          # $ per 1M tokens
    cache_miss: 0.435
    output: 0.87
```

## Why smallm over alternatives?

| | smallm | LiteLLM | llmgateway |
|---|---|---|---|
| **Tool permission enforcement** | ✅ Yes — per profile | ❌ No | ❌ No |
| **Deployment** | Single Docker, SQLite | Docker + PostgreSQL + Redis | Docker |
| **Dashboard** | Server-rendered HTML, no build | React SPA | Next.js SPA |
| **Source available** | ✅ MIT, all features | Puppet villain tier💰 | Source-available, enterprise tears |
| **Memory footprint** | ~100MB + model | 500MB+ service stack | Node, need I say more |
| **Agent-native** | ✅ Built for Hermes agents | Generic API proxy | Generic API proxy |
| **API key management** | ✅ Virtual keys, spend limits | ✅ Virtual keys | ❓ |

smallm is the only LLM gateway that understands **agent tool permissions**. No other open-source
gateway strips tools based on who's calling. If you're running AI agents in production, you need
that.

## Architecture

```
smallm (:8734) — single binary, single port
│
├── Python stdlib (http.server) — no framework overhead
├── SQLite (costs.db) — zero-infra persistence
├── YAML config — human-readable, hot-reloadable
├── Server-rendered dashboard — Chart.js, no SPA
├── Plugin architecture — provider costs, memory backends
└── Docker — FROM python:3.11-slim
```

**What's deliberately NOT here:** PostgreSQL, Redis, Next.js, pnpm, Kubernetes, or anything from
the TypeScript ecosystem. SQLite handles millions of cost rows. One binary, one port, one
`docker compose up`.

## API

| Endpoint | Description |
|---|---|
| `POST /{profile}/v1/chat/completions` | OpenAI-compatible chat completions |
| `GET /health` | Provider health + circuit breaker status |
| `GET /v1/models` | Available models across all providers |
| `GET /` | Dashboard |
| `GET /api/daily-costs` | JSON cost data |
| `POST /api/keys` | Create API key |
| `GET /api/keys` | List API keys |

[Full API reference →](PLAN.md)

## Roadmap

See [PLAN.md](PLAN.md) for phased roadmap. Current status:

| Phase | Status |
|---|---|
| 1–4 | ✅ Shipped — routing, tool stripping, cost tracking, config YAML |
| 5 | 🚧 In progress — multi-tenant users, teams, credit limits (schema done) |
| 6 | 🚧 In progress — dashboard upgrade, time series, per-user utilization |
| 7 | 📋 Planned — fails-closed permission matrix, rate limiting, audit log |
| Memory plugin | 📋 [Spec ready](features/memory.md) — embedded LanceDB, unified memory endpoint |

## Status

smallm runs **24/7 in production** routing all LLM traffic for a multi-agent homelab setup
(6 Hermes profiles, 15+ custom skills, daily cron jobs). It handles ~200 requests/day across
4 profiles with real cost tracking.

**It works.** The rough edges are being sanded down. Phase 5 (multi-tenant auth) is the next
milestone before calling it 1.0.

## AI Attribution

> This project was built with significant assistance from AI coding agents (Hermes Agent with
> DeepSeek V4). Architecture decisions, code, tests, and documentation were all produced in
> collaboration between a human operator and AI. The bugs are mine, the architecture is ours.

## License

MIT — do whatever you want. If you use it, I'd love to hear about it.
