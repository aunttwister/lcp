# LCP v0.5.0 — Benchmark-driven routing is here

**LCP** is a self-hosted LLM control plane: route, meter, and control every AI
agent from one container — no cloud, no Postgres, no Redis. Just SQLite.

This is the first public release and includes the headline feature:

## ✨ Benchmark-driven dynamic routing
- Grade your models with **LiveBench** (or your own benchmark), and LCP
  classifies **every request by task type** and routes it to the best-fit
  **(provider, model)** — balancing capability score, cost bias,
  circuit-breaker health, and your own policy/rules.
- **Three policies** — `eager` (deterministic), `cost_first`, `explore`
  (weighted A/B) — plus a min-score floor.
- **UI rules** — `prefer` / `block` / `policy` per task/profile, including
  model-only rules.
- Runtime **enable toggle** — no restarts, no YAML edits.
- Task taxonomy: `code_generation`, `unit_tests`, `debugging`, `planning`,
  `reasoning_chain`, `agentic_multi_step`, `research_deep`, `casual_chat`.

## 🔌 Provider health that survives restarts
- Circuit-breaker state (status, failure counts, cooldowns, manual overrides)
  is **persisted** — no more resurrecting dead providers after a redeploy.
- Health tab groups providers **per profile** with expandable stacks.

## 💰 Cost visibility
- **OpenCode credits** scraped from the billing page ($9.49, not $949M — yes,
  that was a fixed-point bug we caught).
- Background **cost-cache refresher** with per-provider TTLs, retry/backoff,
  and stale-serving.
- Provider plugins: DeepSeek, OpenCode, Command Code, llama.cpp.
- Global header widget (every page) + sidebar Usage submenu with live credits.

## 🖥️ UI polish
- Collapsible animated sidebar, modal-based Add Profile, tab deep-links
  (`#routing`, `#health`, …), Logs page now shows the actual error reason.

## 🚀 Try it
```bash
git clone https://github.com/aunttwister/lcp.git
cd lcp && cp config/gateway.example.yaml config/gateway.yaml
docker compose up -d --build
```
Open `http://localhost:8734`.

## Stats
- **1473 tests passing**, ~94% coverage, zero external services.
- AGPL-3.0, everything included — no enterprise tier.

[Readme](https://github.com/aunttwister/lcp) · [Changelog](CHANGELOG.md) · [Issues](https://github.com/aunttwister/lcp/issues)
