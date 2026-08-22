# Launch copy — Show HN + r/selfhosted

These are drafts. Trim/personalize as you like before posting. Both lead with
the homelab story and the benchmark-routing hook — the differentiator.

---

## Show HN (title ≤ 80 chars)

> Show HN: I built a self-hosted LLM gateway that routes by benchmark scores

**Body (markdown):**

I run a homelab with 6 AI agents (Hermes profiles, cron jobs, VS Code Copilot)
and was getting wrecked by API costs with zero visibility into which model was
actually good at which task.

So I built **LCP** — a self-hosted LLM control plane. One container, one port,
SQLite, no cloud, no Postgres, no Redis.

**The thing I'm most proud of: benchmark-driven routing.**
- Grade models with LiveBench (or your own benchmark).
- Every request is classified by task type and routed to the best-fit
  (provider, model) — balancing capability score, cost bias, circuit-breaker
  health, and your own rules.
- Policies: eager (deterministic) / cost_first / explore (weighted A/B).
- All tunable live in the UI — no restarts.

**Also included:**
- Per-agent tool permissions, budgets, and virtual keys
- Circuit-breaker health that **persists across restarts**
- Cost plugins for DeepSeek / OpenCode / Command Code / llama.cpp, incl.
  OpenCode credits scraped from the billing page
- Encrypted provider credentials (no env vars)
- 1473 tests, ~94% coverage, AGPL-3.0 (everything included, no enterprise tier)

Screenshots: dashboard, Providers → Routing tab, Usage.

Try it:
```bash
git clone https://github.com/aunttwister/lcp.git
cd lcp && cp config/gateway.example.yaml config/gateway.yaml
docker compose up -d --build
```

Happy to answer questions — especially about the routing scoring formula or the
billing-page scraper (which involved a fun 8-decimal fixed-point bug: it
initially reported $949,260,397 as your balance instead of $9.49 😅).

---

## r/selfhosted

**Title:** Self-hosted LLM gateway — routes each request to the best model by
benchmark score, one container, no cloud

**Body:**

After running a multi-agent homelab (Hermes agents, cron, Copilot) for a while,
I hit two problems: (1) I couldn't see which model was actually good at which
task, and (2) I was paying for top-tier models for requests a cheaper one could
handle.

I built **LCP** — a self-hosted LLM control plane:

- **Benchmark-driven routing** — LiveBench-grade your models, and LCP routes
  each request to the best (provider, model) by task type, weighted by cost and
  provider health. Tunable in the UI (eager / cost-first / explore + rules).
- **Multi-agent control** — per-profile tool permissions, budgets, virtual keys.
  Run many agents behind one gateway and know who spent what, with what tools.
- **No external services** — one container, SQLite, hot-reloadable YAML. No
  Postgres/Redis.
- **Persistent health** — circuit-breaker state survives restarts (no more
  dead providers coming back to life).
- **Cost plugins** — DeepSeek / OpenCode / Command Code / llama.cpp, with
  OpenCode credits scraped from their billing page.
- Encrypted credentials, server-rendered dashboard (no SPA), 1473 tests.

Try it: `git clone https://github.com/aunttwister/lcp.git && cd lcp && cp
config/gateway.example.yaml config/gateway.yaml && docker compose up -d --build`

Comparison vs LiteLLM/OpenRouter/one-api is in the README. Feedback welcome!
