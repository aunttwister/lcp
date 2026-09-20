# Changelog

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Shared log-table module** — one implementation of "a table of log rows",
  used by all four Logs tabs (conversations, requests, provider decisions, board
  decisions):
  - `src/ui/tables.py` — the server half. Owns the page window and the ORDER BY,
    exposes an **allow-list of named sort keys** (`newest` / `oldest`, plus
    `asked` for the ledger) instead of interpolating `?sort=` into SQL, and
    clamps `?page=` to the real range. The default sort key on every tab is
    `newest`, so logs are newest-first unless asked otherwise.
  - `static/js/logtable.js` — the client half. Fetches the JSON view and renders
    the rows, the sort + page-size + filter controls, the pager and the
    "Showing X–Y of N" line from the server's `filter` payload, keeping state in
    the URL. Conversations expand their timeline on demand (one request per
    expansion) instead of inlining 50 events for every row on the page.

### Fixed
- **Logs: two of four tabs were silently capped at 20 rows.** The requests and
  provider-decisions tabs paginated in the data layer but never rendered a pager,
  so rows 21+ were unreachable — 54,244 requests showed 20. Both now page.
- **Logs: the board-decisions tab dumped the entire ledger** into the page (the
  funnel is still computed over the whole ledger, never over one page).
- **Logs: the Requests tab's conversation link did nothing** — it linked to
  `?qid=<id>`, which no view read. `conversations_view` now honours `qid`.
- Page-size `all` is no longer offered as a button on the log tabs (the API
  still honours it) — it was one click away from a 15 MB page.

## [0.5.0] — 2026-08-22

This release is the first public `main` release — the culmination of the
dynamic-routing program plus the provider-health and cost-visibility work.

### Added
- **Benchmark-driven dynamic routing** (runtime-enabled via a UI toggle):
  - Task classifier (`code_generation`, `debugging`, `unit_tests`, `planning`,
    `reasoning_chain`, `agentic_multi_step`, `research_deep`, `casual_chat`).
  - Capability scoring from LiveBench / manual / gateway-YAML sources, with a
    per-task cost-bias boost, circuit-breaker health bonus, and credit penalty.
  - Routing policies: `eager` (deterministic), `cost_first`, `explore`
    (weighted), plus a `min_score` floor.
  - UI-defined rules (`prefer` / `block` / `policy`), including model-only rules
    (provider `*` wildcard).
  - Providers → Routing tab: enable toggle, policy editor, rules editor,
    per-task recommendations, recent-decisions log.
- **Persistent provider health**: circuit-breaker state (status, failure counts,
  cooldowns, manual overrides) survives restarts via a new `provider_health`
  table (migration 019). Health tab groups providers per profile with
  expandable stacks.
- **OpenCode billing tracking**: scrape available credits from the billing page
  (8-decimal fixed-point balance), shown in the sidebar, Usage page, and the
  global header widget on every page.
- **Cost-cache background refresher** with per-provider TTLs, retry/backoff,
  and stale-serving; Providers → Cache tab.
- **Setup wizard** rework: shared modal for provider installs, reinstall
  support, remove-only UI for installed steps.

### Changed
- Sidebar: collapsible Usage submenu (animated), circular chevron, two-row
  provider details (credits + monthly %).
- Providers page: profile-grouped health, compact override selects, tab hash
  persistence (`#health` / `#config` / `#cache` / `#routing`).
- Logs page: `/api/logs` now returns `error_detail` and shows the reason inline
  (truncated) + hover tooltip.
- Profiles page: Add Profile uses the shared modal (no more `prompt()`); row
  actions spaced and renamed.
- Usage page: fixed overflow of the By Model / By Profile cards.

### Fixed
- OpenCode credits parser: `balance` is 8-decimal fixed point (e.g.
  `949260397` → `$9.49`), not raw dollars.
- Rule engine: `*` / `''` treated as wildcards so model-only rules work.
- Routing toggle and per-task recommendations no longer require a restart.

## [0.4.0] — earlier
- Cost-cache + settings page; per-provider refresh TTLs; cache tab.
- Provider plugins for DeepSeek, OpenCode, Command Code, llama.cpp.
- Benchmark import (LiveBench) + model registry + capability matrix.

[0.5.0]: https://github.com/aunttwister/lcp/releases/tag/v0.5.0
[0.4.0]: https://github.com/aunttwister/lcp/releases/tag/v0.4.0
