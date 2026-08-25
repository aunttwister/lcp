# Contributing to LCP

Thanks for your interest! LCP is a self-hosted LLM gateway. This guide covers
how to set up a dev environment, run tests, and open a good PR.

## Development setup

```bash
# Clone
git clone https://github.com/aunttwister/lcp.git
cd lcp

# Create + activate the venv
python -m venv .venv
source .venv/bin/activate          # (Windows: .venv\Scripts\activate)

# Install with dev dependencies
pip install -e '.[dev]'
```

## Running tests

```bash
# Full suite (804 tests)
pytest

# With coverage
pytest --cov=src --cov-report=term-missing -q

# A single file
pytest tests/test_router.py -q
```

Keep the suite green and add tests for new behaviour. Target the existing
coverage level (~94%) — if you add a feature, add tests that exercise it.

## Project layout

- `src/main.py` — entry point (boots config, DB, plugins, server)
- `src/api/` — core: config, router, cost estimation, prompt cache, key
  manager, alert manager, circuit breaker, credential store, plugins
- `src/server/` — HTTP layer (`http.server`-based, no framework)
- `src/ui/` — server-rendered dashboard (Jinja2 templates + static assets)
- `src/api/cost_plugins/` — provider cost/balance/subscription plugins
- `tests/` — pytest suite (unit + in-process handler tests)

## Commit style

We use [conventional commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `ui:` — dashboard / front-end change
- `docs:` — documentation
- `chore:` — housekeeping (no behaviour change)
- `refactor:` — behaviour-preserving code change
- `ci:` — CI / tooling
- `test:` — tests

Example: `fix: include upstream error body in 401/403 auth failures`

## Opening a PR

1. Branch from `main` (e.g. `feat/rate-limiter`).
2. Make small, focused changes with clear commit messages.
3. Run `pytest` locally — everything must pass.
4. Update `README.md` if user-facing behaviour or endpoints change.
5. Open the PR against `main` with a short description of what/why.

## Style

- Python 3.11+, stdlib-first where reasonable.
- Follow the existing structure: config is DB-backed (seeded from
  `src/api/config.py` → `SEED_CONFIG`), providers are added via the dashboard
  (encrypted credential store), not new env vars or a YAML file.
- Log with `structlog` (structured key/value pairs) rather than `print`.

## Reporting issues / security

- Feature requests and bugs → GitHub Issues (see templates).
- Security vulnerabilities → see [SECURITY.md](SECURITY.md). Do **not** file
  them as public issues.
