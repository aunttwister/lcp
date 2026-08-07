# ────────────────────────────────────────────────────────────────────────────
# Stage 1: test — runs the full suite; the build FAILS if any test fails.
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS test

WORKDIR /app

# Dependencies (layer-cached: only reinstalls when pyproject.toml changes)
COPY pyproject.toml .
COPY src/__init__.py ./src/__init__.py
RUN pip install --no-cache-dir .[dev]

# Pre-download tiktoken BPE vocabulary so tests never hit the CDN mid-run
ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken_cache
RUN mkdir -p /app/data/tiktoken_cache && \
    python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Application + tests
COPY src/ ./src/
COPY tests/ ./tests/

# Fail the build if the test suite is not green
RUN python3 -m pytest -q

# ────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — slim final image, tests/dev deps NOT included.
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# ── Dependencies (layer-cached: only reinstalls when pyproject.toml changes) ──
COPY pyproject.toml .
COPY src/__init__.py ./src/__init__.py
RUN pip install --no-cache-dir .

# ── Pre-download tiktoken BPE vocabulary so it never hits the CDN at runtime ──
ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken_cache
RUN mkdir -p /app/data/tiktoken_cache && \
    python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ── Application code ──
COPY src/ ./src/
COPY config/ ./config/
COPY alembic.ini .
COPY alembic/ ./alembic/

RUN mkdir -p /app/data

EXPOSE 8734

CMD ["sh", "-c", "cd /app && python3 -m alembic upgrade head && python3 -m src.main"]
