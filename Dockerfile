# ── Runtime image ───────────────────────────────────────────────────────────
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

# ── LiveBench (OPTIONAL — benchmark "plugin") ────────────────────────────────
# The in-UI benchmark runner is opt-in. Set the build arg WITH_BENCH=1 to bake
# a LiveBench checkout into the image; the default (WITH_BENCH=0) keeps the
# gateway image lean and the benchmark runner reports "unavailable".
#
#   docker compose build --build-arg WITH_BENCH=1 lcp
#
# Full LiveBench (core + code_runner/requirements_eval.txt) covers all 6
# categories including `coding` (which executes generated code and needs
# TensorFlow + the scientific stack). Core-only covers the other 5.
ARG WITH_BENCH=0
RUN if [ "$WITH_BENCH" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends git ca-certificates \
        && git clone --depth 1 https://github.com/LiveBench/LiveBench.git /opt/livebench \
        && cd /opt/livebench \
        && pip install --no-cache-dir -e . \
        && pip install --no-cache-dir -r code_runner/requirements_eval.txt \
        && apt-get purge -y git \
        && apt-get autoremove -y \
        && rm -rf /var/lib/apt/lists/* ; \
    fi
ENV LCP_LIVEBENCH_DIR=/opt/livebench

# ── Application code ──
COPY src/ ./src/
COPY config/ ./config/
COPY alembic.ini .
COPY alembic/ ./alembic/

RUN mkdir -p /app/data

EXPOSE 8734

CMD ["sh", "-c", "cd /app && python3 -m alembic upgrade head && python3 -m src.main"]
