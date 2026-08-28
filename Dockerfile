# ── Runtime image ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# ── git — required by the in-UI runtime install of LiveBench ────────────────
# (the "benchmark plugin" clones its checkout at runtime when not baked in)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

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
#
# Checkouts live under LCP_MODULES_DIR (default /opt/lcp-modules). The runtime
# installer clones to $LCP_MODULES_DIR/livebench; set LCP_MODULES_DIR to a
# Docker volume mount so installs survive container recreation.
ARG WITH_BENCH=0
ENV LCP_MODULES_DIR=/opt/lcp-modules
RUN if [ "$WITH_BENCH" = "1" ]; then \
        mkdir -p "${LCP_MODULES_DIR}" \
        && git clone --depth 1 https://github.com/LiveBench/LiveBench.git "${LCP_MODULES_DIR}/livebench" \
        && cd "${LCP_MODULES_DIR}/livebench" \
        && pip install --no-cache-dir -e . \
        && pip install --no-cache-dir -r code_runner/requirements_eval.txt ; \
    fi

# ── Semantic routing + memory (baked GLOBALLY — in-process modules) ──────────
# These two modules run IN-PROCESS (the classifier and the memory backend), so
# their deps MUST live in the image's site-packages — NOT pip --target'd into
# the bind-mounted /opt/lcp-modules (PYTHONPATH shadowing there caused the
# tokenizers version conflict). LiveBench is the ONLY module that stays on the
# bind mount: it runs in subprocesses that inject their own PYTHONPATH.
#
# sentence-transformers + tokenizers are shared by BOTH modules; lancedb is
# memory-only. The SAME embedding model (bge-small) serves both, so the weights
# are cached ONCE under /app/models/embedding (not bind-mounted).
#
# Set WITH_ROUTER=0 / WITH_MEMORY=0 for a lean image (the module then degrades
# until installed at runtime from the Setup page).
ARG WITH_ROUTER=1
ARG WITH_MEMORY=1
RUN if [ "$WITH_ROUTER" = "1" ] || [ "$WITH_MEMORY" = "1" ]; then \
        pip install --no-cache-dir sentence-transformers "tokenizers==0.22.2" ; \
    fi \
    && if [ "$WITH_MEMORY" = "1" ]; then \
        pip install --no-cache-dir lancedb ; \
    fi \
    && if [ "$WITH_ROUTER" = "1" ]; then \
        mkdir -p /app/models/embedding \
        && python3 -c \
            "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='/app/models/embedding')" ; \
    fi

# ── Application code ──
COPY src/ ./src/
COPY config/ ./config/
COPY alembic.ini .
COPY alembic/ ./alembic/
# Dev/ops helper scripts (judge_routing, probe_intent, seed_data, ...) so they
# can be run inside the container: docker compose exec lcp python3 scripts/...
COPY scripts/ ./scripts/

RUN mkdir -p /app/data

EXPOSE 8734

CMD ["sh", "-c", "cd /app && python3 -m alembic upgrade head && python3 -m src.main"]
