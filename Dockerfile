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

# ── LiveBench checkout (benchmark runner shells out to run_livebench.py) ─────
# Core LiveBench deps only (no PyTorch, no TensorFlow): openai/anthropic clients,
# pandas, numpy, sympy, nltk, spacy, datasets, litellm, etc. This is enough to
# generate answers and grade every category EXCEPT 'coding', which additionally
# needs code_runner/requirements_eval.txt (TensorFlow + the scientific stack) to
# execute generated code. `git` is only required at build time and is removed.
ENV LCP_LIVEBENCH_DIR=/opt/livebench
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && git clone --depth 1 https://github.com/LiveBench/LiveBench.git /opt/livebench \
    && cd /opt/livebench \
    && pip install --no-cache-dir -e . \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ── Application code ──
COPY src/ ./src/
COPY config/ ./config/
COPY alembic.ini .
COPY alembic/ ./alembic/

RUN mkdir -p /app/data

EXPOSE 8734

CMD ["sh", "-c", "cd /app && python3 -m alembic upgrade head && python3 -m src.main"]
