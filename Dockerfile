FROM python:3.11-alpine

WORKDIR /app

# Install build deps, then remove them
RUN apk add --no-cache gcc musl-dev && \
    pip install --no-cache-dir structlog sqlalchemy alembic pyyaml tiktoken jinja2 && \
    apk del gcc musl-dev

# Copy application code
COPY pyproject.toml .
COPY src/ ./src/
COPY config/ ./config/
COPY alembic.ini .
COPY alembic/ ./alembic/

# Data directory
RUN mkdir -p /app/data

EXPOSE 8734

# Run migrations then start server
CMD ["sh", "-c", "cd /app && python3 -m alembic upgrade head && python3 -m src.main"]
