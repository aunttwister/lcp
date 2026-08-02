FROM python:3.11-alpine

WORKDIR /app

# Copy dependency manifest first (for layer caching)
COPY pyproject.toml .

# Install runtime deps from pyproject.toml, then clean up build tools
COPY src/__init__.py ./src/__init__.py
RUN apk add --no-cache gcc musl-dev && \
    pip install --no-cache-dir . && \
    apk del gcc musl-dev

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY alembic.ini .
COPY alembic/ ./alembic/

# Data directory
RUN mkdir -p /app/data

EXPOSE 8734

# Run migrations then start server
CMD ["sh", "-c", "cd /app && python3 -m alembic upgrade head && python3 -m src.main"]
