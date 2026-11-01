# Multi-stage Dockerfile for Modular RAG Evaluation Framework

# --- Stage 1: Base ---
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md* ./
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# --- Stage 2: Development ---
FROM base AS development

# Development workspace mounts code at runtime
ENV PYTHONPATH="/app/src"

CMD ["pytest", "-v"]

# --- Stage 3: Production ---
FROM base AS production

# Create non-root user
RUN useradd -m -u 1000 appuser

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY docs/ /app/docs/

RUN chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["bench-rag"]
CMD ["--help"]
