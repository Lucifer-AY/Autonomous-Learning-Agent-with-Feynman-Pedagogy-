# ============================================
# Deep Research From Scratch - Dockerfile
# Python 3.11 + Node.js + uv package manager
# ============================================

FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000 \
    UV_SYSTEM_PYTHON=1

# Install system dependencies including Node.js (required for MCP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager from official image for reproducible builds
COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uvx /usr/local/bin/uvx

# Set working directory
WORKDIR /app

# Copy dependency manifests first to improve Docker layer caching
COPY pyproject.toml uv.lock langgraph.json ./

# Install dependencies (locked and reproducible)
RUN uv sync --frozen

# Copy project sources
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# Ensure report output directory exists
RUN mkdir -p /app/src/deep_research_from_scratch/files

# Expose port for LangGraph server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Default command - run LangGraph dev server from installed env
CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--port", "8000", "--allow-blocking"]
