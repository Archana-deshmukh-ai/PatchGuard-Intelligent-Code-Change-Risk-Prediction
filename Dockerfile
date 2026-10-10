# syntax=docker/dockerfile:1
# =============================================================================
# PatchGuard: Intelligent Code Change Risk Prediction
# Multi-Stage Production Dockerfile for REST API and Developer Dashboard
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Dependency Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install build dependencies for compiling binary extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for reproducible isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install dependencies into virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    PATCHGUARD_HOST="0.0.0.0" \
    PATCHGUARD_PORT=8000 \
    PATCHGUARD_REPOS_ROOT="/repositories" \
    PATCHGUARD_MODELS_DIR="/app/models"

# Install Git (commit mining & diff extraction), curl (healthcheck), and libgomp1 (OpenMP for XGBoost/Scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated unprivileged user and group
RUN groupadd -g 10001 patchguard && \
    useradd -u 10001 -g patchguard -m -s /bin/bash patchguard

# Scope Git safe-directory ownership exceptions narrowly to the authorized repositories mount root
RUN git config --system --add safe.directory "/repositories/*" && \
    git config --system --add safe.directory "/repositories"

# Set up application workspace and mount points
WORKDIR /app
RUN mkdir -p /app/models /repositories /app/reports/experiments/phase8 && \
    chown -R patchguard:patchguard /app /repositories

# Copy virtual environment from builder stage (read-only for non-root user)
COPY --from=builder /opt/venv /opt/venv

# Copy application source code, models, and metadata
COPY --chown=patchguard:patchguard src/ /app/src/
COPY --chown=patchguard:patchguard models/ /app/models/
COPY --chown=patchguard:patchguard reports/experiments/phase8/ /app/reports/experiments/phase8/
COPY --chown=patchguard:patchguard requirements.txt /app/

# Switch to unprivileged user
USER patchguard

# Expose REST API (8000) and Streamlit Dashboard (8501)
EXPOSE 8000 8501

# Container liveness health check querying IPv4 loopback
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Default runtime entrypoint: Launch FastAPI REST API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
