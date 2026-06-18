# ── VeriTune Production Dockerfile ────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (builder) — install Python deps into a venv
#   Stage 2 (runtime) — lean image with just the venv + app code
#
# Build:  docker build -t veritune:latest .
# Run:    docker run -p 8000:8000 --env-file .env veritune:latest

FROM python:3.10-slim AS builder

WORKDIR /build

# System deps needed for bitsandbytes / torch compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 veritune
RUN mkdir -p /app/outputs /app/data && chown -R veritune:veritune /app

# Copy application source
COPY --chown=veritune:veritune . .

USER veritune

# Environment defaults (override via --env or .env file)
ENV PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=2 \
    LOG_LEVEL=INFO \
    DEVICE=cpu \
    ROUTER_PATH=/app/outputs/router \
    REGISTRY_PATH=/app/outputs/checkpoints/checkpoint_registry.json \
    WANDB_DISABLED=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", \
     "uvicorn serving.main:app \
      --host ${HOST} \
      --port ${PORT} \
      --workers ${WORKERS} \
      --log-level ${LOG_LEVEL} \
      --loop asyncio \
      --timeout-keep-alive 30"]
