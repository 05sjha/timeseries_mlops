# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Demand Forecasting Model Server
# ─────────────────────────────────────────────────────────────────────────────
#
# MULTI-STAGE BUILD — the most important Docker concept for MLOps.
#
# WHY MULTI-STAGE?
#   A naive Dockerfile copies everything in and ends up with a 2–4 GB image
#   containing compilers, test tools, build artifacts, and source code.
#   Multi-stage builds separate WHAT YOU NEED TO BUILD from WHAT YOU NEED TO RUN.
#
#   Stage 1 (builder): installs build tools, compiles dependencies
#   Stage 2 (runtime): copies only the compiled output — no build tools
#
#   Result: production image is 300–500 MB instead of 2+ GB.
#   Smaller image = faster pull times, smaller attack surface, lower egress cost.
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Use the full Python image — we need pip, compilers, and dev headers
FROM python:3.11-slim AS builder

# Set working directory inside the container
WORKDIR /build

# LAYER CACHING STRATEGY:
# Copy dependency files FIRST (before source code).
# Docker caches each layer. If only your code changes (not dependencies),
# the pip install layer is reused — cutting rebuild time from minutes to seconds.
# copy dependency declaration first
COPY pyproject.toml .

# install dependencies (this layer gets cached)
RUN pip install --no-cache-dir --prefix=/install \
    "fastapi>=0.110" \
    "uvicorn>=0.29" \
    "pandas>=2.0" \
    "numpy>=1.25" \
    "scikit-learn>=1.3" \
    "xgboost>=2.0" \
    "mlflow>=2.10" \
    "pydantic>=2.0" \
    "pyyaml>=6.0" \
    "google-cloud-storage>=2.14"

# copy source code last
COPY src/ src/

# install the local package
# RUN pip install --no-cache-dir --prefix=/install --no-deps -e .


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Minimal base image — no pip, no compilers, no unnecessary packages
FROM python:3.11-slim AS runtime

# Security: run as non-root user
# Running as root inside a container is a security risk —
# if the process is compromised, the attacker has root access to the host.
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy ONLY the installed packages from the builder stage
# The build tools (pip, gcc, etc.) are NOT copied — they stay in Stage 1
COPY --from=builder /install /usr/local

# Copy application source
COPY --from=builder /build/src ./src

# add this line below the src copy
ENV PYTHONPATH=/app/src

# Copy config
COPY configs/ configs/

# Environment variables (non-secret configuration)
# Secrets (API keys, passwords) MUST come from environment at runtime,
# never baked into the image — they'd be visible in docker history
ENV PYTHONUNBUFFERED=1 \        
    PYTHONDONTWRITEBYTECODE=1 \ 
    MODEL_PATH=/app/model \     
    PORT=8080

# Create model directory for mounted model artifacts
RUN mkdir -p /app/model && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port Vertex AI / Cloud Run expect
EXPOSE 8080

# Health check — container orchestrators (GKE, Cloud Run) ping this
# to decide if the container is healthy and ready to receive traffic.
# Without this, a crashed server still receives traffic until timeout.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Use exec form (not shell form) — signals reach the process directly.
# Shell form: CMD "uvicorn ..." → PID 1 is /bin/sh, uvicorn is PID 2.
#   SIGTERM goes to sh which may not forward it → graceful shutdown fails.
# Exec form: CMD ["uvicorn", ...] → PID 1 is uvicorn → SIGTERM received directly.
CMD ["uvicorn", "demand_forecast.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2"]
