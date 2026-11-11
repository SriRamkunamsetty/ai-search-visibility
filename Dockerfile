# syntax=docker/dockerfile:1.4
# ==============================================================================
# Stage 1: Build & Packaging
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels .

# ==============================================================================
# Stage 2: Hardened, Non-Root Production Runtime
# ==============================================================================
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="ai-search-visibility" \
      org.opencontainers.image.description="AI assistant visibility audit and evaluation engine" \
      org.opencontainers.image.authors="SriRamKunamsetty <mohansriramkunamsetty@gmail.com>" \
      org.opencontainers.image.source="https://github.com/SriRamkunamsetty/ai-search-visibility" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Create unprivileged service user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

# Install built wheel from builder
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/* && rm -rf /tmp/wheels

COPY src/ ./src/
COPY docs/ ./docs/
COPY examples/ ./examples/

RUN chown -R appuser:appgroup /app

USER appuser:appgroup

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + str(os.environ.get('PORT', 8000)) + '/livez')" || exit 1

ENTRYPOINT ["aivisibility"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
