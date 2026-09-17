# syntax=docker/dockerfile:1
#
# Multi-stage build for the churn package itself (CLI entry points now;
# the same image is reused as the API base image from Phase 6 onward via
# a CMD override). Non-root runtime user, healthcheck, dependency layer
# cached separately from source so source-only changes don't reinstall
# the world.

FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv==0.11.32

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app
COPY --from=builder /build/.venv /app/.venv
COPY src/ ./src/
COPY sql/ ./sql/
COPY config/ ./config/
COPY alembic.ini ./
COPY alembic/ ./alembic/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import churn" || exit 1

CMD ["python", "-m", "churn.ingestion.cli", "--help"]
