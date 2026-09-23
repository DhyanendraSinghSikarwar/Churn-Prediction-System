.PHONY: setup db-up db-down migrate ingest train score serve test lint format clean

# Note: `make` is not preinstalled on Windows. If it's unavailable, run the
# body of the relevant target directly (each is one or two commands) —
# see README.md "Quickstart" for the equivalent uv/docker commands.

setup:
	uv sync
	uv run pre-commit install

db-up:
	docker compose up -d postgres pgadmin minio minio-init mlflow
	uv run alembic upgrade head

db-down:
	docker compose down

migrate:
	uv run alembic upgrade head

ingest:
	uv run python -m churn.ingestion.cli

train:
	uv run python pipelines/train.py

score:
	@echo "Phase 6 not yet built: pipelines/batch_score.py"

serve:
	@echo "Phase 6 not yet built: api/"

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run black --check src tests
	uv run mypy src

format:
	uv run ruff check --fix src tests
	uv run black src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
