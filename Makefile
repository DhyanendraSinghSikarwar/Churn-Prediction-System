.PHONY: setup db-up db-down migrate ingest train score serve backfill monitor dashboard report flow test lint format clean

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
	uv run python pipelines/batch_score.py

serve:
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

backfill:
	uv run python pipelines/backfill_predictions.py

monitor:
	uv run python pipelines/monitor.py

dashboard:
	uv run streamlit run dashboards/app.py

report:
	uv run python dashboards/report.py

flow:
	uv run python pipelines/flows.py

test:
	uv run pytest

lint:
	uv run ruff check src tests pipelines api dashboards
	uv run black --check src tests pipelines api dashboards
	uv run mypy src pipelines api dashboards

format:
	uv run ruff check --fix src tests pipelines api dashboards
	uv run black src tests pipelines api dashboards

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
