#!/usr/bin/env bash
# Runs once, on first container init (docker-entrypoint-initdb.d
# convention), to create a second database + role for MLflow's backend
# store on the same Postgres instance — see DECISIONS.md (Phase 0,
# tech-stack: "MLflow backend store").
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER ${MLFLOW_BACKEND_DB_USER} WITH PASSWORD '${MLFLOW_BACKEND_DB_PASSWORD}';
    CREATE DATABASE ${MLFLOW_BACKEND_DB_NAME} OWNER ${MLFLOW_BACKEND_DB_USER};
EOSQL
