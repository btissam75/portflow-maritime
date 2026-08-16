#!/bin/sh
set -eu

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
: "${MLFLOW_ALLOWED_HOSTS:?MLFLOW_ALLOWED_HOSTS is required}"

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@timescaledb:5432/mlflow" \
  --artifacts-destination "s3://mlflow-artifacts" \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}"
