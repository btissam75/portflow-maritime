#!/bin/sh
set -eu

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

until pg_isready -h timescaledb -p 5432 -U "${POSTGRES_USER}" -d postgres; do
  sleep 2
done

exists="$(psql -h timescaledb -U "${POSTGRES_USER}" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = 'airflow'")"

if [ "${exists}" != "1" ]; then
  psql -h timescaledb -U "${POSTGRES_USER}" -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE airflow"
  echo "Created database: airflow"
else
  echo "Database already exists: airflow"
fi

psql -h timescaledb -U "${POSTGRES_USER}" -d postgres -v ON_ERROR_STOP=1 -c "ALTER DATABASE airflow SET timezone TO 'UTC'"
