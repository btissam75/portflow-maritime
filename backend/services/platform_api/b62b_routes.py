from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel


MODEL_VERSION = "b62b-vintage-weather-wave-shadow-v1"
AUDIT_SOURCE = "b62b_vintage_forecast_shadow_validation"
PREDICTION_TABLE = "serving.maritime_metocean_vintage_shadow_v1"
METRIC_TABLE = "serving.maritime_metocean_vintage_metric_v1"


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


_pool: SimpleConnectionPool | None = None


def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        dsn = (
            f"host={_env('SMART_PORT_DB_HOST', 'timescaledb')} "
            f"port={_env('SMART_PORT_DB_PORT', '5432')} "
            f"dbname={_env('SMART_PORT_DB_NAME', 'maritime')} "
            f"user={_env('SMART_PORT_DB_USER', 'smartport')} "
            f"password={_env('SMART_PORT_DB_PASSWORD')}"
        )
        _pool = SimpleConnectionPool(1, int(os.getenv("B62B_DB_POOL_SIZE", "4")), dsn=dsn)
    return _pool


@contextmanager
def _connection() -> Iterator[Any]:
    pool = _get_pool()
    connection = pool.getconn()
    try:
        connection.set_session(readonly=True, autocommit=True)
        yield connection
    finally:
        pool.putconn(connection)


class B62BStatus(BaseModel):
    audit_status: str
    decision: str
    model_version: str
    rows: int
    archive_origins: int
    valid_origins: int
    test_origins: int
    fresh_origins: int
    fresh_span_days: float
    selected_model: str | None = None
    reference_model: str | None = None
    valid_accepted: bool
    archive_confirmed: bool
    fresh_confirmed: bool
    critical_gates_passed: bool
    production_promotion_allowed: bool
    limited_pilot_allowed: bool
    automatic_action_allowed: bool
    test_role: str
    next_block: str | None = None
    finished_at: datetime | None = None


class Metric(BaseModel):
    evaluation_role: str
    model: str
    rows: int
    origins: int
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None
    coverage: float | None = None
    mean_interval_width: float | None = None
    quantile_crossings: int


class Prediction(BaseModel):
    evaluation_role: str
    issue_at: datetime
    valid_at: datetime
    horizon_h: int
    variable: str
    model: str
    actual: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None


router = APIRouter(
    prefix="/api/v1/maritime/metocean-vintage-validation",
    tags=["Maritime authentic vintage forecast validation"],
)


@router.get("/status", response_model=B62BStatus)
def status() -> B62BStatus:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT status,row_count,metadata,finished_at FROM audit.ingestion_run
            WHERE source_name=%s ORDER BY started_at DESC LIMIT 1
            """,
            (AUDIT_SOURCE,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B62B audit unavailable")
    metadata = dict(row[2] or {})
    return B62BStatus(
        audit_status=str(row[0]),
        decision=str(metadata.get("decision", "UNKNOWN")),
        model_version=str(metadata.get("model_version", MODEL_VERSION)),
        rows=int(row[1] or metadata.get("rows", 0)),
        archive_origins=int(metadata.get("archive_origins", 0)),
        valid_origins=int(metadata.get("valid_origins", 0)),
        test_origins=int(metadata.get("test_origins", 0)),
        fresh_origins=int(metadata.get("fresh_origins", 0)),
        fresh_span_days=float(metadata.get("fresh_span_days", 0.0)),
        selected_model=metadata.get("selected_model"),
        reference_model=metadata.get("reference_model"),
        valid_accepted=bool(metadata.get("valid_accepted", False)),
        archive_confirmed=bool(metadata.get("archive_confirmed", False)),
        fresh_confirmed=bool(metadata.get("fresh_confirmed", False)),
        critical_gates_passed=bool(metadata.get("critical_gates_passed", False)),
        production_promotion_allowed=bool(metadata.get("production_promotion_allowed", False)),
        limited_pilot_allowed=bool(metadata.get("limited_pilot_allowed", False)),
        automatic_action_allowed=bool(metadata.get("automatic_action_allowed", False)),
        test_role=str(metadata.get("test_role", "NOT_CONSUMED")),
        next_block=metadata.get("next_block"),
        finished_at=row[3],
    )


@router.get("/metrics", response_model=list[Metric])
def metrics(evaluation_role: str | None = Query(default=None)) -> list[Metric]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT evaluation_role,model,rows,origins,mae,rmse,bias,coverage,
                   mean_interval_width,quantile_crossings
            FROM {METRIC_TABLE}
            WHERE model_version=%s
              AND (%s::text IS NULL OR evaluation_role=%s)
            ORDER BY evaluation_role,mae
            """,
            (MODEL_VERSION, evaluation_role, evaluation_role),
        )
        rows = cursor.fetchall()
    return [Metric(**dict(zip(Metric.model_fields, row))) for row in rows]


@router.get("/predictions", response_model=list[Prediction])
def predictions(
    evaluation_role: str = Query(default="FRESH_FORWARD_CONFIRMATORY"),
    model: str | None = Query(default=None),
    limit: int = Query(default=250, ge=1, le=2_000),
) -> list[Prediction]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT evaluation_role,issue_at,valid_at,horizon_h,variable,model,
                   actual,p10,p50,p90
            FROM {PREDICTION_TABLE}
            WHERE model_version=%s AND evaluation_role=%s
              AND (%s::text IS NULL OR model=%s)
            ORDER BY issue_at DESC,model LIMIT %s
            """,
            (MODEL_VERSION, evaluation_role, model, model, limit),
        )
        rows = cursor.fetchall()
    return [Prediction(**dict(zip(Prediction.model_fields, row))) for row in rows]


@router.get("/model-card")
def model_card() -> dict[str, Any]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT metadata FROM audit.ingestion_run
            WHERE source_name=%s AND status='SUCCESS'
            ORDER BY finished_at DESC LIMIT 1
            """,
            (AUDIT_SOURCE,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B62B model card unavailable")
    metadata = dict(row[0] or {})
    return {
        "model_version": metadata.get("model_version", MODEL_VERSION),
        "decision": metadata.get("decision"),
        "selected_model": metadata.get("selected_model"),
        "selection_role": metadata.get("selection_role"),
        "test_role": metadata.get("test_role"),
        "archive_role": metadata.get("archive_role"),
        "fresh_role": metadata.get("fresh_role"),
        "production_contract": metadata.get("production_contract"),
        "production_promotion_allowed": metadata.get("production_promotion_allowed", False),
        "automatic_action_allowed": False,
        "artifacts": metadata.get("artifacts", {}),
    }
