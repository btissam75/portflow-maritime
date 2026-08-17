from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel

from platform_api import local_demo


MODEL_VERSION = "b62-weather-wave-vessel-autogluon-v1"
AUDIT_SOURCE = "b62_weather_wave_vessel_autogluon"
FORECAST_TABLE = "serving.maritime_metocean_forecast_shadow_v1"
IMPACT_TABLE = "serving.maritime_metocean_vessel_impact_shadow_v1"
TRACKS = {"RESEARCH_REANALYSIS_SHADOW", "ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT"}


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
        _pool = SimpleConnectionPool(1, int(os.getenv("B62_DB_POOL_SIZE", "4")), dsn=dsn)
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


class B62Status(BaseModel):
    audit_status: str
    decision: str
    model_version: str
    rows: int
    selected_chronos_tasks: int
    issue_time_ready: bool
    issue_time_span_days: float
    critical_gates_passed: bool
    test_used_for_selection: bool
    production_promotion_allowed: bool
    automatic_action_allowed: bool
    serving_forecast_rows: int
    serving_impact_rows: int
    next_block: str | None = None
    finished_at: datetime | None = None


class ForecastPoint(BaseModel):
    track: str
    issue_at: datetime
    valid_at: datetime
    horizon_h: int
    variable: str
    p10: float | None = None
    p50: float
    p90: float | None = None
    source_model: str
    uncertainty_status: str
    operationally_available: bool
    production_claim_allowed: bool


class VesselImpact(BaseModel):
    port_call_id: str
    vessel_name: str | None = None
    port_code: str | None = None
    terminal_code: str | None = None
    vessel_type: str | None = None
    cargo_group: str | None = None
    source_decision_at: datetime
    forecast_issue_at: datetime
    valid_at: datetime
    horizon_h: int
    base_temporal_risk: float
    metocean_severity: float
    vessel_exposure: float
    combined_priority_score: float
    metocean_tier: str
    priority_tier: str
    forecast_track: str
    score_semantics: str
    automatic_action_allowed: bool
    production_claim_allowed: bool


router = APIRouter(
    prefix="/api/v1/maritime/metocean-cascade",
    tags=["Maritime weather-wave-vessel cascade"],
)


@router.get("/status", response_model=B62Status)
def status() -> B62Status:
    if local_demo.enabled():
        forecasts = local_demo.metocean_forecast("ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT")
        impacts = local_demo.vessel_impacts()
        return B62Status(
            audit_status="DEMO",
            decision="LOCAL_DEMO_SHADOW_ONLY",
            model_version=local_demo.DEMO_METOCEAN_VERSION,
            rows=len(forecasts),
            selected_chronos_tasks=0,
            issue_time_ready=True,
            issue_time_span_days=3.0,
            critical_gates_passed=True,
            test_used_for_selection=False,
            production_promotion_allowed=False,
            automatic_action_allowed=False,
            serving_forecast_rows=len(forecasts),
            serving_impact_rows=len(impacts),
            next_block="CONNECT_REAL_B62_MATERIALIZATION",
            finished_at=forecasts[0]["issue_at"],
        )
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
        raise HTTPException(status_code=404, detail="B62 audit unavailable")
    metadata = dict(row[2] or {})
    return B62Status(
        audit_status=str(row[0]),
        decision=str(metadata.get("decision", "UNKNOWN")),
        model_version=str(metadata.get("model_version", MODEL_VERSION)),
        rows=int(row[1] or metadata.get("rows", 0)),
        selected_chronos_tasks=int(metadata.get("selected_chronos_tasks", 0)),
        issue_time_ready=bool(metadata.get("issue_time_ready", False)),
        issue_time_span_days=float(metadata.get("issue_time_span_days", 0.0)),
        critical_gates_passed=bool(metadata.get("critical_gates_passed", False)),
        test_used_for_selection=bool(metadata.get("test_used_for_selection", False)),
        production_promotion_allowed=bool(metadata.get("production_promotion_allowed", False)),
        automatic_action_allowed=bool(metadata.get("automatic_action_allowed", False)),
        serving_forecast_rows=int(metadata.get("serving_forecast_rows", 0)),
        serving_impact_rows=int(metadata.get("serving_impact_rows", 0)),
        next_block=metadata.get("next_block"),
        finished_at=row[3],
    )


@router.get("/forecast", response_model=list[ForecastPoint])
def forecast(
    track: str = Query(default="ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT"),
    variable: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2_000),
) -> list[ForecastPoint]:
    if track not in TRACKS:
        raise HTTPException(status_code=422, detail="Unsupported B62 forecast track")
    if local_demo.enabled():
        rows = local_demo.metocean_forecast(track)
        if variable is not None:
            rows = [row for row in rows if row["variable"] == variable]
        return [ForecastPoint(**row) for row in rows[:limit]]
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            WITH latest AS (
                SELECT MAX(issue_at) AS issue_at FROM {FORECAST_TABLE}
                WHERE model_version=%s AND track=%s
            )
            SELECT track,f.issue_at,valid_at,horizon_h,variable,p10,p50,p90,
                   source_model,uncertainty_status,operationally_available,
                   production_claim_allowed
            FROM {FORECAST_TABLE} f JOIN latest l ON l.issue_at=f.issue_at
            WHERE model_version=%s AND track=%s
              AND (%s::text IS NULL OR variable=%s)
            ORDER BY valid_at,variable LIMIT %s
            """,
            (MODEL_VERSION, track, MODEL_VERSION, track, variable, variable, limit),
        )
        rows = cursor.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No B62 forecast found")
    return [ForecastPoint(**dict(zip(ForecastPoint.model_fields, row))) for row in rows]


@router.get("/vessel-impact", response_model=list[VesselImpact])
def vessel_impact(
    horizon_h: int | None = Query(default=None, ge=1, le=168),
    priority_tier: str | None = Query(default=None),
    limit: int = Query(default=250, ge=1, le=1_000),
) -> list[VesselImpact]:
    if local_demo.enabled():
        rows = local_demo.vessel_impacts()
        if horizon_h is not None:
            rows = [row for row in rows if row["horizon_h"] == horizon_h]
        if priority_tier is not None:
            rows = [row for row in rows if row["priority_tier"] == priority_tier]
        return [VesselImpact(**row) for row in rows[:limit]]
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT port_call_id,vessel_name,port_code,terminal_code,vessel_type,cargo_group,
                   source_decision_at,forecast_issue_at,valid_at,horizon_h,
                   base_temporal_risk,metocean_severity,vessel_exposure,
                   combined_priority_score,metocean_tier,priority_tier,forecast_track,
                   score_semantics,
                   automatic_action_allowed,production_claim_allowed
            FROM {IMPACT_TABLE}
            WHERE model_version=%s
              AND (%s::integer IS NULL OR horizon_h=%s)
              AND (%s::text IS NULL OR priority_tier=%s)
            ORDER BY forecast_issue_at DESC,combined_priority_score DESC LIMIT %s
            """,
            (MODEL_VERSION, horizon_h, horizon_h, priority_tier, priority_tier, limit),
        )
        rows = cursor.fetchall()
    return [VesselImpact(**dict(zip(VesselImpact.model_fields, row))) for row in rows]


@router.get("/model-card")
def model_card() -> dict[str, Any]:
    if local_demo.enabled():
        return {
            "model_version": local_demo.DEMO_METOCEAN_VERSION,
            "decision": "LOCAL_DEMO_SHADOW_ONLY",
            "runtime": "LOCAL_DEMO",
            "issue_time_ready": True,
            "production_promotion_allowed": False,
            "automatic_action_allowed": False,
            "test_role": "NOT_CONSUMED",
            "artifacts": {},
        }
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
        raise HTTPException(status_code=404, detail="B62 model card unavailable")
    metadata = dict(row[0] or {})
    return {
        "model_version": metadata.get("model_version", MODEL_VERSION),
        "decision": metadata.get("decision"),
        "runtime": metadata.get("runtime"),
        "issue_time_ready": metadata.get("issue_time_ready"),
        "production_promotion_allowed": False,
        "automatic_action_allowed": False,
        "test_role": metadata.get("test_role"),
        "artifacts": metadata.get("artifacts", {}),
    }
