from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import psycopg2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel, Field

from platform_api import local_demo


SERVICE_VERSION = "b56h-b56gv21-platform-api-v1"
FORECAST_VERSION = "b56g-v2.1-asymmetric-aci-v1"
POINT_FORECAST_VERSION = "b56e-arrival-flow-probabilistic-ensemble-v1"
SERVING_TABLE = "serving.maritime_arrival_flow_asymmetric_backtest_v21"
AUDIT_SOURCE = "b56g_v21_asymmetric_calibration"
AUDIT_DATASET = "port_arrival_flow_asymmetric_intervals"
SHADOW_AUDIT_SOURCE = "b56g_v21_prospective_shadow_monitor"
SHADOW_AUDIT_DATASET = "port_arrival_flow_prospective_shadow"
ALLOWED_HORIZONS = (6, 12, 24)


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _database_dsn() -> str:
    return (
        f"host={_env('SMART_PORT_DB_HOST', 'timescaledb')} "
        f"port={_env('SMART_PORT_DB_PORT', '5432')} "
        f"dbname={_env('SMART_PORT_DB_NAME', 'maritime')} "
        f"user={_env('SMART_PORT_DB_USER', 'smartport')} "
        f"password={_env('SMART_PORT_DB_PASSWORD')}"
    )


_pool: SimpleConnectionPool | None = None


def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("B56F_DB_POOL_SIZE", "8")),
            dsn=_database_dsn(),
        )
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


class ForecastPoint(BaseModel):
    as_of_time: datetime
    target_time: datetime
    horizon_h: int
    selected_model: str
    actual_arrivals: float | None
    point_prediction: float
    p10: float
    p50: float
    p90: float
    absolute_error: float | None
    source_mode: str


class ReplayRange(BaseModel):
    first_as_of_time: datetime
    last_as_of_time: datetime
    timestamps: int
    serving_rows: int
    horizons_h: list[int]
    forecast_version: str
    source_mode: str


class SourceStatus(BaseModel):
    audit_status: str
    decision: str
    source_status: str
    source_break: datetime | None
    latest_eligible: datetime | None
    historical_replay_allowed: bool
    live_serving_allowed: bool
    training_executed: bool
    selection_used_test: bool
    finished_at: datetime | None


class HorizonMetric(BaseModel):
    horizon_h: int
    observations: int
    mae: float
    rmse: float
    wape_pct: float
    bias: float
    coverage_p10_p90: float
    mean_interval_width: float


class ReplaySnapshot(BaseModel):
    requested_as_of: datetime | None
    resolved_as_of: datetime
    forecasts: list[ForecastPoint]
    source_mode: str = "HISTORICAL_REPLAY"
    live: bool = False


class PortCallItem(BaseModel):
    port_call_id: str
    port_code: str
    terminal_code: str | None
    imo: int | None
    vessel_name: str
    voyage_id: str | None
    planned_eta: datetime
    actual_ata: datetime | None
    planned_etd: datetime | None
    actual_atd: datetime | None
    cargo_type: str | None
    vessel_type: str | None
    status: str
    arrival_delay_h: float | None


class WeatherPoint(BaseModel):
    observed_at: datetime
    latitude: float
    longitude: float
    wave_height_m: float | None
    wave_period_s: float | None
    wave_direction_deg: float | None
    wind_speed_ms: float | None
    wind_direction_deg: float | None
    surface_current_ms: float | None
    visibility_m: float | None
    pressure_hpa: float | None
    quality_flag: int


class OperationalSummary(BaseModel):
    resolved_as_of: datetime
    expected_next_24h: int
    expected_next_72h: int
    arrived_previous_24h: int
    overdue_calls: int
    vessels_in_port: int
    active_call_window: int
    wave_height_m: float | None
    wave_period_s: float | None
    wind_speed_ms: float | None
    weather_observed_at: datetime | None
    ais_positions_72h: int
    ais_vessels_72h: int
    ais_last_observed_at: datetime | None


class DataHealthSource(BaseModel):
    source: str
    label: str
    status: str
    rows: int
    latest_event_time: datetime | None
    age_hours: float | None
    detail: str


class DataHealth(BaseModel):
    resolved_as_of: datetime
    sources: list[DataHealthSource]


class HorizonGovernance(BaseModel):
    horizon_h: int
    selected_policy: str
    window_days: int | None
    gamma: float | None
    coverage_30d: float
    mae_30d: float
    interval_width_30d: float
    gate_status: str


class ModelGovernance(BaseModel):
    model_version: str
    point_source: str
    mode: str = "HISTORICAL_REPLAY"
    calibration_decision: str
    shadow_decision: str
    replay_allowed: bool
    live_allowed: bool
    integrity_passed: bool
    point_fidelity_passed: bool
    coherence_passed: bool
    recent30_gates_passed: bool
    formal_promotion_allowed: bool
    promotion_blocker: str | None
    prospective_forecasts: int
    paired_forecasts: int
    last_audit_at: datetime | None
    horizons: list[HorizonGovernance]


class PerformancePoint(BaseModel):
    period_start: datetime
    horizon_h: int
    observations: int
    mae: float
    bias: float
    coverage_p10_p90: float
    mean_interval_width: float


class ErrorHeatmapCell(BaseModel):
    day_of_week: int
    hour_of_day: int
    observations: int
    mae: float
    bias: float
    coverage_p10_p90: float


def _normalize_time(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _range_row() -> tuple[Any, ...]:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    MIN(as_of_time),
                    MAX(as_of_time),
                    COUNT(DISTINCT as_of_time),
                    COUNT(*),
                    ARRAY_AGG(DISTINCT horizon_h ORDER BY horizon_h),
                    MIN(source_mode)
                FROM {SERVING_TABLE}
                WHERE forecast_version=%s
                """,
                (FORECAST_VERSION,),
            )
            row = cursor.fetchone()
    if row is None or row[0] is None:
        raise HTTPException(status_code=503, detail="Historical replay data is unavailable")
    return row


def _resolve_as_of(requested: datetime | None) -> datetime:
    requested = _normalize_time(requested)
    first_time, last_time, *_ = _range_row()
    if requested is None:
        return last_time
    if requested < first_time:
        raise HTTPException(
            status_code=422,
            detail=f"as_of precedes replay range ({first_time.isoformat()})",
        )
    if requested > last_time:
        raise HTTPException(
            status_code=422,
            detail=f"as_of exceeds latest eligible replay time ({last_time.isoformat()})",
        )
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT MAX(as_of_time)
                FROM {SERVING_TABLE}
                WHERE forecast_version=%s AND as_of_time<=%s
                """,
                (FORECAST_VERSION, requested),
            )
            resolved = cursor.fetchone()[0]
    if resolved is None:
        raise HTTPException(status_code=404, detail="No replay snapshot found")
    return resolved


def _forecast_rows(resolved: datetime, horizon_h: int | None = None) -> list[ForecastPoint]:
    parameters: list[Any] = [FORECAST_VERSION, resolved]
    horizon_clause = ""
    if horizon_h is not None:
        if horizon_h not in ALLOWED_HORIZONS:
            raise HTTPException(status_code=422, detail="horizon_h must be 6, 12, or 24")
        horizon_clause = " AND horizon_h=%s"
        parameters.append(horizon_h)

    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    as_of_time,
                    as_of_time + make_interval(hours => horizon_h),
                    horizon_h,
                    selected_policy,
                    actual_arrivals::double precision,
                    point_prediction::double precision,
                    p10::double precision,
                    p50::double precision,
                    p90::double precision,
                    CASE
                        WHEN actual_arrivals IS NULL THEN NULL
                        ELSE ABS(point_prediction-actual_arrivals)::double precision
                    END,
                    source_mode
                FROM {SERVING_TABLE}
                WHERE forecast_version=%s AND as_of_time=%s
                {horizon_clause}
                ORDER BY horizon_h
                """,
                tuple(parameters),
            )
            rows = cursor.fetchall()
    return [
        ForecastPoint(
            as_of_time=row[0],
            target_time=row[1],
            horizon_h=int(row[2]),
            selected_model=row[3],
            actual_arrivals=row[4],
            point_prediction=row[5],
            p10=row[6],
            p50=row[7],
            p90=row[8],
            absolute_error=row[9],
            source_mode=row[10],
        )
        for row in rows
    ]


app = FastAPI(
    title="Smart Port Maritime Probabilistic Replay API",
    description=(
        "Read-only API for B56E point forecasts with B56G-v2.1 asymmetric intervals. "
        "This service never presents replay data as live."
    ),
    version=SERVICE_VERSION,
)

origins = [
    item.strip()
    for item in os.getenv(
        "B56F_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:4173,http://localhost:8088",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def _shutdown() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "maritime-historical-replay-api",
        "version": SERVICE_VERSION,
        "mode": "local-demo-shadow" if local_demo.enabled() else "database",
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    if local_demo.enabled():
        return {
            "status": "ready",
            "service": "maritime-historical-replay-api",
            "mode": "local-demo-shadow",
            "serving_rows": 0,
            "last_as_of_time": local_demo.capacity_snapshot_times()[-1],
            "live": False,
            "scientific_claim_allowed": False,
        }
    replay_range = get_replay_range()
    return {
        "status": "ready",
        "service": "maritime-historical-replay-api",
        "serving_rows": replay_range.serving_rows,
        "last_as_of_time": replay_range.last_as_of_time,
        "live": False,
    }


@app.get("/api/v1/maritime/replay/config")
def config() -> dict[str, Any]:
    return {
        "api_version": SERVICE_VERSION,
        "forecast_version": FORECAST_VERSION,
        "point_forecast_version": POINT_FORECAST_VERSION,
        "interval_calibration": "ASYMMETRIC_ADAPTIVE_CONFORMAL",
        "allowed_horizons_h": list(ALLOWED_HORIZONS),
        "source_mode": "HISTORICAL_REPLAY",
        "live": False,
        "timezone": "UTC",
    }


@app.get("/api/v1/maritime/replay/range", response_model=ReplayRange)
def get_replay_range() -> ReplayRange:
    first_time, last_time, timestamps, rows, horizons, source_mode = _range_row()
    return ReplayRange(
        first_as_of_time=first_time,
        last_as_of_time=last_time,
        timestamps=int(timestamps),
        serving_rows=int(rows),
        horizons_h=[int(value) for value in horizons],
        forecast_version=FORECAST_VERSION,
        source_mode=source_mode,
    )


@app.get("/api/v1/maritime/replay/source-status", response_model=SourceStatus)
def source_status() -> SourceStatus:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, finished_at, metadata
                FROM audit.ingestion_run
                WHERE source_name=%s AND dataset_name=%s
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (AUDIT_SOURCE, AUDIT_DATASET),
            )
            row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="B56G-v2.1 audit status is unavailable")
    audit_status, finished_at, metadata = row
    metadata = dict(metadata or {})
    return SourceStatus(
        audit_status=audit_status,
        decision=str(metadata.get("status", "UNKNOWN")),
        source_status=str(metadata.get("objective", "ASYMMETRIC_INTERVAL_CALIBRATION")),
        source_break=metadata.get("source_completeness_break_start"),
        latest_eligible=metadata.get("latest_model_eligible_time"),
        historical_replay_allowed=bool(metadata.get("historical_replay_allowed", False)),
        live_serving_allowed=False,
        training_executed=bool(metadata.get("training_executed", False)),
        selection_used_test=bool(metadata.get("selection_used_test", False)),
        finished_at=finished_at,
    )


@app.get("/api/v1/maritime/replay/snapshot", response_model=ReplaySnapshot)
def snapshot(
    as_of: datetime | None = Query(default=None),
    horizon_h: int | None = Query(default=None),
) -> ReplaySnapshot:
    resolved = _resolve_as_of(as_of)
    forecasts = _forecast_rows(resolved, horizon_h)
    if not forecasts:
        raise HTTPException(status_code=404, detail="No forecast found for this snapshot")
    return ReplaySnapshot(
        requested_as_of=_normalize_time(as_of),
        resolved_as_of=resolved,
        forecasts=forecasts,
    )


@app.get("/api/v1/maritime/replay/timeline", response_model=list[ForecastPoint])
def timeline(
    horizon_h: int = Query(default=24),
    end: datetime | None = Query(default=None),
    hours: int = Query(default=168, ge=24, le=720),
) -> list[ForecastPoint]:
    if horizon_h not in ALLOWED_HORIZONS:
        raise HTTPException(status_code=422, detail="horizon_h must be 6, 12, or 24")
    resolved_end = _resolve_as_of(end)
    start = resolved_end - timedelta(hours=hours - 1)
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    as_of_time,
                    as_of_time + make_interval(hours => horizon_h),
                    horizon_h,
                    selected_policy,
                    actual_arrivals::double precision,
                    point_prediction::double precision,
                    p10::double precision,
                    p50::double precision,
                    p90::double precision,
                    CASE
                        WHEN actual_arrivals IS NULL THEN NULL
                        ELSE ABS(point_prediction-actual_arrivals)::double precision
                    END,
                    source_mode
                FROM {SERVING_TABLE}
                WHERE forecast_version=%s
                  AND horizon_h=%s
                  AND as_of_time BETWEEN %s AND %s
                ORDER BY as_of_time
                """,
                (FORECAST_VERSION, horizon_h, start, resolved_end),
            )
            rows = cursor.fetchall()
    return [
        ForecastPoint(
            as_of_time=row[0],
            target_time=row[1],
            horizon_h=int(row[2]),
            selected_model=row[3],
            actual_arrivals=row[4],
            point_prediction=row[5],
            p10=row[6],
            p50=row[7],
            p90=row[8],
            absolute_error=row[9],
            source_mode=row[10],
        )
        for row in rows
    ]


@app.get("/api/v1/maritime/replay/metrics", response_model=list[HorizonMetric])
def metrics(
    end: datetime | None = Query(default=None),
    days: int = Query(default=30, ge=7, le=365),
) -> list[HorizonMetric]:
    resolved_end = _resolve_as_of(end)
    start = resolved_end - timedelta(days=days)
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    horizon_h,
                    COUNT(*)::integer,
                    AVG(ABS(point_prediction-actual_arrivals))::double precision,
                    SQRT(AVG(POWER(point_prediction-actual_arrivals, 2)))::double precision,
                    (
                        100.0 * SUM(ABS(point_prediction-actual_arrivals))
                        / NULLIF(SUM(ABS(actual_arrivals)), 0)
                    )::double precision,
                    AVG(point_prediction-actual_arrivals)::double precision,
                    AVG(
                        CASE WHEN actual_arrivals BETWEEN p10 AND p90
                        THEN 1.0 ELSE 0.0 END
                    )::double precision,
                    AVG(p90-p10)::double precision
                FROM {SERVING_TABLE}
                WHERE forecast_version=%s
                  AND as_of_time BETWEEN %s AND %s
                  AND actual_arrivals IS NOT NULL
                GROUP BY horizon_h
                ORDER BY horizon_h
                """,
                (FORECAST_VERSION, start, resolved_end),
            )
            rows = cursor.fetchall()
    return [
        HorizonMetric(
            horizon_h=int(row[0]),
            observations=int(row[1]),
            mae=float(row[2]),
            rmse=float(row[3]),
            wape_pct=float(row[4]),
            bias=float(row[5]),
            coverage_p10_p90=float(row[6]),
            mean_interval_width=float(row[7]),
        )
        for row in rows
    ]


@app.get("/api/v1/maritime/replay/model-governance", response_model=ModelGovernance)
def model_governance() -> ModelGovernance:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT finished_at, metadata
                FROM audit.ingestion_run
                WHERE source_name=%s AND dataset_name=%s AND status='SUCCESS'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (AUDIT_SOURCE, AUDIT_DATASET),
            )
            calibration_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT finished_at, metadata
                FROM audit.ingestion_run
                WHERE source_name=%s AND dataset_name=%s AND status='SUCCESS'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (SHADOW_AUDIT_SOURCE, SHADOW_AUDIT_DATASET),
            )
            shadow_row = cursor.fetchone()
            cursor.execute(
                f"""
                WITH bounds AS (
                    SELECT MAX(as_of_time) AS end_time
                    FROM {SERVING_TABLE}
                    WHERE forecast_version=%s
                )
                SELECT
                    horizon_h,
                    MAX(selected_policy),
                    AVG(
                        CASE WHEN actual_arrivals BETWEEN p10 AND p90
                        THEN 1.0 ELSE 0.0 END
                    )::double precision,
                    AVG(ABS(point_prediction-actual_arrivals))::double precision,
                    AVG(p90-p10)::double precision
                FROM {SERVING_TABLE}, bounds
                WHERE forecast_version=%s
                  AND actual_arrivals IS NOT NULL
                  AND as_of_time BETWEEN bounds.end_time-INTERVAL '30 days'
                                     AND bounds.end_time
                GROUP BY horizon_h
                ORDER BY horizon_h
                """,
                (FORECAST_VERSION, FORECAST_VERSION),
            )
            horizon_rows = cursor.fetchall()

    if calibration_row is None:
        raise HTTPException(status_code=503, detail="Calibration governance is unavailable")

    calibration_finished_at, calibration_metadata_raw = calibration_row
    calibration_metadata = dict(calibration_metadata_raw or {})
    shadow_finished_at = shadow_row[0] if shadow_row else None
    shadow_metadata = dict(shadow_row[1] or {}) if shadow_row else {}
    selected_policies = dict(calibration_metadata.get("selected_policies") or {})

    horizon_governance: list[HorizonGovernance] = []
    for horizon_h, policy, coverage, mae, interval_width in horizon_rows:
        policy_metadata = dict(selected_policies.get(str(horizon_h)) or {})
        if 0.77 <= coverage <= 0.83:
            gate_status = "PASS"
        elif 0.74 <= coverage <= 0.86:
            gate_status = "WATCH"
        else:
            gate_status = "FAIL"
        horizon_governance.append(
            HorizonGovernance(
                horizon_h=int(horizon_h),
                selected_policy=str(policy),
                window_days=policy_metadata.get("window_days"),
                gamma=policy_metadata.get("gamma"),
                coverage_30d=float(coverage),
                mae_30d=float(mae),
                interval_width_30d=float(interval_width),
                gate_status=gate_status,
            )
        )

    return ModelGovernance(
        model_version=FORECAST_VERSION,
        point_source=str(calibration_metadata.get("point_source", POINT_FORECAST_VERSION)),
        calibration_decision=str(calibration_metadata.get("status", "UNKNOWN")),
        shadow_decision=str(shadow_metadata.get("status", "NOT_STARTED")),
        replay_allowed=bool(calibration_metadata.get("historical_replay_allowed", False)),
        live_allowed=bool(calibration_metadata.get("live_serving_allowed", False)),
        integrity_passed=bool(calibration_metadata.get("integrity_gates_passed", False)),
        point_fidelity_passed=bool(calibration_metadata.get("point_fidelity_passed", False)),
        coherence_passed=bool(calibration_metadata.get("coherence_gates_passed", False)),
        recent30_gates_passed=bool(calibration_metadata.get("recent30_gates_passed", False)),
        formal_promotion_allowed=bool(
            calibration_metadata.get("formal_promotion_allowed", False)
        ),
        promotion_blocker=calibration_metadata.get("formal_promotion_blocker"),
        prospective_forecasts=int(shadow_metadata.get("forecast_rows", 0)),
        paired_forecasts=int(shadow_metadata.get("paired_rows", 0)),
        last_audit_at=shadow_finished_at or calibration_finished_at,
        horizons=horizon_governance,
    )


@app.get(
    "/api/v1/maritime/replay/performance-history",
    response_model=list[PerformancePoint],
)
def performance_history(
    horizon_h: int = Query(default=24),
    end: datetime | None = Query(default=None),
    days: int = Query(default=60, ge=14, le=365),
) -> list[PerformancePoint]:
    if horizon_h not in ALLOWED_HORIZONS:
        raise HTTPException(status_code=422, detail="horizon_h must be 6, 12, or 24")
    resolved_end = _resolve_as_of(end)
    start = resolved_end - timedelta(days=days)
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    DATE_TRUNC('day', as_of_time),
                    horizon_h,
                    COUNT(*)::integer,
                    AVG(ABS(point_prediction-actual_arrivals))::double precision,
                    AVG(point_prediction-actual_arrivals)::double precision,
                    AVG(
                        CASE WHEN actual_arrivals BETWEEN p10 AND p90
                        THEN 1.0 ELSE 0.0 END
                    )::double precision,
                    AVG(p90-p10)::double precision
                FROM {SERVING_TABLE}
                WHERE forecast_version=%s
                  AND horizon_h=%s
                  AND as_of_time BETWEEN %s AND %s
                  AND actual_arrivals IS NOT NULL
                GROUP BY DATE_TRUNC('day', as_of_time), horizon_h
                ORDER BY DATE_TRUNC('day', as_of_time)
                """,
                (FORECAST_VERSION, horizon_h, start, resolved_end),
            )
            rows = cursor.fetchall()
    return [
        PerformancePoint(
            period_start=row[0],
            horizon_h=int(row[1]),
            observations=int(row[2]),
            mae=float(row[3]),
            bias=float(row[4]),
            coverage_p10_p90=float(row[5]),
            mean_interval_width=float(row[6]),
        )
        for row in rows
    ]


@app.get(
    "/api/v1/maritime/replay/error-heatmap",
    response_model=list[ErrorHeatmapCell],
)
def error_heatmap(
    horizon_h: int = Query(default=24),
    end: datetime | None = Query(default=None),
    days: int = Query(default=120, ge=30, le=365),
) -> list[ErrorHeatmapCell]:
    if horizon_h not in ALLOWED_HORIZONS:
        raise HTTPException(status_code=422, detail="horizon_h must be 6, 12, or 24")
    resolved_end = _resolve_as_of(end)
    start = resolved_end - timedelta(days=days)
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    EXTRACT(ISODOW FROM as_of_time)::integer,
                    EXTRACT(HOUR FROM as_of_time)::integer,
                    COUNT(*)::integer,
                    AVG(ABS(point_prediction-actual_arrivals))::double precision,
                    AVG(point_prediction-actual_arrivals)::double precision,
                    AVG(
                        CASE WHEN actual_arrivals BETWEEN p10 AND p90
                        THEN 1.0 ELSE 0.0 END
                    )::double precision
                FROM {SERVING_TABLE}
                WHERE forecast_version=%s
                  AND horizon_h=%s
                  AND as_of_time BETWEEN %s AND %s
                  AND actual_arrivals IS NOT NULL
                GROUP BY
                    EXTRACT(ISODOW FROM as_of_time),
                    EXTRACT(HOUR FROM as_of_time)
                ORDER BY 1, 2
                """,
                (FORECAST_VERSION, horizon_h, start, resolved_end),
            )
            rows = cursor.fetchall()
    return [
        ErrorHeatmapCell(
            day_of_week=int(row[0]),
            hour_of_day=int(row[1]),
            observations=int(row[2]),
            mae=float(row[3]),
            bias=float(row[4]),
            coverage_p10_p90=float(row[5]),
        )
        for row in rows
    ]


@app.get("/api/v1/maritime/operations/port-calls", response_model=list[PortCallItem])
def port_calls(
    as_of: datetime | None = Query(default=None),
    before_h: int = Query(default=24, ge=0, le=168),
    after_h: int = Query(default=72, ge=1, le=336),
    limit: int = Query(default=250, ge=1, le=1000),
) -> list[PortCallItem]:
    resolved = _resolve_as_of(as_of)
    start = resolved - timedelta(hours=before_h)
    end = resolved + timedelta(hours=after_h)
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    port_call_id::text,
                    port_code,
                    NULLIF(BTRIM(terminal_code), ''),
                    imo,
                    COALESCE(NULLIF(BTRIM(vessel_name), ''), 'Navire non renseigne'),
                    NULLIF(BTRIM(voyage_id), ''),
                    planned_eta,
                    CASE WHEN actual_ata<=%s THEN actual_ata END,
                    planned_etd,
                    CASE WHEN actual_atd<=%s THEN actual_atd END,
                    NULLIF(BTRIM(cargo_type), ''),
                    NULLIF(BTRIM(vessel_type), ''),
                    CASE
                        WHEN actual_atd IS NOT NULL AND actual_atd<=%s THEN 'DEPARTED'
                        WHEN actual_atb IS NOT NULL AND actual_atb<=%s
                             AND (actual_atd IS NULL OR actual_atd>%s) THEN 'BERTHED'
                        WHEN actual_ata IS NOT NULL AND actual_ata<=%s THEN 'ARRIVED'
                        WHEN planned_eta<%s THEN 'OVERDUE'
                        ELSE 'EXPECTED'
                    END,
                    CASE
                        WHEN actual_ata IS NOT NULL AND actual_ata<=%s
                        THEN EXTRACT(EPOCH FROM (actual_ata-planned_eta))/3600.0
                    END::double precision
                FROM core.port_call
                WHERE port_code='MAPTM'
                  AND planned_eta BETWEEN %s AND %s
                ORDER BY planned_eta
                LIMIT %s
                """,
                (
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    start,
                    end,
                    limit,
                ),
            )
            rows = cursor.fetchall()
    return [
        PortCallItem(
            port_call_id=row[0],
            port_code=row[1],
            terminal_code=row[2],
            imo=row[3],
            vessel_name=row[4],
            voyage_id=row[5],
            planned_eta=row[6],
            actual_ata=row[7],
            planned_etd=row[8],
            actual_atd=row[9],
            cargo_type=row[10],
            vessel_type=row[11],
            status=row[12],
            arrival_delay_h=row[13],
        )
        for row in rows
    ]


@app.get("/api/v1/maritime/operations/weather", response_model=list[WeatherPoint])
def weather(
    as_of: datetime | None = Query(default=None),
    hours: int = Query(default=168, ge=6, le=720),
) -> list[WeatherPoint]:
    resolved = _resolve_as_of(as_of)
    start = resolved - timedelta(hours=hours - 1)
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    observed_at,
                    latitude,
                    longitude,
                    wave_height_m::double precision,
                    wave_period_s::double precision,
                    wave_direction_deg::double precision,
                    wind_speed_ms::double precision,
                    wind_direction_deg::double precision,
                    surface_current_ms::double precision,
                    visibility_m::double precision,
                    pressure_hpa::double precision,
                    quality_flag
                FROM core.maritime_observation
                WHERE observed_at BETWEEN %s AND %s
                ORDER BY observed_at
                """,
                (start, resolved),
            )
            rows = cursor.fetchall()
    return [
        WeatherPoint(
            observed_at=row[0],
            latitude=float(row[1]),
            longitude=float(row[2]),
            wave_height_m=row[3],
            wave_period_s=row[4],
            wave_direction_deg=row[5],
            wind_speed_ms=row[6],
            wind_direction_deg=row[7],
            surface_current_ms=row[8],
            visibility_m=row[9],
            pressure_hpa=row[10],
            quality_flag=int(row[11]),
        )
        for row in rows
    ]


@app.get("/api/v1/maritime/operations/summary", response_model=OperationalSummary)
def operational_summary(
    as_of: datetime | None = Query(default=None),
) -> OperationalSummary:
    resolved = _resolve_as_of(as_of)
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE planned_eta>%s AND planned_eta<=%s+INTERVAL '24 hours'
                    )::integer,
                    COUNT(*) FILTER (
                        WHERE planned_eta>%s AND planned_eta<=%s+INTERVAL '72 hours'
                    )::integer,
                    COUNT(*) FILTER (
                        WHERE actual_ata>%s-INTERVAL '24 hours' AND actual_ata<=%s
                    )::integer,
                    COUNT(*) FILTER (
                        WHERE planned_eta<%s
                          AND (actual_ata IS NULL OR actual_ata>%s)
                    )::integer,
                    COUNT(*) FILTER (
                        WHERE actual_atb<=%s
                          AND (actual_atd IS NULL OR actual_atd>%s)
                    )::integer,
                    COUNT(*) FILTER (
                        WHERE planned_eta BETWEEN %s-INTERVAL '24 hours'
                                              AND %s+INTERVAL '72 hours'
                    )::integer
                FROM core.port_call
                WHERE port_code='MAPTM'
                """,
                (
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                    resolved,
                ),
            )
            call_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT
                    observed_at,
                    wave_height_m::double precision,
                    wave_period_s::double precision,
                    wind_speed_ms::double precision
                FROM core.maritime_observation
                WHERE observed_at<=%s
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                (resolved,),
            )
            weather_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT
                    COUNT(*)::integer,
                    COUNT(DISTINCT mmsi)::integer,
                    MAX(observed_at)
                FROM core.vessel_position
                WHERE observed_at BETWEEN %s-INTERVAL '72 hours' AND %s
                """,
                (resolved, resolved),
            )
            ais_row = cursor.fetchone()
    return OperationalSummary(
        resolved_as_of=resolved,
        expected_next_24h=int(call_row[0]),
        expected_next_72h=int(call_row[1]),
        arrived_previous_24h=int(call_row[2]),
        overdue_calls=int(call_row[3]),
        vessels_in_port=int(call_row[4]),
        active_call_window=int(call_row[5]),
        weather_observed_at=weather_row[0] if weather_row else None,
        wave_height_m=weather_row[1] if weather_row else None,
        wave_period_s=weather_row[2] if weather_row else None,
        wind_speed_ms=weather_row[3] if weather_row else None,
        ais_positions_72h=int(ais_row[0]),
        ais_vessels_72h=int(ais_row[1]),
        ais_last_observed_at=ais_row[2],
    )


def _health_status(rows: int, age_hours: float | None, ready_age_h: float) -> str:
    if rows == 0 or age_hours is None:
        return "MISSING"
    if age_hours <= ready_age_h:
        return "READY"
    return "STALE"


@app.get("/api/v1/maritime/operations/data-health", response_model=DataHealth)
def data_health(
    as_of: datetime | None = Query(default=None),
) -> DataHealth:
    resolved = _resolve_as_of(as_of)
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*)::integer, MAX(as_of_time)
                FROM {SERVING_TABLE}
                WHERE forecast_version=%s AND as_of_time<=%s
                """,
                (FORECAST_VERSION, resolved),
            )
            forecast_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT COUNT(*)::integer, MAX(observed_at)
                FROM core.maritime_observation
                WHERE observed_at<=%s
                """,
                (resolved,),
            )
            weather_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT COUNT(*)::integer, MAX(planned_eta)
                FROM core.port_call
                WHERE port_code='MAPTM' AND planned_eta<=%s
                """,
                (resolved,),
            )
            call_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT COUNT(*)::integer, MAX(observed_at)
                FROM core.vessel_position
                WHERE observed_at<=%s
                """,
                (resolved,),
            )
            ais_row = cursor.fetchone()

    source_rows = [
        (
            "FORECAST",
            "Prévisions B56G-v2.1",
            forecast_row,
            1.0,
            "Point B56E et intervalles asymétriques adaptatifs",
        ),
        ("PORT_CALLS", "Escales", call_row, 72.0, "Planning et evenements portuaires"),
        ("WEATHER", "Meteo et vagues", weather_row, 3.0, "Observation maritime horaire"),
        ("AIS", "Positions AIS", ais_row, 1.0, "Collecte de positions navires"),
    ]
    sources: list[DataHealthSource] = []
    for source, label, row, ready_age_h, detail in source_rows:
        rows = int(row[0])
        latest = row[1]
        age_hours = (
            max(0.0, (resolved - latest).total_seconds() / 3600.0)
            if latest is not None
            else None
        )
        sources.append(
            DataHealthSource(
                source=source,
                label=label,
                status=_health_status(rows, age_hours, ready_age_h),
                rows=rows,
                latest_event_time=latest,
                age_hours=age_hours,
                detail=detail,
            )
        )
    return DataHealth(resolved_as_of=resolved, sources=sources)

# B61C_SHADOW_DECISION_ROUTER
from platform_api.b61c_routes import router as b61c_router
app.include_router(b61c_router)

# B61D_CONTEXTUAL_HSMM_ROUTER
from platform_api.b61d_routes import router as b61d_router
app.include_router(b61d_router)

# B61D_V11_ANCHORED_HSMM_ROUTER
from platform_api.b61d_v11_routes import router as b61d_v11_router
app.include_router(b61d_v11_router)

# B61D_V12_STATE_POLICY_ROUTER
from platform_api.b61d_v12_routes import router as b61d_v12_router
app.include_router(b61d_v12_router)

# B61D_V13_DUAL_STAGE_ROUTER
from platform_api.b61d_v13_routes import router as b61d_v13_router
app.include_router(b61d_v13_router)

# B61D_V131_CONTRACT_RECALIBRATION_ROUTER
from platform_api.b61d_v131_routes import router as b61d_v131_router
app.include_router(b61d_v131_router)

# B61E_CAPACITY_AWARE_TEMPORAL_RANKING_ROUTER
from platform_api.b61e_routes import router as b61e_router
app.include_router(b61e_router)

# B62_WEATHER_WAVE_VESSEL_AUTOGLUON_ROUTER
from platform_api.b62_routes import router as b62_router
app.include_router(b62_router)

# B62A_GOVERNED_METOCEAN_AUGMENTATION_ROUTER
from platform_api.b62a_routes import router as b62a_router
app.include_router(b62a_router)

# B62B_VINTAGE_FORECAST_SHADOW_VALIDATION_ROUTER
from platform_api.b62b_routes import router as b62b_router
app.include_router(b62b_router)

# PORTFLOW_CONTROL_TOWER_MVP_ROUTER
from platform_api.control_tower_routes import router as control_tower_router
app.include_router(control_tower_router)
