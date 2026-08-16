from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_values

from model_trainer.arrival_flow_baselines import (
    HORIZONS,
    _clean_json,
    _db_connection,
    _json_default,
    _s3_client,
    _upload,
)


MONITOR_VERSION = "b56g-v2.1-prospective-shadow-monitor-v1"
FORECAST_VERSION = "b56g-v2.1-asymmetric-aci-v1"
SOURCE_NAME = "b56g_v21_prospective_shadow_monitor"
DATASET_NAME = "port_arrival_flow_prospective_shadow"

FORECAST_TABLE = "serving.maritime_arrival_flow_shadow_forecast_v21"
OBSERVATION_TABLE = "serving.maritime_arrival_flow_shadow_observation_v21"
CALIBRATION_SOURCE = "b56g_v21_asymmetric_calibration"
CALIBRATION_DATASET = "port_arrival_flow_asymmetric_intervals"

EXPECTED_COVERAGE = 0.80
COVERAGE_MIN = 0.77
COVERAGE_MAX = 0.83
TAIL_MIN = 0.07
TAIL_MAX = 0.13
ROLLING_WINDOWS_DAYS = (7, 30, 90)
MIN_PROMOTION_DAYS = 30
MIN_PROMOTION_ROWS_PER_HORIZON = 500
LABEL_SETTLEMENT_HOURS = 2
MAX_ISSUANCE_LAG_MINUTES = 90

REFERENCE_MAE = {6: 1.226, 12: 1.907913, 24: 3.144254}
MAE_LIMIT = {
    horizon: max(value * 1.25, value + 0.50)
    for horizon, value in REFERENCE_MAE.items()
}


def _utc(value: datetime | str | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return timestamp.tz_convert("UTC")


def _ensure_schema() -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE SCHEMA IF NOT EXISTS serving;

                CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
                    forecast_id UUID PRIMARY KEY,
                    as_of_time TIMESTAMPTZ NOT NULL,
                    target_time TIMESTAMPTZ NOT NULL,
                    horizon_h INTEGER NOT NULL,
                    forecast_version TEXT NOT NULL,
                    selected_policy TEXT NOT NULL,
                    point_prediction DOUBLE PRECISION NOT NULL,
                    p10 DOUBLE PRECISION NOT NULL,
                    p50 DOUBLE PRECISION NOT NULL,
                    p90 DOUBLE PRECISION NOT NULL,
                    issued_at TIMESTAMPTZ NOT NULL,
                    source_snapshot_time TIMESTAMPTZ NOT NULL,
                    source_mode TEXT NOT NULL,
                    payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT maritime_shadow_horizon_check
                        CHECK (horizon_h IN (6, 12, 24)),
                    CONSTRAINT maritime_shadow_nonnegative_check
                        CHECK (
                            point_prediction >= 0 AND p10 >= 0
                            AND p50 >= 0 AND p90 >= 0
                        ),
                    CONSTRAINT maritime_shadow_quantile_order_check
                        CHECK (p10 <= p50 AND p50 <= p90),
                    CONSTRAINT maritime_shadow_point_interval_check
                        CHECK (p10 <= point_prediction
                               AND point_prediction <= p90),
                    CONSTRAINT maritime_shadow_issuance_check
                        CHECK (
                            source_snapshot_time <= issued_at
                            AND issued_at < target_time
                        ),
                    CONSTRAINT maritime_shadow_source_mode_check
                        CHECK (source_mode = 'PROSPECTIVE_SHADOW'),
                    CONSTRAINT maritime_shadow_forecast_unique
                        UNIQUE (
                            as_of_time, horizon_h, forecast_version
                        )
                );

                CREATE INDEX IF NOT EXISTS
                    ix_maritime_shadow_forecast_maturity
                ON {FORECAST_TABLE} (target_time, horizon_h);

                CREATE TABLE IF NOT EXISTS {OBSERVATION_TABLE} (
                    observation_id UUID PRIMARY KEY,
                    forecast_id UUID NOT NULL REFERENCES
                        {FORECAST_TABLE}(forecast_id),
                    as_of_time TIMESTAMPTZ NOT NULL,
                    target_time TIMESTAMPTZ NOT NULL,
                    horizon_h INTEGER NOT NULL,
                    actual_arrivals DOUBLE PRECISION NOT NULL,
                    source TEXT NOT NULL,
                    source_watermark TIMESTAMPTZ NOT NULL,
                    available_at TIMESTAMPTZ NOT NULL,
                    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    CONSTRAINT maritime_shadow_observation_nonnegative
                        CHECK (actual_arrivals >= 0),
                    CONSTRAINT maritime_shadow_observation_maturity
                        CHECK (
                            available_at >= target_time
                            AND source_watermark >= target_time
                        ),
                    CONSTRAINT maritime_shadow_observation_unique
                        UNIQUE (forecast_id, source, source_watermark)
                );

                CREATE INDEX IF NOT EXISTS
                    ix_maritime_shadow_observation_latest
                ON {OBSERVATION_TABLE}
                    (forecast_id, available_at DESC, received_at DESC);
                """
            )


def _load_calibration_decision() -> dict[str, Any]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, metadata
                FROM audit.ingestion_run
                WHERE source_name=%s AND dataset_name=%s
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (CALIBRATION_SOURCE, CALIBRATION_DATASET),
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("B56G-v2.1 calibration result is missing")
    status, metadata = row
    metadata = dict(metadata or {})
    if status != "SUCCESS":
        raise RuntimeError(f"Latest B56G-v2.1 calibration is {status}")
    if metadata.get("asymmetric_version") != FORECAST_VERSION:
        raise RuntimeError("Unexpected B56G-v2.1 calibration version")
    if metadata.get("point_fidelity_passed") is not True:
        raise RuntimeError("B56G-v2.1 point fidelity gate did not pass")
    if metadata.get("coherence_gates_passed") is not True:
        raise RuntimeError("B56G-v2.1 coherence gate did not pass")
    return metadata


def register_shadow_forecast(
    *,
    as_of_time: datetime | str,
    horizon_h: int,
    selected_policy: str,
    point_prediction: float,
    p10: float,
    p50: float,
    p90: float,
    issued_at: datetime | str,
    source_snapshot_time: datetime | str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_schema()
    if horizon_h not in HORIZONS:
        raise ValueError(f"horizon_h must be one of {HORIZONS}")

    as_of = _utc(as_of_time)
    issued = _utc(issued_at)
    snapshot = _utc(source_snapshot_time)
    now = pd.Timestamp.now(tz="UTC")
    target = as_of + pd.Timedelta(hours=horizon_h)
    values = np.asarray(
        [point_prediction, p10, p50, p90], dtype="float64"
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecast values must be finite and nonnegative")
    if not (p10 <= p50 <= p90 and p10 <= point_prediction <= p90):
        raise ValueError("Forecast quantiles or point are incoherent")
    if issued > now + pd.Timedelta(minutes=5):
        raise ValueError("issued_at cannot be in the future")
    if abs((issued - as_of).total_seconds()) > (
        MAX_ISSUANCE_LAG_MINUTES * 60
    ):
        raise ValueError(
            "Prospective forecast must be registered near its as_of_time; "
            "historical backfill is forbidden"
        )
    if snapshot > issued:
        raise ValueError("source_snapshot_time cannot follow issued_at")
    if issued >= target:
        raise ValueError("Forecast must be issued before target maturity")

    decision = _load_calibration_decision()
    expected = decision.get("selected_policies", {}).get(str(horizon_h), {})
    expected_policy = expected.get("policy")
    if expected_policy and selected_policy != expected_policy:
        raise ValueError(
            f"Policy mismatch for {horizon_h}h: expected {expected_policy}"
        )

    forecast_id = str(uuid.uuid4())
    row_payload = {
        "monitor_version": MONITOR_VERSION,
        "calibration_run_status": decision.get("status"),
        **(payload or {}),
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT forecast_id, selected_policy, point_prediction,
                       p10, p50, p90, issued_at
                FROM {FORECAST_TABLE}
                WHERE as_of_time=%s AND horizon_h=%s
                  AND forecast_version=%s
                """,
                (
                    as_of.to_pydatetime(),
                    horizon_h,
                    FORECAST_VERSION,
                ),
            )
            existing = cursor.fetchone()
            if existing is not None:
                stored = np.asarray(existing[2:6], dtype="float64")
                if (
                    str(existing[1]) != selected_policy
                    or not np.allclose(stored, values, atol=1e-12, rtol=0)
                ):
                    raise RuntimeError(
                        "Immutable forecast already exists with different values"
                    )
                return {
                    "status": "SUCCESS",
                    "forecast_id": str(existing[0]),
                    "reused": True,
                    "target_time": target.isoformat(),
                }
            cursor.execute(
                f"""
                INSERT INTO {FORECAST_TABLE} (
                    forecast_id, as_of_time, target_time, horizon_h,
                    forecast_version, selected_policy, point_prediction,
                    p10, p50, p90, issued_at, source_snapshot_time,
                    source_mode, payload
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, 'PROSPECTIVE_SHADOW', %s
                )
                """,
                (
                    forecast_id,
                    as_of.to_pydatetime(),
                    target.to_pydatetime(),
                    horizon_h,
                    FORECAST_VERSION,
                    selected_policy,
                    float(point_prediction),
                    float(p10),
                    float(p50),
                    float(p90),
                    issued.to_pydatetime(),
                    snapshot.to_pydatetime(),
                    Json(row_payload),
                ),
            )
    return {
        "status": "SUCCESS",
        "forecast_id": forecast_id,
        "reused": False,
        "target_time": target.isoformat(),
    }


def register_shadow_observation(
    *,
    forecast_id: str,
    actual_arrivals: float,
    source: str,
    source_watermark: datetime | str,
    available_at: datetime | str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_schema()
    availability = _utc(available_at)
    watermark = _utc(source_watermark)
    now = pd.Timestamp.now(tz="UTC")
    if availability > now + pd.Timedelta(minutes=5):
        raise ValueError("available_at cannot be in the future")
    if watermark > availability:
        raise ValueError("source_watermark cannot follow available_at")
    if not np.isfinite(actual_arrivals) or actual_arrivals < 0:
        raise ValueError("actual_arrivals must be finite and nonnegative")

    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT as_of_time, target_time, horizon_h, issued_at
                FROM {FORECAST_TABLE}
                WHERE forecast_id=%s AND source_mode='PROSPECTIVE_SHADOW'
                """,
                (forecast_id,),
            )
            forecast = cursor.fetchone()
            if forecast is None:
                raise ValueError("Unknown prospective forecast_id")
            as_of, target, horizon, issued = forecast
            if availability < pd.Timestamp(target):
                raise ValueError("Observation is not mature yet")
            if watermark < pd.Timestamp(target):
                raise ValueError("Source watermark has not reached target_time")
            if availability <= pd.Timestamp(issued):
                raise ValueError("Observation must become available after issue")
            observation_id = str(uuid.uuid4())
            cursor.execute(
                f"""
                INSERT INTO {OBSERVATION_TABLE} (
                    observation_id, forecast_id, as_of_time, target_time,
                    horizon_h, actual_arrivals, source, source_watermark,
                    available_at, payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (forecast_id, source, source_watermark)
                DO NOTHING
                RETURNING observation_id
                """,
                (
                    observation_id,
                    forecast_id,
                    as_of,
                    target,
                    horizon,
                    float(actual_arrivals),
                    source,
                    watermark.to_pydatetime(),
                    availability.to_pydatetime(),
                    Json(payload or {}),
                ),
            )
            inserted = cursor.fetchone()
    return {
        "status": "SUCCESS",
        "observation_id": (
            observation_id if inserted is not None else None
        ),
        "reused": inserted is None,
    }


def _capture_mature_observations(as_of: pd.Timestamp) -> dict[str, Any]:
    _ensure_schema()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT max(available_at)
                FROM lineage.source_event
                WHERE source_table='core.port_call'
                """
            )
            value = cursor.fetchone()[0]
            if value is None:
                return {
                    "watermark": None,
                    "eligible_forecasts": 0,
                    "inserted_observations": 0,
                    "status": "NO_B56D1_PORT_CALL_WATERMARK",
                }
            watermark = min(pd.Timestamp(value), as_of)
            cutoff = watermark - pd.Timedelta(
                hours=LABEL_SETTLEMENT_HOURS
            )
            cursor.execute(
                f"""
                SELECT f.forecast_id, f.as_of_time, f.target_time,
                       f.horizon_h
                FROM {FORECAST_TABLE} f
                WHERE f.source_mode='PROSPECTIVE_SHADOW'
                  AND f.target_time <= %s
                ORDER BY f.target_time, f.horizon_h
                """,
                (cutoff.to_pydatetime(),),
            )
            forecasts = cursor.fetchall()
            values: list[tuple[Any, ...]] = []
            for forecast_id, start, target, horizon in forecasts:
                cursor.execute(
                    """
                    SELECT count(*)::double precision
                    FROM core.port_call
                    WHERE actual_ata >= %s AND actual_ata < %s
                    """,
                    (start, target),
                )
                actual = float(cursor.fetchone()[0])
                cursor.execute(
                    f"""
                    SELECT actual_arrivals
                    FROM {OBSERVATION_TABLE}
                    WHERE forecast_id=%s
                    ORDER BY available_at DESC, received_at DESC
                    LIMIT 1
                    """,
                    (forecast_id,),
                )
                latest = cursor.fetchone()
                if latest is not None and float(latest[0]) == actual:
                    continue
                values.append(
                    (
                        str(uuid.uuid4()),
                        forecast_id,
                        start,
                        target,
                        horizon,
                        actual,
                        "B56D1_CORE_PORT_CALL_WATERMARK",
                        watermark.to_pydatetime(),
                        as_of.to_pydatetime(),
                        Json(
                            {
                                "settlement_hours": LABEL_SETTLEMENT_HOURS,
                                "capture_version": MONITOR_VERSION,
                            }
                        ),
                    )
                )
            if values:
                execute_values(
                    cursor,
                    f"""
                    INSERT INTO {OBSERVATION_TABLE} (
                        observation_id, forecast_id, as_of_time,
                        target_time, horizon_h, actual_arrivals, source,
                        source_watermark, available_at, payload
                    ) VALUES %s
                    ON CONFLICT (
                        forecast_id, source, source_watermark
                    ) DO NOTHING
                    """,
                    values,
                    page_size=1000,
                )
    return {
        "watermark": watermark.isoformat(),
        "eligible_forecasts": len(forecasts),
        "inserted_observations": len(values),
        "status": "CAPTURED",
    }


def _inventory(as_of: pd.Timestamp) -> pd.DataFrame:
    _ensure_schema()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    f.horizon_h,
                    count(DISTINCT f.forecast_id) AS forecasts,
                    count(DISTINCT f.forecast_id) FILTER (
                        WHERE f.target_time <= %s
                    ) AS time_mature_forecasts,
                    count(DISTINCT o.forecast_id) AS observed_forecasts,
                    min(f.as_of_time) AS first_forecast,
                    max(f.as_of_time) AS last_forecast,
                    max(o.available_at) AS last_observation_available_at
                FROM {FORECAST_TABLE} f
                LEFT JOIN {OBSERVATION_TABLE} o
                  ON o.forecast_id=f.forecast_id
                 AND o.available_at <= %s
                WHERE f.forecast_version=%s
                  AND f.source_mode='PROSPECTIVE_SHADOW'
                GROUP BY f.horizon_h
                ORDER BY f.horizon_h
                """,
                (
                    as_of.to_pydatetime(),
                    as_of.to_pydatetime(),
                    FORECAST_VERSION,
                ),
            )
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


def _paired_data(as_of: pd.Timestamp) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                WITH latest_observation AS (
                    SELECT DISTINCT ON (forecast_id)
                        forecast_id, actual_arrivals, source,
                        source_watermark, available_at
                    FROM {OBSERVATION_TABLE}
                    WHERE available_at <= %s
                    ORDER BY forecast_id, available_at DESC, received_at DESC
                )
                SELECT
                    f.forecast_id, f.as_of_time, f.target_time,
                    f.horizon_h, f.selected_policy, f.point_prediction,
                    f.p10, f.p50, f.p90, f.issued_at,
                    f.source_snapshot_time, o.actual_arrivals,
                    o.source AS observation_source,
                    o.source_watermark, o.available_at
                FROM {FORECAST_TABLE} f
                JOIN latest_observation o USING (forecast_id)
                WHERE f.forecast_version=%s
                  AND f.source_mode='PROSPECTIVE_SHADOW'
                  AND f.issued_at < o.available_at
                  AND f.target_time <= o.source_watermark
                ORDER BY f.as_of_time, f.horizon_h
                """,
                (as_of.to_pydatetime(), FORECAST_VERSION),
            )
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    for column in (
        "as_of_time",
        "target_time",
        "issued_at",
        "source_snapshot_time",
        "source_watermark",
        "available_at",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _window(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    if frame.empty or days == 0:
        return frame
    end = frame["as_of_time"].max()
    return frame.loc[
        frame["as_of_time"] > end - pd.Timedelta(days=days)
    ]


def _metric_row(
    frame: pd.DataFrame,
    horizon: int,
    window_days: int,
) -> dict[str, Any]:
    part = frame.loc[frame["horizon_h"].eq(horizon)].copy()
    part = _window(part, window_days)
    if part.empty:
        return {
            "horizon_h": horizon,
            "window_days": window_days,
            "window": "FULL" if window_days == 0 else f"LAST_{window_days}D",
            "n": 0,
            "calendar_days": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias_prediction_minus_actual": np.nan,
            "coverage_p10_p90": np.nan,
            "below_p10_rate": np.nan,
            "above_p90_rate": np.nan,
            "mean_interval_width": np.nan,
            "winkler_interval_score": np.nan,
            "coverage_gate_passed": False,
            "lower_tail_gate_passed": False,
            "upper_tail_gate_passed": False,
            "point_mae_gate_passed": False,
            "all_quality_gates_passed": False,
        }
    actual = part["actual_arrivals"].to_numpy(dtype="float64")
    point = part["point_prediction"].to_numpy(dtype="float64")
    low = part["p10"].to_numpy(dtype="float64")
    high = part["p90"].to_numpy(dtype="float64")
    error = point - actual
    below = float(np.mean(actual < low))
    above = float(np.mean(actual > high))
    coverage = 1.0 - below - above
    alpha = 1.0 - EXPECTED_COVERAGE
    interval_score = (
        high
        - low
        + (2.0 / alpha) * (low - actual) * (actual < low)
        + (2.0 / alpha) * (actual - high) * (actual > high)
    )
    if len(part) <= 1:
        days = 0.0
    else:
        days = float(
            (
                part["as_of_time"].max() - part["as_of_time"].min()
            ).total_seconds()
            / 86400.0
        )
    coverage_gate = COVERAGE_MIN <= coverage <= COVERAGE_MAX
    lower_gate = TAIL_MIN <= below <= TAIL_MAX
    upper_gate = TAIL_MIN <= above <= TAIL_MAX
    point_gate = float(np.mean(np.abs(error))) <= MAE_LIMIT[horizon]
    return {
        "horizon_h": horizon,
        "window_days": window_days,
        "window": "FULL" if window_days == 0 else f"LAST_{window_days}D",
        "n": len(part),
        "calendar_days": days,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias_prediction_minus_actual": float(np.mean(error)),
        "coverage_p10_p90": coverage,
        "below_p10_rate": below,
        "above_p90_rate": above,
        "mean_interval_width": float(np.mean(high - low)),
        "winkler_interval_score": float(np.mean(interval_score)),
        "coverage_gate_passed": coverage_gate,
        "lower_tail_gate_passed": lower_gate,
        "upper_tail_gate_passed": upper_gate,
        "point_mae_gate_passed": point_gate,
        "all_quality_gates_passed": bool(
            coverage_gate and lower_gate and upper_gate and point_gate
        ),
    }


def _rolling_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _metric_row(frame, horizon, window_days)
            for horizon in HORIZONS
            for window_days in (0, *ROLLING_WINDOWS_DAYS)
        ]
    )


def _contract_checks(
    forecasts: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    forecast_count = int(forecasts["forecasts"].sum()) if not forecasts.empty else 0
    observed_count = (
        int(forecasts["observed_forecasts"].sum())
        if not forecasts.empty
        else 0
    )
    invalid_temporal = 0
    invalid_source = 0
    if not paired.empty:
        invalid_temporal = int(
            (
                (paired["issued_at"] >= paired["available_at"])
                | (paired["target_time"] > paired["source_watermark"])
                | (
                    paired["source_snapshot_time"]
                    > paired["issued_at"]
                )
            ).sum()
        )
        invalid_source = int(
            paired["observation_source"].eq("HISTORICAL_BACKFILL").sum()
        )
    return pd.DataFrame(
        [
            {
                "check": "forecast_source_is_prospective_only",
                "value": forecast_count,
                "passed": True,
            },
            {
                "check": "observations_are_post_issue_and_mature",
                "value": invalid_temporal,
                "passed": invalid_temporal == 0,
            },
            {
                "check": "historical_backfill_is_forbidden",
                "value": invalid_source,
                "passed": invalid_source == 0,
            },
            {
                "check": "observed_forecasts_not_above_forecasts",
                "value": f"{observed_count}/{forecast_count}",
                "passed": observed_count <= forecast_count,
            },
        ]
    )


def _source_checksum(as_of: pd.Timestamp) -> str:
    _ensure_schema()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT count(*), max(issued_at)
                FROM {FORECAST_TABLE}
                WHERE forecast_version=%s
                """,
                (FORECAST_VERSION,),
            )
            forecasts = cursor.fetchone()
            cursor.execute(
                f"""
                SELECT count(*), max(available_at)
                FROM {OBSERVATION_TABLE}
                WHERE available_at <= %s
                """,
                (as_of.to_pydatetime(),),
            )
            observations = cursor.fetchone()
    payload = {
        "version": MONITOR_VERSION,
        "forecast_rows": int(forecasts[0]),
        "last_issued_at": forecasts[1],
        "observation_rows": int(observations[0]),
        "last_available_at": observations[1],
    }
    return hashlib.sha256(
        json.dumps(
            payload, default=_json_default, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, metadata
                FROM audit.ingestion_run
                WHERE source_name=%s AND dataset_name=%s
                  AND checksum=%s AND status='SUCCESS'
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _start_run(checksum: str) -> str:
    metadata = {
        "monitor_version": MONITOR_VERSION,
        "forecast_version": FORECAST_VERSION,
        "policy": (
            "PROSPECTIVE_ONLY_IMMUTABLE_FORECAST_LEDGER_"
            "MATURED_AVAILABLE_AT_LABELS_NO_TEST_REUSE"
        ),
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit.ingestion_run (
                    source_name, dataset_name, object_uri, checksum, metadata
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    SOURCE_NAME,
                    DATASET_NAME,
                    f"timescaledb://{FORECAST_TABLE}",
                    checksum,
                    Json(metadata),
                ),
            )
            return str(cursor.fetchone()[0])


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at=now(), status=%s, row_count=%s,
                    metadata=metadata || %s, error_message=%s
                WHERE run_id=%s
                """,
                (
                    status,
                    row_count,
                    Json(
                        _clean_json(metadata),
                        dumps=lambda value: json.dumps(
                            value, default=_json_default, allow_nan=False
                        ),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _decision(
    inventory: pd.DataFrame,
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
    checks: pd.DataFrame,
    capture: dict[str, Any],
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    forecast_rows = (
        int(inventory["forecasts"].sum()) if not inventory.empty else 0
    )
    observed_rows = (
        int(inventory["observed_forecasts"].sum())
        if not inventory.empty
        else 0
    )
    contract_passed = bool(checks["passed"].all())
    full = metrics.loc[metrics["window_days"].eq(0)].copy()
    evidence_passed = bool(
        not full.empty
        and (full["n"] >= MIN_PROMOTION_ROWS_PER_HORIZON).all()
        and (full["calendar_days"] >= MIN_PROMOTION_DAYS).all()
    )
    quality_passed = bool(
        evidence_passed and full["all_quality_gates_passed"].all()
    )

    if not contract_passed:
        status = "NEED_SHADOW_CONTRACT_REPAIR"
    elif forecast_rows == 0:
        status = "WAITING_FOR_PROSPECTIVE_FORECASTS"
    elif observed_rows == 0:
        status = "WAITING_FOR_MATURE_LABELS"
    elif not evidence_passed:
        status = "COLLECTING_PROSPECTIVE_EVIDENCE"
    elif not quality_passed:
        status = "PROSPECTIVE_SHADOW_WARNING"
    else:
        status = "READY_FOR_CONTROLLED_CANARY"

    return {
        "status": status,
        "monitor_version": MONITOR_VERSION,
        "forecast_version": FORECAST_VERSION,
        "as_of": as_of.isoformat(),
        "forecast_rows": forecast_rows,
        "observed_forecasts": observed_rows,
        "paired_rows": len(paired),
        "contract_gates_passed": contract_passed,
        "minimum_evidence_passed": evidence_passed,
        "quality_gates_passed": quality_passed,
        "minimum_days": MIN_PROMOTION_DAYS,
        "minimum_rows_per_horizon": MIN_PROMOTION_ROWS_PER_HORIZON,
        "coverage_gate": [COVERAGE_MIN, COVERAGE_MAX],
        "tail_gate": [TAIL_MIN, TAIL_MAX],
        "reference_mae": REFERENCE_MAE,
        "mae_limits": MAE_LIMIT,
        "automatic_observation_capture": capture,
        "training_executed": False,
        "selection_used_test": False,
        "historical_test_reused_as_prospective": False,
        "controlled_canary_allowed": status == "READY_FOR_CONTROLLED_CANARY",
        "formal_production_promotion_allowed": False,
        "formal_promotion_blocker": (
            "REQUIRE_CONTROLLED_CANARY_AFTER_PROSPECTIVE_SHADOW"
        ),
        "next_block": (
            "B56G_V21_SHADOW_CONTRACT_REPAIR"
            if status == "NEED_SHADOW_CONTRACT_REPAIR"
            else "B56G_V21_LIVE_FORECAST_ISSUER"
            if forecast_rows == 0
            else "CONTINUE_PROSPECTIVE_SHADOW_COLLECTION"
            if status
            in {
                "WAITING_FOR_MATURE_LABELS",
                "COLLECTING_PROSPECTIVE_EVIDENCE",
                "PROSPECTIVE_SHADOW_WARNING",
            }
            else "B56G_V21_CONTROLLED_CANARY"
            if status == "READY_FOR_CONTROLLED_CANARY"
            else "B56G_V21_SHADOW_REVIEW"
        ),
    }


def _log_mlflow(
    output_dir: Path,
    decision: dict[str, Any],
    metrics: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment(
            "maritime-arrival-flow-prospective-shadow"
        )
        with mlflow.start_run(run_name=MONITOR_VERSION):
            mlflow.log_params(
                {
                    "version": MONITOR_VERSION,
                    "forecast_version": FORECAST_VERSION,
                    "status": decision["status"],
                }
            )
            values: dict[str, float] = {}
            full = metrics.loc[metrics["window_days"].eq(0)]
            for row in full.itertuples(index=False):
                if int(row.n) == 0:
                    continue
                horizon = int(row.horizon_h)
                values[f"prospective_mae_{horizon}h"] = float(row.mae)
                values[f"prospective_coverage_{horizon}h"] = float(
                    row.coverage_p10_p90
                )
            if values:
                mlflow.log_metrics(values)
            mlflow.log_artifacts(str(output_dir))
        return "LOGGED"
    except Exception as exc:
        return f"SKIPPED:{type(exc).__name__}"


def run_b56g_v21_shadow_monitor(
    *,
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    as_of: datetime | str | None = None,
    auto_capture_observations: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    _load_calibration_decision()
    _ensure_schema()
    as_of_time = _utc(as_of)
    capture = (
        _capture_mature_observations(as_of_time)
        if auto_capture_observations
        else {
            "status": "DISABLED",
            "watermark": None,
            "eligible_forecasts": 0,
            "inserted_observations": 0,
        }
    )
    checksum = _source_checksum(as_of_time)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        run_id, metadata = previous
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": True,
            "results": metadata,
        }

    run_id = _start_run(checksum)
    try:
        inventory = _inventory(as_of_time)
        paired = _paired_data(as_of_time)
        metrics = _rolling_metrics(paired)
        checks = _contract_checks(inventory, paired)
        decision = _decision(
            inventory, paired, metrics, checks, capture, as_of_time
        )

        with tempfile.TemporaryDirectory(
            prefix="b56g-v21-shadow-"
        ) as temporary:
            output_dir = Path(temporary)
            reports_dir = output_dir / "reports"
            configs_dir = output_dir / "configs"
            reports_dir.mkdir(parents=True)
            configs_dir.mkdir(parents=True)

            checks.to_csv(
                reports_dir / "00_prospective_contract_checks.csv",
                index=False,
            )
            inventory.to_csv(
                reports_dir / "01_shadow_inventory.csv", index=False
            )
            metrics.to_csv(
                reports_dir / "02_rolling_shadow_metrics.csv", index=False
            )
            paired.to_csv(
                reports_dir / "03_matured_forecast_observations.csv",
                index=False,
            )
            (reports_dir / "README_B56G_V21_SHADOW.md").write_text(
                "\n".join(
                    [
                        "# B56G-v2.1 prospective shadow monitor",
                        "",
                        f"Decision: {decision['status']}",
                        "",
                        "- Historical TEST rows are never prospective evidence.",
                        "- Forecasts are immutable and registered at issue time.",
                        "- Labels require a post-target available_at watermark.",
                        "- Promotion requires 30 days and 500 rows per horizon.",
                        "- Passing shadow permits only a controlled canary.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = (
                configs_dir / "04_b56g_v21_shadow_decision.json"
            )
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    allow_nan=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            mlflow_status = _log_mlflow(
                output_dir, decision, metrics
            )

            client = _s3_client()
            outputs: dict[str, str] = {}
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(output_dir).as_posix()
                category, name = relative.split("/", 1)
                key = (
                    f"{category}/b56gv21-shadow/"
                    f"{output_prefix.strip('/')}/{name}"
                )
                outputs[name] = _upload(
                    client, path, output_bucket, key
                )

        decision.update(
            {
                "mlflow_status": mlflow_status,
                "outputs": outputs,
                "checksum": checksum,
            }
        )
        _finish_run(run_id, "SUCCESS", len(paired), decision)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "results": _clean_json(decision),
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {"monitor_version": MONITOR_VERSION},
            error_message=str(exc),
        )
        raise
