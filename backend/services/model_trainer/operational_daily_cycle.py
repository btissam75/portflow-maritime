from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extras import Json

from model_trainer.adaptive_recalibration import (
    API_VERSION as B57E_API_VERSION,
    forecast_b57e_daily,
    initialize_b57e,
    recalibrate_b57e,
    register_b57e_observations,
    runtime_status_b57e,
)


CYCLE_VERSION = "b57f-operational-daily-cycle-v1"
SOURCE_NAME = "b57f_operational_daily_cycle"
DATASET_NAME = "tir_prequential_daily_collection"
DEFAULT_BUCKET = "gold-maritime"
STABILITY_OBSERVATIONS_REQUIRED = 2
MIN_DURATION_LABEL_RATE = 0.50
SETTLEMENT_HOUR_UTC = 6
TARGET_MAP = {
    "TIR_VOLUME": "target_tir_rows",
    "DURATION_MEDIAN": "target_duration_median_h",
    "LONG_24H_RATE": "target_long_24h_rate",
}


def _db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, allow_nan=False)


def _ensure_schema() -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE SCHEMA IF NOT EXISTS serving;

                CREATE TABLE IF NOT EXISTS serving.tir_operational_cycle_state (
                    cycle_version TEXT PRIMARY KEY,
                    initialized_at TIMESTAMPTZ NOT NULL,
                    baseline_cutoff_date DATE NOT NULL,
                    last_run_at TIMESTAMPTZ,
                    last_source_observed_date DATE,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS
                    serving.tir_daily_outcome_candidate (
                    prediction_date DATE PRIMARY KEY,
                    first_seen_at TIMESTAMPTZ NOT NULL,
                    payload_first_seen_at TIMESTAMPTZ NOT NULL,
                    last_seen_at TIMESTAMPTZ NOT NULL,
                    payload_hash TEXT NOT NULL,
                    stable_observations INTEGER NOT NULL,
                    target_payload JSONB NOT NULL,
                    quality_payload JSONB NOT NULL,
                    prior_operational_forecast BOOLEAN NOT NULL DEFAULT false,
                    status TEXT NOT NULL,
                    registered_at TIMESTAMPTZ,
                    registered_payload_hash TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                ALTER TABLE serving.tir_daily_outcome_candidate
                    ADD COLUMN IF NOT EXISTS registered_payload_hash TEXT;

                ALTER TABLE serving.tir_daily_outcome_candidate
                    ADD COLUMN IF NOT EXISTS payload_first_seen_at TIMESTAMPTZ;

                UPDATE serving.tir_daily_outcome_candidate
                SET payload_first_seen_at=first_seen_at
                WHERE payload_first_seen_at IS NULL;

                ALTER TABLE serving.tir_daily_outcome_candidate
                    ALTER COLUMN payload_first_seen_at SET NOT NULL;
                """
            )


def _start_audit_run() -> str:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit.ingestion_run (
                    source_name, dataset_name, object_uri, checksum, metadata
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    SOURCE_NAME,
                    DATASET_NAME,
                    "timescaledb://features.tir_daily_event_aware_v1",
                    CYCLE_VERSION,
                    Json(
                        {
                            "cycle_version": CYCLE_VERSION,
                            "policy": (
                                "FUTURE_ROW_FIRST_STABLE_OUTCOME_TWICE_"
                                "FORECAST_PRECEDES_AVAILABILITY_NO_BACKFILL"
                            ),
                        }
                    ),
                ),
            )
            return str(cursor.fetchone()[0])


def _finish_audit_run(
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
                SET status=%s, row_count=%s, finished_at=now(),
                    metadata=metadata || %s, error_message=%s
                WHERE run_id=%s
                """,
                (
                    status,
                    row_count,
                    Json(metadata, dumps=_json_dumps),
                    error_message,
                    run_id,
                ),
            )


def _source_summary() -> dict[str, Any]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH boundary AS (
                    SELECT max(prediction_date) FILTER (
                        WHERE COALESCE(
                            (
                                quality_flags
                                ->>'tir_source_day_observed_flag'
                            )::int,
                            0
                        ) = 1
                    ) AS last_observed_date
                    FROM features.tir_daily_event_aware_v1
                )
                SELECT
                    min(prediction_date),
                    max(prediction_date),
                    boundary.last_observed_date,
                    count(*),
                    count(*) FILTER (
                        WHERE prediction_date > boundary.last_observed_date
                    )
                FROM features.tir_daily_event_aware_v1
                CROSS JOIN boundary
                GROUP BY boundary.last_observed_date
                """
            )
            row = cursor.fetchone()
    if row is None or row[2] is None:
        raise RuntimeError("B57B materialized daily source is empty")
    return {
        "first_date": row[0],
        "last_feature_date": row[1],
        "last_observed_date": row[2],
        "rows": int(row[3]),
        "future_rows": int(row[4]),
    }


def _initialize_state(
    now: pd.Timestamp,
    source: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT initialized_at, baseline_cutoff_date,
                       last_run_at, last_source_observed_date, metadata
                FROM serving.tir_operational_cycle_state
                WHERE cycle_version=%s
                """,
                (CYCLE_VERSION,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """
                    INSERT INTO serving.tir_operational_cycle_state (
                        cycle_version, initialized_at, baseline_cutoff_date,
                        last_run_at, last_source_observed_date, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        CYCLE_VERSION,
                        now.to_pydatetime(),
                        source["last_observed_date"],
                        now.to_pydatetime(),
                        source["last_observed_date"],
                        Json(
                            {
                                "historical_rows_imported": False,
                                "baseline_policy": (
                                    "EXCLUDE_ALL_ROWS_OBSERVED_BEFORE_FIRST_RUN"
                                ),
                            }
                        ),
                    ),
                )
                return {
                    "initialized_at": now,
                    "baseline_cutoff_date": source["last_observed_date"],
                    "last_run_at": now,
                    "last_source_observed_date": source["last_observed_date"],
                }, True
    return {
        "initialized_at": row[0],
        "baseline_cutoff_date": row[1],
        "last_run_at": row[2],
        "last_source_observed_date": row[3],
        "metadata": dict(row[4] or {}),
    }, False


def _new_observed_rows(baseline_cutoff_date) -> list[dict[str, Any]]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    prediction_date,
                    targets,
                    quality_flags,
                    updated_at
                FROM features.tir_daily_event_aware_v1
                WHERE prediction_date > %s
                  AND prediction_date < (now() AT TIME ZONE 'UTC')::date
                  AND COALESCE(
                        (quality_flags->>'tir_source_day_observed_flag')::int,
                        0
                      ) = 1
                ORDER BY prediction_date
                """,
                (baseline_cutoff_date,),
            )
            rows = cursor.fetchall()
    return [
        {
            "prediction_date": row[0],
            "targets": dict(row[1] or {}),
            "quality": dict(row[2] or {}),
            "source_updated_at": row[3],
        }
        for row in rows
    ]


def _finite_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_payload(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    targets = row["targets"]
    values = {
        target_name: _finite_value(targets, source_column)
        for target_name, source_column in TARGET_MAP.items()
    }
    duration_label_rate = _finite_value(targets, "target_duration_label_rate")
    quality = {
        "duration_label_rate": duration_label_rate,
        "source_updated_at": row["source_updated_at"],
        "tir_source_day_observed_flag": row["quality"].get(
            "tir_source_day_observed_flag"
        ),
    }
    return values, quality


def _payload_hash(values: dict[str, Any], quality: dict[str, Any]) -> str:
    stable_quality = {
        key: value for key, value in quality.items() if key != "source_updated_at"
    }
    payload = _json_dumps({"values": values, "quality": stable_quality})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _has_prior_operational_forecast(
    prediction_date,
    available_at: pd.Timestamp,
) -> bool:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(DISTINCT target_name) = 3
                FROM serving.tir_daily_forecast_ledger
                WHERE api_version=%s
                  AND prediction_date::date=%s
                  AND operating_mode='OPERATIONAL'
                  AND issued_at < %s
                """,
                (
                    B57E_API_VERSION,
                    prediction_date,
                    available_at.to_pydatetime(),
                ),
            )
            return bool(cursor.fetchone()[0])


def _update_candidate(
    row: dict[str, Any],
    now: pd.Timestamp,
) -> dict[str, Any]:
    values, quality = _candidate_payload(row)
    digest = _payload_hash(values, quality)
    missing = [name for name, value in values.items() if value is None]
    label_rate = quality["duration_label_rate"]
    settlement_at = pd.Timestamp(row["prediction_date"], tz="UTC") + pd.Timedelta(
        days=1, hours=SETTLEMENT_HOUR_UTC
    )

    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload_hash, stable_observations, registered_at,
                       registered_payload_hash, payload_first_seen_at
                FROM serving.tir_daily_outcome_candidate
                WHERE prediction_date=%s
                """,
                (row["prediction_date"],),
            )
            existing = cursor.fetchone()
            stable = (
                int(existing[1]) + 1
                if existing is not None and existing[0] == digest
                else 1
            )
            registered_at = None if existing is None else existing[2]
            registered_hash = None if existing is None else existing[3]
            payload_first_seen_at = (
                existing[4]
                if existing is not None and existing[0] == digest
                else now.to_pydatetime()
            )
            payload_available_at = pd.Timestamp(payload_first_seen_at)
            if payload_available_at.tzinfo is None:
                payload_available_at = payload_available_at.tz_localize("UTC")
            else:
                payload_available_at = payload_available_at.tz_convert("UTC")
            prior_forecast = _has_prior_operational_forecast(
                row["prediction_date"],
                payload_available_at,
            )

            if registered_at is not None and registered_hash == digest:
                status = "REGISTERED"
            elif missing:
                status = "MISSING_TARGETS"
            elif label_rate is None or label_rate < MIN_DURATION_LABEL_RATE:
                status = "INSUFFICIENT_DURATION_LABEL_RATE"
            elif now < settlement_at:
                status = "WAITING_FOR_SETTLEMENT"
            elif stable < STABILITY_OBSERVATIONS_REQUIRED:
                status = (
                    "WAITING_FOR_STABILITY_REVISION"
                    if registered_at is not None
                    else "WAITING_FOR_STABILITY"
                )
            elif not prior_forecast:
                status = "NO_PRIOR_OPERATIONAL_FORECAST"
            else:
                status = "READY_TO_REGISTER"

            cursor.execute(
                """
                INSERT INTO serving.tir_daily_outcome_candidate (
                    prediction_date, first_seen_at, payload_first_seen_at,
                    last_seen_at,
                    payload_hash, stable_observations, target_payload,
                    quality_payload, prior_operational_forecast, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_date) DO UPDATE SET
                    last_seen_at=EXCLUDED.last_seen_at,
                    payload_first_seen_at=EXCLUDED.payload_first_seen_at,
                    payload_hash=EXCLUDED.payload_hash,
                    stable_observations=EXCLUDED.stable_observations,
                    target_payload=EXCLUDED.target_payload,
                    quality_payload=EXCLUDED.quality_payload,
                    prior_operational_forecast=
                        EXCLUDED.prior_operational_forecast,
                    status=CASE
                        WHEN serving.tir_daily_outcome_candidate.registered_at
                             IS NOT NULL
                        THEN 'REGISTERED'
                        ELSE EXCLUDED.status
                    END,
                    updated_at=now()
                """,
                (
                    row["prediction_date"],
                    now.to_pydatetime(),
                    payload_available_at.to_pydatetime(),
                    now.to_pydatetime(),
                    digest,
                    stable,
                    Json(values, dumps=_json_dumps),
                    Json(quality, dumps=_json_dumps),
                    prior_forecast,
                    status,
                ),
            )
    return {
        "prediction_date": str(row["prediction_date"]),
        "status": status,
        "stable_observations": stable,
        "prior_operational_forecast": prior_forecast,
        "values": values,
        "payload_hash": digest,
        "available_at": payload_available_at.isoformat(),
    }


def _mark_registered(
    prediction_date,
    registered_at: pd.Timestamp,
    payload_hash: str,
) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE serving.tir_daily_outcome_candidate
                SET status='REGISTERED', registered_at=%s,
                    registered_payload_hash=%s, updated_at=now()
                WHERE prediction_date=%s
                """,
                (
                    registered_at.to_pydatetime(),
                    payload_hash,
                    prediction_date,
                ),
            )


def _collect_eligible_outcomes(
    state: dict[str, Any],
    now: pd.Timestamp,
) -> dict[str, Any]:
    rows = _new_observed_rows(state["baseline_cutoff_date"])
    candidates = [_update_candidate(row, now) for row in rows]
    registered = []
    for candidate in candidates:
        if candidate["status"] != "READY_TO_REGISTER":
            continue
        result = register_b57e_observations(
            prediction_date=candidate["prediction_date"],
            available_at=candidate["available_at"],
            source="B57F_TIR_DAILY_GOLD_STABLE_2",
            values=candidate["values"],
        )
        _mark_registered(
            candidate["prediction_date"],
            now,
            candidate["payload_hash"],
        )
        registered.append(
            {
                "prediction_date": candidate["prediction_date"],
                "inserted": result["inserted"],
                "duplicates": result["duplicates"],
            }
        )
    return {
        "scanned_rows": len(rows),
        "candidate_statuses": {
            status: sum(item["status"] == status for item in candidates)
            for status in sorted({item["status"] for item in candidates})
        },
        "registered_days": len(registered),
        "registered": registered,
    }


def _update_state(now: pd.Timestamp, source: dict[str, Any]) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE serving.tir_operational_cycle_state
                SET last_run_at=%s, last_source_observed_date=%s,
                    metadata=metadata || %s
                WHERE cycle_version=%s
                """,
                (
                    now.to_pydatetime(),
                    source["last_observed_date"],
                    Json(
                        {
                            "last_feature_date": str(source["last_feature_date"]),
                            "future_rows": source["future_rows"],
                        }
                    ),
                    CYCLE_VERSION,
                ),
            )


def run_b57f_operational_cycle(
    artifact_bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    _ensure_schema()
    run_id = _start_audit_run()
    now = pd.Timestamp.now(tz="UTC")
    try:
        source = _source_summary()
        if int(source["future_rows"]) != 1:
            raise RuntimeError("B57B must expose exactly one unobserved future row")
        expected_future = source["last_observed_date"] + timedelta(days=1)
        if source["last_feature_date"] != expected_future:
            raise RuntimeError(
                "B57B future row is not exactly one day after observed data"
            )

        state, initialized_now = _initialize_state(now, source)
        collection = (
            {
                "scanned_rows": 0,
                "candidate_statuses": {"BASELINE_INITIALIZED": 1},
                "registered_days": 0,
                "registered": [],
            }
            if initialized_now
            else _collect_eligible_outcomes(state, now)
        )

        initialization = initialize_b57e(
            artifact_bucket=artifact_bucket,
            output_bucket=artifact_bucket,
            output_prefix="version=1",
            force=False,
        )
        runtime = runtime_status_b57e()
        forecast_date = str(source["last_feature_date"])
        forecast = forecast_b57e_daily(forecast_date)
        recalibration = recalibrate_b57e()
        _update_state(now, source)

        decision = (
            "LIVE_CYCLE_ACTIVE"
            if forecast["mode"] == "OPERATIONAL"
            else "LIVE_CYCLE_INSTALLED_WAITING_FRESH_SOURCE"
        )
        metadata = {
            "status": decision,
            "cycle_version": CYCLE_VERSION,
            "baseline_initialized_now": initialized_now,
            "baseline_cutoff_date": state["baseline_cutoff_date"],
            "source": source,
            "collection": collection,
            "forecast_date": forecast_date,
            "forecast_mode": forecast["mode"],
            "b57e_decision": runtime["decision"],
            "adaptive_targets": runtime["adaptive_target_count"],
            "registered_days": collection["registered_days"],
            "training_executed": False,
            "historical_backfill_used": False,
            "target_exposed_by_forecast": False,
            "gates_passed": True,
            "next_block": (
                "B57F_ACCUMULATE_ELIGIBLE_LIVE_DAYS"
                if forecast["mode"] == "OPERATIONAL"
                else "B57F_REFRESH_UPSTREAM_TIR_SOURCE"
            ),
        }
        _finish_audit_run(
            run_id,
            "SUCCESS",
            int(source["rows"]),
            metadata,
        )
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "decision": decision,
            "results": metadata,
            "initialization": {
                "status": initialization.get("status"),
                "reused": initialization.get("reused"),
            },
            "forecast": forecast,
            "recalibration": recalibration,
        }
    except Exception as exc:
        _finish_audit_run(
            run_id,
            "FAILED",
            None,
            {"cycle_version": CYCLE_VERSION},
            error_message=str(exc),
        )
        raise


def monitoring_b57f() -> dict[str, Any]:
    _ensure_schema()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT initialized_at, baseline_cutoff_date, last_run_at,
                       last_source_observed_date, metadata
                FROM serving.tir_operational_cycle_state
                WHERE cycle_version=%s
                """,
                (CYCLE_VERSION,),
            )
            state = cursor.fetchone()
            cursor.execute(
                """
                SELECT status, count(*), min(prediction_date),
                       max(prediction_date), max(last_seen_at)
                FROM serving.tir_daily_outcome_candidate
                GROUP BY status
                ORDER BY status
                """
            )
            candidates = cursor.fetchall()
    return {
        "status": "READY" if state else "NOT_INITIALIZED",
        "cycle_version": CYCLE_VERSION,
        "state": (
            None
            if state is None
            else {
                "initialized_at": state[0],
                "baseline_cutoff_date": state[1],
                "last_run_at": state[2],
                "last_source_observed_date": state[3],
                "metadata": dict(state[4] or {}),
            }
        ),
        "candidates": [
            {
                "status": row[0],
                "days": int(row[1]),
                "first_date": row[2],
                "last_date": row[3],
                "last_seen_at": row[4],
            }
            for row in candidates
        ],
        "rules": {
            "historical_backfill": "FORBIDDEN",
            "stable_observations_required": STABILITY_OBSERVATIONS_REQUIRED,
            "minimum_duration_label_rate": MIN_DURATION_LABEL_RATE,
            "settlement_hour_utc": SETTLEMENT_HOUR_UTC,
            "prior_operational_forecast_required": True,
        },
    }
