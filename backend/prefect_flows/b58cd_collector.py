from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import boto3
import httpx
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from prefect_flows.b58cd_core import (
    build_issue_time_forecast_frame,
    canonical_payload_bytes,
    clean_json_value,
    payload_sha256,
)


SOURCE_NAME = "b58cd_issue_time_weather_forecast_collection"
DATASET_NAME = "maritime_issue_time_weather_forecast"
DATASET_VERSION = "b58cd-issue-time-weather-forecast-v1"
MODEL_VERSION = "open-meteo-best-match-live-v1"
PROVIDER = "OPEN_METEO"
PROVIDER_MODEL = "BEST_MATCH"
RAW_BUCKET = "bronze-maritime"
REPORT_BUCKET = "gold-maritime"
REPORT_PREFIX = "reports/b58cd/version=1"
FORECAST_TABLE = "features.maritime_issue_time_weather_forecast_v1"
SNAPSHOT_TABLE = "audit.weather_forecast_snapshot_v1"
WEATHER_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
MARINE_ENDPOINT = "https://marine-api.open-meteo.com/v1/marine"
FORECAST_HOURS = 168
WEATHER_VARIABLES = (
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "visibility",
    "temperature_2m",
)
MARINE_VARIABLES = (
    "wave_height",
    "wave_direction",
    "wave_period",
    "ocean_current_velocity",
    "ocean_current_direction",
    "sea_surface_temperature",
)


def _clean_json(value: Any) -> Any:
    return clean_json_value(value)


def _nullable_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _ensure_schema() -> None:
    sql = f"""
    CREATE SCHEMA IF NOT EXISTS features;
    CREATE TABLE IF NOT EXISTS {SNAPSHOT_TABLE} (
        snapshot_id text PRIMARY KEY,
        collection_run_id text NOT NULL,
        provider text NOT NULL,
        endpoint_family text NOT NULL,
        provider_model text NOT NULL,
        issue_at timestamptz NOT NULL,
        requested_at timestamptz NOT NULL,
        available_at timestamptz NOT NULL,
        requested_latitude double precision NOT NULL,
        requested_longitude double precision NOT NULL,
        grid_latitude double precision,
        grid_longitude double precision,
        payload_sha256 text NOT NULL,
        object_uri text NOT NULL,
        response_ms double precision,
        request_parameters jsonb NOT NULL,
        response_metadata jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CHECK (available_at >= requested_at),
        UNIQUE (endpoint_family, issue_at)
    );
    CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
        provider text NOT NULL,
        provider_model text NOT NULL,
        collection_run_id text NOT NULL,
        issue_at timestamptz NOT NULL,
        available_at timestamptz NOT NULL,
        weather_available_at timestamptz,
        marine_available_at timestamptz,
        valid_at timestamptz NOT NULL,
        lead_time_h double precision NOT NULL,
        requested_latitude double precision NOT NULL,
        requested_longitude double precision NOT NULL,
        weather_grid_latitude double precision,
        weather_grid_longitude double precision,
        marine_grid_latitude double precision,
        marine_grid_longitude double precision,
        wind_speed_ms double precision,
        wind_direction_deg double precision,
        pressure_hpa double precision,
        visibility_m double precision,
        air_temperature_c double precision,
        wave_height_m double precision,
        wave_direction_deg double precision,
        wave_period_s double precision,
        ocean_current_ms double precision,
        ocean_current_direction_deg double precision,
        sea_surface_temperature_c double precision,
        atmosphere_available_flag boolean NOT NULL,
        visibility_available_flag boolean NOT NULL,
        wave_available_flag boolean NOT NULL,
        marine_current_available_flag boolean NOT NULL,
        full_weather_available_flag boolean NOT NULL,
        weather_payload_sha256 text,
        marine_payload_sha256 text,
        captured_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (
            provider, provider_model, issue_at, valid_at,
            requested_latitude, requested_longitude
        ),
        CHECK (available_at >= issue_at),
        CHECK (valid_at >= issue_at),
        CHECK (valid_at >= available_at),
        CHECK (lead_time_h >= 0),
        CHECK (wind_direction_deg IS NULL OR (wind_direction_deg >= 0 AND wind_direction_deg < 360)),
        CHECK (wave_direction_deg IS NULL OR (wave_direction_deg >= 0 AND wave_direction_deg < 360)),
        CHECK (ocean_current_direction_deg IS NULL OR (ocean_current_direction_deg >= 0 AND ocean_current_direction_deg < 360))
    );
    CREATE INDEX IF NOT EXISTS ix_issue_time_weather_valid
        ON {FORECAST_TABLE} (valid_at DESC, available_at DESC);
    CREATE INDEX IF NOT EXISTS ix_issue_time_weather_available
        ON {FORECAST_TABLE} (available_at DESC);
    """
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            try:
                cursor.execute(
                    f"SELECT create_hypertable('{FORECAST_TABLE}', 'issue_at', if_not_exists => TRUE)"
                )
            except Exception:
                connection.rollback()
                with connection.cursor() as fallback:
                    fallback.execute(sql)


def _query_one(query: str, parameters: tuple[Any, ...] | None = None) -> dict[str, Any]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            row = cursor.fetchone()
            if row is None:
                return {}
            return {
                column.name: value
                for column, value in zip(cursor.description, row)
            }


def _location() -> tuple[float, float]:
    configured_latitude = os.getenv("SMART_PORT_FORECAST_LATITUDE", "").strip()
    configured_longitude = os.getenv("SMART_PORT_FORECAST_LONGITUDE", "").strip()
    if configured_latitude and configured_longitude:
        return float(configured_latitude), float(configured_longitude)
    row = _query_one(
        """
        SELECT avg(latitude)::double precision AS latitude,
               avg(longitude)::double precision AS longitude
        FROM core.maritime_observation
        WHERE quality_flag=0
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
        """
    )
    if row.get("latitude") is None or row.get("longitude") is None:
        raise RuntimeError("No forecast location is available")
    return float(row["latitude"]), float(row["longitude"])


def _require_upstream_contract() -> dict[str, Any]:
    row = _query_one(
        """
        SELECT status, metadata
        FROM audit.ingestion_run
        WHERE source_name='b58cc_weather_feature_ablation'
          AND status='SUCCESS'
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    if not row:
        raise RuntimeError("B58C-C v3 SUCCESS contract is required")
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if int(metadata.get("critical_temporal_leakage_checks_failed", -1)) != 0:
        raise RuntimeError("B58C-C critical leakage gate is not zero")
    expected = "RESEARCH_SIGNAL_FOUND_NEED_ISSUE_TIME_FORECASTS"
    if metadata.get("decision") != expected:
        raise RuntimeError(f"B58C-C decision is not ready: {metadata.get('decision')}")
    return dict(metadata)


def _start_run(checksum: str, latitude: float, longitude: float) -> str:
    metadata = {
        "dataset_version": DATASET_VERSION,
        "model_version": MODEL_VERSION,
        "orchestrator": "PREFECT",
        "collection_mode": "LIVE_ISSUE_TIME_ONLY",
        "historical_backfill_allowed": False,
        "production_promotion_allowed": False,
        "latitude": latitude,
        "longitude": longitude,
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit.ingestion_run
                    (source_name, dataset_name, object_uri, checksum, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    SOURCE_NAME,
                    DATASET_NAME,
                    f"s3://{RAW_BUCKET}/external/open-meteo/issue-time/",
                    checksum,
                    Json(metadata),
                ),
            )
            return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **values: Any) -> None:
    progress = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **_clean_json(values),
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
                (Json({"progress": progress}), run_id),
            )


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
                SET status=%s, row_count=%s, finished_at=now(),
                    metadata=metadata || %s, error_message=%s
                WHERE run_id=%s
                """,
                (status, row_count, Json(_clean_json(metadata)), error_message, run_id),
            )


def _request_json(
    http: httpx.Client,
    endpoint: str,
    parameters: dict[str, Any],
    attempts: int = 4,
) -> tuple[dict[str, Any], datetime, datetime, float]:
    api_key = os.getenv("OPEN_METEO_API_KEY", "").strip()
    request_parameters = dict(parameters)
    if api_key:
        request_parameters["apikey"] = api_key
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        requested_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            response = http.get(endpoint, params=request_parameters)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(str(payload.get("reason", "Open-Meteo error")))
            available_at = datetime.now(timezone.utc)
            return payload, requested_at, available_at, 1000.0 * (time.perf_counter() - started)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Open-Meteo request failed after {attempts} attempts: {last_error}")


def _archive_snapshot(
    client,
    run_id: str,
    endpoint_family: str,
    issue_at: datetime,
    requested_at: datetime,
    available_at: datetime,
    latitude: float,
    longitude: float,
    parameters: dict[str, Any],
    payload: dict[str, Any],
    response_ms: float,
) -> dict[str, Any]:
    digest = payload_sha256(payload)
    stamp = issue_at.strftime("%Y%m%dT%H%M%S.%fZ")
    key = (
        "external/open-meteo/issue-time/"
        f"endpoint={endpoint_family}/date={issue_at:%Y-%m-%d}/"
        f"issue_at={stamp}/{digest}.json"
    )
    archive = {
        "contract_version": DATASET_VERSION,
        "provider": PROVIDER,
        "provider_model": PROVIDER_MODEL,
        "endpoint_family": endpoint_family,
        "issue_at": issue_at.isoformat(),
        "requested_at": requested_at.isoformat(),
        "available_at": available_at.isoformat(),
        "request_parameters": parameters,
        "payload_sha256": digest,
        "response": payload,
    }
    body = canonical_payload_bytes(archive)
    client.put_object(
        Bucket=RAW_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
        Metadata={
            "dataset-version": DATASET_VERSION,
            "payload-sha256": digest,
            "endpoint-family": endpoint_family,
        },
    )
    client.head_object(Bucket=RAW_BUCKET, Key=key)
    snapshot_id = hashlib.sha256(
        f"{endpoint_family}|{issue_at.isoformat()}|{digest}".encode("utf-8")
    ).hexdigest()
    row = {
        "snapshot_id": snapshot_id,
        "collection_run_id": run_id,
        "provider": PROVIDER,
        "endpoint_family": endpoint_family,
        "provider_model": PROVIDER_MODEL,
        "issue_at": issue_at,
        "requested_at": requested_at,
        "available_at": available_at,
        "requested_latitude": latitude,
        "requested_longitude": longitude,
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "payload_sha256": digest,
        "object_uri": f"s3://{RAW_BUCKET}/{key}",
        "response_ms": response_ms,
        "request_parameters": parameters,
        "response_metadata": {
            "generationtime_ms": payload.get("generationtime_ms"),
            "timezone": payload.get("timezone"),
            "utc_offset_seconds": payload.get("utc_offset_seconds"),
            "hourly_units": payload.get("hourly_units", {}),
        },
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {SNAPSHOT_TABLE} (
                    snapshot_id, collection_run_id, provider, endpoint_family,
                    provider_model, issue_at, requested_at, available_at,
                    requested_latitude, requested_longitude,
                    grid_latitude, grid_longitude, payload_sha256, object_uri,
                    response_ms, request_parameters, response_metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (snapshot_id) DO NOTHING
                """,
                (
                    row["snapshot_id"], row["collection_run_id"], row["provider"],
                    row["endpoint_family"], row["provider_model"], row["issue_at"],
                    row["requested_at"], row["available_at"],
                    row["requested_latitude"], row["requested_longitude"],
                    row["grid_latitude"], row["grid_longitude"], row["payload_sha256"],
                    row["object_uri"], row["response_ms"],
                    Json(row["request_parameters"]), Json(row["response_metadata"]),
                ),
            )
    return row


def _materialize_forecasts(
    frame: pd.DataFrame,
    run_id: str,
    latitude: float,
    longitude: float,
    weather_snapshot: dict[str, Any] | None,
    marine_snapshot: dict[str, Any] | None,
) -> int:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            (
                PROVIDER, PROVIDER_MODEL, run_id, row.issue_at, row.available_at,
                row.weather_available_at if pd.notna(row.weather_available_at) else None,
                row.marine_available_at if pd.notna(row.marine_available_at) else None,
                row.valid_at, float(row.lead_time_h), latitude, longitude,
                weather_snapshot.get("grid_latitude") if weather_snapshot else None,
                weather_snapshot.get("grid_longitude") if weather_snapshot else None,
                marine_snapshot.get("grid_latitude") if marine_snapshot else None,
                marine_snapshot.get("grid_longitude") if marine_snapshot else None,
                _nullable_float(row.wind_speed_ms),
                _nullable_float(row.wind_direction_deg),
                _nullable_float(row.pressure_hpa),
                _nullable_float(row.visibility_m),
                _nullable_float(row.air_temperature_c),
                _nullable_float(row.wave_height_m),
                _nullable_float(row.wave_direction_deg),
                _nullable_float(row.wave_period_s),
                _nullable_float(row.ocean_current_ms),
                _nullable_float(row.ocean_current_direction_deg),
                _nullable_float(row.sea_surface_temperature_c),
                bool(row.atmosphere_available_flag), bool(row.visibility_available_flag),
                bool(row.wave_available_flag), bool(row.marine_current_available_flag),
                bool(row.full_weather_available_flag),
                weather_snapshot.get("payload_sha256") if weather_snapshot else None,
                marine_snapshot.get("payload_sha256") if marine_snapshot else None,
            )
        )
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            execute_values(
                cursor,
                f"""
                INSERT INTO {FORECAST_TABLE} (
                    provider, provider_model, collection_run_id, issue_at,
                    available_at, weather_available_at, marine_available_at,
                    valid_at, lead_time_h, requested_latitude, requested_longitude,
                    weather_grid_latitude, weather_grid_longitude,
                    marine_grid_latitude, marine_grid_longitude,
                    wind_speed_ms, wind_direction_deg, pressure_hpa, visibility_m,
                    air_temperature_c, wave_height_m, wave_direction_deg,
                    wave_period_s, ocean_current_ms, ocean_current_direction_deg,
                    sea_surface_temperature_c, atmosphere_available_flag,
                    visibility_available_flag, wave_available_flag,
                    marine_current_available_flag, full_weather_available_flag,
                    weather_payload_sha256, marine_payload_sha256
                ) VALUES %s
                ON CONFLICT (
                    provider, provider_model, issue_at, valid_at,
                    requested_latitude, requested_longitude
                ) DO UPDATE SET
                    available_at=EXCLUDED.available_at,
                    collection_run_id=EXCLUDED.collection_run_id,
                    weather_available_at=EXCLUDED.weather_available_at,
                    marine_available_at=EXCLUDED.marine_available_at,
                    wind_speed_ms=EXCLUDED.wind_speed_ms,
                    wind_direction_deg=EXCLUDED.wind_direction_deg,
                    pressure_hpa=EXCLUDED.pressure_hpa,
                    visibility_m=EXCLUDED.visibility_m,
                    air_temperature_c=EXCLUDED.air_temperature_c,
                    wave_height_m=EXCLUDED.wave_height_m,
                    wave_direction_deg=EXCLUDED.wave_direction_deg,
                    wave_period_s=EXCLUDED.wave_period_s,
                    ocean_current_ms=EXCLUDED.ocean_current_ms,
                    ocean_current_direction_deg=EXCLUDED.ocean_current_direction_deg,
                    sea_surface_temperature_c=EXCLUDED.sea_surface_temperature_c,
                    atmosphere_available_flag=EXCLUDED.atmosphere_available_flag,
                    visibility_available_flag=EXCLUDED.visibility_available_flag,
                    wave_available_flag=EXCLUDED.wave_available_flag,
                    marine_current_available_flag=EXCLUDED.marine_current_available_flag,
                    full_weather_available_flag=EXCLUDED.full_weather_available_flag,
                    weather_payload_sha256=EXCLUDED.weather_payload_sha256,
                    marine_payload_sha256=EXCLUDED.marine_payload_sha256
                """,
                rows,
                page_size=500,
            )
    return len(rows)


def _history_stats() -> dict[str, Any]:
    row = _query_one(
        f"""
        SELECT count(DISTINCT issue_at)::bigint AS collections,
               min(issue_at) AS first_issue_at,
               max(issue_at) AS last_issue_at,
               EXTRACT(epoch FROM (max(issue_at)-min(issue_at)))/86400.0 AS span_days,
               avg(atmosphere_available_flag::int)::double precision AS atmosphere_coverage,
               avg(wave_available_flag::int)::double precision AS wave_coverage,
               avg(full_weather_available_flag::int)::double precision AS full_coverage
        FROM {FORECAST_TABLE}
        """
    )
    return _clean_json(row)


def _store_decision_report(client, issue_at: datetime, decision: dict[str, Any]) -> dict[str, str]:
    body = json.dumps(
        _clean_json(decision), sort_keys=True, indent=2, ensure_ascii=True
    ).encode("utf-8")
    stamp = issue_at.strftime("%Y%m%dT%H%M%S.%fZ")
    run_key = f"{REPORT_PREFIX}/runs/issue_at={stamp}/decision.json"
    latest_key = f"{REPORT_PREFIX}/latest_decision.json"
    for key in (run_key, latest_key):
        client.put_object(
            Bucket=REPORT_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    return {
        "run_report_uri": f"s3://{REPORT_BUCKET}/{run_key}",
        "latest_report_uri": f"s3://{REPORT_BUCKET}/{latest_key}",
    }


def run_b58cd_issue_time_collection() -> dict[str, Any]:
    _ensure_schema()
    upstream = _require_upstream_contract()
    latitude, longitude = _location()
    checksum_payload = {
        "dataset_version": DATASET_VERSION,
        "model_version": MODEL_VERSION,
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
        "upstream_model_version": upstream.get("model_version"),
    }
    checksum = hashlib.sha256(
        json.dumps(checksum_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_id = _start_run(checksum, latitude, longitude)
    issue_at = datetime.now(timezone.utc)
    client = _s3_client()
    try:
        _update_progress(run_id, "REQUESTING_PROVIDER", issue_at=issue_at)
        common = {
            "latitude": round(latitude, 5),
            "longitude": round(longitude, 5),
            "forecast_hours": FORECAST_HOURS,
            "timezone": "UTC",
            "cell_selection": "sea",
        }
        weather_parameters = {
            **common,
            "hourly": ",".join(WEATHER_VARIABLES),
            "wind_speed_unit": "ms",
        }
        marine_parameters = {
            **common,
            "hourly": ",".join(MARINE_VARIABLES),
            "wind_speed_unit": "ms",
        }
        payloads: dict[str, dict[str, Any]] = {}
        snapshots: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        headers = {"User-Agent": "smart-port-maritime-issue-time-collector/1.0"}
        with httpx.Client(timeout=120.0, headers=headers, follow_redirects=True) as http:
            for family, endpoint, parameters in (
                ("weather", WEATHER_ENDPOINT, weather_parameters),
                ("marine", MARINE_ENDPOINT, marine_parameters),
            ):
                try:
                    payload, requested, available, elapsed_ms = _request_json(
                        http, endpoint, parameters
                    )
                    snapshot = _archive_snapshot(
                        client, run_id, family, issue_at, requested, available,
                        latitude, longitude, parameters, payload, elapsed_ms,
                    )
                    payloads[family] = payload
                    snapshots[family] = snapshot
                except Exception as exc:
                    errors[family] = str(exc)
        if not payloads:
            raise RuntimeError(f"All issue-time providers failed: {errors}")

        _update_progress(
            run_id,
            "NORMALIZING_FORECASTS",
            endpoints=list(payloads),
            endpoint_errors=errors,
        )
        frame = build_issue_time_forecast_frame(
            payloads.get("weather"),
            payloads.get("marine"),
            issue_at,
            snapshots.get("weather", {}).get("available_at"),
            snapshots.get("marine", {}).get("available_at"),
            FORECAST_HOURS,
        )
        rows = _materialize_forecasts(
            frame, run_id, latitude, longitude,
            snapshots.get("weather"), snapshots.get("marine"),
        )
        history = _history_stats()
        validation_ready = bool(
            int(history.get("collections") or 0) >= 600
            and float(history.get("span_days") or 0.0) >= 30.0
            and float(history.get("atmosphere_coverage") or 0.0) >= 0.90
            and float(history.get("wave_coverage") or 0.0) >= 0.90
        )
        current_atmosphere_coverage = float(
            frame["atmosphere_available_flag"].mean()
        )
        current_wave_coverage = float(frame["wave_available_flag"].mean())
        complete = bool(
            "weather" in payloads
            and "marine" in payloads
            and current_atmosphere_coverage >= 0.90
            and current_wave_coverage >= 0.90
        )
        decision_name = (
            "READY_FOR_ISSUE_TIME_BACKTEST"
            if validation_ready
            else "ISSUE_TIME_COLLECTION_ACTIVE"
            if complete
            else "ISSUE_TIME_COLLECTION_PARTIAL_PROVIDER_OUTAGE"
        )
        decision = {
            "status": "SUCCESS",
            "decision": decision_name,
            "dataset_version": DATASET_VERSION,
            "model_version": MODEL_VERSION,
            "collection_mode": "LIVE_ISSUE_TIME_ONLY",
            "provider": PROVIDER,
            "provider_model": PROVIDER_MODEL,
            "issue_at": issue_at,
            "available_at": frame["available_at"].max(),
            "first_valid_at": frame["valid_at"].min(),
            "last_valid_at": frame["valid_at"].max(),
            "forecast_rows": rows,
            "successful_endpoints": sorted(payloads),
            "endpoint_errors": errors,
            "weather_payload_sha256": snapshots.get("weather", {}).get("payload_sha256"),
            "marine_payload_sha256": snapshots.get("marine", {}).get("payload_sha256"),
            "atmosphere_coverage": current_atmosphere_coverage,
            "visibility_coverage": float(frame["visibility_available_flag"].mean()),
            "wave_coverage": current_wave_coverage,
            "marine_current_coverage": float(frame["marine_current_available_flag"].mean()),
            "full_weather_coverage": float(frame["full_weather_available_flag"].mean()),
            "history": history,
            "validation_ready": validation_ready,
            "minimum_validation_days": 30,
            "historical_backfill_allowed": False,
            "synthetic_rows_created": 0,
            "production_promotion_allowed": False,
            "navigation_use_allowed": False,
            "next_block": (
                "B58C_E_ISSUE_TIME_WEATHER_BACKTEST"
                if validation_ready
                else "B58C_D_CONTINUE_HOURLY_COLLECTION"
            ),
        }
        decision.update(_store_decision_report(client, issue_at, decision))
        _finish_run(run_id, "SUCCESS", rows, decision)
        return _clean_json(decision)
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "dataset_version": DATASET_VERSION,
                "model_version": MODEL_VERSION,
                "issue_at": issue_at,
                "production_promotion_allowed": False,
            },
            str(exc),
        )
        raise
