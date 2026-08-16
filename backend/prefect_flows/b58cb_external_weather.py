from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import httpx
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from prefect_flows.b58cb_core import (
    normalize_direction_degrees,
    normalize_open_meteo_payload,
)


DATASET_VERSION = "b58cb-external-weather-hourly-v1"
SOURCE_NAME = "b58cb_prefect_external_weather_enrichment"
DATASET_NAME = "maritime_external_weather_hourly"
OUTPUT_TABLE = "features.maritime_external_weather_hourly_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"
RAW_BUCKET = "bronze-maritime"

ATMOSPHERE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
VISIBILITY_ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MARINE_ENDPOINT = "https://marine-api.open-meteo.com/v1/marine"

ATMOSPHERE_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)
MARINE_VARIABLES = (
    "ocean_current_velocity",
    "ocean_current_direction",
    "sea_surface_temperature",
)
CANONICAL_VARIABLES = (
    "wind_speed_ms",
    "wind_direction_deg",
    "surface_current_ms",
    "visibility_m",
    "pressure_hpa",
)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


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


def _query_frame(query: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _source_extent() -> dict[str, Any]:
    frame = _query_frame(
        """
        SELECT
            min(observed_at) AS first_observed_at,
            max(observed_at) AS last_observed_at,
            avg(latitude)::double precision AS latitude,
            avg(longitude)::double precision AS longitude,
            count(*)::bigint AS source_rows
        FROM core.maritime_observation
        WHERE quality_flag=0
        """
    )
    if frame.empty or pd.isna(frame.iloc[0]["first_observed_at"]):
        raise RuntimeError("core.maritime_observation has no quality rows")
    row = frame.iloc[0]
    return {
        "first_observed_at": pd.Timestamp(row["first_observed_at"]),
        "last_observed_at": pd.Timestamp(row["last_observed_at"]),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "source_rows": int(row["source_rows"]),
    }


def _upstream_decision() -> dict[str, Any]:
    frame = _query_frame(
        """
        SELECT status, metadata
        FROM audit.ingestion_run
        WHERE source_name='b58ca_prefect_missingness_audit'
          AND dataset_name='maritime_weather_missingness_diagnostics'
          AND status='SUCCESS'
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    if frame.empty:
        raise RuntimeError("B58C-A SUCCESS decision is required")
    metadata = frame.iloc[0]["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    absent = set((metadata or {}).get("structurally_absent_variables", []))
    expected = set(CANONICAL_VARIABLES)
    if not expected.issubset(absent):
        raise RuntimeError(
            "B58C-A contract changed; review variables before external enrichment"
        )
    return dict(metadata or {})


def _year_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start.normalize(), pd.Timestamp(f"{year}-01-01", tz="UTC"))
        chunk_end = min(end.normalize(), pd.Timestamp(f"{year}-12-31", tz="UTC"))
        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))
    return chunks


def _read_cached_json(client, key: str) -> dict[str, Any] | None:
    try:
        body = client.get_object(Bucket=RAW_BUCKET, Key=key)["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    return json.loads(body.decode("utf-8"))


def _store_raw_json(client, key: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    client.put_object(
        Bucket=RAW_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
        Metadata={"dataset-version": DATASET_VERSION},
    )


def _request_json(
    http: httpx.Client,
    endpoint: str,
    parameters: dict[str, Any],
    attempts: int = 5,
) -> dict[str, Any]:
    api_key = os.getenv("OPEN_METEO_API_KEY", "").strip()
    if api_key:
        parameters = {**parameters, "apikey": api_key}
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = http.get(endpoint, params=parameters)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(str(payload.get("reason", "Open-Meteo error")))
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Open-Meteo request failed after {attempts} attempts: {last_error}")


def _collect_family(
    family: str,
    endpoint: str,
    variables: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    latitude: float,
    longitude: float,
    force_download: bool,
    required: bool,
    extra_parameters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = _s3_client()
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    headers = {"User-Agent": "smart-port-maritime-research/1.0"}
    with httpx.Client(timeout=180.0, headers=headers, follow_redirects=True) as http:
        for chunk_start, chunk_end in _year_chunks(start, end):
            key = (
                f"external/open-meteo/{family}/{DATASET_VERSION}/"
                f"year={chunk_start.year}/response.json"
            )
            payload = None if force_download else _read_cached_json(client, key)
            source = "MINIO_CACHE" if payload is not None else "OPEN_METEO_API"
            try:
                if payload is None:
                    parameters = {
                        "latitude": round(latitude, 5),
                        "longitude": round(longitude, 5),
                        "start_date": chunk_start.strftime("%Y-%m-%d"),
                        "end_date": chunk_end.strftime("%Y-%m-%d"),
                        "hourly": ",".join(variables),
                        "timezone": "UTC",
                        "cell_selection": "sea",
                        **(extra_parameters or {}),
                    }
                    payload = _request_json(http, endpoint, parameters)
                    _store_raw_json(client, key, payload)
                frame = normalize_open_meteo_payload(payload)
                for variable in variables:
                    if variable not in frame.columns:
                        frame[variable] = np.nan
                frame = frame[["observed_at", *variables]]
                frames.append(frame)
                status_rows.append(
                    {
                        "family": family,
                        "year": chunk_start.year,
                        "status": "SUCCESS",
                        "rows": len(frame),
                        "source": source,
                        "raw_uri": f"s3://{RAW_BUCKET}/{key}",
                        "error": None,
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "family": family,
                        "year": chunk_start.year,
                        "status": "FAILED",
                        "rows": 0,
                        "source": source,
                        "raw_uri": f"s3://{RAW_BUCKET}/{key}",
                        "error": str(exc),
                    }
                )
                if required:
                    raise
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("observed_at")
        .drop_duplicates("observed_at", keep="last")
        if frames
        else pd.DataFrame(columns=["observed_at", *variables])
    )
    return combined, pd.DataFrame(status_rows)


def collect_external_weather(
    extent: dict[str, Any], force_download: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(extent["first_observed_at"]).tz_convert("UTC")
    end = pd.Timestamp(extent["last_observed_at"]).tz_convert("UTC")
    latitude = float(extent["latitude"])
    longitude = float(extent["longitude"])

    atmosphere, atmosphere_status = _collect_family(
        "era5_atmosphere",
        ATMOSPHERE_ENDPOINT,
        ATMOSPHERE_VARIABLES,
        start,
        end,
        latitude,
        longitude,
        force_download,
        required=True,
        extra_parameters={"models": "era5", "wind_speed_unit": "ms"},
    )
    marine, marine_status = _collect_family(
        "era5_ocean",
        MARINE_ENDPOINT,
        MARINE_VARIABLES,
        start,
        end,
        latitude,
        longitude,
        force_download,
        required=False,
        extra_parameters={"length_unit": "metric"},
    )

    visibility_start = max(start, pd.Timestamp("2022-01-01", tz="UTC"))
    visibility, visibility_status = _collect_family(
        "historical_forecast_visibility",
        VISIBILITY_ENDPOINT,
        ("visibility",),
        visibility_start,
        end,
        latitude,
        longitude,
        force_download,
        required=False,
    )

    grid = pd.DataFrame(
        {"observed_at": pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")}
    )
    external = grid.merge(atmosphere, on="observed_at", how="left", validate="one_to_one")
    if not marine.empty:
        external = external.merge(marine, on="observed_at", how="left", validate="one_to_one")
    else:
        for column in MARINE_VARIABLES:
            external[column] = np.nan
    if not visibility.empty:
        external = external.merge(visibility, on="observed_at", how="left", validate="one_to_one")
    else:
        external["visibility"] = np.nan

    external["wind_speed_ms"] = external["wind_speed_10m"]
    external["wind_direction_deg"] = normalize_direction_degrees(
        external["wind_direction_10m"]
    )
    external["pressure_hpa"] = external["pressure_msl"]
    external["visibility_m"] = external["visibility"]
    # Open-Meteo marine metric current velocity is km/h.
    external["surface_current_ms"] = external["ocean_current_velocity"] / 3.6
    external["dataset_version"] = DATASET_VERSION
    external["collected_at"] = pd.Timestamp.now(tz="UTC")
    external["availability_semantics"] = "RETROSPECTIVE_REANALYSIS_RESEARCH_ONLY"
    external["latitude"] = latitude
    external["longitude"] = longitude

    request_status = pd.concat(
        [atmosphere_status, marine_status, visibility_status], ignore_index=True
    )
    return external, request_status


def _coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in CANONICAL_VARIABLES:
        available = int(frame[variable].notna().sum())
        rows.append(
            {
                "variable": variable,
                "hourly_rows": len(frame),
                "available_rows": available,
                "missing_rows": len(frame) - available,
                "coverage_pct": 100.0 * available / len(frame),
                "source": {
                    "wind_speed_ms": "ERA5 wind_speed_10m",
                    "wind_direction_deg": "ERA5 wind_direction_10m",
                    "pressure_hpa": "ERA5 pressure_msl",
                    "surface_current_ms": "Open-Meteo ERA5-Ocean current velocity",
                    "visibility_m": "Historical Forecast visibility",
                }[variable],
                "status": "RESEARCH_ONLY",
            }
        )
    return pd.DataFrame(rows)


def _integrity_report(frame: pd.DataFrame) -> pd.DataFrame:
    bounds = {
        "wind_speed_ms": (0.0, 100.0),
        "wind_direction_deg": (0.0, 360.0),
        "surface_current_ms": (0.0, 10.0),
        "visibility_m": (0.0, 200_000.0),
        "pressure_hpa": (800.0, 1_100.0),
    }
    rows = []
    for variable, (lower, upper) in bounds.items():
        values = frame[variable].dropna()
        invalid = int(((values < lower) | (values >= upper)).sum())
        rows.append(
            {
                "variable": variable,
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "observed_rows": len(values),
                "outside_bound_rows": invalid,
                "bound_role": "BROAD_INTEGRITY_GUARDRAIL_ONLY",
            }
        )
    return pd.DataFrame(rows)


def _materialize(frame: pd.DataFrame, run_id: str) -> int:
    columns = [
        "observed_at",
        "wind_speed_ms",
        "wind_direction_deg",
        "surface_current_ms",
        "visibility_m",
        "pressure_hpa",
        "wind_gusts_10m",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "sea_surface_temperature",
        "dataset_version",
        "collected_at",
        "availability_semantics",
        "latitude",
        "longitude",
        "ingestion_run_id",
    ]
    payload = frame.copy()
    payload["ingestion_run_id"] = run_id
    payload = payload[columns].replace({np.nan: None})
    values = [tuple(row) for row in payload.itertuples(index=False, name=None)]

    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS features.maritime_external_weather_hourly_v1 (
                    observed_at TIMESTAMPTZ PRIMARY KEY,
                    wind_speed_ms REAL,
                    wind_direction_deg REAL,
                    surface_current_ms REAL,
                    visibility_m REAL,
                    pressure_hpa REAL,
                    wind_gusts_10m REAL,
                    temperature_2m REAL,
                    relative_humidity_2m REAL,
                    precipitation REAL,
                    cloud_cover REAL,
                    sea_surface_temperature REAL,
                    dataset_version TEXT NOT NULL,
                    collected_at TIMESTAMPTZ NOT NULL,
                    availability_semantics TEXT NOT NULL,
                    latitude DOUBLE PRECISION NOT NULL,
                    longitude DOUBLE PRECISION NOT NULL,
                    ingestion_run_id UUID NOT NULL REFERENCES audit.ingestion_run(run_id)
                )
                """
            )
            updates = ", ".join(
                f"{column}=EXCLUDED.{column}" for column in columns if column != "observed_at"
            )
            execute_values(
                cursor,
                f"""
                INSERT INTO {OUTPUT_TABLE} ({', '.join(columns)}) VALUES %s
                ON CONFLICT (observed_at) DO UPDATE SET {updates}
                """,
                values,
                page_size=2_000,
            )
    return len(values)


def _checksum(extent: dict[str, Any]) -> str:
    payload = json.dumps(
        _clean_json({"version": DATASET_VERSION, **extent}), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _start_run(checksum: str) -> str:
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
                    "https://open-meteo.com/",
                    checksum,
                    Json(
                        {
                            "dataset_version": DATASET_VERSION,
                            "orchestrator": "PREFECT",
                            "source_modified": False,
                            "training_executed": False,
                        }
                    ),
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
                (status, row_count, Json(_clean_json(metadata)), error_message, run_id),
            )


def _upload(path: Path, key: str) -> str:
    client = _s3_client()
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/octet-stream",
    }.get(path.suffix, "application/octet-stream")
    client.upload_file(str(path), OUTPUT_BUCKET, key, ExtraArgs={"ContentType": content_type})
    client.head_object(Bucket=OUTPUT_BUCKET, Key=key)
    return f"s3://{OUTPUT_BUCKET}/{key}"


def run_b58cb_external_enrichment(
    force_download: bool = False,
    materialize_timescale: bool = True,
) -> dict[str, Any]:
    upstream = _upstream_decision()
    extent = _source_extent()
    checksum = _checksum(extent)
    run_id = _start_run(checksum)
    try:
        external, requests = collect_external_weather(extent, force_download)
        coverage = _coverage_report(external)
        integrity = _integrity_report(external)
        coverage_lookup = coverage.set_index("variable")["coverage_pct"]
        atmospheric_coverage_ready = bool(
            coverage_lookup[["wind_speed_ms", "wind_direction_deg", "pressure_hpa"]]
            .ge(95.0)
            .all()
        )
        full_coverage_ready = bool(coverage_lookup.ge(95.0).all())
        integrity_passed = bool(integrity["outside_bound_rows"].eq(0).all())
        atmospheric_ready = atmospheric_coverage_ready and integrity_passed
        full_ready = full_coverage_ready and integrity_passed
        decision_name = (
            "NEED_EXTERNAL_WEATHER_INTEGRITY_REPAIR"
            if not integrity_passed
            else "READY_FOR_FULL_EXTERNAL_WEATHER_BENCHMARK"
            if full_ready
            else "READY_FOR_PARTIAL_EXTERNAL_WEATHER_BENCHMARK"
            if atmospheric_ready
            else "EXTERNAL_WEATHER_COLLECTION_INSUFFICIENT"
        )
        next_block = (
            "B58C_B2_EXTERNAL_INTEGRITY_REPAIR"
            if not integrity_passed
            else "B58C_C_OBSERVED_MASKING_AND_FEATURE_ABLATION"
            if atmospheric_ready
            else "B58C_B2_EXTERNAL_PROVIDER_REPAIR"
        )
        decision = {
            "status": "SUCCESS",
            "decision": decision_name,
            "dataset_version": DATASET_VERSION,
            "rows": len(external),
            "first_observed_at": external["observed_at"].min(),
            "last_observed_at": external["observed_at"].max(),
            "latitude": extent["latitude"],
            "longitude": extent["longitude"],
            "integrity_passed": integrity_passed,
            "atmospheric_coverage_ready": atmospheric_coverage_ready,
            "full_external_weather_coverage_ready": full_coverage_ready,
            "atmospheric_track_ready": atmospheric_ready,
            "full_external_weather_ready": full_ready,
            "coverage_pct": coverage_lookup.to_dict(),
            "failed_request_chunks": int(requests["status"].eq("FAILED").sum()),
            "outside_integrity_bound_rows": int(integrity["outside_bound_rows"].sum()),
            "source_modified": False,
            "training_executed": False,
            "synthetic_rows_created": 0,
            "reanalysis_status": "RESEARCH_ONLY_NOT_HISTORICALLY_AVAILABLE",
            "historical_replay_allowed": False,
            "navigation_use_allowed": False,
            "upstream_decision": upstream.get("decision"),
            "next_block": next_block,
        }

        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b58cb-") as temporary:
            directory = Path(temporary)
            dataset_path = directory / "maritime_external_weather_hourly_v1.parquet"
            external.to_parquet(dataset_path, index=False)
            coverage_path = directory / "01_external_variable_coverage.csv"
            coverage.to_csv(coverage_path, index=False)
            requests_path = directory / "02_external_request_status.csv"
            requests.to_csv(requests_path, index=False)
            integrity_path = directory / "03_external_integrity.csv"
            integrity.to_csv(integrity_path, index=False)
            decision_path = directory / "04_b58cb_final_decision.json"
            decision_path.write_text(
                json.dumps(_clean_json(decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            readme_path = directory / "README_B58CB.md"
            readme_path.write_text(
                "\n".join(
                    [
                        "# B58C-B external weather enrichment",
                        "",
                        f"Decision: {decision_name}",
                        "",
                        "ERA5 and ERA5-Ocean values are retrospective reanalysis.",
                        "They are research features, not historically available port observations.",
                        "Marine current values are not suitable for coastal navigation.",
                        "No Bronze/Core row was modified and no synthetic row was created.",
                    ]
                ),
                encoding="utf-8",
            )

            mapping = {
                dataset_path: f"datasets/b58cb/{OUTPUT_PREFIX}/{dataset_path.name}",
                decision_path: f"configs/b58cb/{OUTPUT_PREFIX}/{decision_path.name}",
                coverage_path: f"reports/b58cb/{OUTPUT_PREFIX}/{coverage_path.name}",
                requests_path: f"reports/b58cb/{OUTPUT_PREFIX}/{requests_path.name}",
                integrity_path: f"reports/b58cb/{OUTPUT_PREFIX}/{integrity_path.name}",
                readme_path: f"reports/b58cb/{OUTPUT_PREFIX}/{readme_path.name}",
            }
            for path, key in mapping.items():
                outputs[path.name] = _upload(path, key)

        materialized_rows = _materialize(external, run_id) if materialize_timescale else 0
        metadata = {
            **decision,
            "checksum": checksum,
            "materialized_timescale_rows": materialized_rows,
            "timescale_table": OUTPUT_TABLE if materialized_rows else None,
            "outputs": outputs,
        }
        _finish_run(run_id, "SUCCESS", len(external), metadata)
        return {"status": "SUCCESS", "run_id": run_id, "results": _clean_json(metadata)}
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {"dataset_version": DATASET_VERSION, "orchestrator": "PREFECT"},
            str(exc),
        )
        raise


def verify_b58cb(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("results") or {}
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"B58C-B failed: {result}")
    if metadata.get("source_modified") is not False:
        raise RuntimeError("Source immutability gate failed")
    if int(metadata.get("synthetic_rows_created", -1)) != 0:
        raise RuntimeError("B58C-B unexpectedly created synthetic rows")
    return {
        "run_id": result["run_id"],
        "decision": metadata.get("decision"),
        "rows": metadata.get("rows"),
        "next_block": metadata.get("next_block"),
        "safety_contract": "PASS",
    }
