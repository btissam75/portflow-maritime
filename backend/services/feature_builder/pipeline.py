from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
import xarray as xr
from botocore.exceptions import ClientError
from psycopg2.extras import Json, execute_values


TARGET_LATITUDE = float(os.getenv("TARGET_LATITUDE", "35.8892"))
TARGET_LONGITUDE = float(os.getenv("TARGET_LONGITUDE", "-5.5000"))
TARGET_PORT_CODE = os.getenv("TARGET_PORT_CODE", "MAPTM")
FEATURE_VERSION = os.getenv("FEATURE_VERSION", "copernicus-point-v1")
SOURCE_NAME = "copernicus_ibi_wave"

TIME_CANDIDATES = (
    "time",
    "valid_time",
    "datetime",
    "forecast_time",
    "forecast_reference_time",
)
LATITUDE_CANDIDATES = ("latitude", "lat", "nav_lat", "y")
LONGITUDE_CANDIDATES = ("longitude", "lon", "nav_lon", "x")

VARIABLE_CANDIDATES = {
    "wave_height_m": ("VHM0", "swh", "significant_wave_height"),
    "wave_period_s": ("VTM02", "VTM10", "mwp", "mean_wave_period"),
    "peak_period_s": ("VTPK", "pp1d", "peak_wave_period"),
    "wave_direction_deg": ("VMDR", "mwd", "mean_wave_direction"),
    "wind_wave_height_m": ("VHM0_WW", "wind_wave_height"),
    "swell_height_1_m": ("VHM0_SW1", "primary_swell_wave_height"),
    "swell_height_2_m": ("VHM0_SW2", "secondary_swell_wave_height"),
    "wind_speed_ms": ("wind_speed", "si10", "wind_speed_10m"),
    "wind_direction_deg": ("wind_direction", "wind_direction_10m"),
    "surface_current_u_ms": ("uo", "eastward_sea_water_velocity"),
    "surface_current_v_ms": ("vo", "northward_sea_water_velocity"),
}


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def check_dependencies() -> dict:
    client = s3_client()
    buckets = sorted(item["Name"] for item in client.list_buckets().get("Buckets", []))
    required = {"bronze-maritime", "silver-maritime"}
    missing = sorted(required.difference(buckets))
    if missing:
        raise RuntimeError(f"Missing S3 buckets: {missing}")

    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), now()")
            database_name, database_time = cursor.fetchone()

    return {
        "s3_endpoint": os.environ["SMART_PORT_S3_ENDPOINT"],
        "buckets": buckets,
        "database": database_name,
        "database_time": database_time.isoformat(),
    }


def resolve_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    exact_names = set(dataset.variables) | set(dataset.coords) | set(dataset.dims)
    for candidate in candidates:
        if candidate in exact_names:
            return candidate

    lower_to_original = {str(name).lower(): str(name) for name in exact_names}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def resolve_variable(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in dataset.data_vars:
            return candidate
    lower_to_original = {str(name).lower(): str(name) for name in dataset.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def open_dataset(path: Path) -> tuple[xr.Dataset, str]:
    errors = []
    for engine in ("netcdf4", "h5netcdf"):
        try:
            dataset = xr.open_dataset(
                path,
                engine=engine,
                decode_times=True,
                mask_and_scale=True,
                cache=False,
            )
            return dataset, engine
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
    raise RuntimeError("Unable to open NetCDF. " + " | ".join(errors))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(value))


def select_nearest_valid_point(
    dataset: xr.Dataset,
    latitude_name: str,
    longitude_name: str,
    reference_variable: str,
) -> tuple[xr.Dataset, float, float, float]:
    latitude = dataset[latitude_name]
    longitude = dataset[longitude_name]

    if latitude.ndim == 1 and longitude.ndim == 1:
        latitude_values = np.asarray(latitude.values, dtype=float)
        longitude_values = np.asarray(longitude.values, dtype=float)
        latitude_index = int(np.nanargmin(np.abs(latitude_values - TARGET_LATITUDE)))
        longitude_index = int(np.nanargmin(np.abs(longitude_values - TARGET_LONGITUDE)))
        latitude_dim = latitude.dims[0]
        longitude_dim = longitude.dims[0]

        candidates = []
        for lat_offset in range(-3, 4):
            for lon_offset in range(-3, 4):
                lat_index = latitude_index + lat_offset
                lon_index = longitude_index + lon_offset
                if not (0 <= lat_index < len(latitude_values)):
                    continue
                if not (0 <= lon_index < len(longitude_values)):
                    continue
                point_values = np.asarray(
                    dataset[reference_variable]
                    .isel({latitude_dim: lat_index, longitude_dim: lon_index})
                    .values
                )
                valid_fraction = float(np.isfinite(point_values).mean()) if point_values.size else 0.0
                distance = haversine_km(
                    TARGET_LATITUDE,
                    TARGET_LONGITUDE,
                    float(latitude_values[lat_index]),
                    float(longitude_values[lon_index]),
                )
                candidates.append((valid_fraction, -distance, lat_index, lon_index))

        valid_fraction, negative_distance, latitude_index, longitude_index = max(candidates)
        if valid_fraction <= 0:
            raise RuntimeError("No valid sea point found around Tanger Med")

        point = dataset.isel({latitude_dim: latitude_index, longitude_dim: longitude_index})
        selected_latitude = float(latitude_values[latitude_index])
        selected_longitude = float(longitude_values[longitude_index])
        return point, selected_latitude, selected_longitude, -negative_distance

    if latitude.shape == longitude.shape and latitude.ndim == 2:
        distance_score = (
            (np.asarray(latitude.values, dtype=float) - TARGET_LATITUDE) ** 2
            + (np.asarray(longitude.values, dtype=float) - TARGET_LONGITUDE) ** 2
        )
        flat_index = int(np.nanargmin(distance_score))
        indexes = np.unravel_index(flat_index, latitude.shape)
        indexers = {dim: index for dim, index in zip(latitude.dims, indexes)}
        point = dataset.isel(indexers)
        selected_latitude = float(np.asarray(latitude.values)[indexes])
        selected_longitude = float(np.asarray(longitude.values)[indexes])
        distance = haversine_km(
            TARGET_LATITUDE, TARGET_LONGITUDE, selected_latitude, selected_longitude
        )
        if not np.isfinite(np.asarray(point[reference_variable].values)).any():
            raise RuntimeError("Nearest curvilinear grid point contains no valid wave data")
        return point, selected_latitude, selected_longitude, distance

    raise RuntimeError(
        f"Unsupported coordinate geometry: latitude={latitude.dims}, longitude={longitude.dims}"
    )


def infer_row_count(point: xr.Dataset, time_name: str | None, reference_variable: str) -> int:
    if time_name and time_name in point.sizes:
        return int(point.sizes[time_name])
    if time_name and time_name in point.variables:
        return max(1, int(point[time_name].size))
    reference = point[reference_variable]
    return max(1, int(reference.size))


def extract_times(
    point: xr.Dataset,
    original: xr.Dataset,
    time_name: str | None,
    row_count: int,
    fallback_time: datetime,
) -> tuple[pd.DatetimeIndex, bool]:
    raw_values = None
    if time_name and time_name in point.variables:
        raw_values = np.atleast_1d(point[time_name].values)
    elif time_name and time_name in original.variables:
        raw_values = np.atleast_1d(original[time_name].values)

    if raw_values is not None:
        parsed = pd.to_datetime(raw_values, errors="coerce", utc=True)
        parsed = pd.DatetimeIndex(parsed)
        if len(parsed) == row_count and parsed.notna().all():
            return parsed, False
        if len(parsed) == 1 and parsed.notna().all():
            if row_count == 1:
                return parsed, False
            return pd.date_range(end=parsed[0], periods=row_count, freq="h"), True

    fallback_timestamp = pd.Timestamp(fallback_time)
    if fallback_timestamp.tzinfo is None:
        fallback_timestamp = fallback_timestamp.tz_localize("UTC")
    else:
        fallback_timestamp = fallback_timestamp.tz_convert("UTC")
    return pd.date_range(end=fallback_timestamp, periods=row_count, freq="h"), True


def extract_variable_values(
    point: xr.Dataset,
    variable_name: str | None,
    time_name: str | None,
    row_count: int,
) -> np.ndarray:
    if variable_name is None:
        return np.full(row_count, np.nan, dtype=float)

    data_array = point[variable_name]
    for dimension in list(data_array.dims):
        if dimension != time_name:
            data_array = data_array.isel({dimension: 0})

    values = np.asarray(data_array.values, dtype=float).reshape(-1)
    if values.size == row_count:
        return values
    if values.size == 1:
        return np.repeat(values[0], row_count)
    raise RuntimeError(
        f"Variable {variable_name} has {values.size} values; expected {row_count}"
    )


def clean_and_engineer_features(frame: pd.DataFrame, time_fallback_used: bool) -> pd.DataFrame:
    frame = frame.sort_values("observed_at").drop_duplicates("observed_at", keep="last")
    frame = frame.reset_index(drop=True)
    quality = np.zeros(len(frame), dtype=np.int16)

    physical_ranges = {
        "wave_height_m": (0.0, 30.0, 2),
        "wave_period_s": (0.0, 40.0, 4),
        "peak_period_s": (0.0, 50.0, 8),
        "wind_speed_ms": (0.0, 100.0, 16),
        "surface_current_u_ms": (-10.0, 10.0, 32),
        "surface_current_v_ms": (-10.0, 10.0, 32),
    }

    if frame["wave_height_m"].isna().to_numpy().any():
        quality |= np.where(frame["wave_height_m"].isna(), 1, 0).astype(np.int16)

    for column, (lower, upper, flag) in physical_ranges.items():
        invalid = frame[column].notna() & ~frame[column].between(lower, upper)
        quality |= np.where(invalid, flag, 0).astype(np.int16)
        frame.loc[invalid, column] = np.nan

    for column in ("wave_direction_deg", "wind_direction_deg"):
        frame[column] = frame[column] % 360.0

    if time_fallback_used:
        quality |= np.int16(64)
    frame["quality_flag"] = quality

    frame["surface_current_ms"] = np.sqrt(
        frame["surface_current_u_ms"] ** 2 + frame["surface_current_v_ms"] ** 2
    )
    radians = np.deg2rad(frame["wave_direction_deg"])
    frame["wave_direction_sin"] = np.sin(radians)
    frame["wave_direction_cos"] = np.cos(radians)
    frame["wave_energy_proxy"] = frame["wave_height_m"] ** 2 * frame["wave_period_s"]
    frame["high_wave_ge_2p5m"] = (frame["wave_height_m"] >= 2.5).astype("int8")
    frame["severe_wave_ge_4m"] = (frame["wave_height_m"] >= 4.0).astype("int8")

    indexed = frame.set_index("observed_at")
    for window in (3, 6, 12, 24):
        rolling = indexed["wave_height_m"].rolling(f"{window}h", min_periods=1)
        frame[f"wave_height_mean_{window}h"] = rolling.mean().to_numpy()
        frame[f"wave_height_max_{window}h"] = rolling.max().to_numpy()

    frame["wave_height_delta_3_steps"] = frame["wave_height_m"] - frame["wave_height_m"].shift(3)
    return frame


def extract_point_features(
    dataset: xr.Dataset,
    fallback_time: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    latitude_name = resolve_name(dataset, LATITUDE_CANDIDATES)
    longitude_name = resolve_name(dataset, LONGITUDE_CANDIDATES)
    time_name = resolve_name(dataset, TIME_CANDIDATES)
    if latitude_name is None or longitude_name is None:
        raise RuntimeError("Latitude/longitude coordinates were not found")

    variable_mapping = {
        output_name: resolve_variable(dataset, candidates)
        for output_name, candidates in VARIABLE_CANDIDATES.items()
    }
    reference_variable = variable_mapping["wave_height_m"]
    if reference_variable is None:
        raise RuntimeError(
            f"Significant wave height was not found. Available variables: {list(dataset.data_vars)}"
        )

    point, selected_latitude, selected_longitude, distance_km = select_nearest_valid_point(
        dataset, latitude_name, longitude_name, reference_variable
    )
    row_count = infer_row_count(point, time_name, reference_variable)
    observed_at, time_fallback_used = extract_times(
        point, dataset, time_name, row_count, fallback_time
    )

    frame = pd.DataFrame({"observed_at": observed_at})
    for output_name, variable_name in variable_mapping.items():
        frame[output_name] = extract_variable_values(
            point, variable_name, time_name, row_count
        )

    frame["source"] = SOURCE_NAME
    frame["port_code"] = TARGET_PORT_CODE
    frame["latitude"] = selected_latitude
    frame["longitude"] = selected_longitude
    frame["grid_distance_to_port_km"] = distance_km
    frame["feature_version"] = FEATURE_VERSION
    frame = clean_and_engineer_features(frame, time_fallback_used)

    metadata = {
        "time_coordinate": time_name,
        "latitude_coordinate": latitude_name,
        "longitude_coordinate": longitude_name,
        "selected_latitude": selected_latitude,
        "selected_longitude": selected_longitude,
        "grid_distance_to_port_km": distance_km,
        "time_fallback_used": time_fallback_used,
        "variable_mapping": variable_mapping,
    }
    return frame, metadata


def create_audit_run(source_uri: str) -> str:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit.ingestion_run (source_name, dataset_name, status, object_uri, metadata)
                VALUES (%s, %s, 'RUNNING', %s, %s)
                RETURNING run_id::text
                """,
                (SOURCE_NAME, FEATURE_VERSION, source_uri, Json({"port_code": TARGET_PORT_CODE})),
            )
            return cursor.fetchone()[0]


def finish_audit_run(
    run_id: str,
    status: str,
    output_uri: str | None,
    row_count: int | None,
    checksum: str | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at = now(), status = %s, object_uri = COALESCE(%s, object_uri),
                    row_count = %s, checksum = %s, metadata = %s, error_message = %s
                WHERE run_id = %s::uuid
                """,
                (
                    status,
                    output_uri,
                    row_count,
                    checksum,
                    Json(metadata),
                    error_message,
                    run_id,
                ),
            )


def nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def upsert_timescale_observations(frame: pd.DataFrame, audit_run_id: str) -> int:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            (
                row.observed_at.to_pydatetime(),
                row.source,
                float(row.latitude),
                float(row.longitude),
                nullable_float(row.wave_height_m),
                nullable_float(row.wave_period_s),
                nullable_float(row.wave_direction_deg),
                nullable_float(row.wind_speed_ms),
                nullable_float(row.wind_direction_deg),
                nullable_float(row.surface_current_ms),
                int(row.quality_flag),
                audit_run_id,
            )
        )

    with db_connection() as connection:
        with connection.cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO core.maritime_observation (
                    observed_at, source, latitude, longitude,
                    wave_height_m, wave_period_s, wave_direction_deg,
                    wind_speed_ms, wind_direction_deg, surface_current_ms,
                    quality_flag, ingestion_run_id
                ) VALUES %s
                ON CONFLICT (observed_at, source, latitude, longitude)
                DO UPDATE SET
                    wave_height_m = EXCLUDED.wave_height_m,
                    wave_period_s = EXCLUDED.wave_period_s,
                    wave_direction_deg = EXCLUDED.wave_direction_deg,
                    wind_speed_ms = EXCLUDED.wind_speed_ms,
                    wind_direction_deg = EXCLUDED.wind_direction_deg,
                    surface_current_ms = EXCLUDED.surface_current_ms,
                    quality_flag = EXCLUDED.quality_flag,
                    ingestion_run_id = EXCLUDED.ingestion_run_id
                """,
                rows,
                page_size=1000,
            )
    return len(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def process_netcdf_object(
    source_bucket: str,
    source_key: str,
    output_bucket: str,
    source_last_modified: datetime | None,
    force: bool,
) -> dict:
    if not source_key.lower().endswith((".nc", ".nc4")):
        raise RuntimeError(f"Unsupported source extension: {source_key}")

    client = s3_client()
    source_head = client.head_object(Bucket=source_bucket, Key=source_key)
    source_etag = source_head["ETag"].strip('"')
    fallback_time = source_last_modified or source_head["LastModified"]
    source_uri = f"s3://{source_bucket}/{source_key}"
    audit_run_id = create_audit_run(source_uri)

    try:
        with tempfile.TemporaryDirectory(prefix="smart-port-feature-") as temp_directory:
            local_source = Path(temp_directory) / Path(source_key).name
            client.download_file(source_bucket, source_key, str(local_source))
            dataset, engine = open_dataset(local_source)
            try:
                frame, extraction_metadata = extract_point_features(dataset, fallback_time)
            finally:
                dataset.close()

            first_observed_at = pd.Timestamp(frame["observed_at"].min())
            partition = f"year={first_observed_at:%Y}/month={first_observed_at:%m}/day={first_observed_at:%d}"
            output_name = f"{local_source.stem}_{source_etag[:12]}_{FEATURE_VERSION}.parquet"
            output_key = f"features/copernicus/tanger_med/{partition}/{output_name}"
            output_uri = f"s3://{output_bucket}/{output_key}"

            if object_exists(client, output_bucket, output_key) and not force:
                upserted_rows = upsert_timescale_observations(frame, audit_run_id)
                finish_audit_run(
                    audit_run_id,
                    "SUCCESS",
                    output_uri,
                    upserted_rows,
                    None,
                    {
                        "status_detail": "REUSED_PARQUET_UPSERTED_TIMESCALE",
                        "source_uri": source_uri,
                        "source_etag": source_etag,
                        **extraction_metadata,
                    },
                )
                return {
                    "status": "REUSED_PARQUET",
                    "source_uri": source_uri,
                    "output_uri": output_uri,
                    "row_count": upserted_rows,
                    "audit_run_id": audit_run_id,
                }

            local_output = Path(temp_directory) / output_name
            frame.to_parquet(local_output, index=False, compression="zstd")
            output_checksum = sha256_file(local_output)
            client.upload_file(
                str(local_output),
                output_bucket,
                output_key,
                ExtraArgs={
                    "ContentType": "application/vnd.apache.parquet",
                    "Metadata": {
                        "feature-version": FEATURE_VERSION,
                        "source-etag": source_etag,
                        "sha256": output_checksum,
                    },
                },
            )

            upserted_rows = upsert_timescale_observations(frame, audit_run_id)
            metadata = {
                "status_detail": "PROCESSED",
                "source_uri": source_uri,
                "source_etag": source_etag,
                "output_uri": output_uri,
                "netcdf_engine": engine,
                "feature_version": FEATURE_VERSION,
                "parquet_columns": list(frame.columns),
                **extraction_metadata,
            }
            finish_audit_run(
                audit_run_id,
                "SUCCESS",
                output_uri,
                upserted_rows,
                output_checksum,
                metadata,
            )
            return {
                "status": "PROCESSED",
                "source_uri": source_uri,
                "output_uri": output_uri,
                "row_count": upserted_rows,
                "audit_run_id": audit_run_id,
                "sha256": output_checksum,
                "selected_latitude": extraction_metadata["selected_latitude"],
                "selected_longitude": extraction_metadata["selected_longitude"],
                "time_fallback_used": extraction_metadata["time_fallback_used"],
            }
    except Exception as exc:
        finish_audit_run(
            audit_run_id,
            "FAILED",
            None,
            None,
            None,
            {"source_uri": source_uri, "source_etag": source_etag},
            error_message=str(exc)[:4000],
        )
        raise
