from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
import httpx
import joblib
import numpy as np
import pandas as pd
import psycopg2
from botocore.exceptions import ClientError
from psycopg2.extras import Json, execute_values

from prefect_flows.b62b_core import (
    ARCHIVE_VARIABLES,
    DATASET_VERSION,
    FEATURE_COLUMNS,
    HORIZON_H,
    MODEL_VERSION,
    TARGET,
    assign_frozen_temporal_roles,
    attach_wave_truth_and_lags,
    fit_seasonal_interval,
    fit_vintage_task,
    forecast_metrics,
    normalize_previous_runs_payload,
    paired_block_bootstrap,
    predict_frozen_b62a,
    predict_vintage_task,
    previous_run_variable,
    production_contract,
    seasonal_predictions,
    select_on_valid,
    split_train_calibration,
)


SOURCE_NAME = "b62b_vintage_forecast_shadow_validation"
DATASET_NAME = "maritime_vintage_weather_wave_validation_v1"
OUTPUT_BUCKET = "gold-maritime"
RAW_BUCKET = "bronze-maritime"
OUTPUT_PREFIX = "version=1"
PREVIOUS_RUNS_ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_MODEL = "gfs_seamless"
B62A_MODEL_KEY = "models/b62a/version=1/tail_challenger.joblib"
PREDICTION_TABLE = "serving.maritime_metocean_vintage_shadow_v1"
METRIC_TABLE = "serving.maritime_metocean_vintage_metric_v1"


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _query_frame(sql: str, parameters: tuple[Any, ...] = ()) -> pd.DataFrame:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(sql, parameters)
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _query_one(sql: str, parameters: tuple[Any, ...] = ()) -> dict[str, Any]:
    frame = _query_frame(sql, parameters)
    return {} if frame.empty else frame.iloc[0].to_dict()


def _relation_exists(relation: str) -> bool:
    row = _query_one("SELECT to_regclass(%s) AS relation", (relation,))
    return row.get("relation") is not None


def _require_upstream() -> dict[str, Any]:
    rows = _query_frame(
        """
        SELECT DISTINCT ON (source_name) source_name,status,run_id,metadata,finished_at
        FROM audit.ingestion_run
        WHERE source_name IN (
          'b62_weather_wave_vessel_autogluon',
          'b62a_governed_metocean_augmentation'
        )
        ORDER BY source_name,started_at DESC
        """
    )
    status = {str(row.source_name): row for row in rows.itertuples(index=False)}
    for source in (
        "b62_weather_wave_vessel_autogluon",
        "b62a_governed_metocean_augmentation",
    ):
        if source not in status or str(status[source].status) != "SUCCESS":
            raise RuntimeError(f"A successful {source} run is required before B62B")
    return {
        source: {
            "run_id": str(status[source].run_id),
            "metadata": dict(status[source].metadata or {}),
        }
        for source in status
    }


def _location() -> tuple[float, float]:
    latitude = os.getenv("SMART_PORT_FORECAST_LATITUDE", "").strip()
    longitude = os.getenv("SMART_PORT_FORECAST_LONGITUDE", "").strip()
    if latitude and longitude:
        return float(latitude), float(longitude)
    if _relation_exists("features.maritime_issue_time_weather_forecast_v1"):
        row = _query_one(
            """
            SELECT requested_latitude AS latitude,requested_longitude AS longitude
            FROM features.maritime_issue_time_weather_forecast_v1
            ORDER BY issue_at DESC LIMIT 1
            """
        )
        if row.get("latitude") is not None and row.get("longitude") is not None:
            return float(row["latitude"]), float(row["longitude"])
    row = _query_one(
        """
        SELECT avg(latitude)::double precision AS latitude,
               avg(longitude)::double precision AS longitude
        FROM core.maritime_observation
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
    )
    if row.get("latitude") is None or row.get("longitude") is None:
        raise RuntimeError("Unable to resolve the smart-port forecast location")
    return float(row["latitude"]), float(row["longitude"])


def _load_wave_observations() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT observed_at,AVG(wave_period_s)::double precision AS wave_period_s
        FROM core.maritime_observation
        WHERE quality_flag=0 AND wave_period_s IS NOT NULL
        GROUP BY observed_at ORDER BY observed_at
        """
    )
    if frame.empty:
        raise RuntimeError("No quality-controlled wave-period truth is available")
    return frame


def _load_atmosphere_for_b62a() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT observed_at,wind_speed_ms,wind_direction_deg,pressure_hpa,
               visibility_m,temperature_2m,relative_humidity_2m,
               precipitation,cloud_cover,wind_gusts_10m,
               surface_current_ms,sea_surface_temperature,
               availability_semantics,dataset_version
        FROM features.maritime_external_weather_hourly_v1
        ORDER BY observed_at
        """
    )


def _load_wave_source_for_b62a() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT observed_at,
               AVG(wave_height_m)::double precision AS wave_height_m,
               AVG(wave_period_s)::double precision AS wave_period_s,
               MOD((DEGREES(ATAN2(
                   AVG(SIN(RADIANS(wave_direction_deg))),
                   AVG(COS(RADIANS(wave_direction_deg)))
               )) + 360.0)::numeric,360.0)::double precision AS wave_direction_deg
        FROM core.maritime_observation
        WHERE quality_flag=0 AND wave_height_m IS NOT NULL
          AND wave_period_s IS NOT NULL AND wave_direction_deg IS NOT NULL
        GROUP BY observed_at ORDER BY observed_at
        """
    )


def _put_bytes(client, bucket: str, key: str, payload: bytes, content_type: str) -> str:
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)
    return f"s3://{bucket}/{key}"


def _put_json(client, key: str, payload: Any) -> str:
    return _put_bytes(
        client,
        OUTPUT_BUCKET,
        key,
        json.dumps(_json_ready(payload), sort_keys=True, indent=2).encode("utf-8"),
        "application/json",
    )


def _put_csv(client, key: str, frame: pd.DataFrame) -> str:
    return _put_bytes(client, OUTPUT_BUCKET, key, frame.to_csv(index=False).encode(), "text/csv")


def _put_parquet(client, key: str, frame: pd.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return _put_bytes(
        client, OUTPUT_BUCKET, key, buffer.getvalue(), "application/vnd.apache.parquet"
    )


def _put_model(client, key: str, model: Any) -> str:
    with tempfile.TemporaryDirectory(prefix="b62b-model-") as temporary:
        path = Path(temporary) / "vintage_task.joblib"
        joblib.dump(model, path, compress=3)
        return _put_bytes(client, OUTPUT_BUCKET, key, path.read_bytes(), "application/octet-stream")


def _get_bytes(client, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def _raw_cache_key(start: date, end: date, latitude: float, longitude: float) -> str:
    location = f"lat={latitude:.4f}/lon={longitude:.4f}"
    return (
        "external/open-meteo/previous-runs/b62b/"
        f"model={ARCHIVE_MODEL}/{location}/start={start}/end={end}/response.json"
    )


def _request_previous_runs(
    client,
    start: date,
    end: date,
    latitude: float,
    longitude: float,
    force_download: bool,
) -> tuple[dict[str, Any], str, bool]:
    key = _raw_cache_key(start, end, latitude, longitude)
    if not force_download:
        try:
            payload = json.loads(_get_bytes(client, RAW_BUCKET, key))
            return payload, f"s3://{RAW_BUCKET}/{key}", True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"NoSuchKey", "404", "NoSuchObject"}:
                raise
    parameters = {
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(previous_run_variable(item, 1) for item in ARCHIVE_VARIABLES),
        "models": ARCHIVE_MODEL,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "cell_selection": "sea",
    }
    headers = {"User-Agent": "smart-port-maritime-b62b-vintage-validator/1.0"}
    last_error: Exception | None = None
    with httpx.Client(timeout=180.0, headers=headers, follow_redirects=True) as http:
        for attempt in range(1, 5):
            try:
                response = http.get(PREVIOUS_RUNS_ENDPOINT, params=parameters)
                response.raise_for_status()
                payload = response.json()
                if payload.get("error"):
                    raise RuntimeError(str(payload.get("reason", "Open-Meteo error")))
                archive = {
                    "contract_version": DATASET_VERSION,
                    "provider_endpoint": PREVIOUS_RUNS_ENDPOINT,
                    "request_parameters": parameters,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "response": payload,
                }
                _put_bytes(
                    client,
                    RAW_BUCKET,
                    key,
                    json.dumps(archive, sort_keys=True).encode(),
                    "application/json",
                )
                return archive, f"s3://{RAW_BUCKET}/{key}", False
            except Exception as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(5 * attempt)
    raise RuntimeError(f"Previous-runs request failed for {start}/{end}: {last_error}")


def _download_archive(
    client,
    start: date,
    end: date,
    latitude: float,
    longitude: float,
    chunk_days: int,
    force_download: bool,
    progress: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    manifest = []
    cursor = start
    chunks = []
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        wrapped, uri, cached = _request_previous_runs(
            client, chunk_start, chunk_end, latitude, longitude, force_download
        )
        payload = dict(wrapped.get("response") or wrapped)
        frame = normalize_previous_runs_payload(payload, 1, ARCHIVE_MODEL)
        frames.append(frame)
        manifest.append(
            {
                "start_date": chunk_start,
                "end_date": chunk_end,
                "rows": len(frame),
                "cached": cached,
                "object_uri": uri,
            }
        )
        if progress is not None:
            progress(index, len(chunks), chunk_start, chunk_end)
    archive = pd.concat(frames, ignore_index=True)
    archive = archive.sort_values("valid_at").drop_duplicates("valid_at")
    return archive, pd.DataFrame(manifest)


def _load_b62a_task(client) -> Any:
    payload = _get_bytes(client, OUTPUT_BUCKET, B62A_MODEL_KEY)
    with tempfile.TemporaryDirectory(prefix="b62a-frozen-") as temporary:
        path = Path(temporary) / "tail_challenger.joblib"
        path.write_bytes(payload)
        tasks = joblib.load(path)
    task = tasks.get((TARGET, HORIZON_H))
    if task is None:
        raise RuntimeError("Frozen B62A wave_period_s/h24 task is unavailable")
    return task


def _b62a_supervised_features() -> pd.DataFrame:
    from prefect_flows.b62_core import assign_temporal_roles, prepare_hourly_frame
    from prefect_flows.b62a_core import build_supervised_frame

    waves = _load_wave_source_for_b62a()
    atmosphere = _load_atmosphere_for_b62a()
    hourly, _ = prepare_hourly_frame(waves, atmosphere)
    # Reproduce the frozen B62A feature contract before loading its task.
    hourly, _ = assign_temporal_roles(hourly, validation_days=365, test_days=365)
    return build_supervised_frame(hourly)


def _aligned_b62a_features(
    frame: pd.DataFrame, supervised: pd.DataFrame, task: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = supervised.drop_duplicates("issue_at").set_index("issue_at")
    source = frame.loc[frame["issue_at"].isin(lookup.index)].copy()
    features = lookup.reindex(pd.DatetimeIndex(source["issue_at"]))[task.feature_columns]
    features.index = source.index
    finite = np.isfinite(features.to_numpy(dtype="float64")).all(axis=1)
    return source.loc[finite].copy(), features.loc[finite].copy()


def _role_predictions(
    frame: pd.DataFrame,
    role: str,
    vintage_task: Any,
    residual_quantiles: tuple[float, float],
    b62a_task: Any,
    supervised: pd.DataFrame,
    require_b62a: bool = True,
) -> pd.DataFrame:
    source = frame.copy()
    if source.empty:
        return pd.DataFrame()
    predictions = [
        seasonal_predictions(source, role, residual_quantiles),
        predict_vintage_task(vintage_task, source, role),
    ]
    b62a_source, b62a_features = _aligned_b62a_features(source, supervised, b62a_task)
    if require_b62a:
        if len(b62a_source) != len(source):
            source = b62a_source
            predictions = [
                seasonal_predictions(source, role, residual_quantiles),
                predict_vintage_task(vintage_task, source, role),
            ]
        if source.empty:
            return pd.DataFrame()
        predictions.append(predict_frozen_b62a(b62a_task, b62a_features, source, role))
    output = pd.concat(predictions, ignore_index=True)
    common = output.groupby("model")["issue_at"].nunique()
    if common.nunique() != 1:
        raise RuntimeError(f"B62B candidates are not aligned: {common.to_dict()}")
    return output


def _load_fresh_matured(waves: pd.DataFrame) -> pd.DataFrame:
    if not _relation_exists("features.maritime_issue_time_weather_forecast_v1"):
        return pd.DataFrame()
    max_actual = pd.Timestamp(pd.to_datetime(waves["observed_at"], utc=True).max())
    frame = _query_frame(
        """
        SELECT DISTINCT ON (issue_at)
               issue_at,valid_at,lead_time_h,
               air_temperature_c AS temperature_2m,
               pressure_hpa AS pressure_msl,
               wind_speed_ms AS wind_speed_10m,
               wind_direction_deg AS wind_direction_10m,
               provider,provider_model
        FROM features.maritime_issue_time_weather_forecast_v1
        WHERE available_at>=issue_at AND valid_at>=available_at
          AND abs(lead_time_h-24.0)<=3.1
          AND valid_at<=%s
        ORDER BY issue_at,abs(lead_time_h-24.0),available_at
        """,
        (max_actual.to_pydatetime(),),
    )
    if frame.empty:
        return frame
    frame["availability_semantics"] = "EXACT_LIVE_ISSUE_TIME_CAPTURE"
    frame["operationally_available_at_issue"] = True
    frame["requested_latitude"] = np.nan
    frame["requested_longitude"] = np.nan
    return attach_wave_truth_and_lags(frame, waves, origin_step_h=6)


def _checksum(upstream: dict[str, Any], parameters: dict[str, Any]) -> str:
    payload = {
        "model_version": MODEL_VERSION,
        "upstream": {key: value["run_id"] for key, value in upstream.items()},
        "parameters": parameters,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT run_id,metadata FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s AND checksum=%s AND status='SUCCESS'
            ORDER BY finished_at DESC LIMIT 1
            """,
            (SOURCE_NAME, DATASET_NAME, checksum),
        )
        row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _test_already_consumed() -> bool:
    row = _query_one(
        """
        SELECT count(*)::integer AS count
        FROM audit.ingestion_run
        WHERE source_name=%s
          AND (
            status='SUCCESS'
            OR metadata->'progress'->>'stage' IN (
              'FROZEN_TEST_CONFIRMATORY_ONCE',
              'WRITING_VERSIONED_ARTIFACTS',
              'COMPLETE'
            )
          )
        """,
        (SOURCE_NAME,),
    )
    return int(row.get("count") or 0) > 0


def _start_run(checksum: str, parameters: dict[str, Any]) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "parameters": parameters,
        "synthetic_rows": 0,
        "test_used_for_selection": False,
        "production_promotion_allowed": False,
        "automatic_action_allowed": False,
    }
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO audit.ingestion_run
              (source_name,dataset_name,object_uri,checksum,metadata)
            VALUES (%s,%s,%s,%s,%s) RETURNING run_id
            """,
            (
                SOURCE_NAME,
                DATASET_NAME,
                f"s3://{OUTPUT_BUCKET}/datasets/b62b/{OUTPUT_PREFIX}/",
                checksum,
                Json(metadata),
            ),
        )
        return str(cursor.fetchone()[0])


def _progress(run_id: str, stage: str, **details: Any) -> None:
    payload = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc),
        **details,
    }
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
            (Json(_json_ready({"progress": payload})), run_id),
        )


def _finish(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE audit.ingestion_run
            SET status=%s,row_count=%s,finished_at=now(),metadata=metadata || %s,error_message=%s
            WHERE run_id=%s
            """,
            (status, row_count, Json(_json_ready(metadata)), error_message, run_id),
        )


def _materialize(predictions: pd.DataFrame, metrics: pd.DataFrame, run_id: str) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS serving;
    CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} (
      model_version TEXT NOT NULL,evaluation_role TEXT NOT NULL,
      issue_at TIMESTAMPTZ NOT NULL,valid_at TIMESTAMPTZ NOT NULL,
      horizon_h INTEGER NOT NULL,variable TEXT NOT NULL,model TEXT NOT NULL,
      actual DOUBLE PRECISION,p10 DOUBLE PRECISION,p50 DOUBLE PRECISION,p90 DOUBLE PRECISION,
      materialization_run_id TEXT NOT NULL,
      PRIMARY KEY(model_version,evaluation_role,issue_at,valid_at,model)
    );
    CREATE TABLE IF NOT EXISTS {METRIC_TABLE} (
      model_version TEXT NOT NULL,evaluation_role TEXT NOT NULL,model TEXT NOT NULL,
      rows INTEGER NOT NULL,origins INTEGER NOT NULL,mae DOUBLE PRECISION,
      rmse DOUBLE PRECISION,bias DOUBLE PRECISION,coverage DOUBLE PRECISION,
      mean_interval_width DOUBLE PRECISION,quantile_crossings INTEGER NOT NULL,
      materialization_run_id TEXT NOT NULL,
      PRIMARY KEY(model_version,evaluation_role,model)
    );
    """
    prediction_columns = [
        "model_version", "evaluation_role", "issue_at", "valid_at", "horizon_h",
        "variable", "model", "actual", "p10", "p50", "p90", "materialization_run_id",
    ]
    prediction_rows = predictions.copy()
    prediction_rows.insert(0, "model_version", MODEL_VERSION)
    prediction_rows["materialization_run_id"] = run_id
    prediction_rows = prediction_rows[prediction_columns].replace({np.nan: None})
    metric_rows = metrics.rename(
        columns={
            "MAE": "mae", "RMSE": "rmse", "BIAS": "bias",
            "P10_P90_COVERAGE": "coverage",
            "MEAN_INTERVAL_WIDTH": "mean_interval_width",
        }
    ).copy()
    metric_rows.insert(0, "model_version", MODEL_VERSION)
    metric_rows["materialization_run_id"] = run_id
    metric_columns = [
        "model_version", "evaluation_role", "model", "rows", "origins", "mae",
        "rmse", "bias", "coverage", "mean_interval_width", "quantile_crossings",
        "materialization_run_id",
    ]
    metric_rows = metric_rows[metric_columns].replace({np.nan: None})
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(ddl)
        cursor.execute(f"DELETE FROM {PREDICTION_TABLE} WHERE model_version=%s", (MODEL_VERSION,))
        cursor.execute(f"DELETE FROM {METRIC_TABLE} WHERE model_version=%s", (MODEL_VERSION,))
        if not prediction_rows.empty:
            execute_values(
                cursor,
                f"INSERT INTO {PREDICTION_TABLE} ({','.join(prediction_columns)}) VALUES %s",
                [tuple(row) for row in prediction_rows.itertuples(index=False, name=None)],
                page_size=1_000,
            )
        if not metric_rows.empty:
            execute_values(
                cursor,
                f"INSERT INTO {METRIC_TABLE} ({','.join(metric_columns)}) VALUES %s",
                [tuple(row) for row in metric_rows.itertuples(index=False, name=None)],
                page_size=100,
            )


def _quality_gates(
    dataset: pd.DataFrame,
    boundaries: dict[str, Any],
    selection: dict[str, Any],
    contract: dict[str, Any],
    first_test_use: bool,
) -> pd.DataFrame:
    train = dataset.loc[dataset["evaluation_role"].eq("TRAIN_ARCHIVE")]
    valid = dataset.loc[dataset["evaluation_role"].eq("VALID_ARCHIVE")]
    test = dataset.loc[dataset["evaluation_role"].eq("TEST_CONFIRMATORY_ONCE")]
    checks = [
        ("NO_SYNTHETIC_TARGETS", True, "CRITICAL", 0),
        ("FIXED_LEAD_ARCHIVE_DISCLOSED", dataset["availability_semantics"].eq("FIXED_LEAD_ARCHIVE_NO_EXACT_AVAILABLE_AT").all(), "CRITICAL", DATASET_VERSION),
        ("TRAIN_BEFORE_VALID", train["valid_at"].max() < valid["issue_at"].min(), "CRITICAL", boundaries["train_end"]),
        ("VALID_BEFORE_TEST", valid["valid_at"].max() < test["issue_at"].min(), "CRITICAL", boundaries["test_start"]),
        ("VALID_SELECTION_ONLY", not bool(selection.get("test_used_for_selection")), "CRITICAL", selection.get("selection_role")),
        ("TEST_USE_ROLE_DISCLOSED", True, "CRITICAL", "TEST_CONFIRMATORY_ONCE" if first_test_use else "TEST_REUSED_DIAGNOSTIC_ONLY"),
        ("ARCHIVE_TEST_CONFIRMED", bool(contract["archive_confirmed"]), "MODEL_GATE", contract["test_bootstrap"].get("gain_ci_low_pct")),
        ("FRESH_FORWARD_CONFIRMED", bool(contract["fresh_confirmed"]), "PRODUCTION_BLOCKER", contract["fresh_origins"]),
        ("AUTOMATIC_ACTION_DISABLED", not bool(contract["automatic_action_allowed"]), "CRITICAL", False),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "severity", "observed_value"])


def run_b62b(
    force: bool = False,
    force_download: bool = False,
    backfill_days: int = 900,
    valid_days: int = 180,
    test_days: int = 180,
    calibration_days: int = 90,
    chunk_days: int = 90,
    min_fresh_origins: int = 60,
    min_fresh_days: int = 30,
    bootstrap_iterations: int = 500,
    min_gain_pct: float = 5.0,
    max_iter: int = 160,
    seed: int = 20260811,
) -> dict[str, Any]:
    upstream = _require_upstream()
    parameters = {
        "backfill_days": backfill_days,
        "valid_days": valid_days,
        "test_days": test_days,
        "calibration_days": calibration_days,
        "chunk_days": chunk_days,
        "min_fresh_origins": min_fresh_origins,
        "min_fresh_days": min_fresh_days,
        "bootstrap_iterations": bootstrap_iterations,
        "min_gain_pct": min_gain_pct,
        "max_iter": max_iter,
        "seed": seed,
    }
    checksum = _checksum(upstream, parameters)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {"status": "SUCCESS", "run_id": previous[0], "reused": True, "results": previous[1]}
    first_test_use = not _test_already_consumed()
    run_id = _start_run(checksum, parameters)
    client = _s3_client()
    try:
        latitude, longitude = _location()
        waves = _load_wave_observations()
        last_truth = pd.Timestamp(pd.to_datetime(waves["observed_at"], utc=True).max()).floor("D")
        end = min(last_truth.date(), (datetime.now(timezone.utc) - timedelta(days=8)).date())
        start = end - timedelta(days=backfill_days - 1)
        _progress(run_id, "DOWNLOADING_AUTHENTIC_FIXED_LEAD_ARCHIVE", start=start, end=end)

        def download_progress(completed: int, total: int, chunk_start: date, chunk_end: date) -> None:
            _progress(
                run_id,
                "DOWNLOADING_AUTHENTIC_FIXED_LEAD_ARCHIVE",
                completed=completed,
                total=total,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )

        archive, manifest = _download_archive(
            client,
            start,
            end,
            latitude,
            longitude,
            chunk_days,
            force_download,
            download_progress,
        )
        dataset = attach_wave_truth_and_lags(archive, waves, origin_step_h=6)
        dataset, boundaries = assign_frozen_temporal_roles(dataset, valid_days, test_days)
        model_train, calibration, calibration_cutoff = split_train_calibration(
            dataset, calibration_days
        )
        _progress(
            run_id,
            "FITTING_REAL_VINTAGE_WEATHER_TO_WAVE_EXPERT",
            archive_rows=len(dataset),
            fit_rows=len(model_train),
            calibration_rows=len(calibration),
        )
        vintage_task = fit_vintage_task(model_train, calibration, max_iter, seed)
        residual_quantiles = fit_seasonal_interval(calibration)
        b62a_task = _load_b62a_task(client)
        supervised = _b62a_supervised_features()

        valid_frame = dataset.loc[dataset["evaluation_role"].eq("VALID_ARCHIVE")]
        valid_predictions = _role_predictions(
            valid_frame,
            "VALID_ARCHIVE",
            vintage_task,
            residual_quantiles,
            b62a_task,
            supervised,
        )
        selection, valid_metrics = select_on_valid(
            valid_predictions, bootstrap_iterations, min_gain_pct, seed
        )
        _progress(
            run_id,
            "FROZEN_TEST_CONFIRMATORY_ONCE",
            selected_model=selection["selected_model"],
            first_test_use=first_test_use,
        )
        test_frame = dataset.loc[dataset["evaluation_role"].eq("TEST_CONFIRMATORY_ONCE")]
        test_predictions = _role_predictions(
            test_frame,
            "TEST_CONFIRMATORY_ONCE",
            vintage_task,
            residual_quantiles,
            b62a_task,
            supervised,
        )
        fresh_frame = _load_fresh_matured(waves)
        fresh_predictions = _role_predictions(
            fresh_frame,
            "FRESH_FORWARD_CONFIRMATORY",
            vintage_task,
            residual_quantiles,
            b62a_task,
            supervised,
            require_b62a=False,
        ) if not fresh_frame.empty else pd.DataFrame()

        contract = production_contract(
            test_predictions,
            fresh_predictions,
            "VINTAGE_WEATHER_TO_WAVE_HGB_CONFORMAL",
            "B62A_AUGMENTED_QUANTILE_HGB_CONFORMAL_FROZEN",
            "B62_SEASONAL_NAIVE_168H",
            min_fresh_origins,
            min_fresh_days,
            bootstrap_iterations,
            min_gain_pct,
            seed,
        )
        if not first_test_use:
            contract["archive_confirmed"] = False
            contract["limited_production_pilot_allowed"] = False
            contract["test_reuse_disclosure"] = "TEST_REUSED_DIAGNOSTIC_ONLY"
        all_predictions = pd.concat(
            [item for item in (valid_predictions, test_predictions, fresh_predictions) if not item.empty],
            ignore_index=True,
        )
        all_metrics = forecast_metrics(all_predictions)
        gates = _quality_gates(dataset, boundaries, selection, contract, first_test_use)
        critical_passed = bool(gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all())
        if not critical_passed:
            decision = "B62B_CRITICAL_GOVERNANCE_FAILURE"
        elif not bool(selection["valid_accepted"]):
            decision = "B62B_ARCHIVE_CHALLENGER_REJECTED_KEEP_B62A"
        elif not bool(contract["archive_confirmed"]):
            decision = "B62B_TEST_NOT_CONFIRMED_KEEP_SHADOW"
        elif bool(contract["limited_production_pilot_allowed"]):
            decision = "READY_FOR_LIMITED_METOCEAN_PILOT_MANUAL_DECISIONS_ONLY"
        else:
            decision = "READY_FOR_FRESH_FORWARD_SHADOW_VALIDATION"
        production_allowed = bool(
            critical_passed
            and bool(selection["valid_accepted"])
            and contract["limited_production_pilot_allowed"]
        )
        next_block = (
            "B62C_LIMITED_MANUAL_DECISION_PILOT"
            if production_allowed
            else "B62B_CONTINUE_FRESH_FORWARD_COLLECTION"
        )

        _materialize(all_predictions, all_metrics, run_id)
        report_root = f"reports/b62b/{OUTPUT_PREFIX}"
        dataset_root = f"datasets/b62b/{OUTPUT_PREFIX}"
        prediction_root = f"predictions/b62b/{OUTPUT_PREFIX}"
        model_root = f"models/b62b/{OUTPUT_PREFIX}"
        bootstrap_rows = [
            {"role": "VALID_SELECTION", **{key: value for key, value in selection.items() if "gain_" in key or key in {"reference_model", "candidate_model", "origins", "clusters"}}},
            {"role": "TEST_CONFIRMATORY", **contract["test_bootstrap"]},
        ]
        if contract.get("fresh_bootstrap"):
            bootstrap_rows.append({"role": "FRESH_FORWARD", **contract["fresh_bootstrap"]})
        bootstrap_report = pd.DataFrame(bootstrap_rows)
        split_report = (
            dataset.groupby("evaluation_role")
            .agg(rows=("issue_at", "size"), origins=("issue_at", "nunique"), first_issue_at=("issue_at", "min"), last_valid_at=("valid_at", "max"))
            .reset_index()
        )
        artifacts = {
            "archive_dataset": _put_parquet(client, f"{dataset_root}/vintage_fixed_lead_dataset.parquet", dataset),
            "archive_manifest": _put_csv(client, f"{report_root}/01_archive_manifest.csv", manifest),
            "temporal_split": _put_csv(client, f"{report_root}/02_temporal_split.csv", split_report),
            "valid_metrics": _put_csv(client, f"{report_root}/03_valid_selection_metrics.csv", valid_metrics),
            "all_metrics": _put_csv(client, f"{report_root}/04_test_and_fresh_metrics.csv", all_metrics),
            "bootstrap": _put_csv(client, f"{report_root}/05_block_bootstrap.csv", bootstrap_report),
            "quality_gates": _put_csv(client, f"{report_root}/06_quality_gates.csv", gates),
            "predictions": _put_parquet(client, f"{prediction_root}/shadow_predictions.parquet", all_predictions),
            "vintage_model": _put_model(client, f"{model_root}/vintage_weather_to_wave.joblib", vintage_task),
        }
        metadata = {
            "decision": decision,
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "rows": len(dataset),
            "archive_origins": int(dataset["issue_at"].nunique()),
            "valid_origins": int(valid_predictions["issue_at"].nunique()),
            "test_origins": int(test_predictions["issue_at"].nunique()),
            "fresh_origins": int(contract["fresh_origins"]),
            "fresh_span_days": float(contract["fresh_span_days"]),
            "selected_model": selection["selected_model"],
            "reference_model": selection["reference_model"],
            "valid_accepted": bool(selection["valid_accepted"]),
            "archive_confirmed": bool(contract["archive_confirmed"]),
            "fresh_confirmed": bool(contract["fresh_confirmed"]),
            "selection_role": "VALID_ARCHIVE_ONLY",
            "test_role": "TEST_CONFIRMATORY_ONCE" if first_test_use else "TEST_REUSED_DIAGNOSTIC_ONLY",
            "archive_role": "AUTHENTIC_FIXED_LEAD_FORECAST_BACKTEST_NO_EXACT_AVAILABLE_AT",
            "fresh_role": "EXACT_LIVE_ISSUE_TIME_CONFIRMATORY",
            "synthetic_rows": 0,
            "targets_imputed": False,
            "test_used_for_selection": False,
            "critical_gates_passed": critical_passed,
            "production_promotion_allowed": production_allowed,
            "limited_pilot_allowed": production_allowed,
            "automatic_action_allowed": False,
            "calibration_cutoff": calibration_cutoff,
            "boundaries": boundaries,
            "selection": selection,
            "production_contract": contract,
            "latitude": latitude,
            "longitude": longitude,
            "upstream": upstream,
            "artifacts": artifacts,
            "next_block": next_block,
        }
        model_card = {
            **metadata,
            "architecture": "FROZEN_B62_SEASONAL_PLUS_FROZEN_B62A_PLUS_AUTHENTIC_VINTAGE_WEATHER_TO_WAVE_HGB_CONFORMAL",
            "scientific_contract": [
                "No synthetic target or synthetic evaluation row is used.",
                "VALID selects; frozen TEST is consumed once and never tunes the model.",
                "Previous-runs archive is fixed-lead evidence, not exact availability evidence.",
                "Only fresh issue-time captures can unlock a limited manual pilot.",
                "Automatic vessel or navigation actions remain disabled.",
            ],
            "features": list(FEATURE_COLUMNS),
        }
        artifacts["model_card"] = _put_json(client, f"{model_root}/model_card.json", model_card)
        metadata["artifacts"] = artifacts
        metadata["progress"] = {"stage": "COMPLETE", "updated_at": datetime.now(timezone.utc)}
        _finish(run_id, "SUCCESS", len(dataset), metadata)
        return {"status": "SUCCESS", "run_id": run_id, "reused": False, "results": _json_ready(metadata)}
    except Exception as exc:
        _finish(
            run_id,
            "FAILED",
            None,
            {
                "decision": "FAILED",
                "model_version": MODEL_VERSION,
                "production_promotion_allowed": False,
                "automatic_action_allowed": False,
            },
            str(exc),
        )
        raise


def verify_b62b_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"B62B did not succeed: {result}")
    metadata = dict(result.get("results") or {})
    if not metadata.get("critical_gates_passed"):
        raise RuntimeError("B62B critical governance gates failed")
    if metadata.get("test_used_for_selection"):
        raise RuntimeError("B62B illegally used TEST for model selection")
    if int(metadata.get("synthetic_rows", -1)) != 0:
        raise RuntimeError("B62B must not contain synthetic evaluation rows")
    if metadata.get("automatic_action_allowed"):
        raise RuntimeError("B62B automatic actions must remain disabled")
    return metadata
