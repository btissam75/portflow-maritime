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
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import psycopg2
from matplotlib import pyplot as plt
from psycopg2.extras import Json
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression

AUDIT_VERSION = "b54fd0-independent-train-readiness-v2"
SOURCE_NAME = "b54fc_split_artifacts"
DATASET_NAME = "b54fd0_train_readiness_audit"
SOURCE_VIEW = "features.port_call_model_ready_v1"
TARGET = "target_arrival_delay_h"
CALL_COLUMN = "port_call_id"
TIME_COLUMN = "prediction_at"
OFFICIAL_PROTOCOL = "TEMPORAL_PURGED"
WINDOWS_H = (3, 6, 12, 24, 72)
LAGS_H = (3, 6, 12, 24)
MAX_SEA_AGE_H = float(os.getenv("B54C_MAX_SEA_AGE_H", "2"))
RANDOM_SEED = int(os.getenv("B54FD0_RANDOM_SEED", "42"))
MI_MAX_ROWS = int(os.getenv("B54FD0_MI_MAX_ROWS", "25000"))
MIN_CORR_ROWS = int(os.getenv("B54FD0_MIN_CORR_ROWS", "100"))

CALENDAR_FEATURES = (
    "cutoff_hour",
    "cutoff_dayofweek",
    "cutoff_month",
    "cutoff_year",
    "cutoff_weekend_flag",
    "cutoff_hour_sin",
    "cutoff_hour_cos",
    "cutoff_dow_sin",
    "cutoff_dow_cos",
    "cutoff_month_sin",
    "cutoff_month_cos",
    "eta_hour",
    "eta_dayofweek",
    "eta_month",
    "eta_weekend_flag",
    "eta_hour_sin",
    "eta_hour_cos",
    "eta_dow_sin",
    "eta_dow_cos",
)

HISTORY_FEATURES = (
    "vessel_hist_count",
    "vessel_hist_mean_delay_h",
    "vessel_hist_std_delay_h",
    "vessel_hist_late_gt_1h_rate",
    "vessel_hist_late_gt_6h_rate",
    "global_hist_count",
    "global_hist_mean_delay_h",
    "global_hist_std_delay_h",
    "global_hist_late_gt_1h_rate",
    "global_hist_late_gt_6h_rate",
)

WEATHER_PREFIXES = (
    "wave_",
    "high_wave_",
    "severe_wave_",
    "wind_",
    "surface_current_",
    "visibility_",
    "pressure_",
    "sea_observation_age_h",
    "sea_feature_",
)

BANNED_EXACT = {
    "port_call_id",
    "source_record_id",
    "voyage_id",
    "actual_ata",
    "actual_atd",
    "arrival_delay_h",
    "departure_delay_h",
    "target_arrival_delay_h",
    "target_departure_delay_h",
    "prediction_at",
    "planned_eta",
    "planned_etd",
    "model_ready_flag",
    "arrived_before_cutoff_flag",
    "exclusion_reason",
    "observed_at",
    "vessel_history_event_time",
    "global_history_event_time",
}

BANNED_TOKENS = (
    "actual_",
    "future_",
    "quarantine",
    "exclusion",
    "outlier",
)


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, pd.Series, np.ndarray)):
        return [_json_safe(item) for item in value]
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _strict_json_dumps(value: Any, **kwargs) -> str:
    return json.dumps(_json_safe(value), allow_nan=False, **kwargs)


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _etag(client, bucket: str, key: str) -> str:
    return str(client.head_object(Bucket=bucket, Key=key)["ETag"]).strip('"')


def _download(client, bucket: str, key: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))
    return destination


def _upload(
    client,
    source: Path,
    bucket: str,
    key: str,
    content_type: str,
) -> str:
    client.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{bucket}/{key}"


def _signature(
    client,
    objects: list[tuple[str, str]],
    sample_size: int,
    numeric_atol: float,
    numeric_rtol: float,
) -> str:
    payload = {
        "audit_version": AUDIT_VERSION,
        "objects": [
            {"bucket": bucket, "key": key, "etag": _etag(client, bucket, key)}
            for bucket, key in objects
        ],
        "sample_size": sample_size,
        "numeric_atol": numeric_atol,
        "numeric_rtol": numeric_rtol,
        "max_sea_age_h": MAX_SEA_AGE_H,
        "random_seed": RANDOM_SEED,
    }
    return hashlib.sha256(
        _strict_json_dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _start_run(source_uri: str, checksum: str, metadata: dict[str, Any]) -> str:
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
                    source_uri,
                    checksum,
                    Json(_json_safe(metadata), dumps=_strict_json_dumps),
                ),
            )
            return str(cursor.fetchone()[0])


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at = now(), status = %s, row_count = %s,
                    metadata = metadata || %s, error_message = %s
                WHERE run_id = %s
                """,
                (
                    status,
                    row_count,
                    Json(_json_safe(metadata or {}), dumps=_strict_json_dumps),
                    error_message,
                    run_id,
                ),
            )


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, metadata
                FROM audit.ingestion_run
                WHERE source_name = %s AND dataset_name = %s
                  AND checksum = %s AND status = 'SUCCESS'
                ORDER BY finished_at DESC LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    if row is None:
        return None
    return str(row[0]), dict(row[1] or {})


def _safe_feature_name(column: str) -> bool:
    lowered = column.lower()
    if column in BANNED_EXACT:
        return False
    if any(token in lowered for token in BANNED_TOKENS):
        return False
    if lowered.startswith("target_"):
        return False
    if "delay" in lowered and not lowered.startswith(("vessel_hist_", "global_hist_")):
        return False
    return True


def _is_weather_feature(column: str) -> bool:
    return column.startswith(WEATHER_PREFIXES)


def _normalize_imo(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype("string")


def _numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _missing_mask(series: pd.Series, categorical: bool = False) -> pd.Series:
    missing = series.isna()
    if categorical:
        text = series.astype("string").str.strip().str.upper()
        missing |= text.isin(["", "NAN", "NONE", "<NA>"]).fillna(True)
    return missing


def _load_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT source
                FROM core.maritime_observation
                GROUP BY source
                ORDER BY count(*) DESC, source
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("core.maritime_observation is empty")
            sea_source = str(row[0])

        calls = pd.read_sql_query(
            f"SELECT * FROM {SOURCE_VIEW} ORDER BY planned_eta", connection
        )
        sea = pd.read_sql_query(
            """
            SELECT
                observed_at,
                source AS sea_source,
                latitude AS sea_latitude,
                longitude AS sea_longitude,
                wave_height_m,
                wave_period_s,
                wave_direction_deg,
                wind_speed_ms,
                wind_direction_deg,
                surface_current_ms,
                visibility_m,
                pressure_hpa,
                quality_flag AS sea_quality_flag
            FROM core.maritime_observation
            WHERE source = %s
            ORDER BY observed_at
            """,
            connection,
            params=(sea_source,),
        )

    required_calls = {CALL_COLUMN, "imo", "planned_eta", "actual_ata"}
    missing = sorted(required_calls.difference(calls.columns))
    if missing:
        raise RuntimeError(f"{SOURCE_VIEW} is missing columns: {missing}")
    if "arrival_delay_h" not in calls.columns:
        calls["arrival_delay_h"] = (
            pd.to_datetime(calls["actual_ata"], utc=True)
            - pd.to_datetime(calls["planned_eta"], utc=True)
        ).dt.total_seconds() / 3600.0

    for column in ("planned_eta", "actual_ata"):
        calls[column] = pd.to_datetime(calls[column], errors="coerce", utc=True)
    calls[CALL_COLUMN] = calls[CALL_COLUMN].astype("string")
    calls["imo_key"] = _normalize_imo(calls["imo"])
    calls["arrival_delay_h"] = _numeric(calls["arrival_delay_h"])

    sea["observed_at"] = pd.to_datetime(sea["observed_at"], errors="coerce", utc=True)
    sea = sea.loc[sea["observed_at"].notna()].copy()
    return calls, sea, sea_source


def _prepare_sea(sea: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "wind_speed_ms",
        "wind_direction_deg",
        "surface_current_ms",
        "visibility_m",
        "pressure_hpa",
        "sea_quality_flag",
    ]
    result = sea.copy()
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = (
        result.sort_values(["observed_at", "sea_quality_flag"])
        .drop_duplicates("observed_at", keep="first")
        .reset_index(drop=True)
    )
    radians = np.deg2rad(result["wave_direction_deg"] % 360.0)
    result["wave_direction_sin"] = np.sin(radians)
    result["wave_direction_cos"] = np.cos(radians)
    result["wave_energy_proxy"] = (
        result["wave_height_m"].pow(2) * result["wave_period_s"]
    )
    result["high_wave_ge_1p5m"] = (result["wave_height_m"] >= 1.5).astype("int8")
    result["high_wave_ge_2p5m"] = (result["wave_height_m"] >= 2.5).astype("int8")
    result["severe_wave_ge_4m"] = (result["wave_height_m"] >= 4.0).astype("int8")
    return result


def _calendar_values(prediction_at: pd.Timestamp, planned_eta: pd.Timestamp) -> dict[str, float]:
    prediction = pd.Timestamp(prediction_at)
    eta = pd.Timestamp(planned_eta)
    return {
        "cutoff_hour": prediction.hour,
        "cutoff_dayofweek": prediction.dayofweek,
        "cutoff_month": prediction.month,
        "cutoff_year": prediction.year,
        "cutoff_weekend_flag": int(prediction.dayofweek >= 5),
        "cutoff_hour_sin": math.sin(2.0 * math.pi * prediction.hour / 24.0),
        "cutoff_hour_cos": math.cos(2.0 * math.pi * prediction.hour / 24.0),
        "cutoff_dow_sin": math.sin(2.0 * math.pi * prediction.dayofweek / 7.0),
        "cutoff_dow_cos": math.cos(2.0 * math.pi * prediction.dayofweek / 7.0),
        "cutoff_month_sin": math.sin(2.0 * math.pi * (prediction.month - 1) / 12.0),
        "cutoff_month_cos": math.cos(2.0 * math.pi * (prediction.month - 1) / 12.0),
        "eta_hour": eta.hour,
        "eta_dayofweek": eta.dayofweek,
        "eta_month": eta.month,
        "eta_weekend_flag": int(eta.dayofweek >= 5),
        "eta_hour_sin": math.sin(2.0 * math.pi * eta.hour / 24.0),
        "eta_hour_cos": math.cos(2.0 * math.pi * eta.hour / 24.0),
        "eta_dow_sin": math.sin(2.0 * math.pi * eta.dayofweek / 7.0),
        "eta_dow_cos": math.cos(2.0 * math.pi * eta.dayofweek / 7.0),
    }


def _prefix_index(times: pd.Series, delays: pd.Series) -> dict[str, np.ndarray]:
    work = pd.DataFrame({"time": times, "delay": delays}).dropna()
    work = work.sort_values("time").reset_index(drop=True)
    values = work["delay"].to_numpy(dtype="float64")
    return {
        "times": work["time"].astype("int64").to_numpy(),
        "values": values,
        "sum": np.cumsum(values),
        "sumsq": np.cumsum(values * values),
        "late1": np.cumsum(values > 1.0),
        "late6": np.cumsum(values > 6.0),
    }


def build_history_indexes(calls: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    history = calls.dropna(subset=["actual_ata", "arrival_delay_h"]).copy()
    global_index = _prefix_index(history["actual_ata"], history["arrival_delay_h"])
    vessel_indexes: dict[str, dict[str, np.ndarray]] = {}
    for key, part in history.loc[history["imo_key"].notna()].groupby("imo_key"):
        vessel_indexes[str(key)] = _prefix_index(part["actual_ata"], part["arrival_delay_h"])
    return global_index, vessel_indexes


def _history_stats(
    index: dict[str, np.ndarray] | None,
    prediction_at: pd.Timestamp,
) -> tuple[dict[str, float], pd.Timestamp | None]:
    if not index or len(index["times"]) == 0:
        return {
            "count": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "late1": np.nan,
            "late6": np.nan,
        }, None
    prediction_ns = pd.Timestamp(prediction_at).value
    count = int(np.searchsorted(index["times"], prediction_ns, side="left"))
    if count <= 0:
        return {
            "count": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "late1": np.nan,
            "late6": np.nan,
        }, None
    position = count - 1
    total = float(index["sum"][position])
    mean = total / count
    variance = max(0.0, float(index["sumsq"][position]) / count - mean * mean)
    stats = {
        "count": float(count),
        "mean": mean,
        "std": math.sqrt(variance),
        "late1": float(index["late1"][position]) / count,
        "late6": float(index["late6"][position]) / count,
    }
    return stats, pd.Timestamp(int(index["times"][position]), tz="UTC")


def _history_values(
    prediction_at: pd.Timestamp,
    imo_key: str | None,
    global_index: dict[str, np.ndarray],
    vessel_indexes: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, float], dict[str, pd.Timestamp | None]]:
    vessel, vessel_time = _history_stats(
        vessel_indexes.get(str(imo_key)) if pd.notna(imo_key) else None,
        prediction_at,
    )
    global_stats, global_time = _history_stats(global_index, prediction_at)
    values = {
        "vessel_hist_count": vessel["count"],
        "vessel_hist_mean_delay_h": vessel["mean"],
        "vessel_hist_std_delay_h": vessel["std"],
        "vessel_hist_late_gt_1h_rate": vessel["late1"],
        "vessel_hist_late_gt_6h_rate": vessel["late6"],
        "global_hist_count": global_stats["count"],
        "global_hist_mean_delay_h": global_stats["mean"],
        "global_hist_std_delay_h": global_stats["std"],
        "global_hist_late_gt_1h_rate": global_stats["late1"],
        "global_hist_late_gt_6h_rate": global_stats["late6"],
    }
    return values, {"vessel": vessel_time, "global": global_time}


def _nanmean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def _weather_values(
    sea: pd.DataFrame,
    prediction_at: pd.Timestamp,
) -> tuple[dict[str, float], pd.Timestamp | None]:
    if sea.empty:
        return {}, None
    times = sea["observed_at"].astype("int64").to_numpy()
    prediction = pd.Timestamp(prediction_at)
    latest_position = int(np.searchsorted(times, prediction.value, side="right") - 1)
    if latest_position < 0:
        return {}, None
    latest_time = pd.Timestamp(int(times[latest_position]), tz="UTC")
    if prediction - latest_time > pd.Timedelta(hours=max(3.0, MAX_SEA_AGE_H + 1.0)):
        return {}, None

    latest = sea.iloc[latest_position]
    height = pd.to_numeric(pd.Series([latest["wave_height_m"]]), errors="coerce").iloc[0]
    period = pd.to_numeric(pd.Series([latest["wave_period_s"]]), errors="coerce").iloc[0]
    direction = pd.to_numeric(
        pd.Series([latest["wave_direction_deg"]]), errors="coerce"
    ).iloc[0]
    radians = np.deg2rad(direction % 360.0) if pd.notna(direction) else np.nan
    age_h = (prediction - latest_time).total_seconds() / 3600.0
    quality = pd.to_numeric(pd.Series([latest["sea_quality_flag"]]), errors="coerce").iloc[0]

    values: dict[str, float] = {
        "wave_height_m": height,
        "wave_period_s": period,
        "wave_direction_sin": math.sin(radians) if pd.notna(radians) else np.nan,
        "wave_direction_cos": math.cos(radians) if pd.notna(radians) else np.nan,
        "wave_energy_proxy": height * height * period
        if pd.notna(height) and pd.notna(period)
        else np.nan,
        "high_wave_ge_1p5m": int(pd.notna(height) and height >= 1.5),
        "high_wave_ge_2p5m": int(pd.notna(height) and height >= 2.5),
        "severe_wave_ge_4m": int(pd.notna(height) and height >= 4.0),
        "wind_speed_ms": latest["wind_speed_ms"],
        "surface_current_ms": latest["surface_current_ms"],
        "visibility_m": latest["visibility_m"],
        "pressure_hpa": latest["pressure_hpa"],
        "sea_observation_age_h": age_h,
        "sea_feature_available_flag": int(
            pd.notna(height) and 0.0 <= age_h <= MAX_SEA_AGE_H and quality == 0
        ),
        "sea_feature_stale_flag": int(age_h > MAX_SEA_AGE_H),
    }

    for hours in WINDOWS_H:
        lower_ns = (latest_time - pd.Timedelta(hours=hours)).value
        start = int(np.searchsorted(times, lower_ns, side="right"))
        part = sea.iloc[start : latest_position + 1]
        part_height = pd.to_numeric(part["wave_height_m"], errors="coerce")
        part_period = pd.to_numeric(part["wave_period_s"], errors="coerce")
        part_energy = part_height.pow(2) * part_period
        count = int(part_height.count())
        values[f"wave_height_mean_{hours}h"] = _nanmean(part_height)
        values[f"wave_height_max_{hours}h"] = (
            float(part_height.max()) if part_height.notna().any() else np.nan
        )
        values[f"wave_height_std_{hours}h"] = (
            float(part_height.std(ddof=1)) if count >= 2 else 0.0
        )
        values[f"wave_period_mean_{hours}h"] = _nanmean(part_period)
        values[f"wave_energy_mean_{hours}h"] = _nanmean(part_energy)
        values[f"high_wave_hours_ge_1p5m_{hours}h"] = int((part_height >= 1.5).sum())
        values[f"high_wave_hours_ge_2p5m_{hours}h"] = int((part_height >= 2.5).sum())
        part_direction = pd.to_numeric(part["wave_direction_deg"], errors="coerce")
        part_radians = np.deg2rad(part_direction % 360.0)
        sin_mean = float(np.nanmean(np.sin(part_radians))) if part_direction.notna().any() else np.nan
        cos_mean = float(np.nanmean(np.cos(part_radians))) if part_direction.notna().any() else np.nan
        values[f"wave_direction_concentration_{hours}h"] = (
            math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
            if np.isfinite(sin_mean) and np.isfinite(cos_mean)
            else np.nan
        )
        values[f"wave_observation_count_{hours}h"] = float(count)
        values[f"wave_coverage_pct_{hours}h"] = min(100.0, 100.0 * count / hours)

    for hours in LAGS_H:
        lag_target = latest_time - pd.Timedelta(hours=hours)
        lag_position = int(np.searchsorted(times, lag_target.value, side="right") - 1)
        lag = np.nan
        if lag_position >= 0:
            lag_time = pd.Timestamp(int(times[lag_position]), tz="UTC")
            if lag_target - lag_time <= pd.Timedelta(hours=2):
                lag = pd.to_numeric(
                    pd.Series([sea.iloc[lag_position]["wave_height_m"]]),
                    errors="coerce",
                ).iloc[0]
        values[f"wave_height_lag_{hours}h"] = lag
        values[f"wave_height_trend_{hours}h"] = (
            height - lag if pd.notna(height) and pd.notna(lag) else np.nan
        )
    return values, latest_time


def _deterministic_stratified_sample(frame: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    source = frame.copy()
    source["prediction_year"] = source[TIME_COLUMN].dt.year.astype("Int64")
    source["_hash"] = pd.util.hash_pandas_object(
        source[CALL_COLUMN].astype("string"), index=False
    ).astype("uint64")
    groups = list(source.groupby(["split", "prediction_year"], dropna=False))
    if not groups:
        return source.head(0)
    quota = max(1, sample_size // len(groups))
    selected_indexes: list[int] = []
    for _, part in groups:
        selected_indexes.extend(part.nsmallest(quota, "_hash").index.tolist())
    remaining = sample_size - len(selected_indexes)
    if remaining > 0:
        leftovers = source.drop(index=selected_indexes).nsmallest(remaining, "_hash")
        selected_indexes.extend(leftovers.index.tolist())
    sampled = source.loc[selected_indexes].nsmallest(sample_size, "_hash").copy()
    return sampled.drop(columns="_hash").reset_index(drop=True)


def _value_match(
    stored: Any,
    expected: Any,
    atol: float,
    rtol: float,
) -> tuple[bool, float | None, str]:
    stored_missing = pd.isna(stored)
    expected_missing = pd.isna(expected)
    if stored_missing and expected_missing:
        return True, None, "BOTH_MISSING"
    if stored_missing != expected_missing:
        return False, None, "MISSINGNESS_MISMATCH"
    try:
        left = float(stored)
        right = float(expected)
    except (TypeError, ValueError):
        passed = str(stored) == str(expected)
        return passed, None, "EXACT" if passed else "VALUE_MISMATCH"
    if not np.isfinite(left) or not np.isfinite(right):
        return False, None, "NON_FINITE"
    error = abs(left - right)
    tolerance = atol + rtol * max(1.0, abs(left), abs(right))
    return error <= tolerance, error, "NUMERIC_TOLERANCE"


def build_recalculation_proof(
    sample: pd.DataFrame,
    calls: pd.DataFrame,
    sea: pd.DataFrame,
    numeric_features: list[str],
    numeric_atol: float,
    numeric_rtol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_index, vessel_indexes = build_history_indexes(calls)
    prepared_sea = _prepare_sea(sea)
    check_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for record in sample.to_dict(orient="records"):
        prediction_at = pd.Timestamp(record[TIME_COLUMN])
        planned_eta = pd.Timestamp(record["planned_eta"])
        normalized_imo = _normalize_imo(pd.Series([record.get("imo")])).iloc[0]
        imo_key = None if pd.isna(normalized_imo) else str(normalized_imo)
        calendar = _calendar_values(prediction_at, planned_eta)
        history, history_times = _history_values(
            prediction_at, imo_key, global_index, vessel_indexes
        )
        weather, weather_time = _weather_values(prepared_sea, prediction_at)
        expected = {**calendar, **history, **weather}

        manifest_rows.append(
            {
                CALL_COLUMN: record[CALL_COLUMN],
                "split": record["split"],
                "prediction_year": prediction_at.year,
                TIME_COLUMN: prediction_at,
                "planned_eta": planned_eta,
                "imo": record.get("imo"),
                "latest_weather_source_time": weather_time,
                "latest_vessel_history_time": history_times["vessel"],
                "latest_global_history_time": history_times["global"],
            }
        )

        for feature in numeric_features:
            if feature in CALENDAR_FEATURES:
                family = "CALENDAR"
                source_time = prediction_at
                strict_past = False
            elif feature.startswith("vessel_hist_"):
                family = "VESSEL_HISTORY"
                source_time = history_times["vessel"]
                strict_past = True
            elif feature.startswith("global_hist_"):
                family = "GLOBAL_HISTORY"
                source_time = history_times["global"]
                strict_past = True
            elif _is_weather_feature(feature):
                family = "WEATHER"
                source_time = weather_time
                strict_past = False
            else:
                family = "UNSUPPORTED"
                source_time = None
                strict_past = False

            stored = record.get(feature, np.nan)
            supported = feature in expected
            value_expected = expected.get(feature, np.nan)
            passed, absolute_error, comparison = _value_match(
                stored,
                value_expected,
                numeric_atol,
                numeric_rtol,
            )
            if not supported:
                passed = False
                comparison = "UNSUPPORTED_FEATURE"

            future_violation = False
            if source_time is not None and pd.notna(source_time):
                if strict_past:
                    future_violation = bool(pd.Timestamp(source_time) >= prediction_at)
                else:
                    future_violation = bool(pd.Timestamp(source_time) > prediction_at)
            if future_violation:
                passed = False
                comparison = "FUTURE_SOURCE_TIME"

            check_rows.append(
                {
                    CALL_COLUMN: record[CALL_COLUMN],
                    "split": record["split"],
                    TIME_COLUMN: prediction_at,
                    "family": family,
                    "feature": feature,
                    "stored_value": stored,
                    "recomputed_value": value_expected,
                    "absolute_error": absolute_error,
                    "comparison": comparison,
                    "source_max_time": source_time,
                    "future_source_violation": future_violation,
                    "passed": bool(passed),
                }
            )

    return pd.DataFrame(manifest_rows), pd.DataFrame(check_rows)


def summarize_recalculation(checks: pd.DataFrame) -> pd.DataFrame:
    if checks.empty:
        return pd.DataFrame(
            [
                {
                    "family": "ALL",
                    "checks": 0,
                    "passed": 0,
                    "failed": 0,
                    "future_violations": 0,
                    "max_absolute_error": np.nan,
                    "pass_pct": 0.0,
                }
            ]
        )
    rows = []
    for family, part in list(checks.groupby("family")) + [("ALL", checks)]:
        errors = pd.to_numeric(part["absolute_error"], errors="coerce")
        rows.append(
            {
                "family": family,
                "checks": int(len(part)),
                "passed": int(part["passed"].sum()),
                "failed": int((~part["passed"]).sum()),
                "future_violations": int(part["future_source_violation"].sum()),
                "max_absolute_error": float(errors.max()) if errors.notna().any() else np.nan,
                "pass_pct": 100.0 * float(part["passed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_missingness_reports(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    categorical_set = set(categorical_features)
    for split_name in ("TRAIN", "VALID", "TEST"):
        part = frame.loc[frame["split"] == split_name]
        for feature in feature_columns:
            categorical = feature in categorical_set
            series = part[feature]
            missing = _missing_mask(series, categorical)
            if categorical:
                infinite_count = 0
                usable = series.loc[~missing].astype("string")
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                infinite_count = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
                usable = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "feature_type": "CATEGORICAL" if categorical else "NUMERIC",
                    "n_rows": int(len(part)),
                    "missing_count": int(missing.sum()),
                    "missing_pct": 100.0 * float(missing.mean()) if len(part) else np.nan,
                    "infinite_count": infinite_count,
                    "unique_non_missing": int(usable.nunique(dropna=True)),
                    "all_missing": bool(len(usable) == 0),
                    "constant_non_missing": bool(len(usable) > 0 and usable.nunique(dropna=True) <= 1),
                }
            )
    report = pd.DataFrame(rows)
    train = report.loc[report["split"] == "TRAIN"].set_index("feature")
    pivot = report.pivot(index="feature", columns="split", values="missing_pct")
    drift_rows = []
    for feature in feature_columns:
        values = pivot.loc[feature] if feature in pivot.index else pd.Series(dtype=float)
        split_values = pd.to_numeric(values, errors="coerce")
        drift_rows.append(
            {
                "feature": feature,
                "train_missing_pct": values.get("TRAIN", np.nan),
                "valid_missing_pct": values.get("VALID", np.nan),
                "test_missing_pct": values.get("TEST", np.nan),
                "max_missing_shift_pp": (
                    float(split_values.max() - split_values.min())
                    if split_values.notna().any()
                    else np.nan
                ),
                "high_missing_train_flag": bool(values.get("TRAIN", 0.0) > 95.0),
                "availability_drift_gt10pp_flag": bool(
                    split_values.notna().any()
                    and float(split_values.max() - split_values.min()) > 10.0
                ),
            }
        )
    drift = pd.DataFrame(drift_rows)
    dropped = [
        feature
        for feature in feature_columns
        if feature in train.index
        and bool(train.loc[feature, "all_missing"] or train.loc[feature, "constant_non_missing"])
    ]
    infinite = [
        feature
        for feature in feature_columns
        if feature in train.index and int(train.loc[feature, "infinite_count"]) > 0
    ]
    return report, drift, dropped, infinite


def _target_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in ("TRAIN", "VALID", "TEST"):
        values = _numeric(frame.loc[frame["split"] == split_name, TARGET]).dropna()
        rows.append(
            {
                "split": split_name,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "p50": float(values.quantile(0.50)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
                "early_lt_minus1h_pct": 100.0 * float((values < -1.0).mean()),
                "within_1h_pct": 100.0 * float((values.abs() <= 1.0).mean()),
                "delay_1_6h_pct": 100.0 * float(((values > 1.0) & (values <= 6.0)).mean()),
                "delay_gt6h_pct": 100.0 * float((values > 6.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _correlation_matrices(
    train: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = pd.DataFrame(
        {feature: _numeric(train[feature]) for feature in numeric_features},
        index=train.index,
    )
    pearson = numeric.corr(method="pearson", min_periods=MIN_CORR_ROWS)
    spearman = numeric.corr(method="spearman", min_periods=MIN_CORR_ROWS)
    with_target = numeric.copy()
    with_target[TARGET] = _numeric(train[TARGET])
    pearson_target = with_target.corr(method="pearson", min_periods=MIN_CORR_ROWS)
    spearman_target = with_target.corr(method="spearman", min_periods=MIN_CORR_ROWS)
    return pearson, spearman, pearson_target, spearman_target


def _high_correlation_pairs(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    rows = []
    columns = list(pearson.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            p_value = pearson.at[left, right]
            s_value = spearman.at[left, right]
            maximum = np.nanmax(np.abs([p_value, s_value]))
            if np.isfinite(maximum) and maximum >= threshold:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "pearson": p_value,
                        "spearman": s_value,
                        "max_abs_correlation": maximum,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "max_abs_correlation", ascending=False, ignore_index=True
    ) if rows else pd.DataFrame(
        columns=[
            "feature_left",
            "feature_right",
            "pearson",
            "spearman",
            "max_abs_correlation",
        ]
    )


def _mutual_information_numeric(feature: pd.Series, target: pd.Series) -> tuple[float, int]:
    pair = pd.DataFrame({"feature": _numeric(feature), "target": _numeric(target)}).dropna()
    if len(pair) < MIN_CORR_ROWS or pair["feature"].nunique() <= 1:
        return np.nan, int(len(pair))
    if len(pair) > MI_MAX_ROWS:
        pair = pair.sample(MI_MAX_ROWS, random_state=RANDOM_SEED)
    value = mutual_info_regression(
        pair[["feature"]].to_numpy(),
        pair["target"].to_numpy(),
        random_state=RANDOM_SEED,
    )[0]
    return float(value), int(len(pair))


def build_target_associations(
    train: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    target = _numeric(train[TARGET])
    rows = []
    for feature in numeric_features:
        values = _numeric(train[feature])
        pair = pd.DataFrame({"x": values, "y": target}).dropna()
        pearson = pair["x"].corr(pair["y"], method="pearson") if len(pair) >= MIN_CORR_ROWS else np.nan
        spearman = pair["x"].corr(pair["y"], method="spearman") if len(pair) >= MIN_CORR_ROWS else np.nan
        mi, mi_rows = _mutual_information_numeric(values, target)
        rows.append(
            {
                "feature": feature,
                "family": _feature_family(feature),
                "n_pairwise": int(len(pair)),
                "missing_pct_train": 100.0 * float(values.isna().mean()),
                "pearson_with_target": pearson,
                "spearman_with_target": spearman,
                "mutual_information": mi,
                "mi_rows": mi_rows,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["max_abs_correlation"] = result[
            ["pearson_with_target", "spearman_with_target"]
        ].abs().max(axis=1)
        result = result.sort_values(
            ["mutual_information", "max_abs_correlation"], ascending=False
        ).reset_index(drop=True)
    return result


def _eta_squared(categories: pd.Series, target: pd.Series) -> float:
    pair = pd.DataFrame({"category": categories.astype("string"), "target": _numeric(target)}).dropna()
    if pair.empty or pair["category"].nunique() <= 1:
        return np.nan
    overall = pair["target"].mean()
    total = float(((pair["target"] - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, part in pair.groupby("category"):
        between += len(part) * float(part["target"].mean() - overall) ** 2
    return float(between / total)


def _categorical_mi(categories: pd.Series, target: pd.Series) -> tuple[float, int]:
    pair = pd.DataFrame({"category": categories.astype("string"), "target": _numeric(target)}).dropna()
    if len(pair) < MIN_CORR_ROWS or pair["category"].nunique() <= 1:
        return np.nan, int(len(pair))
    if len(pair) > MI_MAX_ROWS:
        pair = pair.sample(MI_MAX_ROWS, random_state=RANDOM_SEED)
    codes, _ = pd.factorize(pair["category"], sort=True)
    mi = mutual_info_regression(
        codes.reshape(-1, 1),
        pair["target"].to_numpy(),
        discrete_features=True,
        random_state=RANDOM_SEED,
    )[0]
    return float(mi), int(len(pair))


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left.astype("string"), "right": right.astype("string")}).dropna()
    table = pd.crosstab(pair["left"], pair["right"])
    if table.empty or min(table.shape) <= 1:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, columns = table.shape
    corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / max(1, n - 1))
    corrected_rows = rows - ((rows - 1) ** 2) / max(1, n - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / max(1, n - 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return math.sqrt(corrected / denominator) if denominator > 0 else np.nan


def build_categorical_reports(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = frame.loc[frame["split"] == "TRAIN"]
    rows = []
    for feature in categorical_features:
        train_values = train[feature].astype("string")
        train_categories = set(train_values.dropna())
        mi, mi_rows = _categorical_mi(train_values, train[TARGET])
        for split_name in ("TRAIN", "VALID", "TEST"):
            part = frame.loc[frame["split"] == split_name]
            values = part[feature].astype("string")
            unseen = values.notna() & ~values.isin(train_categories)
            rows.append(
                {
                    "feature": feature,
                    "split": split_name,
                    "n_rows": int(len(part)),
                    "missing_pct": 100.0 * float(values.isna().mean()),
                    "unique_categories": int(values.nunique(dropna=True)),
                    "unseen_vs_train_count": int(unseen.sum()),
                    "unseen_vs_train_pct": 100.0 * float(unseen.mean()),
                    "train_eta_squared": _eta_squared(train_values, train[TARGET]),
                    "train_mutual_information": mi,
                    "train_mi_rows": mi_rows,
                }
            )

    mapping_rows = []
    primary_categories = list(categorical_features)
    if {"imo", "vessel_name"}.issubset(frame.columns):
        mapping = frame[["imo", "vessel_name"]].copy()
        mapping["imo"] = _normalize_imo(mapping["imo"])
        mapping["normalized_vessel_name"] = (
            mapping["vessel_name"]
            .astype("string")
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        for imo, part in mapping.dropna(subset=["imo"]).groupby("imo"):
            names = sorted(part["normalized_vessel_name"].dropna().unique().tolist())
            mapping_rows.append(
                {
                    "mapping_direction": "IMO_TO_NAME",
                    "mapping_key": imo,
                    "distinct_values": len(names),
                    "conflict_flag": len(names) > 1,
                    "values": " | ".join(names),
                }
            )
        for name, part in mapping.dropna(subset=["normalized_vessel_name"]).groupby("normalized_vessel_name"):
            imos = sorted(part["imo"].dropna().unique().tolist())
            mapping_rows.append(
                {
                    "mapping_direction": "NAME_TO_IMO",
                    "mapping_key": name,
                    "distinct_values": len(imos),
                    "conflict_flag": len(imos) > 1,
                    "values": " | ".join(imos),
                }
            )
        cramer = _cramers_v(mapping["imo"], mapping["normalized_vessel_name"])
        mapping_rows.append(
            {
                "mapping_direction": "GLOBAL_ASSOCIATION",
                "mapping_key": "CRAMERS_V_IMO_VESSEL_NAME",
                "distinct_values": np.nan,
                "conflict_flag": False,
                "values": cramer,
            }
        )
        if "imo" in categorical_features:
            primary_categories = ["imo"]
    return pd.DataFrame(rows), pd.DataFrame(mapping_rows), primary_categories


def build_missingness_target_associations(
    train: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    target = _numeric(train[TARGET])
    rows = []
    for feature in numeric_features:
        values = _numeric(train[feature])
        missing = values.isna().astype("float64")
        if missing.sum() == 0:
            continue
        pair = pd.DataFrame({"missing": missing, "target": target}).dropna()
        missing_target = pair.loc[pair["missing"] == 1, "target"]
        present_target = pair.loc[pair["missing"] == 0, "target"]
        rows.append(
            {
                "feature": feature,
                "missing_count": int(missing.sum()),
                "missing_pct": 100.0 * float(missing.mean()),
                "missing_indicator_target_pearson": pair["missing"].corr(pair["target"])
                if pair["missing"].nunique() > 1
                else np.nan,
                "target_mean_when_missing": float(missing_target.mean())
                if len(missing_target)
                else np.nan,
                "target_mean_when_present": float(present_target.mean())
                if len(present_target)
                else np.nan,
                "target_mean_difference": (
                    float(missing_target.mean() - present_target.mean())
                    if len(missing_target) and len(present_target)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _feature_family(feature: str) -> str:
    if feature.startswith("cutoff_") or feature.startswith("eta_"):
        return "CALENDAR"
    if feature.startswith("vessel_hist_"):
        return "VESSEL_HISTORY"
    if feature.startswith("global_hist_"):
        return "GLOBAL_HISTORY"
    if feature.startswith(("sea_feature_", "sea_observation_", "wave_observation_", "wave_coverage_")):
        return "WEATHER_QUALITY"
    if _is_weather_feature(feature):
        return "WAVE_WEATHER"
    return "OTHER"


def build_rolling_correlation_stability(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    rolling = assignments.loc[
        (assignments["protocol"] == "ROLLING_TEMPORAL_CV")
        & (assignments["split"] == "TRAIN")
    ].copy()
    rows = []
    for fold, assignment in rolling.groupby("fold"):
        train = assignment[[CALL_COLUMN]].merge(
            frame[[CALL_COLUMN, TARGET] + numeric_features],
            on=CALL_COLUMN,
            how="inner",
            validate="one_to_one",
        )
        target = _numeric(train[TARGET])
        for feature in numeric_features:
            pair = pd.DataFrame({"x": _numeric(train[feature]), "y": target}).dropna()
            rows.append(
                {
                    "feature": feature,
                    "fold": int(float(fold)),
                    "n_pairwise": int(len(pair)),
                    "pearson": pair["x"].corr(pair["y"], method="pearson")
                    if len(pair) >= MIN_CORR_ROWS
                    else np.nan,
                    "spearman": pair["x"].corr(pair["y"], method="spearman")
                    if len(pair) >= MIN_CORR_ROWS
                    else np.nan,
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summary_rows = []
    for feature, part in detail.groupby("feature"):
        row: dict[str, Any] = {"feature": feature, "family": _feature_family(feature)}
        for fold in (1, 2, 3):
            fold_row = part.loc[part["fold"] == fold]
            row[f"pearson_fold_{fold}"] = fold_row["pearson"].iloc[0] if len(fold_row) else np.nan
            row[f"spearman_fold_{fold}"] = fold_row["spearman"].iloc[0] if len(fold_row) else np.nan
        pearson_values = pd.Series([row[f"pearson_fold_{fold}"] for fold in (1, 2, 3)])
        spearman_values = pd.Series([row[f"spearman_fold_{fold}"] for fold in (1, 2, 3)])
        row["pearson_range"] = float(pearson_values.max() - pearson_values.min())
        row["spearman_range"] = float(spearman_values.max() - spearman_values.min())
        row["sign_flip_flag"] = bool(
            (pearson_values.dropna().min() < 0 < pearson_values.dropna().max())
            or (spearman_values.dropna().min() < 0 < spearman_values.dropna().max())
        )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(
        ["spearman_range", "pearson_range"], ascending=False
    )


def build_schema_audit(
    frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)
    feature_set = set(feature_columns)
    rows = []
    for position, column in enumerate(frame.columns):
        if column in numeric_set:
            role = "NUMERIC_FEATURE"
        elif column in categorical_set:
            role = "CATEGORICAL_FEATURE"
        elif column == TARGET:
            role = "TARGET_AUDIT_ONLY"
        elif column == CALL_COLUMN:
            role = "IDENTIFIER_AUDIT_ONLY"
        elif column in {TIME_COLUMN, "planned_eta"}:
            role = "TIME_AUDIT_ONLY"
        elif column in feature_set:
            role = "UNCLASSIFIED_FEATURE"
        else:
            role = "METADATA_AUDIT_ONLY"
        series = frame[column]
        rows.append(
            {
                "position": position,
                "column": column,
                "dtype": str(series.dtype),
                "role": role,
                "missing_count": int(series.isna().sum()),
                "missing_pct": 100.0 * float(series.isna().mean()),
                "unique_values": int(series.nunique(dropna=True)),
                "allowed_in_model_matrix": bool(column in feature_set),
                "safe_feature_name": bool(_safe_feature_name(column))
                if column in feature_set
                else None,
            }
        )
    return pd.DataFrame(rows)


def build_split_leakage_recheck(
    frame: pd.DataFrame,
    official_assignments: pd.DataFrame,
    feature_columns: list[str],
    split_decision: dict[str, Any],
    build_report: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, value: Any, expected: str, critical: bool = True):
        rows.append(
            {
                "check": check,
                "passed": bool(passed),
                "critical": bool(critical),
                "observed": value,
                "expected": expected,
            }
        )

    add(
        "source_one_row_per_port_call",
        frame[CALL_COLUMN].notna().all() and not frame[CALL_COLUMN].duplicated().any(),
        int(frame[CALL_COLUMN].duplicated().sum()),
        "0 duplicate port_call_id",
    )
    add(
        "assignment_one_row_per_port_call",
        official_assignments[CALL_COLUMN].notna().all()
        and not official_assignments[CALL_COLUMN].duplicated().any(),
        int(official_assignments[CALL_COLUMN].duplicated().sum()),
        "0 duplicate official assignments",
    )
    source_ids = set(frame[CALL_COLUMN].astype("string"))
    assignment_ids = set(official_assignments[CALL_COLUMN].astype("string"))
    add(
        "assignment_covers_source",
        source_ids == assignment_ids,
        {
            "missing_assignments": len(source_ids - assignment_ids),
            "unknown_assignments": len(assignment_ids - source_ids),
        },
        "exactly the same port_call_id set",
    )
    split_labels = set(official_assignments["split"].astype(str))
    add(
        "official_split_labels",
        {"TRAIN", "VALID", "TEST", "PURGED"}.issuperset(split_labels)
        and {"TRAIN", "VALID", "TEST"}.issubset(split_labels),
        sorted(split_labels),
        "TRAIN, VALID, TEST and optional PURGED",
    )

    joined = frame[[CALL_COLUMN, TIME_COLUMN, "planned_eta", TARGET]].merge(
        official_assignments[[CALL_COLUMN, "split"]],
        on=CALL_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    participating = joined.loc[joined["split"].isin(["TRAIN", "VALID", "TEST"])].copy()
    cutoff_delta = (
        pd.to_datetime(participating["planned_eta"], utc=True)
        - pd.to_datetime(participating[TIME_COLUMN], utc=True)
    ).dt.total_seconds() / 3600.0
    cutoff_bad = int((~np.isclose(cutoff_delta, 24.0, atol=1e-9)).sum())
    add("prediction_cutoff_is_t_minus_24h", cutoff_bad == 0, cutoff_bad, "0 violations")

    target_values = pd.to_numeric(participating[TARGET], errors="coerce")
    non_finite_target = int((~np.isfinite(target_values.to_numpy(dtype="float64"))).sum())
    add("target_is_finite", non_finite_target == 0, non_finite_target, "0 non-finite targets")

    bounds: dict[str, dict[str, Any]] = {}
    for split_name in ("TRAIN", "VALID", "TEST"):
        times = pd.to_datetime(
            participating.loc[participating["split"] == split_name, TIME_COLUMN],
            utc=True,
        )
        bounds[split_name] = {
            "n": int(len(times)),
            "min": times.min(),
            "max": times.max(),
        }
        add(
            f"{split_name.lower()}_is_non_empty",
            len(times) > 0,
            int(len(times)),
            "> 0 rows",
        )
    chronology_ok = bool(
        bounds["TRAIN"]["max"] < bounds["VALID"]["min"]
        and bounds["VALID"]["max"] < bounds["TEST"]["min"]
    )
    add("strict_split_chronology", chronology_ok, bounds, "TRAIN < VALID < TEST")

    participating["label_available_at"] = pd.to_datetime(
        participating["planned_eta"], utc=True
    ) + pd.to_timedelta(target_values, unit="h")
    train_label_max = participating.loc[
        participating["split"] == "TRAIN", "label_available_at"
    ].max()
    valid_label_max = participating.loc[
        participating["split"] == "VALID", "label_available_at"
    ].max()
    valid_start = bounds["VALID"]["min"]
    test_start = bounds["TEST"]["min"]
    add(
        "train_labels_available_before_valid",
        bool(train_label_max <= valid_start),
        {"train_label_max": train_label_max, "valid_start": valid_start},
        "train_label_max <= valid_start",
    )
    add(
        "valid_labels_available_before_test",
        bool(valid_label_max <= test_start),
        {"valid_label_max": valid_label_max, "test_start": test_start},
        "valid_label_max <= test_start",
    )

    unsafe = sorted(column for column in feature_columns if not _safe_feature_name(column))
    add("no_forbidden_model_features", not unsafe, unsafe, "empty list")
    direct_actual = sorted(
        column for column in frame.columns if column.lower() in BANNED_EXACT and column in feature_columns
    )
    add("no_actual_or_target_in_feature_config", not direct_actual, direct_actual, "empty list")

    upstream_decision = split_decision.get("decision", split_decision)
    add(
        "upstream_split_ready",
        upstream_decision.get("status") == "READY_FOR_SPLIT_STRESS_MODELS",
        upstream_decision.get("status"),
        "READY_FOR_SPLIT_STRESS_MODELS",
    )
    add(
        "official_protocol_is_temporal_purged",
        upstream_decision.get("official_protocol") == OFFICIAL_PROTOCOL,
        upstream_decision.get("official_protocol"),
        OFFICIAL_PROTOCOL,
    )
    add(
        "upstream_build_has_no_temporal_leakage",
        int(build_report.get("temporal_leakage_violations", -1)) == 0,
        build_report.get("temporal_leakage_violations"),
        "0",
    )
    add(
        "training_not_executed_upstream_split_audit",
        split_decision.get("training_executed") is False,
        split_decision.get("training_executed"),
        "false",
    )
    return pd.DataFrame(rows)


def _cluster_order(matrix: pd.DataFrame) -> list[str]:
    if len(matrix) <= 2:
        return list(matrix.columns)
    values = matrix.reindex(index=matrix.columns, columns=matrix.columns).fillna(0.0)
    similarity = np.clip(np.abs(values.to_numpy(dtype="float64")), 0.0, 1.0)
    similarity = (similarity + similarity.T) / 2.0
    np.fill_diagonal(similarity, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if not np.isfinite(condensed).all() or np.allclose(condensed, 0.0):
        return list(matrix.columns)
    return list(matrix.columns[leaves_list(linkage(condensed, method="average"))])


def save_correlation_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    clustered: bool = False,
) -> None:
    data = matrix.copy()
    if clustered and len(data) > 2:
        order = _cluster_order(data)
        data = data.loc[order, order]
    size = max(12.0, min(28.0, 7.0 + len(data) * 0.16))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(data.to_numpy(dtype="float64"), cmap="coolwarm", vmin=-1, vmax=1)
    positions = np.arange(len(data.columns))
    axis.set_xticks(positions)
    axis.set_yticks(positions)
    axis.set_xticklabels(data.columns, rotation=90, fontsize=4)
    axis.set_yticklabels(data.index, fontsize=4)
    axis.set_title(title, fontsize=12)
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_family_correlation_matrix(pearson: pd.DataFrame) -> pd.DataFrame:
    families = sorted({_feature_family(column) for column in pearson.columns})
    output = pd.DataFrame(index=families, columns=families, dtype="float64")
    for left in families:
        left_columns = [column for column in pearson.columns if _feature_family(column) == left]
        for right in families:
            right_columns = [column for column in pearson.columns if _feature_family(column) == right]
            block = pearson.loc[left_columns, right_columns].abs().to_numpy(dtype="float64")
            if left == right and block.shape[0] > 1:
                block = block[~np.eye(block.shape[0], dtype=bool)]
            finite = block[np.isfinite(block)]
            output.loc[left, right] = float(np.mean(finite)) if finite.size else np.nan
    return output


def save_family_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix.to_numpy(dtype="float64"), cmap="viridis", vmin=0, vmax=1)
    positions = np.arange(len(matrix.columns))
    axis.set_xticks(positions)
    axis.set_yticks(positions)
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(matrix.index, fontsize=8)
    axis.set_title("TRAIN mean absolute Pearson correlation by feature family")
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_csv(
    client,
    work: Path,
    data: pd.DataFrame,
    output_bucket: str,
    key: str,
) -> str:
    path = work / Path(key).name
    data.to_csv(path, index=False)
    return _upload(client, path, output_bucket, key, "text/csv")


def _save_parquet(
    client,
    work: Path,
    data: pd.DataFrame,
    output_bucket: str,
    key: str,
) -> str:
    path = work / Path(key).name
    data.to_parquet(path, index=False)
    return _upload(client, path, output_bucket, key, "application/vnd.apache.parquet")


def _save_json(
    client,
    work: Path,
    payload: dict[str, Any],
    output_bucket: str,
    key: str,
) -> str:
    path = work / Path(key).name
    path.write_text(_strict_json_dumps(payload, indent=2), encoding="utf-8")
    return _upload(client, path, output_bucket, key, "application/json")


def run_b54fd0_train_readiness(
    source_bucket: str,
    model_ready_key: str,
    feature_config_key: str,
    split_assignments_key: str,
    split_decision_key: str,
    build_report_key: str,
    output_bucket: str,
    output_prefix: str = "version=1",
    sample_size: int = 600,
    numeric_atol: float = 0.002,
    numeric_rtol: float = 0.0001,
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    source_objects = [
        (source_bucket, model_ready_key),
        (source_bucket, feature_config_key),
        (source_bucket, split_assignments_key),
        (source_bucket, split_decision_key),
        (source_bucket, build_report_key),
    ]
    checksum = _signature(
        client,
        source_objects,
        sample_size,
        numeric_atol,
        numeric_rtol,
    )
    source_uri = f"s3://{source_bucket}/{model_ready_key}"
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            run_id, metadata = previous
            return {
                "status": "SUCCESS",
                "cached": True,
                "run_id": run_id,
                "checksum": checksum,
                "results": metadata,
                "outputs": metadata.get("output_uris", {}),
            }

    parameters = {
        "sample_size": int(sample_size),
        "numeric_atol": float(numeric_atol),
        "numeric_rtol": float(numeric_rtol),
        "official_protocol": OFFICIAL_PROTOCOL,
        "sample_policy": "DETERMINISTIC_STRATIFIED_BY_OFFICIAL_SPLIT_AND_YEAR",
        "correlation_fit_scope": "OFFICIAL_TRAIN_ONLY",
        "training_executed": False,
    }
    run_id = _start_run(
        source_uri,
        checksum,
        {
            "audit_version": AUDIT_VERSION,
            "parameters": parameters,
            "training_executed": False,
        },
    )
    outputs: dict[str, str] = {}
    frame: pd.DataFrame | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="b54fd0-") as temp_dir:
            work = Path(temp_dir)
            model_path = _download(
                client, source_bucket, model_ready_key, work / "model_ready.parquet"
            )
            config_path = _download(
                client, source_bucket, feature_config_key, work / "feature_config.json"
            )
            assignment_path = _download(
                client,
                source_bucket,
                split_assignments_key,
                work / "assignments.parquet",
            )
            split_decision_path = _download(
                client,
                source_bucket,
                split_decision_key,
                work / "split_decision.json",
            )
            build_report_path = _download(
                client,
                source_bucket,
                build_report_key,
                work / "build_report.json",
            )

            frame = pd.read_parquet(model_path).copy()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            assignments = pd.read_parquet(assignment_path).copy()
            split_decision = json.loads(
                split_decision_path.read_text(encoding="utf-8")
            )
            build_report = json.loads(build_report_path.read_text(encoding="utf-8"))

            required_frame = {CALL_COLUMN, TIME_COLUMN, "planned_eta", TARGET}
            missing_required = sorted(required_frame.difference(frame.columns))
            if missing_required:
                raise RuntimeError(
                    f"B54F-D0 model-ready source is missing: {missing_required}"
                )
            required_assignment = {CALL_COLUMN, "protocol", "fold", "split"}
            missing_assignment = sorted(required_assignment.difference(assignments.columns))
            if missing_assignment:
                raise RuntimeError(
                    f"B54F-D0 split assignments are missing: {missing_assignment}"
                )

            frame[CALL_COLUMN] = frame[CALL_COLUMN].astype("string")
            frame[TIME_COLUMN] = pd.to_datetime(
                frame[TIME_COLUMN], errors="coerce", utc=True
            )
            frame["planned_eta"] = pd.to_datetime(
                frame["planned_eta"], errors="coerce", utc=True
            )
            frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
            assignments[CALL_COLUMN] = assignments[CALL_COLUMN].astype("string")

            configured_features = list(
                dict.fromkeys(str(column) for column in config.get("feature_columns", []))
            )
            missing_configured = sorted(
                column for column in configured_features if column not in frame.columns
            )
            if missing_configured:
                raise RuntimeError(
                    f"Configured features absent from Gold dataset: {missing_configured}"
                )
            numeric_features = [
                column
                for column in config.get("numeric_features", [])
                if column in configured_features
            ]
            categorical_features = [
                column
                for column in config.get("categorical_features", [])
                if column in configured_features
            ]
            unclassified = sorted(
                set(configured_features).difference(numeric_features, categorical_features)
            )
            if unclassified:
                raise RuntimeError(
                    f"Feature config has unclassified features: {unclassified}"
                )

            official = assignments.loc[
                (assignments["protocol"] == OFFICIAL_PROTOCOL)
                & assignments["fold"].isna()
            ].copy()
            if official.empty:
                raise RuntimeError("TEMPORAL_PURGED official assignments were not found")
            split_recheck = build_split_leakage_recheck(
                frame,
                official,
                configured_features,
                split_decision,
                build_report,
            )
            frame = frame.merge(
                official[[CALL_COLUMN, "split"]],
                on=CALL_COLUMN,
                how="left",
                validate="one_to_one",
            )
            participating = frame.loc[
                frame["split"].isin(["TRAIN", "VALID", "TEST"])
            ].copy()
            if participating.empty:
                raise RuntimeError("Official temporal split has no participating rows")

            schema_audit = build_schema_audit(
                frame,
                configured_features,
                numeric_features,
                categorical_features,
            )
            missingness, missingness_drift, dropped_train, infinite_train = (
                build_missingness_reports(
                    participating,
                    configured_features,
                    categorical_features,
                )
            )
            infinite_all = sorted(
                missingness.loc[missingness["infinite_count"] > 0, "feature"].unique()
            )
            target_distribution = _target_distribution(participating)

            sample = _deterministic_stratified_sample(participating, sample_size)
            calls, sea, sea_source = _load_source_tables()
            recalculation_manifest, recalculation_checks = build_recalculation_proof(
                sample,
                calls,
                sea,
                numeric_features,
                numeric_atol,
                numeric_rtol,
            )
            recalculation_summary = summarize_recalculation(recalculation_checks)

            train = participating.loc[participating["split"] == "TRAIN"].copy()
            pearson, spearman, pearson_target, spearman_target = _correlation_matrices(
                train, numeric_features
            )
            high_pairs = _high_correlation_pairs(pearson, spearman)
            target_associations = build_target_associations(
                train,
                numeric_features,
            )
            categorical_report, imo_name_report, primary_categories = (
                build_categorical_reports(participating, categorical_features)
            )
            missingness_target = build_missingness_target_associations(
                train, numeric_features
            )
            rolling_stability = build_rolling_correlation_stability(
                frame,
                assignments,
                numeric_features,
            )
            family_correlation = build_family_correlation_matrix(pearson)

            unsafe_features = sorted(
                column for column in configured_features if not _safe_feature_name(column)
            )
            frozen_numeric = [
                column
                for column in numeric_features
                if column not in set(dropped_train) and _safe_feature_name(column)
            ]
            frozen_categorical = [
                column
                for column in primary_categories
                if column not in set(dropped_train) and _safe_feature_name(column)
            ]
            frozen_features = frozen_numeric + frozen_categorical
            no_identity_features = list(frozen_numeric)
            no_weather_features = [
                column
                for column in frozen_numeric
                if not _is_weather_feature(column)
            ] + list(frozen_categorical)
            availability_features = [
                column
                for column in frozen_features
                if column.startswith(
                    ("sea_feature_", "sea_observation_", "wave_observation_", "wave_coverage_")
                )
            ]

            all_recalculation = recalculation_summary.loc[
                recalculation_summary["family"] == "ALL"
            ]
            failed_recalculation = (
                int(all_recalculation["failed"].iloc[0])
                if len(all_recalculation)
                else 1
            )
            future_violations = (
                int(all_recalculation["future_violations"].iloc[0])
                if len(all_recalculation)
                else 1
            )
            critical_split_failures = split_recheck.loc[
                split_recheck["critical"] & ~split_recheck["passed"]
            ]
            fatal_reasons: list[str] = []
            if unsafe_features:
                fatal_reasons.append(f"unsafe_feature_names={unsafe_features}")
            if infinite_all:
                fatal_reasons.append(f"non_finite_features={infinite_all}")
            if failed_recalculation:
                fatal_reasons.append(
                    f"independent_recalculation_failures={failed_recalculation}"
                )
            if future_violations:
                fatal_reasons.append(f"future_source_violations={future_violations}")
            if len(critical_split_failures):
                fatal_reasons.append(
                    "split_or_leakage_gate_failures="
                    + ",".join(critical_split_failures["check"].astype(str))
                )
            if not frozen_features:
                fatal_reasons.append("no_frozen_features")

            warning_reasons: list[str] = []
            high_missing = missingness_drift.loc[
                missingness_drift["high_missing_train_flag"], "feature"
            ].tolist()
            missing_drift = missingness_drift.loc[
                missingness_drift["availability_drift_gt10pp_flag"], "feature"
            ].tolist()
            if dropped_train:
                warning_reasons.append(f"dropped_train_constant_or_empty={dropped_train}")
            if high_missing:
                warning_reasons.append(f"train_missing_gt95pct={high_missing}")
            if missing_drift:
                warning_reasons.append(f"missingness_shift_gt10pp={missing_drift}")
            if len(high_pairs):
                warning_reasons.append(f"high_correlation_pairs={len(high_pairs)}")

            decision_status = (
                "READY_FOR_MODEL_STRESS" if not fatal_reasons else "NEED_FEATURE_REPAIR"
            )
            gates = {
                "schema_passed": not missing_configured and not unclassified,
                "split_and_leakage_passed": len(critical_split_failures) == 0,
                "independent_recalculation_passed": failed_recalculation == 0,
                "future_source_time_passed": future_violations == 0,
                "finite_feature_values_passed": not infinite_all,
                "feature_freeze_passed": bool(frozen_features) and not unsafe_features,
                "all_critical_gates_passed": not fatal_reasons,
            }

            frozen_config = {
                "audit_version": AUDIT_VERSION,
                "upstream_feature_version": config.get("feature_version"),
                "grain": "ONE_ROW_PER_PORT_CALL",
                "prediction_cutoff": "PLANNED_ETA_MINUS_24H",
                "official_protocol": OFFICIAL_PROTOCOL,
                "selection_scope": "OFFICIAL_TRAIN_ONLY",
                "target_column": TARGET,
                "frozen_feature_columns": frozen_features,
                "frozen_numeric_features": frozen_numeric,
                "frozen_categorical_features": frozen_categorical,
                "metadata_only_categorical_features": sorted(
                    set(categorical_features).difference(frozen_categorical)
                ),
                "no_identity_feature_columns": no_identity_features,
                "no_weather_feature_columns": no_weather_features,
                "availability_and_quality_features": availability_features,
                "dropped_train_all_missing_or_constant": dropped_train,
                "redundancy_policy": (
                    "RETAIN_FOR_TREE_MODELS; REPORT_ABS_CORRELATION_GE_0P95; "
                    "REVIEW_WITH_SHAP_IN_B54F_D1"
                ),
                "missing_value_policy": {
                    "numeric": "KEEP_NAN_FOR_CATBOOST_WITH_AVAILABILITY_FLAGS",
                    "categorical": "FIT_TRAIN_ONLY_MISSING_TOKEN_IN_B54F_D1",
                    "global_imputation": "FORBIDDEN",
                    "drop_all_missing_or_constant": True,
                },
                "identity_policy": (
                    "IMO_PRIMARY_OPTIONAL_STRESS_TRACK; VESSEL_NAME_METADATA_ONLY; "
                    "NO_IDENTITY_TRACK_REQUIRED"
                ),
                "weather_policy": (
                    "PAST_ONLY_AT_CUTOFF; WITH_WEATHER_AND_NO_WEATHER_TRACKS_REQUIRED"
                ),
                "training_executed": False,
            }

            prefix = output_prefix.strip("/")
            report_prefix = f"reports/b54fd0/{prefix}"
            config_prefix = f"configs/b54fd0/{prefix}"
            outputs["schema_audit"] = _save_csv(
                client, work, schema_audit, output_bucket, f"{report_prefix}/01_schema_audit.csv"
            )
            outputs["recalculation_manifest"] = _save_csv(
                client,
                work,
                recalculation_manifest,
                output_bucket,
                f"{report_prefix}/02_recalculation_sample_manifest.csv",
            )
            outputs["recalculation_checks"] = _save_parquet(
                client,
                work,
                recalculation_checks,
                output_bucket,
                f"{report_prefix}/03_independent_recalculation_checks.parquet",
            )
            outputs["recalculation_summary"] = _save_csv(
                client,
                work,
                recalculation_summary,
                output_bucket,
                f"{report_prefix}/04_recalculation_summary.csv",
            )
            outputs["missingness"] = _save_csv(
                client, work, missingness, output_bucket, f"{report_prefix}/05_split_missingness.csv"
            )
            outputs["missingness_drift"] = _save_csv(
                client,
                work,
                missingness_drift,
                output_bucket,
                f"{report_prefix}/06_missingness_drift.csv",
            )
            outputs["target_distribution"] = _save_csv(
                client,
                work,
                target_distribution,
                output_bucket,
                f"{report_prefix}/07_target_distribution_by_split.csv",
            )
            outputs["pearson_features"] = _save_csv(
                client,
                work,
                pearson.rename_axis("feature").reset_index(),
                output_bucket,
                f"{report_prefix}/08_train_pearson_feature_matrix.csv",
            )
            outputs["spearman_features"] = _save_csv(
                client,
                work,
                spearman.rename_axis("feature").reset_index(),
                output_bucket,
                f"{report_prefix}/09_train_spearman_feature_matrix.csv",
            )
            outputs["pearson_target"] = _save_csv(
                client,
                work,
                pearson_target.rename_axis("feature").reset_index(),
                output_bucket,
                f"{report_prefix}/10_train_pearson_with_target.csv",
            )
            outputs["spearman_target"] = _save_csv(
                client,
                work,
                spearman_target.rename_axis("feature").reset_index(),
                output_bucket,
                f"{report_prefix}/11_train_spearman_with_target.csv",
            )
            outputs["high_correlation_pairs"] = _save_csv(
                client,
                work,
                high_pairs,
                output_bucket,
                f"{report_prefix}/12_high_correlation_pairs_ge_0p95.csv",
            )
            outputs["target_associations"] = _save_csv(
                client,
                work,
                target_associations,
                output_bucket,
                f"{report_prefix}/13_train_feature_target_associations.csv",
            )
            outputs["categorical_associations"] = _save_csv(
                client,
                work,
                categorical_report,
                output_bucket,
                f"{report_prefix}/14_categorical_associations.csv",
            )
            outputs["imo_name_consistency"] = _save_csv(
                client,
                work,
                imo_name_report,
                output_bucket,
                f"{report_prefix}/15_imo_name_consistency.csv",
            )
            outputs["rolling_stability"] = _save_csv(
                client,
                work,
                rolling_stability,
                output_bucket,
                f"{report_prefix}/16_rolling_train_correlation_stability.csv",
            )
            outputs["missingness_target"] = _save_csv(
                client,
                work,
                missingness_target,
                output_bucket,
                f"{report_prefix}/17_train_missingness_target_associations.csv",
            )
            outputs["split_leakage_recheck"] = _save_csv(
                client,
                work,
                split_recheck,
                output_bucket,
                f"{report_prefix}/18_split_and_leakage_recheck.csv",
            )
            outputs["family_correlation"] = _save_csv(
                client,
                work,
                family_correlation.rename_axis("family").reset_index(),
                output_bucket,
                f"{report_prefix}/19_train_family_correlation.csv",
            )

            heatmaps = {
                "pearson_heatmap": (
                    pearson,
                    "20_train_pearson_heatmap.png",
                    "TRAIN Pearson feature correlation",
                    False,
                ),
                "spearman_heatmap": (
                    spearman,
                    "21_train_spearman_heatmap.png",
                    "TRAIN Spearman feature correlation",
                    False,
                ),
                "clustered_pearson_heatmap": (
                    pearson,
                    "22_train_pearson_clustered_heatmap.png",
                    "TRAIN clustered Pearson feature correlation",
                    True,
                ),
            }
            for label, (matrix, filename, title, clustered) in heatmaps.items():
                image_path = work / filename
                save_correlation_heatmap(matrix, image_path, title, clustered)
                outputs[label] = _upload(
                    client,
                    image_path,
                    output_bucket,
                    f"{report_prefix}/{filename}",
                    "image/png",
                )
            family_image = work / "23_train_family_correlation_heatmap.png"
            save_family_heatmap(family_correlation, family_image)
            outputs["family_heatmap"] = _upload(
                client,
                family_image,
                output_bucket,
                f"{report_prefix}/{family_image.name}",
                "image/png",
            )

            outputs["frozen_feature_config"] = _save_json(
                client,
                work,
                frozen_config,
                output_bucket,
                f"{config_prefix}/b54fd0_frozen_feature_config_v1.json",
            )

            summary = _json_safe(
                {
                    "audit_version": AUDIT_VERSION,
                    "source_rows": int(len(frame)),
                    "participating_rows": int(len(participating)),
                    "source_calls": int(frame[CALL_COLUMN].nunique()),
                    "feature_count": int(len(configured_features)),
                    "numeric_feature_count": int(len(numeric_features)),
                    "categorical_feature_count": int(len(categorical_features)),
                    "frozen_feature_count": int(len(frozen_features)),
                    "frozen_numeric_feature_count": int(len(frozen_numeric)),
                    "frozen_categorical_feature_count": int(len(frozen_categorical)),
                    "weather_feature_count": int(
                        sum(_is_weather_feature(column) for column in configured_features)
                    ),
                    "sea_source": sea_source,
                    "recalculation_sample_rows": int(len(sample)),
                    "recalculation_check_rows": int(len(recalculation_checks)),
                    "recalculation_failures": failed_recalculation,
                    "future_source_violations": future_violations,
                    "dropped_train_all_missing_or_constant": dropped_train,
                    "non_finite_features": infinite_all,
                    "high_correlation_pair_count": int(len(high_pairs)),
                    "high_missing_feature_count": int(len(high_missing)),
                    "missingness_drift_feature_count": int(len(missing_drift)),
                    "parameters": parameters,
                    "gates": gates,
                    "decision": {
                        "status": decision_status,
                        "fatal_reasons": fatal_reasons,
                        "warnings": warning_reasons,
                        "official_protocol": OFFICIAL_PROTOCOL,
                        "next_block": (
                            "B54F_D1_RANDOM_VS_TEMPORAL_MODEL_STRESS_TEST"
                            if decision_status == "READY_FOR_MODEL_STRESS"
                            else "B54F_D0_FEATURE_REPAIR"
                        ),
                    },
                    "scientific_scope": {
                        "correlations": "OFFICIAL_TRAIN_ONLY",
                        "mutual_information": "OFFICIAL_TRAIN_ONLY",
                        "feature_freeze": "OFFICIAL_TRAIN_ONLY",
                        "valid_test_usage": "MISSINGNESS_AND_STABILITY_DIAGNOSTICS_ONLY",
                        "independent_recalculation": True,
                        "training_executed": False,
                    },
                    "training_executed": False,
                    "output_uris": outputs,
                    "generated_at_utc": datetime.now(timezone.utc),
                }
            )
            outputs["decision"] = _save_json(
                client,
                work,
                summary,
                output_bucket,
                f"{config_prefix}/b54fd0_train_readiness_decision_v1.json",
            )
            summary["output_uris"] = outputs

        _finish_run(run_id, "SUCCESS", len(participating), summary)
        return {
            "status": "SUCCESS",
            "cached": False,
            "run_id": run_id,
            "checksum": checksum,
            "results": summary,
            "outputs": outputs,
        }
    except Exception as exc:
        _finish_run(run_id, "FAILED", error_message=str(exc)[:4000])
        raise
