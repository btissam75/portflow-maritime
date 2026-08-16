from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values


AUDIT_VERSION = "b56a-operational-feasibility-v1.1"
FEATURE_VERSION = "b56a-port-hourly-state-v1"
SOURCE_NAME = "b56a_operational_feasibility"
DATASET_NAME = "port_hourly_state_feasibility"
SOURCE_VIEW = "features.port_call_model_ready_v1"
WEATHER_TABLE = "core.maritime_observation"

HORIZONS_H = (6, 12, 24)
AUTOCORRELATION_LAGS = (1, 6, 12, 24, 48, 168, 336)
MIN_CALL_ROWS = 10_000
MIN_YEARS = 3
MIN_ARRIVAL_COVERAGE = 0.95
MIN_WEATHER_COVERAGE = 0.95
MIN_OCCUPANCY_CALL_COVERAGE = 0.90
MAX_QUARANTINED_ACTUAL_SEQUENCE_RATE = 0.001

CALL_COLUMNS = (
    "port_call_id",
    "terminal_code",
    "mmsi",
    "imo",
    "vessel_name",
    "voyage_id",
    "planned_eta",
    "planned_etd",
    "actual_ata",
    "actual_atd",
    "arrival_delay_h",
    "departure_delay_h",
    "cargo_type",
    "vessel_type",
    "source",
    "updated_at",
)

WEATHER_COLUMNS = (
    "wave_height_m",
    "wave_period_s",
    "wave_direction_deg",
    "wind_speed_ms",
    "wind_direction_deg",
    "surface_current_ms",
    "visibility_m",
    "pressure_hpa",
)

FEATURE_COLUMNS = (
    "arrivals_prev_1h",
    "arrivals_last_6h",
    "arrivals_last_24h",
    "arrivals_last_168h",
    "departures_prev_1h",
    "departures_last_6h",
    "departures_last_24h",
    "delayed_gt3_last_24h",
    "mean_arrival_delay_last_24h",
    "vessels_in_port_observed",
    "wave_height_lag_1h_m",
    "wave_period_lag_1h_s",
    "wave_direction_lag_1h_deg",
    "weather_available_flag",
    "hour_of_day",
    "day_of_week",
    "month",
    "weekend_flag",
)

TARGET_COLUMNS = (
    "target_arrivals_next_6h",
    "target_arrivals_next_12h",
    "target_arrivals_next_24h",
    "target_delayed_gt3_next_24h",
    "target_mean_arrival_delay_next_24h",
    "target_max_vessels_in_port_next_24h",
)


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


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


def _query_frame(query: str) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _relation_exists(schema: str, relation: str) -> bool:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (f"{schema}.{relation}",))
            return cursor.fetchone()[0] is not None


def _load_port_calls() -> pd.DataFrame:
    if not _relation_exists("features", "port_call_model_ready_v1"):
        raise RuntimeError(f"Required source view does not exist: {SOURCE_VIEW}")
    frame = _query_frame(
        f"""
        SELECT
            port_call_id::text AS port_call_id,
            terminal_code,
            mmsi,
            imo,
            vessel_name,
            voyage_id,
            planned_eta,
            planned_etd,
            actual_ata,
            actual_atd,
            arrival_delay_h::double precision AS arrival_delay_h,
            departure_delay_h::double precision AS departure_delay_h,
            cargo_type,
            vessel_type,
            source,
            updated_at
        FROM {SOURCE_VIEW}
        ORDER BY port_call_id
        """
    )
    missing = sorted(set(CALL_COLUMNS).difference(frame.columns))
    if missing:
        raise RuntimeError(f"Source view is missing columns: {missing}")
    for column in (
        "planned_eta",
        "planned_etd",
        "actual_ata",
        "actual_atd",
        "updated_at",
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    for column in ("arrival_delay_h", "departure_delay_h"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_weather() -> tuple[pd.DataFrame, int]:
    if not _relation_exists("core", "maritime_observation"):
        raise RuntimeError(f"Required weather table does not exist: {WEATHER_TABLE}")
    raw_count = int(
        _query_frame(
            "SELECT COUNT(*)::bigint AS n FROM core.maritime_observation "
            "WHERE quality_flag=0"
        ).iloc[0]["n"]
    )
    frame = _query_frame(
        """
        SELECT
            observed_at,
            AVG(wave_height_m)::double precision AS wave_height_m,
            AVG(wave_period_s)::double precision AS wave_period_s,
            AVG(wave_direction_deg)::double precision AS wave_direction_deg,
            AVG(wind_speed_ms)::double precision AS wind_speed_ms,
            AVG(wind_direction_deg)::double precision AS wind_direction_deg,
            AVG(surface_current_ms)::double precision AS surface_current_ms,
            AVG(visibility_m)::double precision AS visibility_m,
            AVG(pressure_hpa)::double precision AS pressure_hpa
        FROM core.maritime_observation
        WHERE quality_flag=0
        GROUP BY observed_at
        ORDER BY observed_at
        """
    )
    if frame.empty:
        raise RuntimeError("No quality_flag=0 maritime observations were found")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    for column in WEATHER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("observed_at").reset_index(drop=True), raw_count


def _frame_signature(calls: pd.DataFrame, weather: pd.DataFrame) -> str:
    digest = hashlib.sha256(AUDIT_VERSION.encode("ascii"))
    call_hash_columns = [
        "port_call_id",
        "planned_eta",
        "planned_etd",
        "actual_ata",
        "actual_atd",
        "arrival_delay_h",
        "imo",
        "updated_at",
    ]
    weather_hash_columns = ["observed_at", *WEATHER_COLUMNS]
    for frame, columns in (
        (calls, call_hash_columns),
        (weather, weather_hash_columns),
    ):
        hashed = pd.util.hash_pandas_object(frame[columns], index=False)
        digest.update(hashed.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _start_run(checksum: str) -> str:
    source_uri = f"postgresql://maritime/{SOURCE_VIEW}+{WEATHER_TABLE}"
    metadata = {
        "audit_version": AUDIT_VERSION,
        "feature_version": FEATURE_VERSION,
        "training_executed": False,
        "split_created": False,
        "bronze_modified": False,
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
                (SOURCE_NAME, DATASET_NAME, source_uri, checksum, Json(metadata)),
            )
            return str(cursor.fetchone()[0])


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    payload = _clean_json(metadata)
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
                    Json(payload, dumps=lambda x: json.dumps(x, default=_json_default)),
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
                WHERE source_name=%s AND dataset_name=%s
                  AND checksum=%s AND status='SUCCESS'
                ORDER BY finished_at DESC LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _upload_file(client, source: Path, bucket: str, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/octet-stream",
    }.get(source.suffix.lower(), "application/octet-stream")
    client.upload_file(
        str(source), bucket, key, ExtraArgs={"ContentType": content_type}
    )
    return f"s3://{bucket}/{key}"


def _future_sum(series: pd.Series, horizon: int) -> pd.Series:
    return (
        series.iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .sum()
        .iloc[::-1]
    )


def _future_max(series: pd.Series, horizon: int) -> pd.Series:
    return (
        series.iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .max()
        .iloc[::-1]
    )


def _occupancy_series(
    calls: pd.DataFrame, grid: pd.DatetimeIndex
) -> tuple[np.ndarray, int]:
    valid = calls.dropna(subset=["actual_ata", "actual_atd"]).copy()
    valid = valid[valid["actual_atd"] >= valid["actual_ata"]]
    valid = valid[
        (valid["actual_atd"] >= grid[0]) & (valid["actual_ata"] <= grid[-1])
    ]
    difference = np.zeros(len(grid) + 1, dtype="int64")
    grid_ns = grid.asi8
    end_boundary_ns = (grid[-1] + pd.Timedelta(hours=1)).value
    for row in valid.itertuples(index=False):
        start = pd.Timestamp(row.actual_ata).ceil("h").value
        end = pd.Timestamp(row.actual_atd).ceil("h").value
        start = max(start, grid_ns[0])
        end = min(end, end_boundary_ns)
        start_index = int(np.searchsorted(grid_ns, start, side="left"))
        end_index = int(np.searchsorted(grid_ns, end, side="left"))
        if start_index < end_index and start_index < len(grid):
            difference[start_index] += 1
            difference[min(end_index, len(grid))] -= 1
    return np.cumsum(difference[:-1]), len(valid)


def build_hourly_frame(
    calls: pd.DataFrame, weather: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    arrival_times = calls["actual_ata"].dropna()
    if arrival_times.empty:
        raise RuntimeError("No actual arrivals are available")
    start = max(arrival_times.min().floor("h"), weather["observed_at"].min().floor("h"))
    end = min(arrival_times.max().floor("h"), weather["observed_at"].max().floor("h"))
    if end <= start:
        raise RuntimeError("Port-call and weather time ranges do not overlap")
    grid = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    if len(grid) < 24 * 365:
        raise RuntimeError("Less than one year of overlapping hourly data")

    hourly = pd.DataFrame({"as_of_time": grid})
    arrivals = calls.dropna(subset=["actual_ata"]).copy()
    arrivals = arrivals[
        (arrivals["actual_ata"] >= start)
        & (arrivals["actual_ata"] < end + pd.Timedelta(hours=1))
    ]
    arrivals["event_hour"] = arrivals["actual_ata"].dt.floor("h")
    arrivals["late_gt_1"] = (arrivals["arrival_delay_h"] > 1).astype("int64")
    arrivals["late_gt_3"] = (arrivals["arrival_delay_h"] > 3).astype("int64")
    arrivals["late_gt_6"] = (arrivals["arrival_delay_h"] > 6).astype("int64")
    arrivals["delay_value"] = arrivals["arrival_delay_h"].where(
        arrivals["arrival_delay_h"].notna(), 0.0
    )
    arrivals["delay_valid"] = arrivals["arrival_delay_h"].notna().astype("int64")
    arrival_hourly = arrivals.groupby("event_hour", observed=True).agg(
        arrivals_observed_in_hour=("port_call_id", "size"),
        delayed_gt1_in_hour=("late_gt_1", "sum"),
        delayed_gt3_in_hour=("late_gt_3", "sum"),
        delayed_gt6_in_hour=("late_gt_6", "sum"),
        delay_sum_in_hour=("delay_value", "sum"),
        delay_count_in_hour=("delay_valid", "sum"),
    )

    departures = calls.dropna(subset=["actual_atd"]).copy()
    departures = departures[
        (departures["actual_atd"] >= start)
        & (departures["actual_atd"] < end + pd.Timedelta(hours=1))
    ]
    departures["event_hour"] = departures["actual_atd"].dt.floor("h")
    departure_hourly = departures.groupby("event_hour", observed=True).size()

    hourly = hourly.set_index("as_of_time")
    for column in arrival_hourly.columns:
        hourly[column] = arrival_hourly[column].reindex(grid, fill_value=0)
    hourly["departures_observed_in_hour"] = departure_hourly.reindex(
        grid, fill_value=0
    )
    occupancy, occupancy_calls = _occupancy_series(calls, grid)
    hourly["vessels_in_port_observed"] = occupancy

    weather_hourly = weather.copy()
    weather_hourly["weather_hour"] = weather_hourly["observed_at"].dt.floor("h")
    weather_hourly = weather_hourly.groupby("weather_hour", observed=True)[
        list(WEATHER_COLUMNS)
    ].mean()
    hourly = hourly.join(weather_hourly, how="left")

    arrivals_past = hourly["arrivals_observed_in_hour"].shift(1)
    departures_past = hourly["departures_observed_in_hour"].shift(1)
    delayed3_past = hourly["delayed_gt3_in_hour"].shift(1)
    delay_sum_past = hourly["delay_sum_in_hour"].shift(1)
    delay_count_past = hourly["delay_count_in_hour"].shift(1)
    hourly["arrivals_prev_1h"] = arrivals_past.fillna(0)
    hourly["arrivals_last_6h"] = arrivals_past.rolling(6, min_periods=1).sum()
    hourly["arrivals_last_24h"] = arrivals_past.rolling(24, min_periods=1).sum()
    hourly["arrivals_last_168h"] = arrivals_past.rolling(168, min_periods=1).sum()
    hourly["departures_prev_1h"] = departures_past.fillna(0)
    hourly["departures_last_6h"] = departures_past.rolling(6, min_periods=1).sum()
    hourly["departures_last_24h"] = departures_past.rolling(24, min_periods=1).sum()
    hourly["delayed_gt3_last_24h"] = delayed3_past.rolling(
        24, min_periods=1
    ).sum()
    past_delay_sum = delay_sum_past.rolling(24, min_periods=1).sum()
    past_delay_count = delay_count_past.rolling(24, min_periods=1).sum()
    hourly["mean_arrival_delay_last_24h"] = past_delay_sum.div(
        past_delay_count.replace(0, np.nan)
    )

    hourly["wave_height_lag_1h_m"] = hourly["wave_height_m"].shift(1)
    hourly["wave_period_lag_1h_s"] = hourly["wave_period_s"].shift(1)
    hourly["wave_direction_lag_1h_deg"] = hourly["wave_direction_deg"].shift(1)
    hourly["weather_available_flag"] = hourly[
        [
            "wave_height_lag_1h_m",
            "wave_period_lag_1h_s",
            "wave_direction_lag_1h_deg",
        ]
    ].notna().all(axis=1).astype("int8")

    for horizon in HORIZONS_H:
        hourly[f"target_arrivals_next_{horizon}h"] = _future_sum(
            hourly["arrivals_observed_in_hour"], horizon
        )
    hourly["target_delayed_gt3_next_24h"] = _future_sum(
        hourly["delayed_gt3_in_hour"], 24
    )
    future_delay_sum = _future_sum(hourly["delay_sum_in_hour"], 24)
    future_delay_count = _future_sum(hourly["delay_count_in_hour"], 24)
    hourly["target_mean_arrival_delay_next_24h"] = future_delay_sum.div(
        future_delay_count.replace(0, np.nan)
    )
    hourly["target_max_vessels_in_port_next_24h"] = _future_max(
        hourly["vessels_in_port_observed"], 24
    )

    hourly["hour_of_day"] = hourly.index.hour.astype("int16")
    hourly["day_of_week"] = hourly.index.dayofweek.astype("int16")
    hourly["month"] = hourly.index.month.astype("int16")
    hourly["weekend_flag"] = (hourly.index.dayofweek >= 5).astype("int8")
    hourly["feature_version"] = FEATURE_VERSION
    hourly = hourly.reset_index()

    build_metadata = {
        "grid_start": start,
        "grid_end": end,
        "grid_rows": len(hourly),
        "arrival_calls_in_grid": len(arrivals),
        "departure_calls_in_grid": len(departures),
        "occupancy_calls_in_grid": occupancy_calls,
        "occupancy_call_coverage": occupancy_calls / max(len(arrivals), 1),
    }
    return hourly, build_metadata


def _source_inventory(
    calls: pd.DataFrame,
    weather: pd.DataFrame,
    raw_weather_rows: int,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": SOURCE_VIEW,
                "grain": "ONE_ROW_PER_PORT_CALL",
                "rows": len(calls),
                "unique_keys": calls["port_call_id"].nunique(),
                "min_time": calls["actual_ata"].min(),
                "max_time": calls["actual_ata"].max(),
            },
            {
                "source": WEATHER_TABLE,
                "grain": "RAW_POINT_OBSERVATION",
                "rows": raw_weather_rows,
                "unique_keys": weather["observed_at"].nunique(),
                "min_time": weather["observed_at"].min(),
                "max_time": weather["observed_at"].max(),
            },
            {
                "source": FEATURE_VERSION,
                "grain": "ONE_ROW_PER_UTC_HOUR",
                "rows": len(hourly),
                "unique_keys": hourly["as_of_time"].nunique(),
                "min_time": hourly["as_of_time"].min(),
                "max_time": hourly["as_of_time"].max(),
            },
        ]
    )


def _grain_audit(calls: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": SOURCE_VIEW,
                "rows": len(calls),
                "unique_keys": calls["port_call_id"].nunique(),
                "duplicate_rows": int(calls["port_call_id"].duplicated().sum()),
                "passed": not calls["port_call_id"].duplicated().any(),
            },
            {
                "dataset": FEATURE_VERSION,
                "rows": len(hourly),
                "unique_keys": hourly["as_of_time"].nunique(),
                "duplicate_rows": int(hourly["as_of_time"].duplicated().sum()),
                "passed": not hourly["as_of_time"].duplicated().any(),
            },
        ]
    )


def _timestamp_semantics() -> pd.DataFrame:
    rows = [
        ("as_of_time", "FEATURE_CUTOFF", "KNOWN", "feature", "KEEP"),
        ("planned_eta", "PLANNED_EVENT", "UNPROVEN_REVISION_TIME", "metadata", "AUDIT"),
        ("actual_ata", "ACTUAL_EVENT", "AFTER_EVENT", "target_source", "FORBID_FEATURE"),
        ("actual_atd", "ACTUAL_EVENT", "AFTER_EVENT", "target_source", "FORBID_FEATURE"),
        ("arrival_delay_h", "DERIVED_TARGET", "AFTER_ATA", "target_source", "FORBID_FEATURE"),
        ("observed_at", "WEATHER_VALID_TIME", "NO_AVAILABLE_AT", "source", "AUDIT"),
        ("wave_height_lag_1h_m", "PAST_WEATHER", "ASSUMED_AVAILABLE_WITH_1H_LAG", "feature", "KEEP_WITH_FLAG"),
        ("target_arrivals_next_24h", "FUTURE_WINDOW", "AFTER_HORIZON", "target", "TARGET_ONLY"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "business_time_semantics",
            "availability_semantics",
            "role",
            "decision",
        ],
    )


def _coverage_report(calls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, part in [("ALL", calls)]:
        rows.append(_coverage_row(scope, part))
    with_year = calls.dropna(subset=["actual_ata"]).copy()
    with_year["year"] = with_year["actual_ata"].dt.year
    for year, part in with_year.groupby("year", observed=True):
        rows.append(_coverage_row(str(year), part))
    return pd.DataFrame(rows)


def _coverage_row(scope: str, frame: pd.DataFrame) -> dict[str, Any]:
    total = len(frame)
    both = frame["actual_ata"].notna() & frame["actual_atd"].notna()
    valid_sequence = both & (frame["actual_atd"] >= frame["actual_ata"])
    return {
        "scope": scope,
        "calls": total,
        "planned_eta_coverage_pct": 100 * frame["planned_eta"].notna().mean(),
        "actual_ata_coverage_pct": 100 * frame["actual_ata"].notna().mean(),
        "actual_atd_coverage_pct": 100 * frame["actual_atd"].notna().mean(),
        "arrival_delay_coverage_pct": 100 * frame["arrival_delay_h"].notna().mean(),
        "valid_ata_atd_coverage_pct": 100 * valid_sequence.sum() / max(total, 1),
    }


def _grid_continuity(hourly: pd.DataFrame) -> pd.DataFrame:
    times = hourly["as_of_time"].sort_values()
    expected = int((times.max() - times.min()) / pd.Timedelta(hours=1)) + 1
    deltas = times.diff().dropna()
    return pd.DataFrame(
        [
            {
                "first_hour": times.min(),
                "last_hour": times.max(),
                "rows": len(times),
                "expected_rows": expected,
                "missing_hours": expected - times.nunique(),
                "duplicate_hours": int(times.duplicated().sum()),
                "non_hourly_steps": int((deltas != pd.Timedelta(hours=1)).sum()),
                "continuous": bool(
                    expected == times.nunique()
                    and not times.duplicated().any()
                    and (deltas == pd.Timedelta(hours=1)).all()
                ),
            }
        ]
    )


def _weather_coverage(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in WEATHER_COLUMNS:
        numeric = pd.to_numeric(hourly[column], errors="coerce")
        available = int(numeric.notna().sum())
        rows.append(
            {
                "variable": column,
                "rows": len(hourly),
                "available_rows": available,
                "coverage_pct": 100 * available / max(len(hourly), 1),
                "missing_pct": 100 * numeric.isna().mean(),
                "minimum": numeric.min(),
                "median": numeric.median(),
                "maximum": numeric.max(),
                "operational_available_at_present": False,
            }
        )
    return pd.DataFrame(rows)


def _target_feasibility(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in TARGET_COLUMNS:
        numeric = pd.to_numeric(hourly[column], errors="coerce")
        valid = numeric.dropna()
        rows.append(
            {
                "target": column,
                "rows": len(hourly),
                "available_rows": len(valid),
                "coverage_pct": 100 * len(valid) / max(len(hourly), 1),
                "zero_pct": 100 * (valid == 0).mean() if len(valid) else np.nan,
                "mean": valid.mean(),
                "median": valid.median(),
                "p95": valid.quantile(0.95) if len(valid) else np.nan,
                "maximum": valid.max() if len(valid) else np.nan,
                "non_constant": bool(valid.nunique() > 1),
            }
        )
    return pd.DataFrame(rows)


def _missingness_by_year(calls: pd.DataFrame) -> pd.DataFrame:
    frame = calls.copy()
    frame["year"] = frame["actual_ata"].dt.year.fillna(
        frame["planned_eta"].dt.year
    )
    columns = [
        "planned_eta",
        "planned_etd",
        "actual_ata",
        "actual_atd",
        "arrival_delay_h",
        "departure_delay_h",
        "terminal_code",
        "imo",
    ]
    rows = []
    for year, part in frame.groupby("year", dropna=False, observed=True):
        for column in columns:
            rows.append(
                {
                    "year": year,
                    "column": column,
                    "rows": len(part),
                    "missing_rows": int(part[column].isna().sum()),
                    "missing_pct": 100 * part[column].isna().mean(),
                }
            )
    return pd.DataFrame(rows)


def _missingness_by_vessel(calls: pd.DataFrame) -> pd.DataFrame:
    frame = calls.copy()
    frame["imo_key"] = frame["imo"].astype("string").fillna("MISSING")
    rows = []
    for imo, part in frame.groupby("imo_key", observed=True):
        rows.append(
            {
                "imo": imo,
                "vessel_name": part["vessel_name"].mode().iloc[0]
                if part["vessel_name"].notna().any()
                else None,
                "calls": len(part),
                "actual_ata_missing_pct": 100 * part["actual_ata"].isna().mean(),
                "actual_atd_missing_pct": 100 * part["actual_atd"].isna().mean(),
                "arrival_delay_missing_pct": 100
                * part["arrival_delay_h"].isna().mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("calls", ascending=False)


def _invalid_sequences(calls: pd.DataFrame) -> pd.DataFrame:
    invalid_actual = (
        calls["actual_ata"].notna()
        & calls["actual_atd"].notna()
        & (calls["actual_atd"] < calls["actual_ata"])
    )
    invalid_planned = (
        calls["planned_eta"].notna()
        & calls["planned_etd"].notna()
        & (calls["planned_etd"] < calls["planned_eta"])
    )
    impossible_arrival = calls["arrival_delay_h"].abs() > 72
    flagged = calls[invalid_actual | invalid_planned | impossible_arrival].copy()
    flagged["invalid_actual_sequence"] = invalid_actual[flagged.index]
    flagged["invalid_planned_sequence"] = invalid_planned[flagged.index]
    flagged["abs_arrival_delay_gt72h"] = impossible_arrival[flagged.index]
    columns = [
        "port_call_id",
        "imo",
        "vessel_name",
        "planned_eta",
        "planned_etd",
        "actual_ata",
        "actual_atd",
        "arrival_delay_h",
        "invalid_actual_sequence",
        "invalid_planned_sequence",
        "abs_arrival_delay_gt72h",
    ]
    return flagged[columns].sort_values("port_call_id")


def _join_audit(
    calls: pd.DataFrame,
    weather: pd.DataFrame,
    raw_weather_rows: int,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "join": "RAW_WEATHER_TO_HOURLY_WEATHER",
                "left_rows": raw_weather_rows,
                "right_rows": weather["observed_at"].nunique(),
                "output_rows": len(weather),
                "duplicate_output_keys": int(weather["observed_at"].duplicated().sum()),
                "many_to_many_detected": False,
            },
            {
                "join": "HOURLY_GRID_TO_WEATHER",
                "left_rows": len(hourly),
                "right_rows": len(weather),
                "output_rows": len(hourly),
                "duplicate_output_keys": int(hourly["as_of_time"].duplicated().sum()),
                "many_to_many_detected": bool(
                    weather["observed_at"].duplicated().any()
                    or hourly["as_of_time"].duplicated().any()
                ),
            },
            {
                "join": "PORT_CALL_TO_ARRIVAL_HOUR",
                "left_rows": int(calls["actual_ata"].notna().sum()),
                "right_rows": len(hourly),
                "output_rows": int(hourly["arrivals_observed_in_hour"].sum()),
                "duplicate_output_keys": 0,
                "many_to_many_detected": False,
            },
        ]
    )


def _leakage_audit(hourly: pd.DataFrame) -> pd.DataFrame:
    def equivalent(left: pd.Series, right: pd.Series) -> bool:
        left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype="float64")
        right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype="float64")
        return bool(
            np.allclose(
                left_values,
                right_values,
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )
        )

    arrivals_past = hourly["arrivals_observed_in_hour"].shift(1)
    departures_past = hourly["departures_observed_in_hour"].shift(1)
    delayed_past = hourly["delayed_gt3_in_hour"].shift(1)
    expected_history = {
        "arrivals_prev_1h": arrivals_past.fillna(0),
        "arrivals_last_6h": arrivals_past.rolling(6, min_periods=1).sum(),
        "arrivals_last_24h": arrivals_past.rolling(24, min_periods=1).sum(),
        "arrivals_last_168h": arrivals_past.rolling(168, min_periods=1).sum(),
        "departures_prev_1h": departures_past.fillna(0),
        "departures_last_6h": departures_past.rolling(6, min_periods=1).sum(),
        "departures_last_24h": departures_past.rolling(24, min_periods=1).sum(),
        "delayed_gt3_last_24h": delayed_past.rolling(24, min_periods=1).sum(),
    }
    history_recalculation_passed = all(
        equivalent(hourly[column], expected)
        for column, expected in expected_history.items()
    )
    weather_recalculation_passed = all(
        equivalent(hourly[feature], hourly[source].shift(1))
        for feature, source in (
            ("wave_height_lag_1h_m", "wave_height_m"),
            ("wave_period_lag_1h_s", "wave_period_s"),
            ("wave_direction_lag_1h_deg", "wave_direction_deg"),
        )
    )

    checks = [
        (
            "NO_TARGET_COLUMN_IN_FEATURE_LIST",
            not set(FEATURE_COLUMNS).intersection(TARGET_COLUMNS),
            "Feature and target contracts are disjoint.",
            "CRITICAL",
        ),
        (
            "NO_ACTUAL_ATA_ATD_AS_MODEL_FEATURE",
            not any("actual_" in column for column in FEATURE_COLUMNS),
            "Actual event timestamps are target sources only.",
            "CRITICAL",
        ),
        (
            "ROLLING_FEATURES_SHIFTED_ONE_HOUR",
            history_recalculation_passed,
            "Stored history features equal an independent shift(1) recalculation.",
            "CRITICAL",
        ),
        (
            "WEATHER_FEATURES_SHIFTED_ONE_HOUR",
            weather_recalculation_passed,
            "Stored weather features equal the preceding observed hour.",
            "CRITICAL",
        ),
        (
            "FUTURE_WINDOWS_TARGET_ONLY",
            all(column.startswith("target_") for column in TARGET_COLUMNS),
            "Future windows are explicitly namespaced as targets.",
            "CRITICAL",
        ),
        (
            "WEATHER_AVAILABLE_AT_PROVEN",
            False,
            "core.maritime_observation has valid time but no publication/available_at.",
            "WARNING",
        ),
        (
            "ETA_REVISION_AVAILABLE_AT_PROVEN",
            False,
            "The current port-call view has no ETA revision publication history.",
            "WARNING",
        ),
        (
            "HOURLY_KEY_UNIQUE",
            not hourly["as_of_time"].duplicated().any(),
            "Exactly one row exists per as_of_time.",
            "CRITICAL",
        ),
    ]
    return pd.DataFrame(
        checks, columns=["check", "passed", "evidence", "severity"]
    )


def _target_distribution_by_year(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    frame["year"] = frame["as_of_time"].dt.year
    rows = []
    for year, part in frame.groupby("year", observed=True):
        for target in TARGET_COLUMNS:
            values = pd.to_numeric(part[target], errors="coerce").dropna()
            rows.append(
                {
                    "year": year,
                    "target": target,
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "p95": values.quantile(0.95) if len(values) else np.nan,
                    "zero_pct": 100 * (values == 0).mean() if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _vessel_concentration(calls: pd.DataFrame) -> pd.DataFrame:
    frame = calls.dropna(subset=["actual_ata"]).copy()
    frame["imo_key"] = frame["imo"].astype("string").fillna("MISSING")
    grouped = (
        frame.groupby("imo_key", observed=True)
        .agg(
            calls=("port_call_id", "size"),
            vessel_name=(
                "vessel_name",
                lambda values: values.mode().iloc[0]
                if values.notna().any()
                else None,
            ),
        )
        .reset_index()
        .sort_values("calls", ascending=False)
    )
    grouped["share_pct"] = 100 * grouped["calls"] / max(grouped["calls"].sum(), 1)
    grouped["cumulative_share_pct"] = grouped["share_pct"].cumsum()
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped[
        ["rank", "imo_key", "vessel_name", "calls", "share_pct", "cumulative_share_pct"]
    ]


def _autocorrelation(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    series_names = (
        "arrivals_observed_in_hour",
        "departures_observed_in_hour",
        "target_arrivals_next_24h",
        "vessels_in_port_observed",
    )
    for series_name in series_names:
        series = pd.to_numeric(hourly[series_name], errors="coerce")
        for lag in AUTOCORRELATION_LAGS:
            paired = pd.concat([series, series.shift(lag)], axis=1).dropna()
            correlation = (
                paired.iloc[:, 0].corr(paired.iloc[:, 1])
                if len(paired) >= 100 and paired.iloc[:, 0].nunique() > 1
                else np.nan
            )
            rows.append(
                {
                    "series": series_name,
                    "lag_hours": lag,
                    "n_pairs": len(paired),
                    "pearson_autocorrelation": correlation,
                }
            )
    return pd.DataFrame(rows)


def _seasonality(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = "arrivals_observed_in_hour"
    dimensions = {
        "HOUR_OF_DAY": hourly["as_of_time"].dt.hour,
        "DAY_OF_WEEK": hourly["as_of_time"].dt.dayofweek,
        "MONTH": hourly["as_of_time"].dt.month,
    }
    for dimension, values in dimensions.items():
        temporary = pd.DataFrame({"category": values, "target": hourly[target]})
        for category, part in temporary.groupby("category", observed=True):
            rows.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "n_hours": len(part),
                    "mean_arrivals": part["target"].mean(),
                    "median_arrivals": part["target"].median(),
                    "p95_arrivals": part["target"].quantile(0.95),
                    "zero_pct": 100 * (part["target"] == 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def _decision(
    calls: pd.DataFrame,
    hourly: pd.DataFrame,
    weather_coverage: pd.DataFrame,
    grain: pd.DataFrame,
    continuity: pd.DataFrame,
    leakage: pd.DataFrame,
    invalid: pd.DataFrame,
    build_metadata: dict[str, Any],
) -> dict[str, Any]:
    arrival_coverage = float(calls["actual_ata"].notna().mean())
    departure_coverage = float(calls["actual_atd"].notna().mean())
    years = int(hourly["as_of_time"].dt.year.nunique())
    critical_leakage = leakage[
        (leakage["severity"] == "CRITICAL") & (~leakage["passed"])
    ]
    duplicate_keys = int(grain["duplicate_rows"].sum())
    invalid_actual = int(invalid["invalid_actual_sequence"].sum()) if len(invalid) else 0
    invalid_actual_rate = invalid_actual / max(len(calls), 1)
    continuous = bool(continuity.iloc[0]["continuous"])

    coverage_map = weather_coverage.set_index("variable")["coverage_pct"].to_dict()
    wave_ready = all(
        coverage_map.get(column, 0.0) >= 100 * MIN_WEATHER_COVERAGE
        for column in ("wave_height_m", "wave_period_s", "wave_direction_deg")
    )
    full_weather_ready = all(
        coverage_map.get(column, 0.0) >= 100 * MIN_WEATHER_COVERAGE
        for column in WEATHER_COLUMNS
    )
    arrival_flow_ready = bool(
        len(calls) >= MIN_CALL_ROWS
        and arrival_coverage >= MIN_ARRIVAL_COVERAGE
        and years >= MIN_YEARS
        and continuous
        and duplicate_keys == 0
        and critical_leakage.empty
    )
    occupancy_ready = bool(
        arrival_flow_ready
        and build_metadata["occupancy_call_coverage"]
        >= MIN_OCCUPANCY_CALL_COVERAGE
        and invalid_actual_rate <= MAX_QUARANTINED_ACTUAL_SEQUENCE_RATE
    )
    weather_impact_ready = bool(arrival_flow_ready and wave_ready)

    fatal_reasons = []
    if len(calls) < MIN_CALL_ROWS:
        fatal_reasons.append("insufficient_port_calls")
    if arrival_coverage < MIN_ARRIVAL_COVERAGE:
        fatal_reasons.append("arrival_coverage_below_threshold")
    if years < MIN_YEARS:
        fatal_reasons.append("insufficient_years")
    if not continuous:
        fatal_reasons.append("hourly_grid_not_continuous")
    if duplicate_keys:
        fatal_reasons.append("duplicate_keys")
    if not critical_leakage.empty:
        fatal_reasons.append("critical_temporal_leakage")

    restricted_reasons = []
    if build_metadata["occupancy_call_coverage"] < MIN_OCCUPANCY_CALL_COVERAGE:
        restricted_reasons.append("occupancy_ata_atd_coverage_below_threshold")
    if invalid_actual:
        restricted_reasons.append("invalid_actual_sequences_quarantined_for_occupancy")
    if not wave_ready:
        restricted_reasons.append("wave_variables_below_coverage_threshold")
    if not full_weather_ready:
        restricted_reasons.append("full_weather_variables_unavailable")

    if not arrival_flow_ready:
        status = "NEED_DATA_REPAIR"
        next_block = "B56A_DATA_REPAIR"
    elif occupancy_ready:
        status = "READY_FOR_PORT_PRESSURE_BASELINES"
        next_block = "B56B_PORT_PRESSURE_TEMPORAL_BASELINES"
    else:
        status = "READY_FOR_ARRIVAL_FLOW_ONLY"
        next_block = "B56B_ARRIVAL_FLOW_TEMPORAL_BASELINES"

    return {
        "status": status,
        "audit_version": AUDIT_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_calls": len(calls),
        "hourly_rows": len(hourly),
        "years": years,
        "arrival_coverage_pct": 100 * arrival_coverage,
        "departure_coverage_pct": 100 * departure_coverage,
        "occupancy_call_coverage_pct": 100
        * build_metadata["occupancy_call_coverage"],
        "critical_leakage_violations": len(critical_leakage),
        "duplicate_keys": duplicate_keys,
        "invalid_actual_sequences": invalid_actual,
        "invalid_actual_sequence_rate_pct": 100 * invalid_actual_rate,
        "invalid_actual_sequences_quarantined": invalid_actual,
        "readiness": {
            "arrival_flow": arrival_flow_ready,
            "occupancy": occupancy_ready,
            "wave_impact": weather_impact_ready,
            "full_weather": full_weather_ready,
            "individual_delay_mvp": arrival_flow_ready,
        },
        "limitations": [
            "actual_atd coverage can bias occupancy reconstruction",
            "weather publication/available_at is absent",
            "ETA revision publication history is absent",
            "current weather is wave-dominant; wind/current/visibility may be empty",
        ],
        "fatal_reasons": fatal_reasons,
        "restricted_reasons": restricted_reasons,
        "training_executed": False,
        "split_created": False,
        "bronze_modified": False,
        "next_block": next_block,
    }


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _materialize_hourly(hourly: pd.DataFrame, run_id: str) -> int:
    columns = [
        "as_of_time",
        "arrivals_prev_1h",
        "arrivals_last_6h",
        "arrivals_last_24h",
        "arrivals_last_168h",
        "departures_prev_1h",
        "departures_last_6h",
        "departures_last_24h",
        "delayed_gt3_last_24h",
        "mean_arrival_delay_last_24h",
        "vessels_in_port_observed",
        "wave_height_lag_1h_m",
        "wave_period_lag_1h_s",
        "wave_direction_lag_1h_deg",
        "weather_available_flag",
        "hour_of_day",
        "day_of_week",
        "month",
        "weekend_flag",
        *TARGET_COLUMNS,
    ]
    values = []
    for row in hourly[columns].itertuples(index=False, name=None):
        converted = [row[0]]
        converted.extend(_optional_float(value) for value in row[1:])
        converted.extend([FEATURE_VERSION, run_id])
        values.append(tuple(converted))

    sql_columns = [*columns, "feature_version", "ingestion_run_id"]
    update_columns = [column for column in sql_columns if column != "as_of_time"]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS features")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS features.port_hourly_state_v1 (
                    as_of_time TIMESTAMPTZ NOT NULL,
                    arrivals_prev_1h REAL,
                    arrivals_last_6h REAL,
                    arrivals_last_24h REAL,
                    arrivals_last_168h REAL,
                    departures_prev_1h REAL,
                    departures_last_6h REAL,
                    departures_last_24h REAL,
                    delayed_gt3_last_24h REAL,
                    mean_arrival_delay_last_24h REAL,
                    vessels_in_port_observed REAL,
                    wave_height_lag_1h_m REAL,
                    wave_period_lag_1h_s REAL,
                    wave_direction_lag_1h_deg REAL,
                    weather_available_flag SMALLINT,
                    hour_of_day SMALLINT,
                    day_of_week SMALLINT,
                    month SMALLINT,
                    weekend_flag SMALLINT,
                    target_arrivals_next_6h REAL,
                    target_arrivals_next_12h REAL,
                    target_arrivals_next_24h REAL,
                    target_delayed_gt3_next_24h REAL,
                    target_mean_arrival_delay_next_24h REAL,
                    target_max_vessels_in_port_next_24h REAL,
                    feature_version TEXT NOT NULL,
                    ingestion_run_id UUID REFERENCES audit.ingestion_run(run_id),
                    PRIMARY KEY (as_of_time, feature_version)
                )
                """
            )
            cursor.execute(
                """
                SELECT create_hypertable(
                    'features.port_hourly_state_v1',
                    by_range('as_of_time'),
                    if_not_exists => TRUE
                )
                """
            )
            template = "(" + ",".join(["%s"] * len(sql_columns)) + ")"
            update_sql = ", ".join(
                f"{column}=EXCLUDED.{column}" for column in update_columns
            )
            execute_values(
                cursor,
                f"""
                INSERT INTO features.port_hourly_state_v1
                    ({', '.join(sql_columns)})
                VALUES %s
                ON CONFLICT (as_of_time, feature_version)
                DO UPDATE SET {update_sql}
                """,
                values,
                template=template,
                page_size=2000,
            )
    return len(values)


def _write_readme(path: Path, decision: dict[str, Any]) -> None:
    readiness = decision["readiness"]
    path.write_text(
        "\n".join(
            [
                "# B56A Operational Forecast Dataset Feasibility",
                "",
                f"Decision: **{decision['status']}**",
                "",
                "## Readiness",
                "",
                f"- Arrival flow: {readiness['arrival_flow']}",
                f"- Occupancy: {readiness['occupancy']}",
                f"- Wave impact: {readiness['wave_impact']}",
                f"- Full weather: {readiness['full_weather']}",
                f"- Individual delay MVP: {readiness['individual_delay_mvp']}",
                "",
                "## Guardrails",
                "",
                "No model was trained, no split was created, and Bronze was not modified.",
                "All rolling operational features use information strictly before as_of_time.",
                "Future windows are stored only in target-prefixed columns.",
                "",
                "## Next block",
                "",
                decision["next_block"],
            ]
        ),
        encoding="utf-8",
    )


def run_b56a_operational_feasibility(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    materialize_timescale: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    calls = _load_port_calls()
    weather, raw_weather_rows = _load_weather()
    checksum = _frame_signature(calls, weather)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "results": previous[1],
        }

    run_id = _start_run(checksum)
    try:
        hourly, build_metadata = build_hourly_frame(calls, weather)
        grain = _grain_audit(calls, hourly)
        continuity = _grid_continuity(hourly)
        weather_report = _weather_coverage(hourly)
        invalid = _invalid_sequences(calls)
        leakage = _leakage_audit(hourly)
        decision = _decision(
            calls,
            hourly,
            weather_report,
            grain,
            continuity,
            leakage,
            invalid,
            build_metadata,
        )

        reports = {
            "01_source_inventory.csv": _source_inventory(
                calls, weather, raw_weather_rows, hourly
            ),
            "02_port_call_grain_audit.csv": grain,
            "03_timestamp_semantic_audit.csv": _timestamp_semantics(),
            "04_arrival_departure_coverage.csv": _coverage_report(calls),
            "05_hourly_grid_continuity.csv": continuity,
            "06_weather_coverage.csv": weather_report,
            "07_target_feasibility.csv": _target_feasibility(hourly),
            "08_missingness_by_year.csv": _missingness_by_year(calls),
            "09_missingness_by_vessel.csv": _missingness_by_vessel(calls),
            "10_invalid_sequences.csv": invalid,
            "11_join_cardinality_audit.csv": _join_audit(
                calls, weather, raw_weather_rows, hourly
            ),
            "12_temporal_leakage_audit.csv": leakage,
            "13_target_distribution_by_year.csv": _target_distribution_by_year(
                hourly
            ),
            "14_vessel_concentration.csv": _vessel_concentration(calls),
            "15_hourly_target_autocorrelation.csv": _autocorrelation(hourly),
            "16_seasonality_diagnostics.csv": _seasonality(hourly),
        }

        with tempfile.TemporaryDirectory(prefix="b56a-") as temporary:
            output_dir = Path(temporary)
            for name, report in reports.items():
                report.to_csv(output_dir / name, index=False)

            dataset_path = output_dir / "port_hourly_state_feasibility_v1.parquet"
            export_columns = [
                "as_of_time",
                "feature_version",
                *FEATURE_COLUMNS,
                *TARGET_COLUMNS,
            ]
            hourly[export_columns].to_parquet(dataset_path, index=False)
            decision_path = output_dir / "17_final_feasibility_decision.json"
            decision_path.write_text(
                json.dumps(_clean_json(decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            readme_path = output_dir / "README_B56A.md"
            _write_readme(readme_path, decision)

            materialized_rows = 0
            if materialize_timescale and decision["readiness"]["arrival_flow"]:
                materialized_rows = _materialize_hourly(hourly, run_id)

            uploaded: dict[str, str] = {}
            for path in sorted(output_dir.iterdir()):
                if path == dataset_path:
                    key = f"datasets/b56a/{output_prefix}/{path.name}"
                elif path == decision_path:
                    key = f"configs/b56a/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b56a/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload_file(
                    client, path, output_bucket, key
                )

        metadata = {
            **decision,
            **_clean_json(build_metadata),
            "checksum": checksum,
            "materialized_timescale_rows": materialized_rows,
            "timescale_table": "features.port_hourly_state_v1"
            if materialized_rows
            else None,
            "outputs": uploaded,
            "output_prefix": f"s3://{output_bucket}/reports/b56a/{output_prefix}/",
        }
        _finish_run(run_id, "SUCCESS", len(hourly), metadata)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "results": metadata,
            "outputs": uploaded,
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {"audit_version": AUDIT_VERSION},
            error_message=str(exc),
        )
        raise
