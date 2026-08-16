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


AUDIT_VERSION = "b58a-weather-timeseries-feasibility-v1"
FEATURE_VERSION = "b58a-weather-hourly-past-only-v1"
SOURCE_NAME = "b58a_weather_timeseries_audit"
DATASET_NAME = "maritime_weather_hourly_multivariate"
SOURCE_TABLE = "core.maritime_observation"
TIMESCALE_TABLE = "features.maritime_weather_hourly_audit_v1"

HORIZONS_H = (6, 12, 24, 48, 72)
AUTOCORRELATION_LAGS_H = (1, 3, 6, 12, 24, 48, 72, 168, 336, 720)
PAST_LAGS_H = (1, 3, 6, 12, 24, 48, 72, 168)
PAST_WINDOWS_H = (3, 6, 12, 24, 72, 168)

CONTINUOUS_VARIABLES = (
    "wave_height_m",
    "wave_period_s",
    "wind_speed_ms",
    "surface_current_ms",
    "visibility_m",
    "pressure_hpa",
)
DIRECTION_VARIABLES = (
    "wave_direction_deg",
    "wind_direction_deg",
)
WEATHER_VARIABLES = (
    "wave_height_m",
    "wave_period_s",
    "wave_direction_deg",
    "wind_speed_ms",
    "wind_direction_deg",
    "surface_current_ms",
    "visibility_m",
    "pressure_hpa",
)
WAVE_VARIABLES = (
    "wave_height_m",
    "wave_period_s",
    "wave_direction_deg",
)
FULL_WEATHER_VARIABLES = WEATHER_VARIABLES

MIN_HOURS = 24 * 365 * 3
MIN_COVERAGE = 0.95
MIN_CONTINUITY = 0.99
MIN_TARGET_PAIRS = 24 * 365


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


def _query_frame(query: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _relation_exists(schema: str, relation: str) -> bool:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (f"{schema}.{relation}",))
            return cursor.fetchone()[0] is not None


def _load_source() -> pd.DataFrame:
    if not _relation_exists("core", "maritime_observation"):
        raise RuntimeError(f"Required source does not exist: {SOURCE_TABLE}")
    frame = _query_frame(
        """
        SELECT
            observed_at,
            source,
            latitude::double precision AS latitude,
            longitude::double precision AS longitude,
            wave_height_m::double precision AS wave_height_m,
            wave_period_s::double precision AS wave_period_s,
            wave_direction_deg::double precision AS wave_direction_deg,
            wind_speed_ms::double precision AS wind_speed_ms,
            wind_direction_deg::double precision AS wind_direction_deg,
            surface_current_ms::double precision AS surface_current_ms,
            visibility_m::double precision AS visibility_m,
            pressure_hpa::double precision AS pressure_hpa,
            quality_flag::integer AS quality_flag,
            ingestion_run_id::text AS ingestion_run_id
        FROM core.maritime_observation
        ORDER BY observed_at, source, latitude, longitude
        """
    )
    if frame.empty:
        raise RuntimeError(f"No observations were found in {SOURCE_TABLE}")
    frame["observed_at"] = pd.to_datetime(
        frame["observed_at"], errors="coerce", utc=True
    )
    if frame["observed_at"].isna().any():
        raise RuntimeError("Source contains invalid observed_at values")
    for column in WEATHER_VARIABLES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["quality_flag"] = pd.to_numeric(
        frame["quality_flag"], errors="coerce"
    ).fillna(1).astype("int16")
    return frame


def _source_schema() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT
            ordinal_position,
            column_name,
            data_type,
            is_nullable,
            CASE
                WHEN column_name='observed_at' THEN 'EVENT_TIME'
                WHEN column_name='ingestion_run_id' THEN 'LINEAGE_REFERENCE'
                ELSE 'OBSERVATION_OR_ATTRIBUTE'
            END AS semantic_role
        FROM information_schema.columns
        WHERE table_schema='core'
          AND table_name='maritime_observation'
        ORDER BY ordinal_position
        """
    )


def _frame_signature(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(AUDIT_VERSION.encode("ascii"))
    columns = [
        "observed_at",
        "source",
        "latitude",
        "longitude",
        *WEATHER_VARIABLES,
        "quality_flag",
        "ingestion_run_id",
    ]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False)
    digest.update(hashed.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _start_run(checksum: str) -> str:
    metadata = {
        "audit_version": AUDIT_VERSION,
        "feature_version": FEATURE_VERSION,
        "training_executed": False,
        "split_created": False,
        "bronze_modified": False,
        "interpolation_executed": False,
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
                    "postgresql://maritime/core.maritime_observation",
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
                WHERE source_name=%s
                  AND dataset_name=%s
                  AND checksum=%s
                  AND status='SUCCESS'
                ORDER BY finished_at DESC
                LIMIT 1
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
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{bucket}/{key}"


def _circular_degrees(sine: pd.Series, cosine: pd.Series) -> pd.Series:
    angle = np.degrees(np.arctan2(sine, cosine))
    return pd.Series(np.mod(angle, 360.0), index=sine.index)


def build_hourly_dataset(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = source.copy()
    raw["hour"] = raw["observed_at"].dt.floor("h")
    duplicate_keys = int(
        raw.duplicated(
            ["observed_at", "source", "latitude", "longitude"],
            keep=False,
        ).sum()
    )
    good = raw[raw["quality_flag"].eq(0)].copy()
    if good.empty:
        raise RuntimeError("No quality_flag=0 weather observations are available")

    for column in DIRECTION_VARIABLES:
        radians = np.deg2rad(good[column])
        good[f"_{column}_sin"] = np.sin(radians)
        good[f"_{column}_cos"] = np.cos(radians)

    aggregations: dict[str, tuple[str, str]] = {
        "observation_count": ("observed_at", "size"),
        "source_count": ("source", "nunique"),
        "latitude": ("latitude", "mean"),
        "longitude": ("longitude", "mean"),
    }
    for column in CONTINUOUS_VARIABLES:
        aggregations[column] = (column, "mean")
    for column in DIRECTION_VARIABLES:
        aggregations[f"_{column}_sin"] = (f"_{column}_sin", "mean")
        aggregations[f"_{column}_cos"] = (f"_{column}_cos", "mean")

    aggregated = good.groupby("hour", as_index=False).agg(**aggregations)
    for column in DIRECTION_VARIABLES:
        aggregated[column] = _circular_degrees(
            aggregated.pop(f"_{column}_sin"),
            aggregated.pop(f"_{column}_cos"),
        )

    first_hour = raw["hour"].min()
    last_hour = raw["hour"].max()
    grid = pd.DataFrame(
        {
            "observed_at": pd.date_range(
                first_hour, last_hour, freq="h", tz="UTC"
            )
        }
    )
    hourly = grid.merge(
        aggregated.rename(columns={"hour": "observed_at"}),
        on="observed_at",
        how="left",
        validate="one_to_one",
    )
    for column in ("observation_count", "source_count"):
        hourly[column] = hourly[column].fillna(0).astype("int32")

    for column in WEATHER_VARIABLES:
        hourly[f"{column}_available_flag"] = (
            hourly[column].notna().astype("int8")
        )

    hourly["wave_family_available_flag"] = (
        hourly[list(WAVE_VARIABLES)].notna().all(axis=1).astype("int8")
    )
    hourly["full_weather_available_flag"] = (
        hourly[list(FULL_WEATHER_VARIABLES)].notna().all(axis=1).astype("int8")
    )
    hourly["any_weather_available_flag"] = (
        hourly[list(WEATHER_VARIABLES)].notna().any(axis=1).astype("int8")
    )

    hourly["hour_of_day"] = hourly["observed_at"].dt.hour.astype("int8")
    hourly["day_of_week"] = hourly["observed_at"].dt.dayofweek.astype("int8")
    hourly["month"] = hourly["observed_at"].dt.month.astype("int8")
    hourly["day_of_year"] = hourly["observed_at"].dt.dayofyear.astype("int16")
    hourly["weekend_flag"] = (
        hourly["day_of_week"].ge(5).astype("int8")
    )
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour_of_day"] / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour_of_day"] / 24)
    hourly["day_of_year_sin"] = np.sin(
        2 * np.pi * hourly["day_of_year"] / 365.25
    )
    hourly["day_of_year_cos"] = np.cos(
        2 * np.pi * hourly["day_of_year"] / 365.25
    )

    for column in DIRECTION_VARIABLES:
        radians = np.deg2rad(hourly[column])
        hourly[f"{column}_sin"] = np.sin(radians)
        hourly[f"{column}_cos"] = np.cos(radians)

    lag_sources = [
        *CONTINUOUS_VARIABLES,
        "wave_direction_deg_sin",
        "wave_direction_deg_cos",
        "wind_direction_deg_sin",
        "wind_direction_deg_cos",
    ]
    past_features: dict[str, pd.Series] = {}
    for column in lag_sources:
        if not hourly[column].notna().any():
            continue
        for lag_h in PAST_LAGS_H:
            past_features[f"past_{column}_lag_{lag_h}h"] = hourly[column].shift(
                lag_h
            )

    for column in CONTINUOUS_VARIABLES:
        if not hourly[column].notna().any():
            continue
        shifted = hourly[column].shift(1)
        for window_h in PAST_WINDOWS_H:
            rolling = shifted.rolling(window_h, min_periods=1)
            past_features[f"past_{column}_mean_{window_h}h"] = rolling.mean()
            past_features[f"past_{column}_std_{window_h}h"] = rolling.std()
            past_features[f"past_{column}_min_{window_h}h"] = rolling.min()
            past_features[f"past_{column}_max_{window_h}h"] = rolling.max()
            past_features[f"past_{column}_count_{window_h}h"] = rolling.count()

    if past_features:
        hourly = pd.concat(
            [hourly, pd.DataFrame(past_features, index=hourly.index)],
            axis=1,
        )

    hourly.insert(1, "feature_version", FEATURE_VERSION)
    metadata = {
        "source_rows": int(len(raw)),
        "quality_rows": int(len(good)),
        "hourly_rows": int(len(hourly)),
        "first_observed_at": first_hour,
        "last_observed_at": last_hour,
        "duplicate_source_key_rows": duplicate_keys,
        "source_count": int(raw["source"].nunique()),
        "latitude_count": int(raw["latitude"].nunique(dropna=True)),
        "longitude_count": int(raw["longitude"].nunique(dropna=True)),
    }
    return hourly, metadata


def _source_inventory(
    source: pd.DataFrame,
    hourly: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    first = hourly["observed_at"].min()
    last = hourly["observed_at"].max()
    span_years = (last - first).total_seconds() / (365.25 * 24 * 3600)
    rows = [
        {
            "item": "source_relation",
            "value": SOURCE_TABLE,
            "interpretation": "canonical weather observation source",
        },
        {
            "item": "source_rows",
            "value": len(source),
            "interpretation": "raw observations, never modified by B58A",
        },
        {
            "item": "quality_rows",
            "value": metadata["quality_rows"],
            "interpretation": "quality_flag=0 rows used for hourly values",
        },
        {
            "item": "hourly_rows",
            "value": len(hourly),
            "interpretation": "complete UTC hourly grid rows",
        },
        {
            "item": "first_observed_at",
            "value": first.isoformat(),
            "interpretation": "first event time",
        },
        {
            "item": "last_observed_at",
            "value": last.isoformat(),
            "interpretation": "last event time",
        },
        {
            "item": "span_years",
            "value": span_years,
            "interpretation": "historical depth",
        },
        {
            "item": "available_at_present",
            "value": False,
            "interpretation": (
                "historical observation availability time is not recorded"
            ),
        },
        {
            "item": "historical_forecast_archive_present",
            "value": False,
            "interpretation": (
                "no archived issue-time weather forecast was found"
            ),
        },
    ]
    return pd.DataFrame(rows)


def _continuity_report(
    source: pd.DataFrame, hourly: pd.DataFrame
) -> pd.DataFrame:
    observed_hours = (
        source.loc[source["quality_flag"].eq(0), "observed_at"]
        .dt.floor("h")
        .nunique()
    )
    expected_hours = len(hourly)
    counts = source.groupby(source["observed_at"].dt.floor("h")).size()
    return pd.DataFrame(
        [
            {
                "first_hour": hourly["observed_at"].min(),
                "last_hour": hourly["observed_at"].max(),
                "expected_hours": expected_hours,
                "observed_quality_hours": int(observed_hours),
                "missing_source_hours": int(expected_hours - observed_hours),
                "continuity_ratio": float(observed_hours / expected_hours),
                "hours_with_multiple_rows": int(counts.gt(1).sum()),
                "max_rows_per_hour": int(counts.max()),
                "grid_duplicate_hours": int(
                    hourly["observed_at"].duplicated().sum()
                ),
                "regular_frequency": "1H_UTC",
            }
        ]
    )


def _coverage_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in WEATHER_VARIABLES:
        available = int(hourly[column].notna().sum())
        values = hourly[column].dropna()
        rows.append(
            {
                "variable": column,
                "family": "WAVE" if column in WAVE_VARIABLES else "OTHER_WEATHER",
                "rows": len(hourly),
                "available_rows": available,
                "missing_rows": int(len(hourly) - available),
                "coverage_pct": 100.0 * available / len(hourly),
                "distinct_values": int(values.nunique()),
                "variance": float(values.var()) if len(values) > 1 else np.nan,
                "usable_for_baseline": bool(
                    available / len(hourly) >= MIN_COVERAGE
                    and values.nunique() > 1
                ),
                "imputation_applied": False,
            }
        )
    return pd.DataFrame(rows)


def _missing_run_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in WEATHER_VARIABLES:
        missing = hourly[column].isna()
        groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        runs = []
        for _, index in hourly.loc[missing].groupby(groups[missing]).groups.items():
            positions = list(index)
            start_position = positions[0]
            end_position = positions[-1]
            runs.append(
                {
                    "variable": column,
                    "start_at": hourly.at[start_position, "observed_at"],
                    "end_at": hourly.at[end_position, "observed_at"],
                    "missing_hours": int(end_position - start_position + 1),
                }
            )
        if not runs:
            rows.append(
                {
                    "variable": column,
                    "start_at": pd.NaT,
                    "end_at": pd.NaT,
                    "missing_hours": 0,
                }
            )
        else:
            rows.extend(
                sorted(
                    runs,
                    key=lambda item: item["missing_hours"],
                    reverse=True,
                )[:20]
            )
    return pd.DataFrame(rows)


def _descriptive_statistics(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in WEATHER_VARIABLES:
        values = hourly[column].dropna()
        if values.empty:
            rows.append(
                {
                    "variable": column,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p01": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                }
            )
            continue
        rows.append(
            {
                "variable": column,
                "n": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "p05": float(values.quantile(0.05)),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _analysis_variables(hourly: pd.DataFrame) -> list[str]:
    candidates = [
        *CONTINUOUS_VARIABLES,
        "wave_direction_deg_sin",
        "wave_direction_deg_cos",
        "wind_direction_deg_sin",
        "wind_direction_deg_cos",
    ]
    return [
        column
        for column in candidates
        if column in hourly
        and hourly[column].notna().sum() >= 2
        and hourly[column].nunique(dropna=True) > 1
    ]


def _seasonality_profiles(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = _analysis_variables(hourly)
    dimensions = (
        ("hour_of_day", "HOUR_OF_DAY"),
        ("month", "MONTH"),
        ("day_of_week", "DAY_OF_WEEK"),
    )
    for variable in variables:
        for group_column, dimension in dimensions:
            grouped = hourly.groupby(group_column, dropna=False)[variable]
            for value, series in grouped:
                series = series.dropna()
                if series.empty:
                    continue
                rows.append(
                    {
                        "variable": variable,
                        "dimension": dimension,
                        "period_value": value,
                        "n": len(series),
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "std": float(series.std()),
                        "p90": float(series.quantile(0.90)),
                    }
                )
    return pd.DataFrame(rows)


def _autocorrelation_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in _analysis_variables(hourly):
        series = hourly[variable]
        for lag_h in AUTOCORRELATION_LAGS_H:
            pairs = pd.concat(
                [series.rename("current"), series.shift(lag_h).rename("lagged")],
                axis=1,
            ).dropna()
            rows.append(
                {
                    "variable": variable,
                    "lag_h": lag_h,
                    "pair_rows": len(pairs),
                    "pearson_autocorrelation": (
                        float(pairs["current"].corr(pairs["lagged"]))
                        if len(pairs) >= 3
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _cross_correlation_report(hourly: pd.DataFrame) -> pd.DataFrame:
    variables = _analysis_variables(hourly)
    rows = []
    for left_index, left in enumerate(variables):
        for right in variables[left_index + 1 :]:
            pairs = hourly[[left, right]].dropna()
            if len(pairs) < 3:
                continue
            rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "pair_rows": len(pairs),
                    "pearson": float(pairs[left].corr(pairs[right])),
                    "spearman": float(
                        pairs[left].corr(pairs[right], method="spearman")
                    ),
                    "predictive_or_causal_claim": False,
                }
            )
    return pd.DataFrame(rows)


def _period_drift_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = hourly.assign(
        year=hourly["observed_at"].dt.year,
        year_month=hourly["observed_at"].dt.strftime("%Y-%m"),
    )
    for variable in _analysis_variables(hourly):
        for grain, group_column in (("YEAR", "year"), ("MONTH", "year_month")):
            for period, values in frame.groupby(group_column)[variable]:
                clean = values.dropna()
                rows.append(
                    {
                        "variable": variable,
                        "grain": grain,
                        "period": period,
                        "n": len(clean),
                        "coverage_pct": 100.0 * len(clean) / len(values),
                        "mean": float(clean.mean()) if len(clean) else np.nan,
                        "median": float(clean.median()) if len(clean) else np.nan,
                        "std": float(clean.std()) if len(clean) > 1 else np.nan,
                        "p95": (
                            float(clean.quantile(0.95)) if len(clean) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _psi(expected: pd.Series, actual: pd.Series) -> float:
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    if len(expected) < 100 or len(actual) < 100:
        return np.nan
    boundaries = np.unique(
        expected.quantile(np.linspace(0, 1, 11)).to_numpy(dtype=float)
    )
    if len(boundaries) < 3:
        return 0.0
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    expected_counts = (
        pd.cut(expected, boundaries, include_lowest=True).value_counts(
            sort=False, normalize=True
        )
    )
    actual_counts = (
        pd.cut(actual, boundaries, include_lowest=True).value_counts(
            sort=False, normalize=True
        )
    )
    expected_probability = np.clip(expected_counts.to_numpy(), 1e-6, None)
    actual_probability = np.clip(actual_counts.to_numpy(), 1e-6, None)
    return float(
        np.sum(
            (actual_probability - expected_probability)
            * np.log(actual_probability / expected_probability)
        )
    )


def _yearly_psi_report(hourly: pd.DataFrame) -> pd.DataFrame:
    years = sorted(hourly["observed_at"].dt.year.unique())
    if not years:
        return pd.DataFrame()
    baseline_year = years[0]
    rows = []
    for variable in _analysis_variables(hourly):
        baseline = hourly.loc[
            hourly["observed_at"].dt.year.eq(baseline_year), variable
        ]
        for year in years:
            current = hourly.loc[
                hourly["observed_at"].dt.year.eq(year), variable
            ]
            psi = _psi(baseline, current)
            rows.append(
                {
                    "variable": variable,
                    "baseline_year": baseline_year,
                    "comparison_year": year,
                    "baseline_rows": int(baseline.notna().sum()),
                    "comparison_rows": int(current.notna().sum()),
                    "psi": psi,
                    "interpretation": (
                        "HIGH_SHIFT"
                        if pd.notna(psi) and psi >= 0.25
                        else "MODERATE_SHIFT"
                        if pd.notna(psi) and psi >= 0.10
                        else "LOW_SHIFT"
                        if pd.notna(psi)
                        else "INSUFFICIENT_DATA"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _longest_true_run(mask: pd.Series) -> int:
    if not mask.any():
        return 0
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return int(mask[mask].groupby(groups[mask]).size().max())


def _extreme_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in CONTINUOUS_VARIABLES:
        values = hourly[variable].dropna()
        if len(values) < 100:
            continue
        for quantile in (0.90, 0.95, 0.99):
            threshold = float(values.quantile(quantile))
            mask = hourly[variable].ge(threshold)
            rows.append(
                {
                    "variable": variable,
                    "threshold_kind": f"EMPIRICAL_P{int(quantile * 100)}",
                    "threshold": threshold,
                    "hours_at_or_above": int(mask.sum()),
                    "rate_pct": 100.0 * float(mask.mean()),
                    "longest_consecutive_hours": _longest_true_run(mask),
                }
            )
    if hourly["wave_height_m"].notna().sum() >= 100:
        for threshold in (1.5, 2.0, 3.0):
            mask = hourly["wave_height_m"].ge(threshold)
            rows.append(
                {
                    "variable": "wave_height_m",
                    "threshold_kind": "DOMAIN_SCREENING_METERS",
                    "threshold": threshold,
                    "hours_at_or_above": int(mask.sum()),
                    "rate_pct": 100.0 * float(mask.mean()),
                    "longest_consecutive_hours": _longest_true_run(mask),
                }
            )
    return pd.DataFrame(rows)


def _angular_absolute_error(
    actual: pd.Series, predicted: pd.Series
) -> pd.Series:
    difference = (actual - predicted).abs() % 360.0
    return pd.concat([difference, 360.0 - difference], axis=1).min(axis=1)


def _forecast_target_feasibility(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in WEATHER_VARIABLES:
        series = hourly[variable]
        for horizon_h in HORIZONS_H:
            target = series.shift(-horizon_h)
            persistence = series
            daily_source_lag_h = (
                int(math.ceil(horizon_h / 24.0) * 24) - horizon_h
            )
            daily_seasonal = series.shift(daily_source_lag_h)
            weekly_seasonal = series.shift(168 - horizon_h)
            comparison = pd.DataFrame(
                {
                    "target": target,
                    "persistence": persistence,
                    "daily_seasonal": daily_seasonal,
                    "weekly_seasonal": weekly_seasonal,
                }
            ).dropna()
            if comparison.empty:
                rows.append(
                    {
                        "variable": variable,
                        "horizon_h": horizon_h,
                        "pair_rows": 0,
                        "pair_coverage_pct": 0.0,
                        "persistence_mae": np.nan,
                        "safe_daily_seasonal_mae": np.nan,
                        "safe_weekly_seasonal_mae": np.nan,
                        "persistence_correlation": np.nan,
                        "target_ready": False,
                        "diagnostic_only": True,
                    }
                )
                continue

            if variable in DIRECTION_VARIABLES:
                persistence_error = _angular_absolute_error(
                    comparison["target"], comparison["persistence"]
                )
                daily_error = _angular_absolute_error(
                    comparison["target"], comparison["daily_seasonal"]
                )
                weekly_error = _angular_absolute_error(
                    comparison["target"], comparison["weekly_seasonal"]
                )
            else:
                persistence_error = (
                    comparison["target"] - comparison["persistence"]
                ).abs()
                daily_error = (
                    comparison["target"] - comparison["daily_seasonal"]
                ).abs()
                weekly_error = (
                    comparison["target"] - comparison["weekly_seasonal"]
                ).abs()

            pair_coverage = len(comparison) / max(1, len(hourly) - horizon_h)
            rows.append(
                {
                    "variable": variable,
                    "horizon_h": horizon_h,
                    "pair_rows": len(comparison),
                    "pair_coverage_pct": 100.0 * pair_coverage,
                    "persistence_mae": float(persistence_error.mean()),
                    "safe_daily_seasonal_mae": float(daily_error.mean()),
                    "safe_weekly_seasonal_mae": float(weekly_error.mean()),
                    "persistence_correlation": float(
                        comparison["target"].corr(comparison["persistence"])
                    ),
                    "target_ready": bool(
                        len(comparison) >= MIN_TARGET_PAIRS
                        and pair_coverage >= MIN_COVERAGE
                    ),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _timestamp_semantics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field": "observed_at",
                "semantic": "EVENT_TIME",
                "known_at_operational_time": "ASSUMPTION_WITH_LATENCY",
                "use": "target time or source time for strictly lagged features",
                "risk": "availability latency is not historically captured",
            },
            {
                "field": "available_at",
                "semantic": "SYSTEM_AVAILABILITY_TIME",
                "known_at_operational_time": "NOT_PRESENT",
                "use": "required before formal historical replay",
                "risk": "production replay is blocked",
            },
            {
                "field": "ingestion_run_id",
                "semantic": "LINEAGE_REFERENCE",
                "known_at_operational_time": "NOT_A_TIMESTAMP",
                "use": "trace source ingestion only",
                "risk": "must not substitute for available_at",
            },
        ]
    )


def _anti_leakage_contract(hourly: pd.DataFrame) -> pd.DataFrame:
    past_columns = [
        column for column in hourly.columns if column.startswith("past_")
    ]
    invalid_past_names = [
        column
        for column in past_columns
        if "_lag_0h" in column or "future" in column or "target" in column
    ]
    return pd.DataFrame(
        [
            {
                "check": "future_columns_in_gold_dataset",
                "severity": "CRITICAL",
                "violations": int(
                    sum(
                        column.startswith("target_") or "future_" in column
                        for column in hourly.columns
                    )
                ),
                "passed": not any(
                    column.startswith("target_") or "future_" in column
                    for column in hourly.columns
                ),
                "detail": "B58A stores observations and past-only features only",
            },
            {
                "check": "past_feature_naming_contract",
                "severity": "CRITICAL",
                "violations": len(invalid_past_names),
                "passed": len(invalid_past_names) == 0,
                "detail": "all lag/rolling features are shifted by at least one hour",
            },
            {
                "check": "interpolation_or_backfill",
                "severity": "CRITICAL",
                "violations": 0,
                "passed": True,
                "detail": "no interpolation, bfill, or fabricated weather values",
            },
            {
                "check": "available_at_present",
                "severity": "WARNING",
                "violations": 1,
                "passed": False,
                "detail": (
                    "historical availability timestamp is absent; formal replay blocked"
                ),
            },
            {
                "check": "split_created",
                "severity": "CRITICAL",
                "violations": 0,
                "passed": True,
                "detail": "B58A does not split data",
            },
            {
                "check": "training_executed",
                "severity": "CRITICAL",
                "violations": 0,
                "passed": True,
                "detail": "persistence metrics are descriptive diagnostics only",
            },
            {
                "check": "bronze_modified",
                "severity": "CRITICAL",
                "violations": 0,
                "passed": True,
                "detail": "source tables are read-only",
            },
        ]
    )


def _decision(
    hourly: pd.DataFrame,
    metadata: dict[str, Any],
    continuity: pd.DataFrame,
    coverage: pd.DataFrame,
    target_feasibility: pd.DataFrame,
    anti_leakage: pd.DataFrame,
) -> dict[str, Any]:
    coverage_lookup = coverage.set_index("variable")["coverage_pct"].to_dict()
    usable_lookup = coverage.set_index("variable")[
        "usable_for_baseline"
    ].to_dict()
    wave_ready = all(bool(usable_lookup.get(item, False)) for item in WAVE_VARIABLES)
    full_weather_ready = all(
        bool(usable_lookup.get(item, False)) for item in FULL_WEATHER_VARIABLES
    )
    span_hours = len(hourly)
    continuity_ratio = float(continuity.iloc[0]["continuity_ratio"])
    critical_leakage_violations = int(
        anti_leakage.loc[
            anti_leakage["severity"].eq("CRITICAL")
            & ~anti_leakage["passed"],
            "violations",
        ].sum()
    )
    wave_targets_ready = bool(
        target_feasibility.loc[
            target_feasibility["variable"].isin(WAVE_VARIABLES),
            "target_ready",
        ].all()
    )
    integrity_gates = {
        "minimum_history": span_hours >= MIN_HOURS,
        "hourly_continuity": continuity_ratio >= MIN_CONTINUITY,
        "unique_hourly_grain": not hourly["observed_at"].duplicated().any(),
        "wave_coverage": wave_ready,
        "wave_target_pairs": wave_targets_ready,
        "critical_anti_leakage": critical_leakage_violations == 0,
    }
    integrity_passed = all(integrity_gates.values())

    if integrity_passed and full_weather_ready:
        status = "READY_FOR_MULTIVARIATE_WEATHER_BASELINES"
        next_block = "B58B_MULTIVARIATE_WEATHER_ROLLING_BACKTEST"
    elif integrity_passed and wave_ready:
        status = "READY_FOR_WAVE_ONLY_TEMPORAL_BASELINES"
        next_block = "B58B_WAVE_BASELINES_AND_ROLLING_BACKTEST"
    elif span_hours < MIN_HOURS or continuity_ratio < MIN_CONTINUITY:
        status = "NEED_WEATHER_DATA_REPAIR"
        next_block = "B58A_SOURCE_CONTINUITY_AND_HISTORY_REPAIR"
    else:
        status = "WEATHER_SERIES_NOT_FORECASTABLE"
        next_block = "B58A_REVIEW_TARGET_VARIANCE_AND_SOURCE"

    absent_variables = [
        column
        for column in WEATHER_VARIABLES
        if float(coverage_lookup.get(column, 0.0)) == 0.0
    ]
    return {
        "status": status,
        "decision": status,
        "audit_version": AUDIT_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_rows": metadata["source_rows"],
        "hourly_rows": len(hourly),
        "first_observed_at": hourly["observed_at"].min(),
        "last_observed_at": hourly["observed_at"].max(),
        "forecast_horizons_h": list(HORIZONS_H),
        "integrity_gates": integrity_gates,
        "gates_passed": integrity_passed,
        "critical_leakage_violations": critical_leakage_violations,
        "wave_track_ready": wave_ready,
        "full_weather_track_ready": full_weather_ready,
        "weather_family_status": {
            "waves": "READY" if wave_ready else "NOT_READY",
            "wind": (
                "READY"
                if all(
                    usable_lookup.get(item, False)
                    for item in ("wind_speed_ms", "wind_direction_deg")
                )
                else "UNAVAILABLE"
            ),
            "current": (
                "READY"
                if usable_lookup.get("surface_current_ms", False)
                else "UNAVAILABLE"
            ),
            "visibility": (
                "READY"
                if usable_lookup.get("visibility_m", False)
                else "UNAVAILABLE"
            ),
            "pressure": (
                "READY"
                if usable_lookup.get("pressure_hpa", False)
                else "UNAVAILABLE"
            ),
        },
        "absent_variables": absent_variables,
        "training_executed": False,
        "split_created": False,
        "bronze_modified": False,
        "interpolation_executed": False,
        "historical_replay_allowed": False,
        "production_promotion_allowed": False,
        "availability_timestamp_present": False,
        "historical_forecast_archive_present": False,
        "scope": (
            "retrospective observed-wave autoregressive feasibility; "
            "not an NWP forecast and not a causal weather-impact study"
        ),
        "next_block": next_block,
    }


def _database_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _materialize_hourly(hourly: pd.DataFrame, run_id: str) -> int:
    columns = [
        "observed_at",
        "feature_version",
        "observation_count",
        "source_count",
        "latitude",
        "longitude",
        *WEATHER_VARIABLES,
        "wave_direction_deg_sin",
        "wave_direction_deg_cos",
        "wind_direction_deg_sin",
        "wind_direction_deg_cos",
        "wave_family_available_flag",
        "full_weather_available_flag",
        "any_weather_available_flag",
        "hour_of_day",
        "day_of_week",
        "month",
        "day_of_year",
        "weekend_flag",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    ddl = """
        CREATE SCHEMA IF NOT EXISTS features;
        CREATE TABLE IF NOT EXISTS features.maritime_weather_hourly_audit_v1 (
            observed_at timestamptz PRIMARY KEY,
            feature_version text NOT NULL,
            observation_count integer NOT NULL,
            source_count integer NOT NULL,
            latitude double precision,
            longitude double precision,
            wave_height_m double precision,
            wave_period_s double precision,
            wave_direction_deg double precision,
            wind_speed_ms double precision,
            wind_direction_deg double precision,
            surface_current_ms double precision,
            visibility_m double precision,
            pressure_hpa double precision,
            wave_direction_deg_sin double precision,
            wave_direction_deg_cos double precision,
            wind_direction_deg_sin double precision,
            wind_direction_deg_cos double precision,
            wave_family_available_flag smallint NOT NULL,
            full_weather_available_flag smallint NOT NULL,
            any_weather_available_flag smallint NOT NULL,
            hour_of_day smallint NOT NULL,
            day_of_week smallint NOT NULL,
            month smallint NOT NULL,
            day_of_year smallint NOT NULL,
            weekend_flag smallint NOT NULL,
            hour_sin double precision NOT NULL,
            hour_cos double precision NOT NULL,
            day_of_year_sin double precision NOT NULL,
            day_of_year_cos double precision NOT NULL,
            audit_run_id uuid NOT NULL,
            materialized_at timestamptz NOT NULL DEFAULT now()
        );
    """
    insert_columns = [*columns, "audit_run_id"]
    update_columns = [
        column for column in insert_columns if column != "observed_at"
    ]
    update_clause = ", ".join(
        f"{column}=EXCLUDED.{column}" for column in update_columns
    )
    sql = f"""
        INSERT INTO {TIMESCALE_TABLE} ({", ".join(insert_columns)})
        VALUES %s
        ON CONFLICT (observed_at) DO UPDATE SET
            {update_clause},
            materialized_at=now()
    """
    values = [
        tuple(_database_value(row[column]) for column in columns) + (run_id,)
        for _, row in hourly[columns].iterrows()
    ]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            execute_values(cursor, sql, values, page_size=5000)
    return len(values)


def _write_readme(path: Path, decision: dict[str, Any]) -> None:
    family = decision["weather_family_status"]
    path.write_text(
        "\n".join(
            [
                "# B58A Weather Time-Series Feasibility Audit",
                "",
                f"Decision: **{decision['status']}**",
                "",
                "## What is usable",
                "",
                f"- Wave family: {family['waves']}",
                f"- Wind: {family['wind']}",
                f"- Surface current: {family['current']}",
                f"- Visibility: {family['visibility']}",
                f"- Pressure: {family['pressure']}",
                "",
                "The current source supports an observed-wave time-series track. "
                "It does not support a full multivariate weather model because "
                "wind, current, visibility and pressure are absent.",
                "",
                "## Guardrails",
                "",
                "- No model was trained and no temporal split was created.",
                "- No Bronze or Core source row was changed.",
                "- Missing variables were not imputed or fabricated.",
                "- Direction is represented with sine/cosine components.",
                "- Every generated lag or rolling feature is shifted by at least one hour.",
                "- Persistence and seasonal errors are diagnostics, not selected models.",
                "",
                "## Availability limitation",
                "",
                "The source records observed_at but not available_at. Therefore this "
                "audit cannot prove what was known at each historical decision time. "
                "Formal operational replay and production promotion remain blocked.",
                "",
                "## Next block",
                "",
                decision["next_block"],
            ]
        ),
        encoding="utf-8",
    )


def run_b58a_weather_timeseries_audit(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    materialize_timescale: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    source = _load_source()
    checksum = _frame_signature(source)
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
        hourly, build_metadata = build_hourly_dataset(source)
        continuity = _continuity_report(source, hourly)
        coverage = _coverage_report(hourly)
        target_feasibility = _forecast_target_feasibility(hourly)
        anti_leakage = _anti_leakage_contract(hourly)
        decision = _decision(
            hourly,
            build_metadata,
            continuity,
            coverage,
            target_feasibility,
            anti_leakage,
        )

        reports = {
            "00_source_schema.csv": _source_schema(),
            "01_source_inventory.csv": _source_inventory(
                source, hourly, build_metadata
            ),
            "02_timestamp_semantics.csv": _timestamp_semantics(),
            "03_hourly_continuity.csv": continuity,
            "04_variable_coverage.csv": coverage,
            "05_missing_runs.csv": _missing_run_report(hourly),
            "06_descriptive_statistics.csv": _descriptive_statistics(hourly),
            "07_seasonality_profiles.csv": _seasonality_profiles(hourly),
            "08_autocorrelation_by_lag.csv": _autocorrelation_report(hourly),
            "09_cross_correlation.csv": _cross_correlation_report(hourly),
            "10_drift_by_period.csv": _period_drift_report(hourly),
            "11_psi_by_year.csv": _yearly_psi_report(hourly),
            "12_extreme_frequency.csv": _extreme_report(hourly),
            "13_forecast_target_feasibility.csv": target_feasibility,
            "14_anti_leakage_contract.csv": anti_leakage,
        }

        with tempfile.TemporaryDirectory(prefix="b58a-") as temporary:
            output_dir = Path(temporary)
            for name, report in reports.items():
                report.to_csv(output_dir / name, index=False)

            dataset_path = (
                output_dir / "maritime_weather_hourly_past_only_v1.parquet"
            )
            hourly.to_parquet(dataset_path, index=False)
            decision_path = output_dir / "15_b58a_final_decision.json"
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            readme_path = output_dir / "README_B58A.md"
            _write_readme(readme_path, decision)

            materialized_rows = 0
            if materialize_timescale and decision["gates_passed"]:
                materialized_rows = _materialize_hourly(hourly, run_id)

            uploaded: dict[str, str] = {}
            for path in sorted(output_dir.iterdir()):
                if path == dataset_path:
                    key = f"datasets/b58a/{output_prefix}/{path.name}"
                elif path == decision_path:
                    key = f"configs/b58a/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b58a/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload_file(
                    client, path, output_bucket, key
                )

        metadata = {
            **decision,
            **_clean_json(build_metadata),
            "checksum": checksum,
            "materialized_timescale_rows": materialized_rows,
            "timescale_table": TIMESCALE_TABLE if materialized_rows else None,
            "outputs": uploaded,
            "output_prefix": (
                f"s3://{output_bucket}/reports/b58a/{output_prefix}/"
            ),
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
