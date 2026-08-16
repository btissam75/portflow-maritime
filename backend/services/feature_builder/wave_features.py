from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values


SOURCE_NAME = "tir_port_calls_model_ready"
DATASET_NAME = "arrival_multi_horizon_wave_features"
FEATURE_VERSION = "b54c-wave-history-v1"
SOURCE_VIEW = "features.port_call_model_ready_v1"
DEFAULT_HORIZONS = (24, 12, 6, 3)
MAX_SEA_AGE_H = float(os.getenv("B54C_MAX_SEA_AGE_H", "2"))
SEA_SOURCE = os.getenv("B54C_SEA_SOURCE", "").strip()

REQUIRED_CALL_COLUMNS = {
    "port_call_id",
    "planned_eta",
    "actual_ata",
    "arrival_delay_h",
    "imo",
}

MODEL_EXCLUDED_COLUMNS = {
    "actual_ata",
    "actual_atd",
    "arrival_delay_h",
    "departure_delay_h",
    "has_arrival_label",
    "has_departure_label",
    "target_arrival_delay_h",
    "target_departure_delay_h",
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


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _qualified_view_exists(cursor) -> bool:
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.views
            WHERE table_schema = 'features'
              AND table_name = 'port_call_model_ready_v1'
        )
        """
    )
    return bool(cursor.fetchone()[0])


def _select_sea_source(cursor) -> str:
    if SEA_SOURCE:
        cursor.execute(
            "SELECT count(*) FROM core.maritime_observation WHERE source = %s",
            (SEA_SOURCE,),
        )
        if int(cursor.fetchone()[0]) == 0:
            raise RuntimeError(
                f"B54C_SEA_SOURCE={SEA_SOURCE!r} has no maritime observations"
            )
        return SEA_SOURCE
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
    return str(row[0])


def _database_signature() -> tuple[str, str, dict[str, Any]]:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            if not _qualified_view_exists(cursor):
                raise RuntimeError(
                    f"Required view {SOURCE_VIEW} does not exist. "
                    "Create the B54B model-ready views first."
                )
            sea_source = _select_sea_source(cursor)
            cursor.execute(
                f"""
                SELECT
                    count(*) AS calls,
                    count(arrival_delay_h) AS arrival_labels,
                    min(planned_eta),
                    max(planned_eta),
                    min(arrival_delay_h),
                    max(arrival_delay_h)
                FROM {SOURCE_VIEW}
                """
            )
            call_stats = cursor.fetchone()
            cursor.execute(
                """
                SELECT count(*), min(observed_at), max(observed_at),
                       count(wave_height_m)
                FROM core.maritime_observation
                WHERE source = %s
                """,
                (sea_source,),
            )
            sea_stats = cursor.fetchone()

    metadata = {
        "source_view": SOURCE_VIEW,
        "sea_source": sea_source,
        "call_stats": list(call_stats),
        "sea_stats": list(sea_stats),
        "feature_version": FEATURE_VERSION,
    }
    serialized = json.dumps(metadata, sort_keys=True, default=_json_default)
    checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return checksum, sea_source, metadata


def _start_run(checksum: str, metadata: dict[str, Any]) -> str:
    with db_connection() as connection:
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
                    f"postgresql://maritime/{SOURCE_VIEW}",
                    checksum,
                    Json(metadata, dumps=lambda obj: json.dumps(obj, default=_json_default)),
                ),
            )
            return str(cursor.fetchone()[0])


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, metadata
                FROM audit.ingestion_run
                WHERE source_name = %s
                  AND dataset_name = %s
                  AND checksum = %s
                  AND status = 'SUCCESS'
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    if row is None:
        return None
    return str(row[0]), dict(row[1] or {})


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at = now(),
                    status = %s,
                    row_count = %s,
                    metadata = metadata || %s,
                    error_message = %s
                WHERE run_id = %s
                """,
                (
                    status,
                    row_count,
                    Json(
                        metadata or {},
                        dumps=lambda obj: json.dumps(obj, default=_json_default),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _load_frames(sea_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    call_query = f"SELECT * FROM {SOURCE_VIEW} ORDER BY planned_eta"
    sea_query = """
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
    """
    with db_connection() as connection:
        calls = pd.read_sql_query(call_query, connection)
        sea = pd.read_sql_query(sea_query, connection, params=(sea_source,))

    missing = sorted(REQUIRED_CALL_COLUMNS.difference(calls.columns))
    if missing:
        raise RuntimeError(f"{SOURCE_VIEW} is missing required columns: {missing}")
    if calls.empty:
        raise RuntimeError(f"{SOURCE_VIEW} is empty")
    if sea.empty:
        raise RuntimeError(f"No sea observations found for source {sea_source}")

    for column in ("planned_eta", "planned_etd", "actual_ata", "actual_atd"):
        if column in calls.columns:
            calls[column] = pd.to_datetime(calls[column], errors="coerce", utc=True)
    calls["port_call_id"] = calls["port_call_id"].astype("string")
    for column in ("arrival_delay_h", "departure_delay_h"):
        if column in calls.columns:
            calls[column] = pd.to_numeric(calls[column], errors="coerce")
    sea["observed_at"] = pd.to_datetime(
        sea["observed_at"], errors="coerce", utc=True
    )
    calls = calls.loc[calls["planned_eta"].notna()].copy()
    sea = sea.loc[sea["observed_at"].notna()].copy()
    return calls, sea


def _rolling_std(series: pd.Series, window: str) -> pd.Series:
    return series.rolling(window, min_periods=2).std().fillna(0.0)


def _prepare_wave_history(sea: pd.DataFrame) -> pd.DataFrame:
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
    for column in numeric_columns:
        sea[column] = pd.to_numeric(sea[column], errors="coerce")

    sea = (
        sea.sort_values(["observed_at", "sea_quality_flag"])
        .drop_duplicates("observed_at", keep="first")
        .reset_index(drop=True)
    )
    direction_rad = np.deg2rad(sea["wave_direction_deg"] % 360.0)
    sea["wave_direction_sin"] = np.sin(direction_rad)
    sea["wave_direction_cos"] = np.cos(direction_rad)
    sea["wave_energy_proxy"] = (
        sea["wave_height_m"].pow(2) * sea["wave_period_s"]
    )
    sea["high_wave_ge_1p5m"] = (sea["wave_height_m"] >= 1.5).astype("int8")
    sea["high_wave_ge_2p5m"] = (sea["wave_height_m"] >= 2.5).astype("int8")
    sea["severe_wave_ge_4m"] = (sea["wave_height_m"] >= 4.0).astype("int8")

    indexed = sea.set_index("observed_at")
    for hours in (3, 6, 12, 24, 72):
        window = f"{hours}h"
        sea[f"wave_height_mean_{hours}h"] = (
            indexed["wave_height_m"].rolling(window, min_periods=1).mean().to_numpy()
        )
        sea[f"wave_height_max_{hours}h"] = (
            indexed["wave_height_m"].rolling(window, min_periods=1).max().to_numpy()
        )
        sea[f"wave_height_std_{hours}h"] = _rolling_std(
            indexed["wave_height_m"], window
        ).to_numpy()
        sea[f"wave_period_mean_{hours}h"] = (
            indexed["wave_period_s"].rolling(window, min_periods=1).mean().to_numpy()
        )
        sea[f"wave_energy_mean_{hours}h"] = (
            indexed["wave_energy_proxy"].rolling(window, min_periods=1).mean().to_numpy()
        )
        sea[f"high_wave_hours_ge_1p5m_{hours}h"] = (
            indexed["high_wave_ge_1p5m"].rolling(window, min_periods=1).sum().to_numpy()
        )
        sea[f"high_wave_hours_ge_2p5m_{hours}h"] = (
            indexed["high_wave_ge_2p5m"].rolling(window, min_periods=1).sum().to_numpy()
        )
        sin_mean = (
            indexed["wave_direction_sin"].rolling(window, min_periods=1).mean()
        )
        cos_mean = (
            indexed["wave_direction_cos"].rolling(window, min_periods=1).mean()
        )
        sea[f"wave_direction_concentration_{hours}h"] = np.sqrt(
            sin_mean.to_numpy() ** 2 + cos_mean.to_numpy() ** 2
        )

    index = pd.DatetimeIndex(sea["observed_at"])
    height = pd.Series(sea["wave_height_m"].to_numpy(), index=index)
    for hours in (3, 6, 12, 24):
        target_index = index - pd.Timedelta(hours=hours)
        lagged = height.reindex(
            target_index,
            method="pad",
            tolerance=pd.Timedelta(hours=2),
        ).to_numpy()
        sea[f"wave_height_lag_{hours}h"] = lagged
        sea[f"wave_height_trend_{hours}h"] = sea["wave_height_m"] - lagged

    return sea.sort_values("observed_at").reset_index(drop=True)


def _build_snapshots(calls: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    horizon_frame = pd.DataFrame({"horizon_h": list(horizons)})
    calls = calls.copy()
    calls["_cross_key"] = 1
    horizon_frame["_cross_key"] = 1
    snapshots = calls.merge(horizon_frame, on="_cross_key", how="inner").drop(
        columns="_cross_key"
    )
    snapshots["prediction_time"] = snapshots["planned_eta"] - pd.to_timedelta(
        snapshots["horizon_h"], unit="h"
    )
    snapshots["snapshot_before_actual_flag"] = (
        snapshots["actual_ata"].isna()
        | (snapshots["prediction_time"] < snapshots["actual_ata"])
    )
    prediction_time = snapshots["prediction_time"]
    snapshots["prediction_hour"] = prediction_time.dt.hour.astype("int8")
    snapshots["prediction_dayofweek"] = prediction_time.dt.dayofweek.astype("int8")
    snapshots["prediction_month"] = prediction_time.dt.month.astype("int8")
    snapshots["prediction_weekend_flag"] = (
        prediction_time.dt.dayofweek >= 5
    ).astype("int8")
    snapshots["prediction_hour_sin"] = np.sin(
        2.0 * math.pi * prediction_time.dt.hour / 24.0
    )
    snapshots["prediction_hour_cos"] = np.cos(
        2.0 * math.pi * prediction_time.dt.hour / 24.0
    )
    snapshots["prediction_dow_sin"] = np.sin(
        2.0 * math.pi * prediction_time.dt.dayofweek / 7.0
    )
    snapshots["prediction_dow_cos"] = np.cos(
        2.0 * math.pi * prediction_time.dt.dayofweek / 7.0
    )
    snapshots["prediction_month_sin"] = np.sin(
        2.0 * math.pi * (prediction_time.dt.month - 1) / 12.0
    )
    snapshots["prediction_month_cos"] = np.cos(
        2.0 * math.pi * (prediction_time.dt.month - 1) / 12.0
    )
    return snapshots


def _attach_wave_history(
    snapshots: pd.DataFrame, wave: pd.DataFrame
) -> pd.DataFrame:
    snapshots = snapshots.sort_values("prediction_time").reset_index(drop=True)
    wave = wave.sort_values("observed_at").reset_index(drop=True)
    result = pd.merge_asof(
        snapshots,
        wave,
        left_on="prediction_time",
        right_on="observed_at",
        direction="backward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta(hours=max(3.0, MAX_SEA_AGE_H + 1.0)),
    )
    result["sea_observation_age_h"] = (
        result["prediction_time"] - result["observed_at"]
    ).dt.total_seconds() / 3600.0
    result["sea_feature_available_flag"] = (
        result["observed_at"].notna()
        & result["wave_height_m"].notna()
        & result["sea_observation_age_h"].between(0.0, MAX_SEA_AGE_H)
        & (result["sea_quality_flag"].fillna(999) == 0)
    )
    result["sea_feature_stale_flag"] = (
        result["observed_at"].notna()
        & (result["sea_observation_age_h"] > MAX_SEA_AGE_H)
    )
    leakage = result["observed_at"].notna() & (
        result["observed_at"] > result["prediction_time"]
    )
    if leakage.any():
        raise RuntimeError(
            f"Temporal leakage detected in {int(leakage.sum())} sea joins"
        )
    return result


def _history_timeline(
    calls: pd.DataFrame, group_column: str | None
) -> pd.DataFrame:
    columns = ["actual_ata", "arrival_delay_h"]
    if group_column:
        columns.append(group_column)
    history = calls[columns].copy()
    history["arrival_delay_h"] = pd.to_numeric(
        history["arrival_delay_h"], errors="coerce"
    )
    history = history.dropna(subset=["actual_ata", "arrival_delay_h"])
    if group_column:
        history = history.loc[history[group_column].notna()].copy()
        history[group_column] = history[group_column].astype("string")
        group_keys = [group_column, "actual_ata"]
    else:
        group_keys = ["actual_ata"]
    history["delay_sq"] = history["arrival_delay_h"].pow(2)
    history["late_gt_1h"] = (history["arrival_delay_h"] > 1).astype("int64")
    history["late_gt_6h"] = (history["arrival_delay_h"] > 6).astype("int64")
    aggregated = (
        history.groupby(group_keys, dropna=False)
        .agg(
            event_count=("arrival_delay_h", "size"),
            event_sum=("arrival_delay_h", "sum"),
            event_sumsq=("delay_sq", "sum"),
            event_late_gt_1h=("late_gt_1h", "sum"),
            event_late_gt_6h=("late_gt_6h", "sum"),
        )
        .reset_index()
    )
    if group_column:
        aggregated = aggregated.sort_values(["actual_ata", group_column])
        grouped = aggregated.groupby(group_column, sort=False)
        for column in (
            "event_count",
            "event_sum",
            "event_sumsq",
            "event_late_gt_1h",
            "event_late_gt_6h",
        ):
            aggregated[f"cum_{column}"] = grouped[column].cumsum()
    else:
        aggregated = aggregated.sort_values("actual_ata")
        for column in (
            "event_count",
            "event_sum",
            "event_sumsq",
            "event_late_gt_1h",
            "event_late_gt_6h",
        ):
            aggregated[f"cum_{column}"] = aggregated[column].cumsum()

    count = aggregated["cum_event_count"].astype(float)
    mean = aggregated["cum_event_sum"] / count
    variance = (aggregated["cum_event_sumsq"] / count) - mean.pow(2)
    aggregated["hist_count"] = count
    aggregated["hist_mean_delay_h"] = mean
    aggregated["hist_std_delay_h"] = np.sqrt(variance.clip(lower=0.0))
    aggregated["hist_late_gt_1h_rate"] = aggregated["cum_event_late_gt_1h"] / count
    aggregated["hist_late_gt_6h_rate"] = aggregated["cum_event_late_gt_6h"] / count
    keep = [
        "actual_ata",
        "hist_count",
        "hist_mean_delay_h",
        "hist_std_delay_h",
        "hist_late_gt_1h_rate",
        "hist_late_gt_6h_rate",
    ]
    if group_column:
        keep.insert(0, group_column)
    return aggregated[keep]


def _attach_safe_delay_history(
    snapshots: pd.DataFrame, calls: pd.DataFrame
) -> pd.DataFrame:
    result = snapshots.copy()
    result["imo_history_key"] = pd.to_numeric(
        result["imo"], errors="coerce"
    ).astype("Int64").astype("string")
    calls_for_history = calls.copy()
    calls_for_history["imo_history_key"] = pd.to_numeric(
        calls_for_history["imo"], errors="coerce"
    ).astype("Int64").astype("string")

    vessel = _history_timeline(calls_for_history, "imo_history_key")
    vessel = vessel.rename(
        columns={
            "actual_ata": "vessel_history_event_time",
            "hist_count": "vessel_hist_count",
            "hist_mean_delay_h": "vessel_hist_mean_delay_h",
            "hist_std_delay_h": "vessel_hist_std_delay_h",
            "hist_late_gt_1h_rate": "vessel_hist_late_gt_1h_rate",
            "hist_late_gt_6h_rate": "vessel_hist_late_gt_6h_rate",
        }
    )
    result = pd.merge_asof(
        result.sort_values(["prediction_time", "imo_history_key"]),
        vessel.sort_values(["vessel_history_event_time", "imo_history_key"]),
        left_on="prediction_time",
        right_on="vessel_history_event_time",
        by="imo_history_key",
        direction="backward",
        allow_exact_matches=False,
    )

    global_timeline = _history_timeline(calls_for_history, None).rename(
        columns={
            "actual_ata": "global_history_event_time",
            "hist_count": "global_hist_count",
            "hist_mean_delay_h": "global_hist_mean_delay_h",
            "hist_std_delay_h": "global_hist_std_delay_h",
            "hist_late_gt_1h_rate": "global_hist_late_gt_1h_rate",
            "hist_late_gt_6h_rate": "global_hist_late_gt_6h_rate",
        }
    )
    result = pd.merge_asof(
        result.sort_values("prediction_time"),
        global_timeline.sort_values("global_history_event_time"),
        left_on="prediction_time",
        right_on="global_history_event_time",
        direction="backward",
        allow_exact_matches=False,
    )
    for event_column in (
        "vessel_history_event_time",
        "global_history_event_time",
    ):
        violation = result[event_column].notna() & (
            result[event_column] >= result["prediction_time"]
        )
        if violation.any():
            raise RuntimeError(
                f"Temporal leakage detected in {event_column}: "
                f"{int(violation.sum())} rows"
            )
    return result.drop(columns="imo_history_key")


def _model_feature_columns(frame: pd.DataFrame) -> list[str]:
    identifiers = {
        "port_call_id",
        "source_record_id",
        "voyage_id",
        "observed_at",
        "prediction_time",
        "planned_eta",
        "planned_etd",
        "vessel_history_event_time",
        "global_history_event_time",
    }
    audit_only = {
        "snapshot_before_actual_flag",
        "sea_feature_available_flag",
        "sea_feature_stale_flag",
        "model_ready_flag",
    }
    excluded = MODEL_EXCLUDED_COLUMNS | identifiers | audit_only

    def is_safe(column: str) -> bool:
        if column in excluded:
            return False
        lowered = column.lower()
        if lowered.startswith(("actual_", "target_", "has_arrival", "has_departure")):
            return False
        if any(
            token in lowered
            for token in (
                "outlier",
                "quarantine",
                "quality_reason",
                "invalid_sequence",
                "canonicalization",
                "value_spread",
                "multiple_eta",
                "multiple_etd",
                "multiple_rta",
                "multiple_rtd",
            )
        ):
            return False
        if "delay" in lowered and not lowered.startswith(
            ("vessel_hist_", "global_hist_")
        ):
            return False
        if "label" in lowered:
            return False
        return True

    return [column for column in frame.columns if is_safe(column)]


def _prepare_outputs(
    calls: pd.DataFrame,
    sea: pd.DataFrame,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    wave = _prepare_wave_history(sea)
    snapshots = _build_snapshots(calls, horizons)
    snapshots = _attach_wave_history(snapshots, wave)
    snapshots = _attach_safe_delay_history(snapshots, calls)
    snapshots["target_arrival_delay_h"] = pd.to_numeric(
        snapshots["arrival_delay_h"], errors="coerce"
    )
    if "departure_delay_h" in snapshots.columns:
        snapshots["target_departure_delay_h"] = pd.to_numeric(
            snapshots["departure_delay_h"], errors="coerce"
        )
    snapshots["model_ready_flag"] = (
        snapshots["target_arrival_delay_h"].notna()
        & snapshots["snapshot_before_actual_flag"]
        & snapshots["sea_feature_available_flag"]
    )
    model_ready_audit = snapshots.loc[snapshots["model_ready_flag"]].copy()
    excluded = snapshots.loc[~snapshots["model_ready_flag"]].copy()

    leakage_count = int(
        (
            model_ready_audit["observed_at"].notna()
            & (model_ready_audit["observed_at"] > model_ready_audit["prediction_time"])
        ).sum()
    )
    horizon_report = []
    for horizon in horizons:
        group = snapshots.loc[snapshots["horizon_h"] == horizon]
        ready = group.loc[group["model_ready_flag"]]
        horizon_report.append(
            {
                "horizon_h": int(horizon),
                "snapshots": int(len(group)),
                "model_ready_snapshots": int(len(ready)),
                "model_ready_pct": 100.0 * len(ready) / max(1, len(group)),
                "sea_available_pct": 100.0
                * float(group["sea_feature_available_flag"].mean()),
                "post_arrival_snapshots_excluded": int(
                    (~group["snapshot_before_actual_flag"]).sum()
                ),
                "target_mean_h": float(ready["target_arrival_delay_h"].mean())
                if len(ready)
                else None,
                "target_p95_h": float(
                    ready["target_arrival_delay_h"].quantile(0.95)
                )
                if len(ready)
                else None,
            }
        )

    feature_columns = _model_feature_columns(model_ready_audit)
    metadata_columns = [
        column
        for column in (
            "port_call_id",
            "source_record_id",
            "prediction_time",
            "planned_eta",
            "horizon_h",
            "model_ready_flag",
        )
        if column in model_ready_audit.columns
    ]
    target_columns = [
        column
        for column in (
            "target_arrival_delay_h",
            "target_departure_delay_h",
        )
        if column in model_ready_audit.columns
    ]
    strict_columns = list(dict.fromkeys(metadata_columns + feature_columns + target_columns))
    model_ready = model_ready_audit[strict_columns].copy()
    coverage_pct = 100.0 * len(model_ready_audit) / max(1, len(snapshots))
    report = {
        "feature_version": FEATURE_VERSION,
        "source_view": SOURCE_VIEW,
        "calls_loaded": int(len(calls)),
        "wave_rows_loaded": int(len(sea)),
        "wave_start": sea["observed_at"].min(),
        "wave_end": sea["observed_at"].max(),
        "horizons_h": list(horizons),
        "snapshots_total": int(len(snapshots)),
        "snapshots_model_ready": int(len(model_ready_audit)),
        "snapshots_excluded": int(len(excluded)),
        "model_ready_pct": coverage_pct,
        "temporal_leakage_violations": leakage_count,
        "max_sea_age_h": MAX_SEA_AGE_H,
        "horizon_report": horizon_report,
        "model_feature_count": len(feature_columns),
        "model_feature_columns": feature_columns,
        "target_columns": target_columns,
        "model_ready_physical_column_count": len(strict_columns),
        "model_ready_contains_actual_timestamps": False,
        "anti_leakage_policy": (
            "Sea observations and completed historical calls must be strictly "
            "available at prediction_time. Snapshots after actual arrival are excluded."
        ),
        "split_policy": "NO_SPLIT_IN_B54C",
        "decision": {
            "ready_for_modeling": leakage_count == 0 and coverage_pct >= 95.0,
            "next_block": "B54D_TEMPORAL_SPLIT_AND_BASELINES",
        },
        "generated_at_utc": datetime.now(timezone.utc),
    }
    return snapshots, model_ready, excluded, report


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def _upload(client, path: Path, bucket: str, key: str, content_type: str) -> str:
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{bucket}/{key}"


def _payload_value(value: Any):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _upsert_feature_store(
    frame: pd.DataFrame, feature_columns: list[str]
) -> int:
    if frame.empty:
        return 0
    payload_columns = [
        column
        for column in feature_columns
        if column not in {"wave_height_m"}
    ]
    rows = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        payload = {
            column: _payload_value(values.get(column))
            for column in payload_columns
        }
        rows.append(
            (
                values["prediction_time"],
                str(values["port_call_id"]),
                _payload_value(values.get("wave_height_m")),
                FEATURE_VERSION,
                Json(payload, dumps=lambda obj: json.dumps(obj, default=_json_default)),
            )
        )
    query = """
        INSERT INTO features.vessel_snapshot (
            snapshot_at,
            port_call_id,
            wave_height_now_m,
            feature_version,
            feature_payload
        ) VALUES %s
        ON CONFLICT (snapshot_at, port_call_id, feature_version)
        DO UPDATE SET
            wave_height_now_m = EXCLUDED.wave_height_now_m,
            feature_payload = EXCLUDED.feature_payload
    """
    with db_connection() as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, query, rows, page_size=2000)
    return len(rows)


def process_multi_horizon_wave_features(
    output_bucket: str = "silver-maritime",
    output_prefix: str = "version=1",
    horizons_h: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    horizons = tuple(
        sorted(
            {int(item) for item in (horizons_h or DEFAULT_HORIZONS)},
            reverse=True,
        )
    )
    if not horizons or any(item <= 0 or item > 168 for item in horizons):
        raise RuntimeError("horizons_h must contain unique integers in [1, 168]")

    checksum, sea_source, signature_metadata = _database_signature()
    signature_metadata["horizons_h"] = list(horizons)
    signature_metadata["max_sea_age_h"] = MAX_SEA_AGE_H
    checksum = hashlib.sha256(
        json.dumps(
            signature_metadata, sort_keys=True, default=_json_default
        ).encode("utf-8")
    ).hexdigest()

    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            previous_run_id, metadata = previous
            return {
                "status": "SKIPPED_ALREADY_PROCESSED",
                "run_id": previous_run_id,
                "checksum": checksum,
                "quality": metadata,
                "outputs": metadata.get("output_uris", {}),
            }

    run_id = _start_run(checksum, signature_metadata)
    client = s3_client()
    try:
        calls, sea = _load_frames(sea_source)
        snapshots, model_ready, excluded, report = _prepare_outputs(
            calls, sea, horizons
        )
        with tempfile.TemporaryDirectory(prefix="b54c-") as temporary:
            work_dir = Path(temporary)
            all_path = work_dir / "arrival_multi_horizon_wave_features_v1.parquet"
            ready_path = (
                work_dir / "arrival_multi_horizon_wave_model_ready_v1.parquet"
            )
            excluded_path = work_dir / "arrival_wave_snapshot_exclusions_v1.parquet"
            report_path = work_dir / "b54c_wave_feature_report_v1.json"
            _write_parquet(snapshots, all_path)
            _write_parquet(model_ready, ready_path)
            _write_parquet(excluded, excluded_path)

            prefix = output_prefix.strip("/") or "version=1"
            keys = {
                "all_snapshots": (
                    f"wave_features/{prefix}/"
                    "arrival_multi_horizon_wave_features_v1.parquet"
                ),
                "model_ready": (
                    f"wave_features_model_ready/{prefix}/"
                    "arrival_multi_horizon_wave_model_ready_v1.parquet"
                ),
                "excluded": (
                    f"quarantine/{prefix}/"
                    "arrival_wave_snapshot_exclusions_v1.parquet"
                ),
                "report": f"audits/{prefix}/b54c_wave_feature_report_v1.json",
            }
            output_uris = {
                "all_snapshots": _upload(
                    client,
                    all_path,
                    output_bucket,
                    keys["all_snapshots"],
                    "application/x-parquet",
                ),
                "model_ready": _upload(
                    client,
                    ready_path,
                    output_bucket,
                    keys["model_ready"],
                    "application/x-parquet",
                ),
                "excluded": _upload(
                    client,
                    excluded_path,
                    output_bucket,
                    keys["excluded"],
                    "application/x-parquet",
                ),
            }
            feature_store_rows = _upsert_feature_store(
                model_ready, report["model_feature_columns"]
            )
            output_uris["report"] = f"s3://{output_bucket}/{keys['report']}"
            report.update(
                {
                    "run_id": run_id,
                    "checksum": checksum,
                    "sea_source": sea_source,
                    "feature_store_rows_upserted": feature_store_rows,
                    "output_uris": output_uris,
                }
            )
            report_path.write_text(
                json.dumps(report, indent=2, default=_json_default),
                encoding="utf-8",
            )
            _upload(
                client,
                report_path,
                output_bucket,
                keys["report"],
                "application/json",
            )

        _finish_run(
            run_id,
            "SUCCESS",
            row_count=int(report["snapshots_model_ready"]),
            metadata=report,
        )
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "checksum": checksum,
            "quality": report,
            "outputs": output_uris,
        }
    except Exception as exc:
        _finish_run(run_id, "FAILED", error_message=str(exc)[:4000])
        raise
