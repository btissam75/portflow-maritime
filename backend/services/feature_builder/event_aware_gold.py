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


FEATURE_VERSION = "b57b-event-aware-daily-gold-v1"
SOURCE_NAME = "b57b_event_aware_gold"
DATASET_NAME = "tir_daily_event_aware_gold"
B57A_VERSION = "b57a-temporal-regime-event-audit-v1.2"
TIR_BUCKET = "bronze-maritime"
TIR_KEY = "tir/source/version=1/data1_maritime_minimal_v1.parquet"
TIR_CANONICAL_PREFIX = "tir/canonical/version=2/snapshots/"
ANALYSIS_START = pd.Timestamp("2020-01-01", tz="UTC")
OPERATIONAL_FORECAST_HORIZON_DAYS = 1

TIR_COLUMNS = (
    "SOURCE_ROW_INDEX",
    "UNITE",
    "DECLARANT",
    "VIDE_PLEIN",
    "NATURE_MARCHANDISE",
    "MATIERE_DANGER",
    "POIDS",
    "GROUPAGE",
    "DATE_ZRE",
    "DATE_EMBARQUEMENT",
)

TARGET_COLUMNS = (
    "target_tir_rows",
    "target_unique_units",
    "target_unique_declarants",
    "target_unique_cargo_types",
    "target_total_weight",
    "target_duration_labeled_rows",
    "target_duration_label_rate",
    "target_duration_mean_h",
    "target_duration_median_h",
    "target_duration_p90_h",
    "target_long_12h_rate",
    "target_long_24h_rate",
    "target_long_48h_rate",
    "target_full_rate",
    "target_empty_rate",
    "target_dangerous_rate",
    "target_groupage_rate",
)

IDENTIFIER_COLUMNS = {"prediction_date", "prediction_at", "feature_version"}
QUALITY_COLUMNS = {
    "tir_source_day_observed_flag",
    "tir_history_available_flag",
    "weather_history_available_flag",
    "weather_history_stale_flag",
    "weather_forecast_available_flag",
    "port_history_available_flag",
    "port_source_break_flag",
    "cold_start_28d_flag",
    "model_ready_flag",
}


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
    if value is pd.NA:
        return None
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


def _query_frame(query: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _latest_b57a_metadata() -> dict[str, Any]:
    frame = _query_frame(
        """
        SELECT metadata
        FROM audit.ingestion_run
        WHERE source_name='b57a_temporal_regime_audit'
          AND dataset_name='port_tir_weather_regime_panel'
          AND status='SUCCESS'
          AND metadata->>'audit_version'=%s
        ORDER BY finished_at DESC
        LIMIT 1
        """,
        (B57A_VERSION,),
    )
    if frame.empty:
        raise RuntimeError(f"Required successful B57A audit is missing: {B57A_VERSION}")
    metadata = dict(frame.iloc[0]["metadata"] or {})
    if metadata.get("status") != "READY_FOR_EVENT_AWARE_PRE_BREAK_FEATURES":
        raise RuntimeError(f"B57A does not authorize B57B: {metadata.get('status')}")
    return metadata



def _resolve_tir_key(client) -> str:
    objects = []
    token = None
    while True:
        request = {"Bucket": TIR_BUCKET, "Prefix": TIR_CANONICAL_PREFIX}
        if token:
            request["ContinuationToken"] = token
        page = client.list_objects_v2(**request)
        objects.extend(
            item
            for item in page.get("Contents", [])
            if str(item["Key"]).endswith("/tir_canonical.parquet")
        )
        if not page.get("IsTruncated"):
            break
        token = page["NextContinuationToken"]
    if objects:
        return str(max(objects, key=lambda item: str(item["Key"]))["Key"])
    return TIR_KEY

def _source_signature(client, b57a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tir_key = _resolve_tir_key(client)
    tir = client.head_object(Bucket=TIR_BUCKET, Key=tir_key)
    weather = _query_frame(
        """
        SELECT COUNT(*)::bigint AS rows, MIN(observed_at) AS min_time,
               MAX(observed_at) AS max_time, COUNT(wave_height_m)::bigint AS wave_rows
        FROM core.maritime_observation
        WHERE quality_flag=0
        """
    ).iloc[0].to_dict()
    port = _query_frame(
        """
        SELECT COUNT(*)::bigint AS rows, MIN(planned_eta) AS min_time,
               MAX(planned_eta) AS max_time, MAX(updated_at) AS max_updated_at
        FROM features.port_call_model_ready_v1
        """
    ).iloc[0].to_dict()
    events = _query_frame(
        """
        SELECT COUNT(*)::bigint AS rows, MIN(start_date) AS min_time,
               MAX(end_date) AS max_time, MAX(updated_at) AS max_updated_at
        FROM reference.business_event
        WHERE audit_version=%s
        """,
        (B57A_VERSION,),
    ).iloc[0].to_dict()
    evidence = {
        "feature_version": FEATURE_VERSION,
        "tir": {
            "key": tir_key,
            "etag": str(tir.get("ETag", "")).strip('"'),
            "size": int(tir["ContentLength"]),
            "last_modified": tir["LastModified"].isoformat(),
        },
        "weather": _clean_json(weather),
        "port": _clean_json(port),
        "events": _clean_json(events),
        "b57a_safe_periods": b57a.get("safe_periods", {}),
    }
    payload = json.dumps(evidence, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), evidence


def _start_run(checksum: str, evidence: dict[str, Any]) -> str:
    metadata = {
        "feature_version": FEATURE_VERSION,
        "source_signature": evidence,
        "grain": "ONE_ROW_PER_UTC_DAY",
        "prediction_at": "UTC_DAY_START",
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
                (
                    SOURCE_NAME,
                    DATASET_NAME,
                    f"s3://{TIR_BUCKET}/{TIR_KEY}",
                    checksum,
                    Json(metadata, dumps=lambda item: json.dumps(item, default=_json_default)),
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
                SET status=%s, row_count=%s, finished_at=now(),
                    metadata=metadata || %s, error_message=%s
                WHERE run_id=%s
                """,
                (
                    status,
                    row_count,
                    Json(_clean_json(metadata), dumps=lambda item: json.dumps(item, default=_json_default)),
                    error_message,
                    run_id,
                ),
            )


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    frame = _query_frame(
        """
        SELECT run_id, metadata
        FROM audit.ingestion_run
        WHERE source_name=%s AND dataset_name=%s AND checksum=%s AND status='SUCCESS'
        ORDER BY finished_at DESC LIMIT 1
        """,
        (SOURCE_NAME, DATASET_NAME, checksum),
    )
    if frame.empty:
        return None
    return str(frame.iloc[0]["run_id"]), dict(frame.iloc[0]["metadata"] or {})


def _load_tir_daily(client, tir_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        client.download_file(TIR_BUCKET, tir_key, handle.name)
        frame = pd.read_parquet(handle.name, columns=list(TIR_COLUMNS))
    raw_rows = len(frame)
    frame.columns = [column.lower() for column in frame.columns]
    frame["date_zre"] = pd.to_datetime(
        frame["date_zre"], errors="coerce", utc=True, format="mixed"
    )
    frame["date_embarquement"] = pd.to_datetime(
        frame["date_embarquement"], errors="coerce", utc=True, format="mixed"
    )
    frame = frame[frame["date_zre"].ge(ANALYSIS_START)].copy()
    frame["prediction_date"] = frame["date_zre"].dt.floor("D")
    duration = (
        frame["date_embarquement"] - frame["date_zre"]
    ).dt.total_seconds() / 3600.0
    frame["duration_h"] = duration.where(duration.between(0, 720))
    frame["poids"] = pd.to_numeric(frame["poids"], errors="coerce")

    vide_plein = frame["vide_plein"].astype("string").str.strip().str.upper()
    frame["full_flag"] = vide_plein.eq("PLEIN").astype("float32")
    frame["empty_flag"] = vide_plein.eq("VIDE").astype("float32")
    frame["dangerous_flag"] = frame["matiere_danger"].fillna(False).astype("float32")
    frame["groupage_flag"] = frame["groupage"].notna().astype("float32")
    frame["long_12h"] = frame["duration_h"].ge(12).where(frame["duration_h"].notna())
    frame["long_24h"] = frame["duration_h"].ge(24).where(frame["duration_h"].notna())
    frame["long_48h"] = frame["duration_h"].ge(48).where(frame["duration_h"].notna())

    grouped = frame.groupby("prediction_date", sort=True, observed=True)
    daily = grouped.agg(
        target_tir_rows=("source_row_index", "size"),
        target_unique_units=("unite", "nunique"),
        target_unique_declarants=("declarant", "nunique"),
        target_unique_cargo_types=("nature_marchandise", "nunique"),
        target_total_weight=("poids", "sum"),
        target_duration_labeled_rows=("duration_h", "count"),
        target_duration_mean_h=("duration_h", "mean"),
        target_duration_median_h=("duration_h", "median"),
        target_duration_p90_h=("duration_h", lambda values: values.quantile(0.90)),
        target_long_12h_rate=("long_12h", "mean"),
        target_long_24h_rate=("long_24h", "mean"),
        target_long_48h_rate=("long_48h", "mean"),
        target_full_rate=("full_flag", "mean"),
        target_empty_rate=("empty_flag", "mean"),
        target_dangerous_rate=("dangerous_flag", "mean"),
        target_groupage_rate=("groupage_flag", "mean"),
    ).reset_index()
    daily["target_duration_label_rate"] = (
        daily["target_duration_labeled_rows"] / daily["target_tir_rows"].replace(0, np.nan)
    )
    daily["tir_source_day_observed_flag"] = 1
    stats = {
        "raw_rows": raw_rows,
        "rows_from_2020": len(frame),
        "invalid_or_missing_zre": int(raw_rows - len(frame)),
        "valid_duration_rows": int(frame["duration_h"].notna().sum()),
        "first_day": daily["prediction_date"].min(),
        "last_day": daily["prediction_date"].max(),
    }
    return daily, stats


def _load_weather_daily() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT date_trunc('day', observed_at) AS prediction_date,
               COUNT(*)::bigint AS realized_weather_observation_count,
               COUNT(wave_height_m)::bigint AS realized_wave_observation_count,
               AVG(wave_height_m)::double precision AS realized_wave_height_mean_m,
               MAX(wave_height_m)::double precision AS realized_wave_height_max_m,
               AVG(wave_period_s)::double precision AS realized_wave_period_mean_s,
               SUM(CASE WHEN wave_height_m > 2 THEN 1 ELSE 0 END)::bigint
                   AS realized_storm_gt2_hours
        FROM core.maritime_observation
        WHERE quality_flag=0 AND observed_at >= %s
        GROUP BY 1 ORDER BY 1
        """,
        (ANALYSIS_START.to_pydatetime(),),
    )
    frame["prediction_date"] = pd.to_datetime(
        frame["prediction_date"], errors="coerce", utc=True
    )
    for column in frame.columns.difference(["prediction_date"]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_port_daily(safe_end: pd.Timestamp) -> pd.DataFrame:
    frame = _query_frame(
        """
        WITH planned AS (
            SELECT date_trunc('day', planned_eta) AS prediction_date,
                   COUNT(*)::bigint AS port_planned_calls
            FROM features.port_call_model_ready_v1
            WHERE planned_eta IS NOT NULL AND planned_eta < %s
            GROUP BY 1
        ), arrived AS (
            SELECT date_trunc('day', actual_ata) AS prediction_date,
                   COUNT(*)::bigint AS port_actual_arrivals,
                   AVG(arrival_delay_h)::double precision AS port_arrival_delay_mean_h
            FROM features.port_call_model_ready_v1
            WHERE actual_ata IS NOT NULL AND actual_ata < %s
            GROUP BY 1
        )
        SELECT COALESCE(p.prediction_date, a.prediction_date) AS prediction_date,
               p.port_planned_calls, a.port_actual_arrivals,
               a.port_arrival_delay_mean_h
        FROM planned p FULL OUTER JOIN arrived a USING (prediction_date)
        ORDER BY 1
        """,
        (
            (safe_end + pd.Timedelta(days=1)).to_pydatetime(),
            (safe_end + pd.Timedelta(days=1)).to_pydatetime(),
        ),
    )
    frame["prediction_date"] = pd.to_datetime(
        frame["prediction_date"], errors="coerce", utc=True
    )
    for column in frame.columns.difference(["prediction_date"]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_events() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT event_id, event_name, event_type, start_date, end_date,
               affected_flow, knowledge_policy, confidence
        FROM reference.business_event
        WHERE audit_version=%s
        ORDER BY start_date
        """,
        (B57A_VERSION,),
    )
    frame["start_date"] = pd.to_datetime(frame["start_date"], utc=True)
    frame["end_date"] = pd.to_datetime(frame["end_date"], utc=True)
    return frame


def _calendar_features(grid: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    result = grid.copy()
    date = result["prediction_date"]
    iso = date.dt.isocalendar()
    result["calendar_year"] = date.dt.year.astype("int16")
    result["calendar_month"] = date.dt.month.astype("int8")
    result["calendar_day_of_week"] = date.dt.dayofweek.astype("int8")
    result["calendar_iso_week"] = iso.week.astype("int16")
    result["calendar_weekend_flag"] = date.dt.dayofweek.ge(5).astype("int8")
    result["calendar_dow_sin"] = np.sin(2 * math.pi * date.dt.dayofweek / 7)
    result["calendar_dow_cos"] = np.cos(2 * math.pi * date.dt.dayofweek / 7)
    result["calendar_month_sin"] = np.sin(2 * math.pi * (date.dt.month - 1) / 12)
    result["calendar_month_cos"] = np.cos(2 * math.pi * (date.dt.month - 1) / 12)
    result["calendar_known_event_flag"] = 0
    result["calendar_aid_el_fitr_flag"] = 0
    result["calendar_aid_al_adha_flag"] = 0
    result["retrospective_bad_weather_reported_flag"] = 0
    for event in events.to_dict("records"):
        mask = date.between(event["start_date"], event["end_date"])
        if event["knowledge_policy"] == "KNOWN_CALENDAR_EVENT":
            result.loc[mask, "calendar_known_event_flag"] = 1
            event_id = str(event["event_id"])
            if "FITR" in event_id:
                result.loc[mask, "calendar_aid_el_fitr_flag"] = 1
            if "ADHA" in event_id:
                result.loc[mask, "calendar_aid_al_adha_flag"] = 1
        elif event["event_type"] == "WEATHER_DISRUPTION":
            result.loc[mask, "retrospective_bad_weather_reported_flag"] = 1
    return result


def _add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("prediction_date").reset_index(drop=True).copy()
    history_columns = {
        "target_tir_rows": "tir_rows",
        "target_duration_mean_h": "tir_duration_mean_h",
        "target_duration_median_h": "tir_duration_median_h",
        "target_long_24h_rate": "tir_long_24h_rate",
        "target_full_rate": "tir_full_rate",
        "target_total_weight": "tir_total_weight",
    }
    for source, prefix in history_columns.items():
        values = pd.to_numeric(result[source], errors="coerce")
        shifted = values.shift(1)
        result[f"hist_{prefix}_lag1d"] = shifted
        result[f"hist_{prefix}_lag7d"] = values.shift(7)
        result[f"hist_{prefix}_roll7d_mean"] = shifted.rolling(7, min_periods=1).mean()
        result[f"hist_{prefix}_roll28d_mean"] = shifted.rolling(28, min_periods=7).mean()
    result["tir_history_available_flag"] = (
        result["hist_tir_rows_lag1d"].notna().astype("int8")
    )
    result["cold_start_28d_flag"] = np.arange(len(result)).__lt__(28).astype("int8")
    return result


def _attach_weather_history(
    frame: pd.DataFrame, weather_daily: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    weather = frame[["prediction_date"]].merge(
        weather_daily, on="prediction_date", how="left", validate="one_to_one"
    )
    base_columns = [
        "realized_weather_observation_count",
        "realized_wave_observation_count",
        "realized_wave_height_mean_m",
        "realized_wave_height_max_m",
        "realized_wave_period_mean_s",
        "realized_storm_gt2_hours",
    ]
    for column in base_columns:
        values = pd.to_numeric(weather[column], errors="coerce")
        prefix = column[len("realized_") :]
        shifted = values.shift(1)
        weather[f"hist_{prefix}_lag1d"] = shifted
        weather[f"hist_{prefix}_roll3d_mean"] = shifted.rolling(3, min_periods=1).mean()
        weather[f"hist_{prefix}_roll7d_mean"] = shifted.rolling(7, min_periods=3).mean()
    history_columns = [column for column in weather if column.startswith("hist_")]
    merged = frame.merge(weather, on="prediction_date", how="left", validate="one_to_one")
    merged["weather_history_available_flag"] = (
        merged["hist_wave_observation_count_lag1d"].gt(0).astype("int8")
    )
    merged["weather_history_stale_flag"] = (
        merged["weather_history_available_flag"].eq(0).astype("int8")
    )
    merged["weather_forecast_available_flag"] = 0
    return merged, history_columns


def _attach_port_history(
    frame: pd.DataFrame,
    port_daily: pd.DataFrame,
    safe_end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    calendar = frame[["prediction_date"]].merge(
        port_daily, on="prediction_date", how="left", validate="one_to_one"
    )
    source_columns = [
        "port_planned_calls",
        "port_actual_arrivals",
        "port_arrival_delay_mean_h",
    ]
    for column in source_columns:
        values = pd.to_numeric(calendar[column], errors="coerce")
        shifted = values.shift(1)
        calendar[f"hist_{column}_lag1d"] = shifted
        calendar[f"hist_{column}_roll7d_mean"] = shifted.rolling(
            7, min_periods=3
        ).mean()
    history_columns = [column for column in calendar if column.startswith("hist_")]
    merged = frame.merge(
        calendar[["prediction_date", *history_columns]],
        on="prediction_date",
        how="left",
        validate="one_to_one",
    )
    merged["port_source_break_flag"] = (
        merged["prediction_date"].gt(safe_end).astype("int8")
    )
    merged["port_history_available_flag"] = (
        merged["hist_port_actual_arrivals_lag1d"].notna()
        & merged["port_source_break_flag"].eq(0)
    ).astype("int8")
    merged.loc[merged["port_source_break_flag"].eq(1), history_columns] = np.nan
    return merged, history_columns


def _build_datasets(
    tir_daily: pd.DataFrame,
    weather_daily: pd.DataFrame,
    port_daily: pd.DataFrame,
    events: pd.DataFrame,
    port_safe_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start = tir_daily["prediction_date"].min()
    observed_end = tir_daily["prediction_date"].max()
    end = observed_end + pd.Timedelta(days=OPERATIONAL_FORECAST_HORIZON_DAYS)
    grid = pd.DataFrame(
        {"prediction_date": pd.date_range(start, end, freq="D", tz="UTC")}
    )
    frame = grid.merge(tir_daily, on="prediction_date", how="left", validate="one_to_one")
    frame["tir_source_day_observed_flag"] = (
        frame["tir_source_day_observed_flag"].fillna(0).astype("int8")
    )
    historical_mask = (
        frame["prediction_date"].le(observed_end)
        & frame["tir_source_day_observed_flag"].eq(1)
    )
    for column in (
        "target_tir_rows",
        "target_unique_units",
        "target_unique_declarants",
        "target_unique_cargo_types",
        "target_total_weight",
        "target_duration_labeled_rows",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[historical_mask, column] = (
            frame.loc[historical_mask, column].fillna(0)
        )
        frame.loc[~historical_mask, column] = np.nan
    frame["prediction_at"] = frame["prediction_date"]
    frame["feature_version"] = FEATURE_VERSION
    frame = _calendar_features(frame, events)
    frame = _add_history_features(frame)
    frame, weather_history_columns = _attach_weather_history(frame, weather_daily)
    frame, port_history_columns = _attach_port_history(frame, port_daily, port_safe_end)

    frame["model_ready_flag"] = (
        frame["target_duration_labeled_rows"].gt(0)
        & frame["target_duration_label_rate"].ge(0.50)
        & frame["tir_history_available_flag"].eq(1)
        & frame["cold_start_28d_flag"].eq(0)
    ).astype("int8")

    realized_weather_columns = [
        column for column in frame if column.startswith("realized_")
    ]
    predictive = frame.drop(
        columns=[
            *realized_weather_columns,
            "retrospective_bad_weather_reported_flag",
        ]
    ).copy()
    explanatory = frame.copy()
    metadata = {
        "weather_history_columns": weather_history_columns,
        "port_history_columns": port_history_columns,
        "realized_weather_columns": realized_weather_columns,
    }
    return predictive, explanatory, metadata


def _column_role(column: str, predictive: bool) -> str:
    if column in IDENTIFIER_COLUMNS:
        return "IDENTIFIER_OR_TIME"
    if column in TARGET_COLUMNS:
        return "TARGET_CURRENT_DAY"
    if column in QUALITY_COLUMNS:
        return "QUALITY_FLAG"
    if column.startswith("calendar_"):
        return "KNOWN_CALENDAR_FEATURE"
    if column.startswith("hist_"):
        return "PAST_ONLY_FEATURE"
    if column.startswith("realized_"):
        return "EXPLANATORY_REALIZED_WEATHER"
    if column == "retrospective_bad_weather_reported_flag":
        return "EXPLANATORY_RETROSPECTIVE_EVENT"
    return "PREDICTIVE_FEATURE" if predictive else "EXPLANATORY_FIELD"


def _schema_report(frame: pd.DataFrame, dataset: str, predictive: bool) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        role = _column_role(column, predictive)
        rows.append(
            {
                "dataset": dataset,
                "column": column,
                "dtype": str(frame[column].dtype),
                "role": role,
                "missing_rows": int(frame[column].isna().sum()),
                "missing_pct": 100.0 * frame[column].isna().mean(),
                "n_unique": int(frame[column].nunique(dropna=True)),
                "allowed_in_predictive_x": role
                in {"KNOWN_CALENDAR_FEATURE", "PAST_ONLY_FEATURE", "PREDICTIVE_FEATURE", "QUALITY_FLAG"},
            }
        )
    return pd.DataFrame(rows)


def _leakage_report(
    predictive: pd.DataFrame,
    explanatory: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    predictive_columns = set(predictive.columns)
    checks = [
        {
            "check": "ONE_ROW_PER_DAY",
            "violations": int(predictive["prediction_date"].duplicated().sum()),
            "policy": "prediction_date must be unique",
        },
        {
            "check": "NO_REALIZED_WEATHER_IN_PREDICTIVE",
            "violations": len([column for column in predictive_columns if column.startswith("realized_")]),
            "policy": "current-day realized weather is explanatory only",
        },
        {
            "check": "NO_RETROSPECTIVE_EVENT_IN_PREDICTIVE",
            "violations": int("retrospective_bad_weather_reported_flag" in predictive_columns),
            "policy": "retrospective event labels are forbidden predictive features",
        },
        {
            "check": "TARGET_NAMESPACE_ISOLATED",
            "violations": len(
                [
                    column
                    for column in predictive_columns
                    if any(token in column for token in ("duration_h", "long_24h"))
                    and not column.startswith(("target_", "hist_"))
                ]
            ),
            "policy": "current-day outcomes are targets; prior outcomes require hist_ prefix",
        },
        {
            "check": "EXPLANATORY_DATASET_RETAINS_REALIZED_CONTEXT",
            "violations": int(
                not any(column.startswith("realized_") for column in explanatory.columns)
            ),
            "policy": "realized weather must remain available for retrospective study",
        },
        {
            "check": "NO_TRAINING_OR_SPLIT",
            "violations": 0,
            "policy": "B57B builds Gold only",
        },
    ]
    report = pd.DataFrame(checks)
    report["passed"] = report["violations"].eq(0)
    return report, int(report["violations"].sum())


def _event_coverage(events: pd.DataFrame, explanatory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in events.to_dict("records"):
        mask = explanatory["prediction_date"].between(
            event["start_date"], event["end_date"]
        )
        expected = int((event["end_date"] - event["start_date"]).days) + 1
        covered = int(mask.sum())
        rows.append(
            {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "knowledge_policy": event["knowledge_policy"],
                "expected_days": expected,
                "covered_days": covered,
                "coverage_pct": 100.0 * covered / expected,
                "target_labeled_days": int(
                    explanatory.loc[mask, "target_duration_mean_h"].notna().sum()
                ),
                "predictive_feature_policy": (
                    "ALLOWED_KNOWN_CALENDAR"
                    if event["knowledge_policy"] == "KNOWN_CALENDAR_EVENT"
                    else "EXPLANATORY_ONLY"
                ),
            }
        )
    return pd.DataFrame(rows)


def _materialize_daily(
    predictive: pd.DataFrame,
    predictive_feature_columns: list[str],
) -> int:
    target_columns = [column for column in TARGET_COLUMNS if column in predictive]
    quality_columns = [column for column in QUALITY_COLUMNS if column in predictive]
    rows = []
    for record in predictive.to_dict("records"):
        features = {
            column: _clean_json(record.get(column))
            for column in predictive_feature_columns
        }
        targets = {column: _clean_json(record.get(column)) for column in target_columns}
        quality = {column: _clean_json(record.get(column)) for column in quality_columns}
        rows.append(
            (
                record["prediction_date"].date(),
                record["prediction_at"].to_pydatetime(),
                FEATURE_VERSION,
                Json(features, dumps=lambda item: json.dumps(item, default=_json_default)),
                Json(targets, dumps=lambda item: json.dumps(item, default=_json_default)),
                Json(quality, dumps=lambda item: json.dumps(item, default=_json_default)),
            )
        )
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS features")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS features.tir_daily_event_aware_v1 (
                    prediction_date DATE PRIMARY KEY,
                    prediction_at TIMESTAMPTZ NOT NULL,
                    feature_version TEXT NOT NULL,
                    predictive_features JSONB NOT NULL,
                    targets JSONB NOT NULL,
                    quality_flags JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            execute_values(
                cursor,
                """
                INSERT INTO features.tir_daily_event_aware_v1
                    (prediction_date, prediction_at, feature_version,
                     predictive_features, targets, quality_flags)
                VALUES %s
                ON CONFLICT (prediction_date) DO UPDATE SET
                    prediction_at=EXCLUDED.prediction_at,
                    feature_version=EXCLUDED.feature_version,
                    predictive_features=EXCLUDED.predictive_features,
                    targets=EXCLUDED.targets,
                    quality_flags=EXCLUDED.quality_flags,
                    updated_at=now()
                """,
                rows,
                page_size=500,
            )
    return len(rows)


def _upload_file(client, path: Path, bucket: str, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".parquet": "application/octet-stream",
        ".md": "text/markdown",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
    return f"s3://{bucket}/{key}"


def _feature_registry(predictive: pd.DataFrame) -> dict[str, Any]:
    targets = [column for column in TARGET_COLUMNS if column in predictive]
    quality = [column for column in QUALITY_COLUMNS if column in predictive]
    identifiers = [column for column in predictive if column in IDENTIFIER_COLUMNS]
    features = [
        column
        for column in predictive
        if column not in set(targets) | set(quality) | set(identifiers)
    ]
    return {
        "feature_version": FEATURE_VERSION,
        "grain": "ONE_ROW_PER_UTC_DAY",
        "prediction_at": "UTC_DAY_START",
        "predictive_features": features,
        "targets": targets,
        "quality_flags": quality,
        "identifiers": identifiers,
        "forbidden_predictive_patterns": [
            "realized_*",
            "retrospective_*",
            "target_* as X",
        ],
        "weather_policy": (
            "PAST_OBSERVATIONS_ONLY; CURRENT_DAY FORECAST UNAVAILABLE; "
            "REALIZED CURRENT-DAY WEATHER EXPLANATORY_ONLY"
        ),
        "event_policy": (
            "MOVING_HOLIDAYS_ALLOWED_AS_KNOWN_CALENDAR; "
            "BAD_WEATHER_EVENT_LABEL_EXPLANATORY_ONLY"
        ),
    }


def build_b57b_event_aware_gold(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    materialize_timescale: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    b57a = _latest_b57a_metadata()
    checksum, source_signature = _source_signature(client, b57a)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "quality": previous[1],
        }

    run_id = _start_run(checksum, source_signature)
    try:
        port_safe_end = pd.Timestamp(
            b57a["safe_periods"]["PORT_CALLS"]["end"], tz="UTC"
        )
        tir_daily, tir_stats = _load_tir_daily(
            client, source_signature["tir"]["key"]
        )
        weather_daily = _load_weather_daily()
        port_daily = _load_port_daily(port_safe_end)
        events = _load_events()
        predictive, explanatory, build_metadata = _build_datasets(
            tir_daily, weather_daily, port_daily, events, port_safe_end
        )

        predictive_schema = _schema_report(predictive, "PREDICTIVE", True)
        explanatory_schema = _schema_report(explanatory, "EXPLANATORY", False)
        schema = pd.concat([predictive_schema, explanatory_schema], ignore_index=True)
        missingness = schema[
            ["dataset", "column", "role", "missing_rows", "missing_pct"]
        ].sort_values(["dataset", "missing_pct"], ascending=[True, False])
        leakage, leakage_violations = _leakage_report(predictive, explanatory)
        event_coverage = _event_coverage(events, explanatory)
        registry = _feature_registry(predictive)

        predictive_features = registry["predictive_features"]
        duplicate_days = int(predictive["prediction_date"].duplicated().sum())
        model_ready_rows = int(predictive["model_ready_flag"].sum())
        decision_status = (
            "READY_FOR_EVENT_AWARE_BASELINES"
            if len(predictive) >= 1_000
            and duplicate_days == 0
            and leakage_violations == 0
            and model_ready_rows >= 1_000
            else "NEED_EVENT_AWARE_DATA_REPAIR"
        )
        quality = {
            "status": decision_status,
            "feature_version": FEATURE_VERSION,
            "predictive_rows": len(predictive),
            "explanatory_rows": len(explanatory),
            "predictive_features": len(predictive_features),
            "targets": len(registry["targets"]),
            "duplicate_days": duplicate_days,
            "model_ready_rows": model_ready_rows,
            "model_ready_pct": 100.0 * model_ready_rows / len(predictive),
            "critical_leakage_violations": leakage_violations,
            "port_safe_end": port_safe_end,
            "tir_first_day": predictive["prediction_date"].min(),
            "tir_last_day": predictive["prediction_date"].max(),
            "weather_history_available_pct": 100.0
            * predictive["weather_history_available_flag"].mean(),
            "forecast_weather_available": False,
            "tir_stats": tir_stats,
            "training_executed": False,
            "split_created": False,
            "bronze_modified": False,
            "canonical_dataset": (
                f"s3://{output_bucket}/datasets/b57b/{output_prefix}/"
                "tir_daily_predictive_gold_v1.parquet"
            ),
            "explanatory_dataset": (
                f"s3://{output_bucket}/datasets/b57b/{output_prefix}/"
                "tir_daily_explanatory_gold_v1.parquet"
            ),
            "next_block": "B57C_EVENT_AWARE_TEMPORAL_BASELINES",
        }

        materialized_rows = 0
        if materialize_timescale:
            materialized_rows = _materialize_daily(predictive, predictive_features)
        quality["timescale_materialized_rows"] = materialized_rows
        quality["timescale_table"] = (
            "features.tir_daily_event_aware_v1" if materialized_rows else None
        )

        with tempfile.TemporaryDirectory(prefix="b57b-") as temporary:
            root = Path(temporary)
            predictive_path = root / "tir_daily_predictive_gold_v1.parquet"
            explanatory_path = root / "tir_daily_explanatory_gold_v1.parquet"
            predictive.to_parquet(predictive_path, index=False)
            explanatory.to_parquet(explanatory_path, index=False)
            schema.to_csv(root / "01_schema_and_roles.csv", index=False)
            missingness.to_csv(root / "02_missingness.csv", index=False)
            leakage.to_csv(root / "03_anti_leakage_audit.csv", index=False)
            event_coverage.to_csv(root / "04_event_coverage.csv", index=False)
            events.to_csv(root / "05_business_event_calendar.csv", index=False)
            (root / "b57b_feature_registry_v1.json").write_text(
                json.dumps(_clean_json(registry), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            (root / "b57b_build_decision_v1.json").write_text(
                json.dumps(_clean_json(quality), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            (root / "README_B57B.md").write_text(
                "\n".join(
                    [
                        "# B57B Event-aware Gold datasets",
                        "",
                        f"Decision: **{decision_status}**",
                        "",
                        "- Predictive Gold contains only information available at UTC day start.",
                        "- Explanatory Gold additionally contains realized current-day weather.",
                        "- The retrospective bad-weather label is never a predictive feature.",
                        "- No split was created and no model was trained.",
                    ]
                ),
                encoding="utf-8",
            )

            uploaded: dict[str, str] = {}
            for path in sorted(root.iterdir()):
                if path in {predictive_path, explanatory_path}:
                    key = f"datasets/b57b/{output_prefix}/{path.name}"
                elif path.suffix == ".json" and "registry" in path.name:
                    key = f"configs/b57b/{output_prefix}/{path.name}"
                elif path.suffix == ".json" and "decision" in path.name:
                    key = f"configs/b57b/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b57b/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload_file(client, path, output_bucket, key)

        metadata = {
            **quality,
            "checksum": checksum,
            "source_signature": source_signature,
            "build_metadata": build_metadata,
            "outputs": uploaded,
        }
        _finish_run(run_id, "SUCCESS", len(predictive), metadata)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "quality": metadata,
            "outputs": uploaded,
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {"feature_version": FEATURE_VERSION},
            error_message=str(exc),
        )
        raise
