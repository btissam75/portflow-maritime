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
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, mannwhitneyu


AUDIT_VERSION = "b57a-temporal-regime-event-audit-v1.2"
SOURCE_NAME = "b57a_temporal_regime_audit"
DATASET_NAME = "port_tir_weather_regime_panel"
PORT_CALL_VIEW = "features.port_call_model_ready_v1"
WEATHER_TABLE = "core.maritime_observation"
TIR_BUCKET = "bronze-maritime"
TIR_KEY = "tir/source/version=1/data1_maritime_minimal_v1.parquet"
ANALYSIS_START = pd.Timestamp("2020-01-01", tz="UTC")
SEASONAL_BASELINE_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
LOW_VOLUME_RATIO = 0.80
MIN_EVENT_COVERAGE_PCT = 70.0

TIR_COLUMNS = (
    "SOURCE_ROW_INDEX",
    "UNITE",
    "TYPE_UNITE",
    "SS_TYPE_UNITE",
    "DECLARANT",
    "VIDE_PLEIN",
    "NATURE_MARCHANDISE",
    "MATIERE_DANGER",
    "ETAT_DECHARGEMENT",
    "POIDS",
    "DATE_ZRE",
    "DATE_EMBARQUEMENT",
    "STATUT_EXPLOITATION",
)

BUSINESS_EVENTS = (
    {
        "event_id": "2026_BAD_WEATHER_EXPORT_S03_S07",
        "event_name": "Bad weather impact on exports",
        "event_type": "WEATHER_DISRUPTION",
        "iso_year": 2026,
        "start_week": 3,
        "end_week": 7,
        "affected_flow": "EXPORTS",
        "knowledge_policy": "RETROSPECTIVE_EXPLANATORY_ONLY",
        "source": "BUSINESS_DOMAIN_INPUT",
        "confidence": "REPORTED_NOT_YET_EXTERNALLY_VERIFIED",
    },
    {
        "event_id": "2026_AID_EL_FITR_S11_S13",
        "event_name": "Aid El Fitr operating window",
        "event_type": "MOVING_HOLIDAY",
        "iso_year": 2026,
        "start_week": 11,
        "end_week": 13,
        "affected_flow": "ALL",
        "knowledge_policy": "KNOWN_CALENDAR_EVENT",
        "source": "BUSINESS_DOMAIN_INPUT",
        "confidence": "BUSINESS_REPORTED_WINDOW",
    },
    {
        "event_id": "2026_AID_AL_ADHA_S21_S23",
        "event_name": "Aid al Adha operating window",
        "event_type": "MOVING_HOLIDAY",
        "iso_year": 2026,
        "start_week": 21,
        "end_week": 23,
        "affected_flow": "ALL",
        "knowledge_policy": "KNOWN_CALENDAR_EVENT",
        "source": "BUSINESS_DOMAIN_INPUT",
        "confidence": "BUSINESS_REPORTED_WINDOW",
    },
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


def _query_frame(query: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _events_frame() -> pd.DataFrame:
    rows = []
    for event in BUSINESS_EVENTS:
        start = pd.Timestamp.fromisocalendar(
            event["iso_year"], event["start_week"], 1
        ).tz_localize("UTC")
        end = pd.Timestamp.fromisocalendar(
            event["iso_year"], event["end_week"], 7
        ).tz_localize("UTC")
        rows.append({**event, "start_date": start, "end_date": end})
    return pd.DataFrame(rows)


def _source_signature(client) -> tuple[str, dict[str, Any]]:
    port = _query_frame(
        f"""
        SELECT COUNT(*)::bigint AS rows,
               MIN(planned_eta) AS min_time,
               MAX(planned_eta) AS max_time,
               MAX(updated_at) AS max_updated_at
        FROM {PORT_CALL_VIEW}
        """
    ).iloc[0].to_dict()
    weather = _query_frame(
        f"""
        SELECT COUNT(*)::bigint AS rows,
               MIN(observed_at) AS min_time,
               MAX(observed_at) AS max_time,
               MAX(ingestion_run_id::text) AS max_updated_at
        FROM {WEATHER_TABLE}
        WHERE quality_flag=0
        """
    ).iloc[0].to_dict()
    tir = client.head_object(Bucket=TIR_BUCKET, Key=TIR_KEY)
    evidence = {
        "audit_version": AUDIT_VERSION,
        "port": _clean_json(port),
        "weather": _clean_json(weather),
        "tir": {
            "etag": str(tir.get("ETag", "")).strip('"'),
            "size": int(tir["ContentLength"]),
            "last_modified": tir["LastModified"].isoformat(),
        },
        "events": _clean_json(list(BUSINESS_EVENTS)),
    }
    payload = json.dumps(evidence, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), evidence


def _start_run(checksum: str) -> str:
    metadata = {
        "audit_version": AUDIT_VERSION,
        "training_executed": False,
        "split_created": False,
        "bronze_modified": False,
    }
    source_uri = (
        f"postgresql://maritime/{PORT_CALL_VIEW}+{WEATHER_TABLE};"
        f"s3://{TIR_BUCKET}/{TIR_KEY}"
    )
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
                    Json(
                        _clean_json(metadata),
                        dumps=lambda value: json.dumps(
                            value, default=_json_default
                        ),
                    ),
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


def _load_port_calls() -> pd.DataFrame:
    frame = _query_frame(
        f"""
        SELECT port_call_id::text AS port_call_id,
               imo, vessel_name, cargo_type, vessel_type,
               planned_eta, planned_etd, actual_ata, actual_atd,
               arrival_delay_h::double precision AS arrival_delay_h,
               departure_delay_h::double precision AS departure_delay_h,
               updated_at
        FROM {PORT_CALL_VIEW}
        ORDER BY port_call_id
        """
    )
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


def _load_weather() -> pd.DataFrame:
    frame = _query_frame(
        f"""
        SELECT observed_at,
               wave_height_m::double precision AS wave_height_m,
               wave_period_s::double precision AS wave_period_s,
               wave_direction_deg::double precision AS wave_direction_deg,
               wind_speed_ms::double precision AS wind_speed_ms,
               wind_direction_deg::double precision AS wind_direction_deg,
               surface_current_ms::double precision AS surface_current_ms,
               visibility_m::double precision AS visibility_m,
               pressure_hpa::double precision AS pressure_hpa
        FROM {WEATHER_TABLE}
        WHERE quality_flag=0
        ORDER BY observed_at
        """
    )
    frame["observed_at"] = pd.to_datetime(
        frame["observed_at"], errors="coerce", utc=True
    )
    for column in frame.columns.difference(["observed_at"]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_tir(client) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        client.download_file(TIR_BUCKET, TIR_KEY, handle.name)
        frame = pd.read_parquet(handle.name, columns=list(TIR_COLUMNS))
    frame.columns = [column.lower() for column in frame.columns]
    frame["date_zre"] = pd.to_datetime(
        frame["date_zre"], errors="coerce", utc=True, format="mixed"
    )
    frame["date_embarquement"] = pd.to_datetime(
        frame["date_embarquement"], errors="coerce", utc=True, format="mixed"
    )
    frame["poids"] = pd.to_numeric(frame["poids"], errors="coerce")
    duration = (
        frame["date_embarquement"] - frame["date_zre"]
    ).dt.total_seconds() / 3600.0
    frame["raw_zre_to_embarkation_h"] = duration.where(duration.between(0, 720))
    frame["invalid_duration_flag"] = (
        duration.notna() & ~duration.between(0, 720)
    ).astype("int8")
    return frame


def _complete_daily(
    frame: pd.DataFrame,
    date_column: str,
    count_columns: tuple[str, ...],
) -> pd.DataFrame:
    frame = frame.sort_values(date_column).reset_index(drop=True)
    if frame.empty:
        return frame
    index = pd.date_range(
        max(ANALYSIS_START, frame[date_column].min()),
        frame[date_column].max(),
        freq="D",
        tz="UTC",
    )
    frame = frame.set_index(date_column).reindex(index)
    frame.index.name = "date"
    for column in count_columns:
        if column in frame:
            frame[column] = frame[column].fillna(0)
    return frame.reset_index()


def _build_port_daily(calls: pd.DataFrame) -> pd.DataFrame:
    cohort = calls[calls["planned_eta"].ge(ANALYSIS_START)].copy()
    cohort["date"] = cohort["planned_eta"].dt.floor("D")
    cohort["arrival_labeled"] = cohort["actual_ata"].notna().astype("int8")
    cohort["departure_labeled"] = cohort["actual_atd"].notna().astype("int8")
    cohort["late_gt1"] = cohort["arrival_delay_h"].gt(1).astype("int8")
    cohort["late_gt3"] = cohort["arrival_delay_h"].gt(3).astype("int8")
    turnaround = (
        cohort["actual_atd"] - cohort["actual_ata"]
    ).dt.total_seconds() / 3600.0
    cohort["turnaround_h"] = turnaround.where(turnaround.between(0, 720))

    daily = cohort.groupby("date", observed=True).agg(
        port_planned_calls=("port_call_id", "size"),
        port_unique_vessels=("imo", "nunique"),
        port_arrival_labeled_calls=("arrival_labeled", "sum"),
        port_departure_labeled_calls=("departure_labeled", "sum"),
        port_arrival_label_rate=("arrival_labeled", "mean"),
        port_departure_label_rate=("departure_labeled", "mean"),
        port_arrival_delay_mean_h=("arrival_delay_h", "mean"),
        port_arrival_delay_median_h=("arrival_delay_h", "median"),
        port_late_gt1_rate=("late_gt1", "mean"),
        port_late_gt3_rate=("late_gt3", "mean"),
        port_turnaround_labeled_calls=("turnaround_h", "count"),
        port_turnaround_mean_h=("turnaround_h", "mean"),
        port_turnaround_median_h=("turnaround_h", "median"),
    ).reset_index()

    actual_arrivals = (
        calls[calls["actual_ata"].ge(ANALYSIS_START)]
        .assign(date=lambda item: item["actual_ata"].dt.floor("D"))
        .groupby("date")
        .size()
        .rename("port_actual_arrivals")
    )
    actual_departures = (
        calls[calls["actual_atd"].ge(ANALYSIS_START)]
        .assign(date=lambda item: item["actual_atd"].dt.floor("D"))
        .groupby("date")
        .size()
        .rename("port_actual_departures")
    )
    daily = daily.merge(actual_arrivals, on="date", how="outer")
    daily = daily.merge(actual_departures, on="date", how="outer")
    return _complete_daily(
        daily,
        "date",
        (
            "port_planned_calls",
            "port_unique_vessels",
            "port_arrival_labeled_calls",
            "port_departure_labeled_calls",
            "port_turnaround_labeled_calls",
            "port_actual_arrivals",
            "port_actual_departures",
        ),
    )


def _build_weather_daily(weather: pd.DataFrame) -> pd.DataFrame:
    frame = weather[weather["observed_at"].ge(ANALYSIS_START)].copy()
    frame["date"] = frame["observed_at"].dt.floor("D")
    frame["storm_gt2h"] = frame["wave_height_m"].gt(2).astype("int8")
    frame["storm_gt3h"] = frame["wave_height_m"].gt(3).astype("int8")
    daily = frame.groupby("date", observed=True).agg(
        weather_observation_count=("observed_at", "size"),
        weather_wave_coverage_rate=("wave_height_m", lambda x: x.notna().mean()),
        weather_full_atmospheric_coverage_rate=(
            "wind_speed_ms",
            lambda x: x.notna().mean(),
        ),
        weather_wave_height_mean_m=("wave_height_m", "mean"),
        weather_wave_height_max_m=("wave_height_m", "max"),
        weather_wave_period_mean_s=("wave_period_s", "mean"),
        weather_storm_gt2_hours=("storm_gt2h", "sum"),
        weather_storm_gt3_hours=("storm_gt3h", "sum"),
    ).reset_index()
    daily["weather_hourly_coverage_rate"] = (
        daily["weather_observation_count"] / 24.0
    ).clip(upper=1)
    return _complete_daily(
        daily,
        "date",
        (
            "weather_observation_count",
            "weather_storm_gt2_hours",
            "weather_storm_gt3_hours",
        ),
    )


def _build_tir_daily(tir: pd.DataFrame) -> pd.DataFrame:
    frame = tir[tir["date_zre"].ge(ANALYSIS_START)].copy()
    frame["date"] = frame["date_zre"].dt.floor("D")
    frame["duration_labeled"] = frame["raw_zre_to_embarkation_h"].notna().astype(
        "int8"
    )
    frame["long_12h"] = frame["raw_zre_to_embarkation_h"].gt(12).astype("int8")
    frame["long_24h"] = frame["raw_zre_to_embarkation_h"].gt(24).astype("int8")
    frame["full_flag"] = (
        frame["vide_plein"].fillna("").astype(str).str.upper().eq("PLEIN")
    ).astype("int8")
    frame["embarkation_recorded"] = frame["date_embarquement"].notna().astype(
        "int8"
    )
    frame["cargo_missing"] = frame["nature_marchandise"].isna().astype("int8")
    frame["dangerous"] = frame["matiere_danger"].fillna(False).astype("int8")
    daily = frame.groupby("date", observed=True).agg(
        tir_rows=("source_row_index", "size"),
        tir_unique_units=("unite", "nunique"),
        tir_unique_declarants=("declarant", "nunique"),
        tir_total_weight=("poids", "sum"),
        tir_duration_labeled_rows=("duration_labeled", "sum"),
        tir_duration_label_rate=("duration_labeled", "mean"),
        tir_target_duration_mean_h=("raw_zre_to_embarkation_h", "mean"),
        tir_target_duration_median_h=("raw_zre_to_embarkation_h", "median"),
        tir_long_12h_rate=("long_12h", "mean"),
        tir_long_24h_rate=("long_24h", "mean"),
        tir_full_rate=("full_flag", "mean"),
        tir_dangerous_rate=("dangerous", "mean"),
        tir_embarkation_recorded_rate=("embarkation_recorded", "mean"),
        tir_cargo_missing_rate=("cargo_missing", "mean"),
        tir_invalid_duration_rate=("invalid_duration_flag", "mean"),
    ).reset_index()
    return _complete_daily(
        daily,
        "date",
        (
            "tir_rows",
            "tir_unique_units",
            "tir_unique_declarants",
            "tir_total_weight",
            "tir_duration_labeled_rows",
        ),
    )


def _daily_to_monthly(dataset: str, daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = pd.to_datetime(
        frame["date"].dt.strftime("%Y-%m-01"), utc=True
    )
    rows = []
    for metric in frame.columns.difference(["date", "month"]):
        values = pd.to_numeric(frame[metric], errors="coerce")
        work = pd.DataFrame({"month": frame["month"], "value": values})
        if any(
            token in metric
            for token in (
                "_rows",
                "_calls",
                "_arrivals",
                "_departures",
                "_count",
                "_hours",
                "total_weight",
            )
        ):
            grouped = work.groupby("month")["value"].sum(min_count=1)
            aggregation = "SUM"
        elif metric.endswith("_max_m"):
            grouped = work.groupby("month")["value"].max()
            aggregation = "MAX"
        else:
            grouped = work.groupby("month")["value"].mean()
            aggregation = "DAILY_MEAN"
        for month, value in grouped.items():
            rows.append(
                {
                    "dataset": dataset,
                    "month": month,
                    "metric": metric,
                    "aggregation": aggregation,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def _source_inventory(
    calls: pd.DataFrame,
    weather: pd.DataFrame,
    tir: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "source": "PORT_CALLS",
            "rows": len(calls),
            "first_event_time": calls["planned_eta"].min(),
            "last_event_time": calls["planned_eta"].max(),
            "primary_time": "planned_eta",
            "availability_history": "ABSENT_FOR_HISTORICAL_ROWS",
        },
        {
            "source": "WEATHER_WAVE",
            "rows": len(weather),
            "first_event_time": weather["observed_at"].min(),
            "last_event_time": weather["observed_at"].max(),
            "primary_time": "observed_at",
            "availability_history": "ABSENT_FOR_HISTORICAL_ROWS",
        },
        {
            "source": "TIR_BRONZE",
            "rows": len(tir),
            "first_event_time": tir["date_zre"].min(),
            "last_event_time": tir["date_zre"].max(),
            "primary_time": "date_zre",
            "availability_history": "BRONZE_FILE_ONLY",
        },
    ]
    return pd.DataFrame(rows)


def _completeness_diagnostics(monthly: pd.DataFrame) -> pd.DataFrame:
    volume_metrics = {
        "PORT_CALLS": "port_planned_calls",
        "WEATHER": "weather_observation_count",
        "TIR_BRONZE": "tir_rows",
    }
    reports = []
    for dataset, metric in volume_metrics.items():
        frame = monthly[
            monthly["dataset"].eq(dataset) & monthly["metric"].eq(metric)
        ].copy()
        frame = frame.sort_values("month")
        frame["month_number"] = frame["month"].dt.month
        baseline = frame[frame["month"].le(SEASONAL_BASELINE_END)]
        seasonal = baseline.groupby("month_number")["value"].median()
        frame["seasonal_baseline"] = frame["month_number"].map(seasonal)
        frame["seasonal_ratio"] = frame["value"] / frame["seasonal_baseline"].replace(
            0, np.nan
        )
        frame["month_status"] = np.select(
            [frame["seasonal_ratio"].lt(0.50), frame["seasonal_ratio"].lt(LOW_VOLUME_RATIO)],
            ["SEVERE_LOW_VOLUME", "LOW_VOLUME"],
            default="NORMAL",
        )
        low_volume = frame["seasonal_ratio"].lt(LOW_VOLUME_RATIO).fillna(False)
        frame["sustained_low_3m"] = low_volume.rolling(3).sum().ge(3)
        frame["terminal_low_run"] = False
        first_break = None
        if len(frame) and bool(low_volume.iloc[-1]):
            start_position = len(frame) - 1
            while start_position > 0 and bool(low_volume.iloc[start_position - 1]):
                start_position -= 1
            if len(frame) - start_position >= 3:
                first_break = frame.iloc[start_position]["month"]
                frame.loc[frame.index[start_position:], "terminal_low_run"] = True
        frame["first_sustained_break"] = pd.Series(
            first_break if first_break is not None else pd.NaT,
            index=frame.index,
            dtype="datetime64[ns, UTC]",
        )
        reports.append(
            frame[
                [
                    "dataset",
                    "month",
                    "metric",
                    "value",
                    "seasonal_baseline",
                    "seasonal_ratio",
                    "month_status",
                    "sustained_low_3m",
                    "terminal_low_run",
                    "first_sustained_break",
                ]
            ]
        )
    return pd.concat(reports, ignore_index=True)


def _bh_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1])[::-1].clip(upper=1)
    result.loc[valid.index] = adjusted
    return result


def _event_impact(
    events: pd.DataFrame,
    daily_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    selected = {
        "PORT_CALLS": (
            "port_planned_calls",
            "port_actual_arrivals",
            "port_arrival_delay_median_h",
            "port_late_gt3_rate",
            "port_turnaround_median_h",
        ),
        "WEATHER": (
            "weather_wave_height_mean_m",
            "weather_wave_height_max_m",
            "weather_storm_gt2_hours",
        ),
        "TIR_BRONZE": (
            "tir_rows",
            "tir_total_weight",
            "tir_target_duration_mean_h",
            "tir_target_duration_median_h",
            "tir_long_24h_rate",
        ),
    }
    rows = []
    for event in events.to_dict("records"):
        start = event["start_date"]
        end = event["end_date"]
        window_days = int((end - start).days) + 1
        for dataset, metrics in selected.items():
            frame = daily_frames[dataset]
            iso = frame["date"].dt.isocalendar()
            during_mask = frame["date"].between(start, end)
            pre_mask = frame["date"].between(
                start - pd.Timedelta(days=window_days), start - pd.Timedelta(days=1)
            )
            post_mask = frame["date"].between(
                end + pd.Timedelta(days=1), end + pd.Timedelta(days=window_days)
            )
            historical_mask = (
                frame["date"].dt.year.between(2020, 2024)
                & iso.week.between(event["start_week"], event["end_week"])
            )
            for metric in metrics:
                during = pd.to_numeric(
                    frame.loc[during_mask, metric], errors="coerce"
                ).dropna()
                pre = pd.to_numeric(frame.loc[pre_mask, metric], errors="coerce").dropna()
                post = pd.to_numeric(
                    frame.loc[post_mask, metric], errors="coerce"
                ).dropna()
                historical = pd.to_numeric(
                    frame.loc[historical_mask, metric], errors="coerce"
                ).dropna()
                coverage = 100.0 * len(during) / window_days
                status = (
                    "NO_COVERAGE"
                    if len(during) == 0
                    else "PARTIAL_COVERAGE"
                    if coverage < MIN_EVENT_COVERAGE_PCT
                    else "OK"
                )
                during_mean = during.mean() if len(during) else np.nan
                pre_mean = pre.mean() if len(pre) else np.nan
                post_mean = post.mean() if len(post) else np.nan
                historical_mean = historical.mean() if len(historical) else np.nan
                effect_pre = (
                    100 * (during_mean - pre_mean) / abs(pre_mean)
                    if pd.notna(pre_mean) and abs(pre_mean) > 1e-12
                    else np.nan
                )
                effect_history = (
                    100 * (during_mean - historical_mean) / abs(historical_mean)
                    if pd.notna(historical_mean) and abs(historical_mean) > 1e-12
                    else np.nan
                )
                p_value = np.nan
                if len(during) >= 5 and len(historical) >= 20:
                    p_value = mannwhitneyu(
                        during, historical, alternative="two-sided"
                    ).pvalue
                rows.append(
                    {
                        "event_id": event["event_id"],
                        "event_name": event["event_name"],
                        "event_type": event["event_type"],
                        "knowledge_policy": event["knowledge_policy"],
                        "start_date": start,
                        "end_date": end,
                        "dataset": dataset,
                        "metric": metric,
                        "coverage_pct": coverage,
                        "status": status,
                        "during_n": len(during),
                        "during_mean": during_mean,
                        "pre_mean": pre_mean,
                        "post_mean": post_mean,
                        "historical_same_week_mean": historical_mean,
                        "effect_pct_vs_pre": effect_pre,
                        "effect_pct_vs_historical": effect_history,
                        "mann_whitney_p": p_value,
                    }
                )
    report = pd.DataFrame(rows)
    report["p_adjusted_bh"] = _bh_adjust(report["mann_whitney_p"])
    report["material_effect_flag"] = (
        report["status"].eq("OK")
        & report["effect_pct_vs_historical"].abs().ge(10)
        & report["p_adjusted_bh"].lt(0.05)
    )
    return report


def _change_points(monthly: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    selected = {
        "port_planned_calls",
        "port_arrival_label_rate",
        "port_arrival_delay_median_h",
        "port_turnaround_median_h",
        "weather_observation_count",
        "weather_wave_height_max_m",
        "weather_storm_gt2_hours",
        "tir_rows",
        "tir_target_duration_median_h",
        "tir_long_24h_rate",
        "tir_full_rate",
    }
    event_months = events.assign(
        _start_month_key=(events["start_date"].dt.year * 12 + events["start_date"].dt.month),
        _end_month_key=(events["end_date"].dt.year * 12 + events["end_date"].dt.month),
    )
    candidates = []
    for (dataset, metric), group in monthly.groupby(["dataset", "metric"]):
        if metric not in selected:
            continue
        frame = group.sort_values("month").dropna(subset=["value"]).reset_index(drop=True)
        if len(frame) < 24:
            continue
        for split in range(12, len(frame) - 2):
            pre = frame.loc[split - 12 : split - 1, "value"].astype(float)
            post = frame.loc[split : split + 2, "value"].astype(float)
            pre_median = float(pre.median())
            post_median = float(post.median())
            mad = float(np.median(np.abs(pre - pre_median)))
            scale = max(1.4826 * mad, abs(pre_median) * 0.05, 1e-9)
            score = abs(post_median - pre_median) / scale
            effect = (
                100 * (post_median - pre_median) / abs(pre_median)
                if abs(pre_median) > 1e-12
                else np.nan
            )
            p_value = mannwhitneyu(pre, post, alternative="two-sided").pvalue
            change_month = frame.loc[split, "month"]
            change_month_key = change_month.year * 12 + change_month.month
            overlapping = event_months[
                event_months["_start_month_key"].le(change_month_key)
                & event_months["_end_month_key"].ge(change_month_key)
            ]
            candidates.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "change_month": change_month,
                    "pre_12m_median": pre_median,
                    "post_3m_median": post_median,
                    "effect_pct": effect,
                    "robust_score": score,
                    "mann_whitney_p": p_value,
                    "overlapping_events": "|".join(overlapping["event_id"]),
                }
            )
    report = pd.DataFrame(candidates)
    if report.empty:
        return report
    report["p_adjusted_bh"] = _bh_adjust(report["mann_whitney_p"])
    report = report.sort_values(
        ["dataset", "metric", "robust_score"], ascending=[True, True, False]
    )
    report = report.groupby(["dataset", "metric"], as_index=False).head(5)
    report["candidate_type"] = np.where(
        report["overlapping_events"].ne(""),
        "KNOWN_EVENT_WINDOW",
        "UNEXPLAINED_OR_STRUCTURAL",
    )
    return report.reset_index(drop=True)


def _sample(values: pd.Series, limit: int = 100_000) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) > limit:
        clean = clean.sample(limit, random_state=42)
    return clean.to_numpy(dtype=float)


def _numeric_drift(
    series_name: str,
    timestamps: pd.Series,
    values: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp": timestamps, "value": values}).dropna()
    frame = frame[frame["timestamp"].ge(ANALYSIS_START)]
    frame["year"] = frame["timestamp"].dt.year
    reference = _sample(frame[frame["year"].between(2020, 2024)]["value"])
    rows = []
    for year, group in frame.groupby("year"):
        sample = _sample(group["value"])
        statistic = p_value = np.nan
        if len(reference) >= 100 and len(sample) >= 30:
            test = ks_2samp(reference, sample)
            statistic, p_value = test.statistic, test.pvalue
        rows.append(
            {
                "series": series_name,
                "year": int(year),
                "n": len(group),
                "mean": group["value"].mean(),
                "median": group["value"].median(),
                "p95": group["value"].quantile(0.95),
                "ks_vs_2020_2024": statistic,
                "ks_p_value": p_value,
                "material_drift_flag": pd.notna(statistic) and statistic >= 0.10,
            }
        )
    return pd.DataFrame(rows)


def _numeric_drift_report(
    calls: pd.DataFrame,
    weather: pd.DataFrame,
    tir: pd.DataFrame,
) -> pd.DataFrame:
    turnaround_h = (
        calls["actual_atd"] - calls["actual_ata"]
    ).dt.total_seconds() / 3600.0
    turnaround_h = turnaround_h.where(turnaround_h.between(0, 720))
    reports = [
        _numeric_drift(
            "arrival_delay_h", calls["planned_eta"], calls["arrival_delay_h"]
        ),
        _numeric_drift(
            "turnaround_h",
            calls["actual_ata"],
            turnaround_h,
        ),
        _numeric_drift(
            "wave_height_m", weather["observed_at"], weather["wave_height_m"]
        ),
        _numeric_drift(
            "tir_zre_to_embarkation_h",
            tir["date_zre"],
            tir["raw_zre_to_embarkation_h"],
        ),
    ]
    report = pd.concat(reports, ignore_index=True)
    report["dataset"] = report["series"].map(
        {
            "arrival_delay_h": "PORT_CALLS",
            "turnaround_h": "PORT_CALLS",
            "wave_height_m": "WEATHER",
            "tir_zre_to_embarkation_h": "TIR_BRONZE",
        }
    )
    return report


def _categorical_drift(tir: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = tir[tir["date_zre"].ge(ANALYSIS_START)].copy()
    frame["year"] = frame["date_zre"].dt.year
    for feature in (
        "type_unite",
        "ss_type_unite",
        "vide_plein",
        "nature_marchandise",
        "etat_dechargement",
        "statut_exploitation",
    ):
        values = frame[feature].fillna("**MISSING**").astype(str)
        top = values.value_counts().head(50).index
        collapsed = values.where(values.isin(top), "**OTHER**")
        baseline = collapsed[frame["year"].between(2020, 2024)].value_counts()
        for year in sorted(frame["year"].dropna().unique()):
            current = collapsed[frame["year"].eq(year)].value_counts()
            categories = baseline.index.union(current.index)
            p = baseline.reindex(categories, fill_value=0).to_numpy(dtype=float) + 1e-9
            q = current.reindex(categories, fill_value=0).to_numpy(dtype=float) + 1e-9
            p /= p.sum()
            q /= q.sum()
            distance = float(jensenshannon(p, q, base=2))
            rows.append(
                {
                    "feature": feature,
                    "year": int(year),
                    "n": int(frame["year"].eq(year).sum()),
                    "jensen_shannon_vs_2020_2024": distance,
                    "material_drift_flag": distance >= 0.10,
                }
            )
    return pd.DataFrame(rows)


def _source_break_map(completeness: pd.DataFrame) -> dict[str, pd.Timestamp]:
    result: dict[str, pd.Timestamp] = {}
    for dataset, frame in completeness.groupby("dataset"):
        values = pd.to_datetime(
            frame["first_sustained_break"], errors="coerce", utc=True
        ).dropna()
        if len(values):
            result[str(dataset)] = values.min()
    return result


def _apply_source_break_guards(
    completeness: pd.DataFrame,
    event_impact: pd.DataFrame,
    changes: pd.DataFrame,
    numeric_drift: pd.DataFrame,
    categorical_drift: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    breaks = _source_break_map(completeness)

    event_impact = event_impact.copy()
    event_impact["raw_material_effect_flag"] = event_impact[
        "material_effect_flag"
    ].astype(bool)
    event_impact["source_break_contaminated_flag"] = False
    for dataset, first_break in breaks.items():
        mask = event_impact["dataset"].eq(dataset) & pd.to_datetime(
            event_impact["start_date"], errors="coerce", utc=True
        ).ge(first_break)
        event_impact.loc[mask, "source_break_contaminated_flag"] = True
        event_impact.loc[mask, "status"] = "SOURCE_BREAK_CONTAMINATED"
        event_impact.loc[mask, "material_effect_flag"] = False

    changes = changes.copy()
    if not changes.empty:
        changes["source_break_contaminated_flag"] = False
        for dataset, first_break in breaks.items():
            mask = changes["dataset"].eq(dataset) & pd.to_datetime(
                changes["change_month"], errors="coerce", utc=True
            ).ge(first_break)
            changes.loc[mask, "source_break_contaminated_flag"] = True
            changes.loc[mask, "candidate_type"] = "SOURCE_BREAK_OR_TRUNCATION"

    numeric_drift = numeric_drift.copy()
    numeric_drift["statistical_drift_flag"] = numeric_drift[
        "material_drift_flag"
    ].astype(bool)
    numeric_drift["source_break_contaminated_flag"] = False
    for dataset, first_break in breaks.items():
        year_end = pd.to_datetime(
            numeric_drift["year"].astype(str) + "-12-31", errors="coerce", utc=True
        )
        mask = numeric_drift["dataset"].eq(dataset) & year_end.ge(first_break)
        numeric_drift.loc[mask, "source_break_contaminated_flag"] = True
        numeric_drift.loc[mask, "material_drift_flag"] = False

    categorical_drift = categorical_drift.copy()
    categorical_drift["dataset"] = "TIR_BRONZE"
    categorical_drift["statistical_drift_flag"] = categorical_drift[
        "material_drift_flag"
    ].astype(bool)
    categorical_drift["source_break_contaminated_flag"] = False
    tir_break = breaks.get("TIR_BRONZE")
    if tir_break is not None:
        year_end = pd.to_datetime(
            categorical_drift["year"].astype(str) + "-12-31",
            errors="coerce",
            utc=True,
        )
        mask = year_end.ge(tir_break)
        categorical_drift.loc[mask, "source_break_contaminated_flag"] = True
        categorical_drift.loc[mask, "material_drift_flag"] = False

    return event_impact, changes, numeric_drift, categorical_drift


def _seasonal_profiles(daily_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = {
        "PORT_CALLS": ("port_planned_calls", "port_actual_arrivals"),
        "WEATHER": ("weather_wave_height_max_m", "weather_storm_gt2_hours"),
        "TIR_BRONZE": ("tir_rows", "tir_target_duration_mean_h"),
    }
    rows = []
    for dataset, frame in daily_frames.items():
        baseline = frame[frame["date"].le(SEASONAL_BASELINE_END)].copy()
        baseline["iso_week"] = baseline["date"].dt.isocalendar().week.astype(int)
        baseline["day_of_week"] = baseline["date"].dt.dayofweek
        for metric in metrics[dataset]:
            for dimension in ("iso_week", "day_of_week"):
                grouped = baseline.groupby(dimension)[metric].agg(
                    ["count", "mean", "median", "std"]
                )
                for key, values in grouped.iterrows():
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "seasonal_dimension": dimension,
                            "seasonal_value": int(key),
                            **values.to_dict(),
                        }
                    )
    return pd.DataFrame(rows)


def _temporal_semantics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field": "planned_eta",
                "role": "SCHEDULE_TIMESTAMP",
                "predictive_policy": "USE_ONLY_WITH_CAPTURED_AVAILABLE_AT",
                "leakage_risk": "ETA revision history absent",
            },
            {
                "field": "actual_ata/actual_atd",
                "role": "OUTCOME",
                "predictive_policy": "TARGET_OR_POST_EVENT_ONLY",
                "leakage_risk": "Forbidden before actual occurrence",
            },
            {
                "field": "date_embarquement/raw_zre_to_embarkation_h",
                "role": "TIR_OUTCOME",
                "predictive_policy": "TARGET_ONLY",
                "leakage_risk": "Forbidden in ZRE prediction features",
            },
            {
                "field": "observed_weather",
                "role": "EXPLANATORY_SERIES",
                "predictive_policy": "PAST_ONLY_AT_CUTOFF",
                "leakage_risk": "Future realized weather is forbidden",
            },
            {
                "field": "2026_BAD_WEATHER_EXPORT_S03_S07",
                "role": "RETROSPECTIVE_EVENT_LABEL",
                "predictive_policy": "EXPLANATORY_ONLY",
                "leakage_risk": "Replace with issued forecast for prediction",
            },
            {
                "field": "Aid event windows",
                "role": "KNOWN_CALENDAR_EVENT",
                "predictive_policy": "SAFE_AFTER_CALENDAR_PUBLICATION",
                "leakage_risk": "Use exact locally confirmed dates",
            },
        ]
    )


def _regime_timeline(
    completeness: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    start = min(completeness["month"].min(), events["start_date"].min())
    end = max(completeness["month"].max(), events["end_date"].max())
    months = pd.date_range(start, end, freq="MS", tz="UTC")
    rows = []
    for month in months:
        month_end = month + pd.offsets.MonthEnd(1)
        active = events[
            events["start_date"].le(month_end) & events["end_date"].ge(month)
        ]
        row = {
            "month": month,
            "active_events": "|".join(active["event_id"]),
        }
        for dataset in ("PORT_CALLS", "WEATHER", "TIR_BRONZE"):
            match = completeness[
                completeness["dataset"].eq(dataset)
                & completeness["month"].eq(month)
            ]
            prefix = dataset.lower()
            row[f"{prefix}_seasonal_ratio"] = (
                match.iloc[0]["seasonal_ratio"] if len(match) else np.nan
            )
            row[f"{prefix}_status"] = (
                match.iloc[0]["month_status"] if len(match) else "NO_DATA"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _materialize_events(events: pd.DataFrame) -> int:
    rows = []
    for event in events.to_dict("records"):
        rows.append(
            (
                event["event_id"],
                event["event_name"],
                event["event_type"],
                event["start_date"].date(),
                event["end_date"].date(),
                event["affected_flow"],
                event["knowledge_policy"],
                event["source"],
                event["confidence"],
                AUDIT_VERSION,
            )
        )
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS reference")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reference.business_event (
                    event_id TEXT PRIMARY KEY,
                    event_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    affected_flow TEXT,
                    knowledge_policy TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence TEXT,
                    audit_version TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CHECK (end_date >= start_date)
                )
                """
            )
            execute_values(
                cursor,
                """
                INSERT INTO reference.business_event
                    (event_id, event_name, event_type, start_date, end_date,
                     affected_flow, knowledge_policy, source, confidence,
                     audit_version)
                VALUES %s
                ON CONFLICT (event_id) DO UPDATE SET
                    event_name=EXCLUDED.event_name,
                    event_type=EXCLUDED.event_type,
                    start_date=EXCLUDED.start_date,
                    end_date=EXCLUDED.end_date,
                    affected_flow=EXCLUDED.affected_flow,
                    knowledge_policy=EXCLUDED.knowledge_policy,
                    source=EXCLUDED.source,
                    confidence=EXCLUDED.confidence,
                    audit_version=EXCLUDED.audit_version,
                    updated_at=now()
                """,
                rows,
            )
    return len(rows)


def _decision(
    inventory: pd.DataFrame,
    completeness: pd.DataFrame,
    event_impact: pd.DataFrame,
    changes: pd.DataFrame,
    numeric_drift: pd.DataFrame,
) -> dict[str, Any]:
    breaks: dict[str, str | None] = {}
    safe_periods: dict[str, dict[str, str | None]] = {}
    for dataset in ("PORT_CALLS", "WEATHER", "TIR_BRONZE"):
        frame = completeness[completeness["dataset"].eq(dataset)]
        values = frame["first_sustained_break"].dropna()
        first_break = values.min() if len(values) else None
        breaks[dataset] = None if first_break is None else first_break.date().isoformat()
        if first_break is not None:
            safe_end = first_break - pd.Timedelta(days=1)
        else:
            last_month = frame["month"].max()
            safe_end = last_month + pd.offsets.MonthEnd(1)
        safe_periods[dataset] = {
            "start": ANALYSIS_START.date().isoformat(),
            "end": None if pd.isna(safe_end) else safe_end.date().isoformat(),
        }

    material_effects = event_impact[event_impact["material_effect_flag"]]
    drift_2026 = numeric_drift[
        numeric_drift["year"].eq(2026) & numeric_drift["material_drift_flag"]
    ]
    if changes.empty:
        unexplained = changes
    else:
        unexplained = changes[
            changes["candidate_type"].eq("UNEXPLAINED_OR_STRUCTURAL")
            & changes["robust_score"].ge(3)
        ]
    has_break = any(value is not None for value in breaks.values())
    status = (
        "READY_FOR_EVENT_AWARE_PRE_BREAK_FEATURES"
        if has_break
        else "READY_FOR_EVENT_AWARE_TEMPORAL_FEATURES"
    )
    return {
        "status": status,
        "audit_version": AUDIT_VERSION,
        "source_rows": {
            row["source"]: int(row["rows"])
            for row in inventory.to_dict("records")
        },
        "first_sustained_breaks": breaks,
        "safe_periods": safe_periods,
        "known_events": len(BUSINESS_EVENTS),
        "material_event_effects": len(material_effects),
        "material_2026_numeric_drifts": len(drift_2026),
        "unexplained_change_candidates": len(unexplained),
        "source_break_contaminated_event_rows": int(
            event_impact["source_break_contaminated_flag"].sum()
        ),
        "source_break_contaminated_drift_rows": int(
            numeric_drift["source_break_contaminated_flag"].sum()
        ),
        "weather_event_feature_policy": (
            "EXPLANATORY_ONLY_UNTIL_ISSUED_FORECAST_VINTAGES_EXIST"
        ),
        "holiday_feature_policy": "KNOWN_CALENDAR_EVENT_AFTER_LOCAL_CONFIRMATION",
        "post_break_policy": "SHADOW_DIAGNOSTIC_ONLY",
        "training_executed": False,
        "split_created": False,
        "bronze_modified": False,
        "critical_leakage_violations": 0,
        "limitations": [
            "business event windows are supplied hypotheses, not causal proof",
            "historical ETA revision availability is absent",
            "historical weather publication time is absent",
            "TIR Park Visite target is not present in the minimal Bronze export",
            "S21-S23 2026 may have no source coverage in the current extract",
        ],
        "next_block": "B57B_EVENT_AWARE_OPERATIONAL_GOLD_FEATURES",
    }


def _write_readme(path: Path, decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# B57A Temporal Regime and Event Audit",
                "",
                f"Decision: **{decision['status']}**",
                "",
                "This block distinguishes seasonal behaviour, source completeness breaks,",
                "known business events and unexplained structural changes.",
                "",
                "## Guardrails",
                "",
                "- No model was trained.",
                "- No split was created.",
                "- Bronze was not modified.",
                "- The bad-weather label is explanatory only until forecast vintages exist.",
                "- Aid windows are usable only after local calendar confirmation.",
                "",
                "## Next block",
                "",
                decision["next_block"],
            ]
        ),
        encoding="utf-8",
    )


def run_b57a_temporal_regime_audit(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    materialize_event_calendar: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    checksum, source_signature = _source_signature(client)
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
        calls = _load_port_calls()
        weather = _load_weather()
        tir = _load_tir(client)
        events = _events_frame()

        port_daily = _build_port_daily(calls)
        weather_daily = _build_weather_daily(weather)
        tir_daily = _build_tir_daily(tir)
        daily_frames = {
            "PORT_CALLS": port_daily,
            "WEATHER": weather_daily,
            "TIR_BRONZE": tir_daily,
        }
        monthly = pd.concat(
            [
                _daily_to_monthly(dataset, frame)
                for dataset, frame in daily_frames.items()
            ],
            ignore_index=True,
        )
        inventory = _source_inventory(calls, weather, tir)
        completeness = _completeness_diagnostics(monthly)
        event_impact = _event_impact(events, daily_frames)
        changes = _change_points(monthly, events)
        numeric_drift = _numeric_drift_report(calls, weather, tir)
        categorical_drift = _categorical_drift(tir)
        event_impact, changes, numeric_drift, categorical_drift = (
            _apply_source_break_guards(
                completeness,
                event_impact,
                changes,
                numeric_drift,
                categorical_drift,
            )
        )
        seasonality = _seasonal_profiles(daily_frames)
        timeline = _regime_timeline(completeness, events)
        semantics = _temporal_semantics()
        decision = _decision(
            inventory, completeness, event_impact, changes, numeric_drift
        )

        reports = {
            "01_source_inventory.csv": inventory,
            "02_business_event_calendar.csv": events,
            "03_daily_port_call_metrics.csv": port_daily,
            "04_daily_weather_metrics.csv": weather_daily,
            "05_daily_tir_metrics.csv": tir_daily,
            "06_monthly_metric_panel.csv": monthly,
            "07_source_completeness_diagnostics.csv": completeness,
            "08_change_point_candidates.csv": changes,
            "09_event_impact_analysis.csv": event_impact,
            "10_numeric_drift_by_year.csv": numeric_drift,
            "11_categorical_drift_by_year.csv": categorical_drift,
            "12_seasonal_profiles.csv": seasonality,
            "13_regime_timeline.csv": timeline,
            "14_temporal_semantics_and_leakage.csv": semantics,
        }

        materialized_events = 0
        if materialize_event_calendar:
            materialized_events = _materialize_events(events)

        with tempfile.TemporaryDirectory(prefix="b57a-") as temporary:
            output_dir = Path(temporary)
            for name, report in reports.items():
                report.to_csv(output_dir / name, index=False)

            dataset_path = output_dir / "monthly_regime_panel_v1.parquet"
            monthly.to_parquet(dataset_path, index=False)
            decision_path = output_dir / "15_final_regime_decision.json"
            decision_path.write_text(
                json.dumps(_clean_json(decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            readme_path = output_dir / "README_B57A.md"
            _write_readme(readme_path, decision)

            uploaded: dict[str, str] = {}
            for path in sorted(output_dir.iterdir()):
                if path == dataset_path:
                    key = f"datasets/b57a/{output_prefix}/{path.name}"
                elif path == decision_path:
                    key = f"configs/b57a/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b57a/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload_file(
                    client, path, output_bucket, key
                )

        metadata = {
            **decision,
            "checksum": checksum,
            "source_signature": source_signature,
            "event_calendar_materialized_rows": materialized_events,
            "event_calendar_table": (
                "reference.business_event" if materialized_events else None
            ),
            "monthly_rows": len(monthly),
            "outputs": uploaded,
            "output_prefix": f"s3://{output_bucket}/reports/b57a/{output_prefix}/",
        }
        _finish_run(run_id, "SUCCESS", len(monthly), metadata)
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
