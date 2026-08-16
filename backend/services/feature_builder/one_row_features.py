from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_values

from feature_builder.wave_features import (
    _attach_safe_delay_history,
    _attach_wave_history,
    _load_frames,
    _prepare_wave_history,
    _select_sea_source,
    db_connection,
    s3_client,
)


SOURCE_NAME = "tir_port_calls_model_ready"
DATASET_NAME = "arrival_one_row_24h_no_split"
SOURCE_VIEW = "features.port_call_model_ready_v1"
FEATURE_VERSION = "b54f-one-row-24h-v1"
CUTOFF_HOURS = 24
ROLLING_WINDOWS_H = (3, 6, 12, 24, 72)

IDENTIFIER_COLUMNS = {
    "port_call_id",
    "source_record_id",
    "voyage_id",
}
AUDIT_ONLY_COLUMNS = {
    "actual_ata",
    "actual_atd",
    "arrival_delay_h",
    "departure_delay_h",
    "target_arrival_delay_h",
    "target_departure_delay_h",
    "arrived_before_cutoff_flag",
    "model_ready_flag",
    "exclusion_reason",
    "observed_at",
    "prediction_at",
    "planned_eta",
    "planned_etd",
    "vessel_history_event_time",
    "global_history_event_time",
}


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


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
                    Json(metadata or {}, dumps=lambda obj: json.dumps(obj, default=_json_default)),
                    error_message,
                    run_id,
                ),
            )


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


def _database_signature() -> tuple[str, str, dict[str, Any]]:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT count(*), min(planned_eta), max(planned_eta) FROM {SOURCE_VIEW}")
            call_stats = list(cursor.fetchone())
            sea_source = _select_sea_source(cursor)
            cursor.execute(
                """
                SELECT count(*), min(observed_at), max(observed_at), count(wave_height_m)
                FROM core.maritime_observation
                WHERE source = %s
                """,
                (sea_source,),
            )
            sea_stats = list(cursor.fetchone())
    metadata = {
        "source_view": SOURCE_VIEW,
        "sea_source": sea_source,
        "call_stats": call_stats,
        "sea_stats": sea_stats,
        "feature_version": FEATURE_VERSION,
        "cutoff_hours": CUTOFF_HOURS,
        "rolling_windows_h": list(ROLLING_WINDOWS_H),
    }
    payload = json.dumps(metadata, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), sea_source, metadata


def _add_wave_coverage(wave: pd.DataFrame) -> pd.DataFrame:
    result = wave.sort_values("observed_at").reset_index(drop=True).copy()
    indexed = result.set_index("observed_at")
    for hours in ROLLING_WINDOWS_H:
        count = indexed["wave_height_m"].rolling(f"{hours}h", min_periods=1).count()
        result[f"wave_observation_count_{hours}h"] = count.to_numpy(dtype="float32")
        result[f"wave_coverage_pct_{hours}h"] = (
            100.0 * count / float(hours)
        ).clip(upper=100.0).to_numpy(dtype="float32")
    return result


def _build_cutoff_rows(calls: pd.DataFrame) -> pd.DataFrame:
    frame = calls.copy()
    frame["prediction_time"] = frame["planned_eta"] - pd.Timedelta(hours=CUTOFF_HOURS)
    frame["arrived_before_cutoff_flag"] = (
        frame["actual_ata"].notna() & (frame["actual_ata"] <= frame["prediction_time"])
    )

    prediction = frame["prediction_time"]
    eta = frame["planned_eta"]
    frame["cutoff_hour"] = prediction.dt.hour.astype("int8")
    frame["cutoff_dayofweek"] = prediction.dt.dayofweek.astype("int8")
    frame["cutoff_month"] = prediction.dt.month.astype("int8")
    frame["cutoff_year"] = prediction.dt.year.astype("int16")
    frame["cutoff_weekend_flag"] = (prediction.dt.dayofweek >= 5).astype("int8")
    frame["cutoff_hour_sin"] = np.sin(2.0 * math.pi * prediction.dt.hour / 24.0)
    frame["cutoff_hour_cos"] = np.cos(2.0 * math.pi * prediction.dt.hour / 24.0)
    frame["cutoff_dow_sin"] = np.sin(2.0 * math.pi * prediction.dt.dayofweek / 7.0)
    frame["cutoff_dow_cos"] = np.cos(2.0 * math.pi * prediction.dt.dayofweek / 7.0)
    frame["cutoff_month_sin"] = np.sin(
        2.0 * math.pi * (prediction.dt.month - 1) / 12.0
    )
    frame["cutoff_month_cos"] = np.cos(
        2.0 * math.pi * (prediction.dt.month - 1) / 12.0
    )
    frame["eta_hour"] = eta.dt.hour.astype("int8")
    frame["eta_dayofweek"] = eta.dt.dayofweek.astype("int8")
    frame["eta_month"] = eta.dt.month.astype("int8")
    frame["eta_weekend_flag"] = (eta.dt.dayofweek >= 5).astype("int8")
    frame["eta_hour_sin"] = np.sin(2.0 * math.pi * eta.dt.hour / 24.0)
    frame["eta_hour_cos"] = np.cos(2.0 * math.pi * eta.dt.hour / 24.0)
    frame["eta_dow_sin"] = np.sin(2.0 * math.pi * eta.dt.dayofweek / 7.0)
    frame["eta_dow_cos"] = np.cos(2.0 * math.pi * eta.dt.dayofweek / 7.0)
    return frame


def _allowed_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    categorical_candidates = {"imo", "vessel_name"}
    explicit_calendar = {
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
    }

    def allowed(column: str) -> bool:
        if column.endswith("_direction_deg"):
            return False
        if column in categorical_candidates or column in explicit_calendar:
            return True
        return column.startswith(
            (
                "vessel_hist_",
                "global_hist_",
                "wave_",
                "high_wave_",
                "severe_wave_",
                "wind_",
                "surface_current_",
                "visibility_",
                "pressure_",
                "sea_observation_age_h",
                "sea_feature_available_flag",
                "sea_feature_stale_flag",
            )
        )

    features = [
        column
        for column in frame.columns
        if allowed(column) and column not in AUDIT_ONLY_COLUMNS
    ]
    categorical = [column for column in features if column in categorical_candidates]
    numeric = [column for column in features if column not in categorical]
    wave = [
        column
        for column in features
        if column.startswith(
            (
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
        )
    ]
    return features, categorical, wave


def _prepare_outputs(
    calls: pd.DataFrame,
    sea: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if calls["port_call_id"].duplicated().any():
        raise RuntimeError("SOURCE_VIEW contains duplicate port_call_id values")

    wave = _add_wave_coverage(_prepare_wave_history(sea))
    frame = _build_cutoff_rows(calls)
    frame = _attach_wave_history(frame, wave)
    frame = _attach_safe_delay_history(frame, calls)
    frame = frame.rename(columns={"prediction_time": "prediction_at"})
    frame["target_arrival_delay_h"] = pd.to_numeric(
        frame["arrival_delay_h"], errors="coerce"
    )
    # Cold start explicite :
    # aucune escale historique admissible implique un compteur égal à zéro.
    frame["vessel_hist_count"] = (
        pd.to_numeric(
            frame.get("vessel_hist_count"),
            errors="coerce",
        )
        .fillna(0)
        .astype("int64")
    )

    frame["vessel_history_available_flag"] = (
        frame["vessel_hist_count"] > 0
    ).astype("int8")

    missing_target = frame["target_arrival_delay_h"].isna()
    arrived = frame["arrived_before_cutoff_flag"].fillna(False)
    sea_missing = ~frame["sea_feature_available_flag"].fillna(False)
    frame["exclusion_reason"] = ""
    frame.loc[missing_target, "exclusion_reason"] += "MISSING_TARGET|"
    frame.loc[arrived, "exclusion_reason"] += "ARRIVED_BEFORE_CUTOFF|"
    frame.loc[sea_missing, "exclusion_reason"] += "NO_SEA_AT_CUTOFF|"
    frame["exclusion_reason"] = frame["exclusion_reason"].str.rstrip("|")
    frame["model_ready_flag"] = ~missing_target & ~arrived & ~sea_missing

    future_sea = frame["observed_at"].notna() & (
        frame["observed_at"] > frame["prediction_at"]
    )
    future_vessel_history = frame["vessel_history_event_time"].notna() & (
        frame["vessel_history_event_time"] >= frame["prediction_at"]
    )
    future_global_history = frame["global_history_event_time"].notna() & (
        frame["global_history_event_time"] >= frame["prediction_at"]
    )
    leakage_count = int(
        future_sea.sum() + future_vessel_history.sum() + future_global_history.sum()
    )
    if leakage_count:
        raise RuntimeError(f"B54F temporal leakage detected: {leakage_count}")

    features, categorical, wave_features = _allowed_feature_columns(frame)
    numeric = [column for column in features if column not in categorical]
    metadata_columns = [
        column
        for column in (
            "port_call_id",
            "source_record_id",
            "prediction_at",
            "planned_eta",
            "model_ready_flag",
        )
        if column in frame.columns
    ]
    model_columns = list(
        dict.fromkeys(metadata_columns + features + ["target_arrival_delay_h"])
    )
    model_ready = frame.loc[frame["model_ready_flag"], model_columns].copy()
    quarantine = frame.loc[~frame["model_ready_flag"]].copy()

    if model_ready["port_call_id"].duplicated().any():
        raise RuntimeError("B54F model-ready output is not one row per port_call_id")

    config = {
        "feature_version": FEATURE_VERSION,
        "grain": "ONE_ROW_PER_PORT_CALL",
        "prediction_cutoff": "PLANNED_ETA_MINUS_24H",
        "split_policy": "NO_SPLIT_IN_B54F_A_B",
        "target_column": "target_arrival_delay_h",
        "feature_columns": features,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "wave_features": wave_features,
        "identifier_columns": sorted(IDENTIFIER_COLUMNS),
        "audit_only_columns": sorted(AUDIT_ONLY_COLUMNS),
    }
    report = {
        "feature_version": FEATURE_VERSION,
        "source_view": SOURCE_VIEW,
        "grain": "ONE_ROW_PER_PORT_CALL",
        "cutoff_hours": CUTOFF_HOURS,
        "calls_loaded": int(len(calls)),
        "full_rows": int(len(frame)),
        "unique_port_calls": int(frame["port_call_id"].nunique()),
        "duplicate_port_call_rows": int(frame["port_call_id"].duplicated().sum()),
        "model_ready_rows": int(len(model_ready)),
        "quarantine_rows": int(len(quarantine)),
        "model_ready_pct": 100.0 * len(model_ready) / max(1, len(frame)),
        "missing_target_rows": int(missing_target.sum()),
        "arrived_before_cutoff_rows": int(arrived.sum()),
        "missing_sea_rows": int(sea_missing.sum()),
        "temporal_leakage_violations": leakage_count,
        "feature_count": int(len(features)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "wave_feature_count": int(len(wave_features)),
        "target_min_h": float(model_ready["target_arrival_delay_h"].min()),
        "target_p50_h": float(model_ready["target_arrival_delay_h"].median()),
        "target_p95_h": float(model_ready["target_arrival_delay_h"].quantile(0.95)),
        "target_max_h": float(model_ready["target_arrival_delay_h"].max()),
        "split_created": False,
        "training_executed": False,
        "timescale_policy": (
            "One-row feature vectors are stored in features.port_call_one_row; "
            "Bronze and core.port_call remain immutable."
        ),
        "decision": {
            "status": "READY_FOR_B54F_B_AUDIT"
            if leakage_count == 0 and len(model_ready) >= 5000
            else "NEED_DATA_REPAIR",
            "next_block": "B54F_B_STRUCTURE_DEPENDENCY_AUDIT",
        },
        "generated_at_utc": datetime.now(timezone.utc),
    }
    return frame, model_ready, quarantine, config, report


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


def _upsert_timescale(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> int:
    if frame.empty:
        return 0
    rows = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        payload = {
            column: _payload_value(values.get(column))
            for column in feature_columns
        }
        rows.append(
            (
                str(values["port_call_id"]),
                values["prediction_at"],
                FEATURE_VERSION,
                _payload_value(values["target_arrival_delay_h"]),
                Json(payload, dumps=lambda obj: json.dumps(obj, default=_json_default)),
            )
        )

    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS features.port_call_one_row (
                    port_call_id UUID NOT NULL REFERENCES core.port_call(port_call_id),
                    prediction_at TIMESTAMPTZ NOT NULL,
                    feature_version TEXT NOT NULL,
                    target_arrival_delay_h REAL NOT NULL,
                    feature_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (port_call_id, feature_version)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS port_call_one_row_prediction_idx
                ON features.port_call_one_row (prediction_at DESC)
                """
            )
            execute_values(
                cursor,
                """
                INSERT INTO features.port_call_one_row (
                    port_call_id,
                    prediction_at,
                    feature_version,
                    target_arrival_delay_h,
                    feature_payload
                ) VALUES %s
                ON CONFLICT (port_call_id, feature_version)
                DO UPDATE SET
                    prediction_at = EXCLUDED.prediction_at,
                    target_arrival_delay_h = EXCLUDED.target_arrival_delay_h,
                    feature_payload = EXCLUDED.feature_payload,
                    created_at = now()
                """,
                rows,
                page_size=1000,
            )
    return len(rows)


def _write_and_upload(
    client,
    frame: pd.DataFrame,
    local_path: Path,
    bucket: str,
    key: str,
) -> str:
    frame.to_parquet(local_path, index=False, compression="zstd")
    client.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/x-parquet"},
    )
    return f"s3://{bucket}/{key}"


def process_one_row_port_call_features(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    checksum, sea_source, signature = _database_signature()
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            run_id, metadata = previous
            return {
                "status": "SKIPPED_ALREADY_PROCESSED",
                "run_id": run_id,
                "checksum": checksum,
                "quality": metadata,
                "outputs": metadata.get("output_uris", {}),
            }

    run_id = _start_run(checksum, signature)
    client = s3_client()
    try:
        calls, sea = _load_frames(sea_source)
        full, model_ready, quarantine, config, report = _prepare_outputs(calls, sea)
        prefix = output_prefix.strip("/") or "version=1"
        keys = {
            "full_no_split": (
                f"datasets/b54f/{prefix}/port_call_one_row_full_no_split_v1.parquet"
            ),
            "model_ready_no_split": (
                f"datasets/b54f/{prefix}/port_call_one_row_model_ready_no_split_v1.parquet"
            ),
            "quarantine": (
                f"quarantine/b54f/{prefix}/port_call_one_row_quarantine_v1.parquet"
            ),
            "feature_config": f"configs/b54f/{prefix}/b54f_feature_config_v1.json",
            "build_report": f"reports/b54f/{prefix}/b54fa_build_report_v1.json",
        }
        with tempfile.TemporaryDirectory(prefix="b54f-a-") as temporary:
            work = Path(temporary)
            output_uris = {
                "full_no_split": _write_and_upload(
                    client, full, work / "full.parquet", output_bucket, keys["full_no_split"]
                ),
                "model_ready_no_split": _write_and_upload(
                    client,
                    model_ready,
                    work / "model_ready.parquet",
                    output_bucket,
                    keys["model_ready_no_split"],
                ),
                "quarantine": _write_and_upload(
                    client,
                    quarantine,
                    work / "quarantine.parquet",
                    output_bucket,
                    keys["quarantine"],
                ),
            }
            timescale_rows = _upsert_timescale(model_ready, config["feature_columns"])
            output_uris["feature_config"] = f"s3://{output_bucket}/{keys['feature_config']}"
            output_uris["build_report"] = f"s3://{output_bucket}/{keys['build_report']}"
            report.update(
                {
                    "run_id": run_id,
                    "checksum": checksum,
                    "sea_source": sea_source,
                    "timescale_rows_upserted": timescale_rows,
                    "output_uris": output_uris,
                    "output_keys": keys,
                }
            )
            config.update({"run_id": run_id, "checksum": checksum})
            config_path = work / "config.json"
            report_path = work / "report.json"
            config_path.write_text(
                json.dumps(config, indent=2, default=_json_default), encoding="utf-8"
            )
            report_path.write_text(
                json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
            )
            client.upload_file(
                str(config_path),
                output_bucket,
                keys["feature_config"],
                ExtraArgs={"ContentType": "application/json"},
            )
            client.upload_file(
                str(report_path),
                output_bucket,
                keys["build_report"],
                ExtraArgs={"ContentType": "application/json"},
            )

        _finish_run(run_id, "SUCCESS", len(model_ready), report)
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
