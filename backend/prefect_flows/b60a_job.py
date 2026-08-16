from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from psycopg2 import sql
from psycopg2.extras import Json

from prefect_flows.b60a_core import (
    CONTRACT_VERSION,
    DATASET_VERSION,
    build_multitask_dataset,
    clean_json,
    feature_sets_json,
)


SOURCE_NAME = "b60a_multitask_hourly_dataset"
DATASET_NAME = "maritime_multitask_hourly_v1"
TARGET_RELATION = "features.maritime_multitask_hourly_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"


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


def _required_relations() -> None:
    required = [
        "features.port_hourly_state_v1",
        "core.maritime_observation",
        "features.maritime_external_weather_hourly_v1",
        "core.port_call",
        "reference.business_event",
        "audit.ingestion_run",
    ]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            for relation in required:
                cursor.execute("SELECT to_regclass(%s)", (relation,))
                if cursor.fetchone()[0] is None:
                    raise RuntimeError(f"Required relation does not exist: {relation}")


def load_hourly_sources() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT
            p.as_of_time,
            1::smallint AS port_source_present,
            CASE WHEN w.observed_at IS NOT NULL THEN 1 ELSE 0 END::smallint
                AS wave_source_present,
            CASE WHEN e.observed_at IS NOT NULL THEN 1 ELSE 0 END::smallint
                AS external_source_present,
            p.arrivals_prev_1h,
            p.arrivals_last_6h,
            p.arrivals_last_24h,
            p.arrivals_last_168h,
            p.departures_prev_1h,
            p.departures_last_6h,
            p.departures_last_24h,
            p.delayed_gt3_last_24h,
            p.mean_arrival_delay_last_24h,
            p.vessels_in_port_observed,
            p.weather_available_flag,
            p.target_arrivals_next_6h AS source_target_arrivals_next_6h,
            p.target_arrivals_next_12h AS source_target_arrivals_next_12h,
            p.target_arrivals_next_24h AS source_target_arrivals_next_24h,
            w.wave_height_m,
            w.wave_period_s,
            w.wave_direction_deg,
            e.wind_speed_ms AS ext_wind_speed_ms,
            e.wind_direction_deg AS ext_wind_direction_deg,
            e.surface_current_ms AS ext_surface_current_ms,
            e.visibility_m AS ext_visibility_m,
            e.pressure_hpa AS ext_pressure_hpa,
            e.wind_gusts_10m AS ext_wind_gusts_10m,
            e.temperature_2m AS ext_temperature_2m,
            e.relative_humidity_2m AS ext_relative_humidity_2m,
            e.precipitation AS ext_precipitation,
            e.cloud_cover AS ext_cloud_cover,
            e.sea_surface_temperature AS ext_sea_surface_temperature,
            e.availability_semantics AS ext_availability_semantics
        FROM features.port_hourly_state_v1 p
        LEFT JOIN core.maritime_observation w
          ON w.observed_at=p.as_of_time
         AND w.source='copernicus_ibi_wave'
        LEFT JOIN features.maritime_external_weather_hourly_v1 e
          ON e.observed_at=p.as_of_time
         AND e.dataset_version='b58cb-external-weather-hourly-v1'
        ORDER BY p.as_of_time
        """
    )


def load_arrival_events() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT
            port_call_id::text AS port_call_id,
            actual_ata,
            port_code,
            terminal_code,
            mmsi::text AS mmsi,
            imo::text AS imo,
            vessel_name,
            voyage_id,
            cargo_type,
            vessel_type,
            source
        FROM core.port_call
        WHERE actual_ata IS NOT NULL
        ORDER BY actual_ata, port_call_id
        """
    )


def load_business_events() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT
            event_id,
            event_name,
            event_type,
            start_date,
            end_date,
            affected_flow,
            knowledge_policy,
            source,
            confidence
        FROM reference.business_event
        ORDER BY start_date, event_id
        """
    )


def issue_time_overlap_report(hourly_end: pd.Timestamp) -> pd.DataFrame:
    return _query_frame(
        """
        SELECT
            COUNT(*)::bigint AS rows,
            COUNT(DISTINCT issue_at)::bigint AS issues,
            MIN(issue_at) AS first_issue_at,
            MAX(issue_at) AS last_issue_at,
            MIN(valid_at) AS first_valid_at,
            MAX(valid_at) AS last_valid_at,
            COUNT(*) FILTER (WHERE valid_at <= %s)::bigint AS historical_overlap_rows,
            'LIVE_ONLY_SHADOW_NOT_A_HISTORICAL_TRAINING_FEATURE'::text AS model_role
        FROM features.maritime_issue_time_weather_forecast_v1
        """,
        (hourly_end,),
    )


def _source_signature(hourly: pd.DataFrame, events: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(CONTRACT_VERSION.encode("ascii"))
    hourly_columns = [
        "as_of_time",
        "arrivals_prev_1h",
        "departures_prev_1h",
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "ext_wind_speed_ms",
        "ext_visibility_m",
    ]
    hourly_hash = pd.util.hash_pandas_object(hourly[hourly_columns], index=False)
    digest.update(hourly_hash.to_numpy(dtype="uint64").tobytes())
    event_hash = pd.util.hash_pandas_object(
        events[["port_call_id", "actual_ata"]], index=False
    )
    digest.update(event_hash.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, metadata
                FROM audit.ingestion_run
                WHERE source_name=%s AND dataset_name=%s AND checksum=%s
                  AND status='SUCCESS'
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute("SELECT to_regclass(%s)", (TARGET_RELATION,))
            if cursor.fetchone()[0] is None:
                return None
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM features.maritime_multitask_hourly_v1
                WHERE dataset_version=%s
                """,
                (DATASET_VERSION,),
            )
            if int(cursor.fetchone()[0]) == 0:
                return None
    return str(row[0]), dict(row[1] or {})


def _start_run(checksum: str) -> str:
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "dataset_version": DATASET_VERSION,
        "orchestrator": "PREFECT",
        "source_modified": False,
        "synthetic_rows_created": 0,
        "synthetic_targets_created": 0,
        "target_imputation_used": False,
        "training_executed": False,
        "production_promotion_allowed": False,
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
                    "postgresql://maritime/features.maritime_multitask_hourly_v1",
                    checksum,
                    Json(metadata),
                ),
            )
            return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    progress = clean_json(
        {"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details}
    )
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
                (Json({"progress": progress}), run_id),
            )


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
                    Json(clean_json(metadata)),
                    error_message,
                    run_id,
                ),
            )


def _sql_type(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series.dtype):
        return "TIMESTAMPTZ"
    if is_bool_dtype(series.dtype):
        return "BOOLEAN"
    if is_numeric_dtype(series.dtype):
        return "DOUBLE PRECISION"
    return "TEXT"


def _materialize_dataset(frame: pd.DataFrame) -> int:
    schema_name, table_name = TARGET_RELATION.split(".", 1)
    columns = list(frame.columns)
    for column in columns:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", column):
            raise ValueError(f"Unsafe materialized column name: {column}")
    definitions = [
        sql.SQL("{} {}").format(sql.Identifier(column), sql.SQL(_sql_type(frame[column])))
        for column in columns
    ]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(definitions),
                )
            )
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=%s AND table_name=%s
                """,
                (schema_name, table_name),
            )
            existing = {row[0] for row in cursor.fetchall()}
            for definition, column in zip(definitions, columns):
                if column not in existing:
                    cursor.execute(
                        sql.SQL("ALTER TABLE {}.{} ADD COLUMN {}").format(
                            sql.Identifier(schema_name),
                            sql.Identifier(table_name),
                            definition,
                        )
                    )
            cursor.execute(
                sql.SQL("CREATE TEMP TABLE b60a_stage (LIKE {}.{} INCLUDING DEFAULTS) ON COMMIT DROP").format(
                    sql.Identifier(schema_name), sql.Identifier(table_name)
                )
            )
            column_sql = sql.SQL(", ").join(map(sql.Identifier, columns)).as_string(cursor)
            copy_sql = f"COPY b60a_stage ({column_sql}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
            for start in range(0, len(frame), 5_000):
                chunk = frame.iloc[start : start + 5_000].copy()
                stream = io.StringIO()
                chunk.to_csv(stream, index=False, header=False, na_rep="\\N")
                stream.seek(0)
                cursor.copy_expert(copy_sql, stream)
            cursor.execute(
                sql.SQL("DELETE FROM {}.{} WHERE dataset_version=%s").format(
                    sql.Identifier(schema_name), sql.Identifier(table_name)
                ),
                (DATASET_VERSION,),
            )
            cursor.execute(
                sql.SQL("INSERT INTO {}.{} ({}) SELECT {} FROM b60a_stage").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                )
            )
            cursor.execute(
                sql.SQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS b60a_multitask_hourly_version_time_uidx "
                    "ON {}.{} (dataset_version, as_of_time)"
                ).format(sql.Identifier(schema_name), sql.Identifier(table_name))
            )
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS b60a_multitask_hourly_arrival_split_idx "
                    "ON {}.{} (split_arrival, as_of_time)"
                ).format(sql.Identifier(schema_name), sql.Identifier(table_name))
            )
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS b60a_multitask_hourly_wave_split_idx "
                    "ON {}.{} (split_wave, as_of_time)"
                ).format(sql.Identifier(schema_name), sql.Identifier(table_name))
            )
            cursor.execute(
                sql.SQL("ANALYZE {}.{}").format(
                    sql.Identifier(schema_name), sql.Identifier(table_name)
                )
            )
    return len(frame)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upload(client, path: Path, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/octet-stream",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(
        str(path), OUTPUT_BUCKET, key, ExtraArgs={"ContentType": content_type}
    )
    client.head_object(Bucket=OUTPUT_BUCKET, Key=key)
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _readme(decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B60A maritime multitask hourly dataset",
            "",
            f"Decision: {decision['decision']}",
            "",
            "One continuous UTC hourly grid supports two scientifically separate tasks:",
            "TIR operational-unit arrival forecasting until the detected April 2025 coverage break,",
            "and wave forecasting through March 2026.",
            "",
            "Targets are observed and never imputed. No synthetic row is created.",
            "External ERA5 weather is lagged and research-only because it was not",
            "available at the historical issue time. Live B58C-D snapshots have no",
            "historical overlap and are intentionally excluded from this training set.",
            "",
            "The dataset is ready for retrospective rolling-origin benchmarks only.",
            "Production promotion remains false until prospective issue-time history",
            "and reproducible model gates are available.",
            "The arrival stream is not labelled as ship arrivals because vessel identity",
            "fields are absent and vessel_type is overwhelmingly TIR.",
        ]
    )


def run_b60a_dataset_build(force: bool = False) -> dict[str, Any]:
    _required_relations()
    hourly = load_hourly_sources()
    arrivals = load_arrival_events()
    business_events = load_business_events()
    checksum = _source_signature(hourly, arrivals)
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
        _update_progress(run_id, "BUILDING_POINT_IN_TIME_FEATURES", source_rows=len(hourly))
        result = build_multitask_dataset(hourly, arrivals, business_events)
        dataset = result.dataset.copy()
        dataset["materialization_run_id"] = run_id
        event_sequence = result.event_sequence.copy()
        event_sequence["materialization_run_id"] = run_id

        _update_progress(
            run_id,
            "MATERIALIZING_TIMESCALE",
            rows=len(dataset),
            columns=len(dataset.columns),
        )
        materialized_rows = _materialize_dataset(dataset)

        _update_progress(run_id, "WRITING_VERSIONED_ARTIFACTS")
        client = _s3_client()
        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b60a-") as temporary:
            directory = Path(temporary)
            dataset_path = directory / "maritime_multitask_hourly_v1.parquet"
            events_path = directory / "arrival_event_sequence_v1.parquet"
            dataset.to_parquet(dataset_path, index=False, compression="zstd")
            event_sequence.to_parquet(events_path, index=False, compression="zstd")
            outputs[dataset_path.name] = _upload(
                client,
                dataset_path,
                f"datasets/b60a/{OUTPUT_PREFIX}/{dataset_path.name}",
            )
            outputs[events_path.name] = _upload(
                client,
                events_path,
                f"datasets/b60a/{OUTPUT_PREFIX}/{events_path.name}",
            )

            reports = dict(result.reports)
            reports["09_issue_time_forecast_overlap.csv"] = issue_time_overlap_report(
                dataset["as_of_time"].max()
            )
            for name, report in reports.items():
                path = directory / name
                report.to_csv(path, index=False)
                outputs[name] = _upload(
                    client, path, f"reports/b60a/{OUTPUT_PREFIX}/{name}"
                )

            decision_path = directory / "10_b60a_final_decision.json"
            decision_path.write_text(
                json.dumps(clean_json(result.decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[decision_path.name] = _upload(
                client,
                decision_path,
                f"configs/b60a/{OUTPUT_PREFIX}/{decision_path.name}",
            )
            feature_set_path = directory / "11_feature_sets.json"
            feature_set_path.write_text(
                feature_sets_json(result.feature_sets), encoding="utf-8"
            )
            outputs[feature_set_path.name] = _upload(
                client,
                feature_set_path,
                f"configs/b60a/{OUTPUT_PREFIX}/{feature_set_path.name}",
            )
            readme_path = directory / "README_B60A.md"
            readme_path.write_text(_readme(result.decision), encoding="utf-8")
            outputs[readme_path.name] = _upload(
                client, readme_path, f"reports/b60a/{OUTPUT_PREFIX}/{readme_path.name}"
            )

            manifest_rows = []
            for path in sorted(directory.iterdir()):
                manifest_rows.append(
                    {
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            manifest_path = directory / "12_artifact_manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            outputs[manifest_path.name] = _upload(
                client,
                manifest_path,
                f"reports/b60a/{OUTPUT_PREFIX}/{manifest_path.name}",
            )

        metadata = {
            **result.decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b60a-maritime-multitask-hourly-dataset",
            "materialized_relation": TARGET_RELATION,
            "materialized_rows": materialized_rows,
            "dataset_uri": outputs["maritime_multitask_hourly_v1.parquet"],
            "event_sequence_uri": outputs["arrival_event_sequence_v1.parquet"],
            "feature_sets_uri": outputs["11_feature_sets.json"],
            "report_prefix": f"s3://{OUTPUT_BUCKET}/reports/b60a/{OUTPUT_PREFIX}/",
            "outputs": outputs,
        }
        _finish_run(run_id, "SUCCESS", len(dataset), metadata)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "results": clean_json(metadata),
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "contract_version": CONTRACT_VERSION,
                "dataset_version": DATASET_VERSION,
                "orchestrator": "PREFECT",
            },
            str(exc),
        )
        raise


def verify_b60a_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B60A status: {result.get('status')}")
    metadata = result.get("results") or {}
    if metadata.get("decision") != "READY_FOR_ADVANCED_TIME_SERIES_BENCHMARK":
        raise RuntimeError(f"B60A quality gates failed: {metadata.get('decision')}")
    for field in ("synthetic_rows_created", "synthetic_targets_created"):
        if metadata.get(field) not in (0, "0"):
            raise RuntimeError(f"B60A safety violation: {field}")
    if metadata.get("target_imputation_used") not in (False, "false"):
        raise RuntimeError("B60A safety violation: targets were imputed")
    if metadata.get("production_promotion_allowed") not in (False, "false"):
        raise RuntimeError("B60A cannot promote a retrospective dataset to production")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "row_count": metadata.get("row_count"),
        "column_count": metadata.get("column_count"),
        "arrival_coverage_break_at": metadata.get("arrival_coverage_break_at"),
        "arrival_test_rows": metadata.get("arrival_test_rows"),
        "wave_test_rows": metadata.get("wave_test_rows"),
        "quality_gates_passed": metadata.get("quality_gates_passed"),
        "next_block": metadata.get("next_block"),
        "safety_contract": "PASS",
    }
