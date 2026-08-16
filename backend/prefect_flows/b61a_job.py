from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import gc
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from psycopg2 import sql
from psycopg2.extras import Json

from prefect_flows.b61a_core import (
    CONTRACT_VERSION,
    DATASET_VERSION,
    EVENT_CONTEXT_VERSION,
    SOURCE_DATASET_VERSION,
    build_governed_dataset,
    clean_json,
)


SOURCE_NAME = "b61a_governed_data_completion"
DATASET_NAME = "maritime_port_call_governed_enrichment_v1"
TARGET_RELATION = "features.maritime_port_call_governed_v1"
REGISTRY_RELATION = "governance.maritime_feature_provenance_v1"
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


def _relation_exists(relation: str) -> bool:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (relation,))
            return cursor.fetchone()[0] is not None


def _required_relations() -> None:
    for relation in ("features.maritime_port_call_landmark_v1", "audit.ingestion_run"):
        if not _relation_exists(relation):
            raise RuntimeError(f"Required relation does not exist: {relation}")


def load_landmarks() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT *
        FROM features.maritime_port_call_landmark_v1
        WHERE dataset_version=%s
        ORDER BY landmark_at, port_call_id
        """,
        (SOURCE_DATASET_VERSION,),
    )
    if frame.empty:
        raise RuntimeError(f"No B60C rows found for {SOURCE_DATASET_VERSION}")
    return frame


def load_event_context() -> pd.DataFrame:
    relation = "features.maritime_port_call_event_context_v1"
    if not _relation_exists(relation):
        return pd.DataFrame()
    return _query_frame(
        """
        SELECT *
        FROM features.maritime_port_call_event_context_v1
        WHERE event_context_version=%s
        ORDER BY landmark_at, port_call_id
        """,
        (EVENT_CONTEXT_VERSION,),
    )


def load_issue_time_forecasts() -> pd.DataFrame:
    relation = "features.maritime_issue_time_weather_forecast_v1"
    if not _relation_exists(relation):
        return pd.DataFrame()
    return _query_frame(
        """
        SELECT
            provider,
            provider_model,
            issue_at,
            available_at,
            valid_at,
            lead_time_h,
            wind_speed_ms,
            wind_direction_deg,
            pressure_hpa,
            visibility_m,
            wave_height_m,
            wave_direction_deg,
            wave_period_s,
            ocean_current_ms,
            sea_surface_temperature_c,
            atmosphere_available_flag,
            wave_available_flag,
            full_weather_available_flag
        FROM features.maritime_issue_time_weather_forecast_v1
        ORDER BY valid_at, available_at, issue_at
        """
    )


def _source_signature(
    landmarks: pd.DataFrame,
    event_context: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> str:
    digest = hashlib.sha256(CONTRACT_VERSION.encode("ascii"))
    landmark_columns = [
        "dataset_version",
        "port_call_id",
        "landmark_at",
        "split",
        "target_departure_delay_h",
        "target_remaining_h",
    ]
    landmark_columns = [column for column in landmark_columns if column in landmarks]
    digest.update(
        pd.util.hash_pandas_object(
            landmarks[landmark_columns], index=False
        ).to_numpy(dtype="uint64").tobytes()
    )
    if not event_context.empty:
        columns = [
            column
            for column in ("event_context_version", "port_call_id", "landmark_at")
            if column in event_context
        ]
        digest.update(
            pd.util.hash_pandas_object(
                event_context[columns], index=False
            ).to_numpy(dtype="uint64").tobytes()
        )
    if not forecasts.empty:
        columns = ["provider", "provider_model", "issue_at", "available_at", "valid_at"]
        digest.update(
            pd.util.hash_pandas_object(
                forecasts[columns], index=False
            ).to_numpy(dtype="uint64").tobytes()
        )
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
                "SELECT COUNT(*) FROM features.maritime_port_call_governed_v1 WHERE dataset_version=%s",
                (DATASET_VERSION,),
            )
            if int(cursor.fetchone()[0]) == 0:
                return None
    return str(row[0]), dict(row[1] or {})


def _start_run(checksum: str) -> str:
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "dataset_version": DATASET_VERSION,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "orchestrator": "PREFECT",
        "source_modified": False,
        "targets_imputed": False,
        "main_dataset_synthetic_rows": 0,
        "test_used_for_feature_engineering": False,
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
                    f"postgresql://maritime/{TARGET_RELATION}",
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


def _materialize_frame(
    frame: pd.DataFrame,
    relation: str,
    version_column: str,
    version_value: str,
    unique_columns: list[str],
    index_name: str,
) -> int:
    schema_name, table_name = relation.split(".", 1)
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
            cursor.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name))
            )
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
                sql.SQL("CREATE TEMP TABLE b61a_stage (LIKE {}.{} INCLUDING DEFAULTS) ON COMMIT DROP").format(
                    sql.Identifier(schema_name), sql.Identifier(table_name)
                )
            )
            column_sql = sql.SQL(", ").join(map(sql.Identifier, columns)).as_string(cursor)
            copy_sql = f"COPY b61a_stage ({column_sql}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
            for start in range(0, len(frame), 2_000):
                chunk = frame.iloc[start : start + 2_000]
                stream = io.StringIO()
                chunk.to_csv(stream, index=False, header=False, na_rep="\\N")
                stream.seek(0)
                cursor.copy_expert(copy_sql, stream)
            cursor.execute(
                sql.SQL("DELETE FROM {}.{} WHERE {}=%s").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.Identifier(version_column),
                ),
                (version_value,),
            )
            cursor.execute(
                sql.SQL("INSERT INTO {}.{} ({}) SELECT {} FROM b61a_stage").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                )
            )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{} ({})").format(
                    sql.Identifier(index_name),
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(map(sql.Identifier, unique_columns)),
                )
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
            "# B61A governed port-call data completion",
            "",
            f"Decision: {decision['decision']}",
            "",
            "B61A preserves every real B60C landmark and target. It adds deterministic",
            "physical features, train-only local thresholds, operational pressure",
            "proxies, B60C-H event context and B58C-D issue-time forecasts when a",
            "historically valid forecast exists.",
            "",
            "Retrospective weather remains research-only. Missing operational facts",
            "such as berth capacity, incidents, priorities and ETA revision histories",
            "are never fabricated. Synthetic stress scenarios remain outside this",
            "dataset and cannot be used for training, validation or testing.",
        ]
    )


def run_b61a_governed_completion(force: bool = False) -> dict[str, Any]:
    _required_relations()
    landmarks = load_landmarks()
    event_context = load_event_context()
    forecasts = load_issue_time_forecasts()
    checksum = _source_signature(landmarks, event_context, forecasts)
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
        _update_progress(
            run_id,
            "LOADING_AND_GOVERNING_SOURCES",
            b60c_rows=len(landmarks),
            event_context_rows=len(event_context),
            issue_time_forecast_rows=len(forecasts),
        )
        result = build_governed_dataset(landmarks, event_context, forecasts)
        dataset = result.dataset
        dataset["materialization_run_id"] = run_id
        del landmarks, event_context, forecasts
        gc.collect()
        registry = result.reports["02_feature_registry.csv"].copy()
        registry.insert(0, "dataset_version", DATASET_VERSION)
        registry["materialization_run_id"] = run_id

        _update_progress(
            run_id,
            "MATERIALIZING_GOVERNED_DATASET",
            rows=len(dataset),
            calls=dataset["port_call_id"].nunique(),
            columns=len(dataset.columns),
        )
        materialized_rows = _materialize_frame(
            dataset,
            TARGET_RELATION,
            "dataset_version",
            DATASET_VERSION,
            ["dataset_version", "port_call_id", "landmark_at"],
            "b61a_dataset_version_call_time_uidx",
        )
        registry_rows = _materialize_frame(
            registry,
            REGISTRY_RELATION,
            "dataset_version",
            DATASET_VERSION,
            ["dataset_version", "feature"],
            "b61a_registry_version_feature_uidx",
        )

        _update_progress(run_id, "WRITING_VERSIONED_ARTIFACTS")
        client = _s3_client()
        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b61a-") as temporary:
            directory = Path(temporary)
            dataset_path = directory / "maritime_port_call_governed_v1.parquet"
            dataset.to_parquet(dataset_path, index=False, compression="zstd")
            outputs[dataset_path.name] = _upload(
                client,
                dataset_path,
                f"datasets/b61a/{OUTPUT_PREFIX}/{dataset_path.name}",
            )
            for name, report in result.reports.items():
                path = directory / name
                report.to_csv(path, index=False)
                outputs[name] = _upload(
                    client, path, f"reports/b61a/{OUTPUT_PREFIX}/{name}"
                )
            decision_path = directory / "10_b61a_final_decision.json"
            decision_path.write_text(
                json.dumps(clean_json(result.decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[decision_path.name] = _upload(
                client,
                decision_path,
                f"configs/b61a/{OUTPUT_PREFIX}/{decision_path.name}",
            )
            feature_path = directory / "11_feature_sets.json"
            feature_path.write_text(
                json.dumps(clean_json(result.feature_sets), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[feature_path.name] = _upload(
                client,
                feature_path,
                f"configs/b61a/{OUTPUT_PREFIX}/{feature_path.name}",
            )
            readme_path = directory / "README_B61A.md"
            readme_path.write_text(_readme(result.decision), encoding="utf-8")
            outputs[readme_path.name] = _upload(
                client,
                readme_path,
                f"reports/b61a/{OUTPUT_PREFIX}/{readme_path.name}",
            )
            manifest_rows = [
                {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(directory.iterdir())
            ]
            manifest_path = directory / "12_artifact_manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            outputs[manifest_path.name] = _upload(
                client,
                manifest_path,
                f"reports/b61a/{OUTPUT_PREFIX}/{manifest_path.name}",
            )

        metadata = {
            **result.decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b61a-governed-data-completion",
            "materialized_relation": TARGET_RELATION,
            "materialized_rows": materialized_rows,
            "feature_registry_relation": REGISTRY_RELATION,
            "feature_registry_rows": registry_rows,
            "dataset_uri": outputs["maritime_port_call_governed_v1.parquet"],
            "feature_sets_uri": outputs["11_feature_sets.json"],
            "report_prefix": f"s3://{OUTPUT_BUCKET}/reports/b61a/{OUTPUT_PREFIX}/",
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


def verify_b61a_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B61A status: {result.get('status')}")
    metadata = result.get("results") or {}
    expected = "READY_FOR_GOVERNED_MULTITASK_MODELING"
    if metadata.get("decision") != expected:
        raise RuntimeError(f"B61A quality gates failed: {metadata.get('decision')}")
    if metadata.get("main_dataset_synthetic_rows") not in (0, "0"):
        raise RuntimeError("B61A main dataset contains synthetic rows")
    if metadata.get("targets_imputed") not in (False, "false"):
        raise RuntimeError("B61A targets were imputed")
    if metadata.get("test_used_for_feature_engineering") not in (False, "false"):
        raise RuntimeError("B61A used TEST during feature engineering")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "row_count": metadata.get("row_count"),
        "vessel_calls": metadata.get("vessel_calls"),
        "core_features": metadata.get("core_features"),
        "research_features": metadata.get("research_features"),
        "issue_time_features": metadata.get("issue_time_features"),
        "issue_time_history_ready": metadata.get("issue_time_history_ready"),
        "production_promotion_allowed": metadata.get("production_promotion_allowed"),
        "quality_gates_passed": metadata.get("quality_gates_passed"),
        "next_block": metadata.get("next_block"),
        "safety_contract": "PASS",
    }
