from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from psycopg2 import sql
from psycopg2.extras import Json

from prefect_flows.b60ch_core import (
    CONTEXT_VERSION,
    REGISTRY_VERSION,
    SOURCE_DATASET_VERSION,
    build_event_registry,
    build_historical_event_intelligence,
    clean_json,
)


SOURCE_NAME = "b60ch_historical_event_intelligence"
DATASET_NAME = "maritime_historical_event_context_v1"
REGISTRY_RELATION = "reference.business_event_history_v1"
CONTEXT_RELATION = "features.maritime_port_call_event_context_v1"
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
    required = (
        "features.maritime_port_call_landmark_v1",
        "audit.ingestion_run",
    )
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            for relation in required:
                cursor.execute("SELECT to_regclass(%s)", (relation,))
                if cursor.fetchone()[0] is None:
                    raise RuntimeError(f"Required relation does not exist: {relation}")


def load_b60c_landmarks() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT
            dataset_version,
            data_origin,
            port_call_id,
            landmark_at,
            actual_ata,
            split,
            early_warning_eligible,
            target_departure_delay_h,
            target_delay_gt_3h,
            target_delay_gt_6h,
            target_total_stay_h
        FROM features.maritime_port_call_landmark_v1
        WHERE dataset_version=%s
        ORDER BY landmark_at, port_call_id
        """,
        (SOURCE_DATASET_VERSION,),
    )
    if frame.empty:
        raise RuntimeError(f"No B60C landmarks found for {SOURCE_DATASET_VERSION}")
    return frame


def _source_signature(landmarks: pd.DataFrame, registry: pd.DataFrame) -> str:
    digest = hashlib.sha256(
        f"{REGISTRY_VERSION}|{CONTEXT_VERSION}|association-v2-stabilized".encode("ascii")
    )
    landmark_columns = [
        "dataset_version",
        "port_call_id",
        "landmark_at",
        "split",
        "target_departure_delay_h",
        "target_delay_gt_3h",
        "target_delay_gt_6h",
        "target_total_stay_h",
    ]
    digest.update(
        pd.util.hash_pandas_object(
            landmarks[landmark_columns], index=False
        ).to_numpy(dtype="uint64").tobytes()
    )
    digest.update(
        pd.util.hash_pandas_object(
            registry[
                [
                    "event_id",
                    "event_family",
                    "start_at",
                    "end_at",
                    "known_at",
                    "calendar_role",
                    "source_uri",
                ]
            ],
            index=False,
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
            for relation, version_column, version_value in (
                (REGISTRY_RELATION, "registry_version", REGISTRY_VERSION),
                (CONTEXT_RELATION, "event_context_version", CONTEXT_VERSION),
            ):
                cursor.execute("SELECT to_regclass(%s)", (relation,))
                if cursor.fetchone()[0] is None:
                    return None
                schema_name, table_name = relation.split(".", 1)
                cursor.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}.{} WHERE {}=%s").format(
                        sql.Identifier(schema_name),
                        sql.Identifier(table_name),
                        sql.Identifier(version_column),
                    ),
                    (version_value,),
                )
                if int(cursor.fetchone()[0]) == 0:
                    return None
    return str(row[0]), dict(row[1] or {})


def _start_run(checksum: str) -> str:
    metadata = {
        "registry_version": REGISTRY_VERSION,
        "context_version": CONTEXT_VERSION,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "orchestrator": "PREFECT",
        "source_modified": False,
        "synthetic_event_rows": 0,
        "synthetic_target_rows": 0,
        "test_used_for_selection": False,
        "causal_claim_allowed": False,
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
                    f"postgresql://maritime/{CONTEXT_RELATION}",
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
    stage_table: str,
    unique_index_name: str,
    unique_columns: tuple[str, ...],
) -> int:
    schema_name, table_name = relation.split(".", 1)
    columns = list(frame.columns)
    for name in (*columns, stage_table, unique_index_name):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Unsafe SQL identifier: {name}")
    definitions = [
        sql.SQL("{} {}").format(
            sql.Identifier(column), sql.SQL(_sql_type(frame[column]))
        )
        for column in columns
    ]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(schema_name)
                )
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
                sql.SQL("CREATE TEMP TABLE {} (LIKE {}.{} INCLUDING DEFAULTS) ON COMMIT DROP").format(
                    sql.Identifier(stage_table),
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                )
            )
            column_sql = sql.SQL(", ").join(map(sql.Identifier, columns)).as_string(cursor)
            copy_sql = (
                f"COPY {stage_table} ({column_sql}) FROM STDIN "
                "WITH (FORMAT CSV, NULL '\\N')"
            )
            for start in range(0, len(frame), 5_000):
                stream = io.StringIO()
                frame.iloc[start : start + 5_000].to_csv(
                    stream, index=False, header=False, na_rep="\\N"
                )
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
                sql.SQL("INSERT INTO {}.{} ({}) SELECT {} FROM {}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.Identifier(stage_table),
                )
            )
            cursor.execute(
                sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{} ({})").format(
                    sql.Identifier(unique_index_name),
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
            "# B60C-H historical event intelligence",
            "",
            f"Decision: {decision['decision']}",
            "",
            "This block creates a sourced 2020-2025 Morocco/Tanger Med event registry",
            "and point-in-time event context for every real B60C landmark.",
            "",
            "Fixed and officially announced calendar events may become predictive",
            "features only when known_at is not later than the landmark. Marhaba, COVID",
            "and capacity-transition rows remain research-only because their historical",
            "publication availability is not captured.",
            "",
            "Effect tables are matched associations, not causal estimates. Latent",
            "periods are investigation candidates only; they are never inserted as",
            "event labels or model features. No target or event is synthesized.",
        ]
    )


def run_b60ch_event_intelligence(force: bool = False) -> dict[str, Any]:
    _required_relations()
    landmarks = load_b60c_landmarks()
    registry = build_event_registry()
    checksum = _source_signature(landmarks, registry)
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
            "BUILDING_SOURCED_EVENT_REGISTRY",
            source_landmarks=len(landmarks),
            source_calls=landmarks["port_call_id"].nunique(),
            registry_events=len(registry),
        )
        result = build_historical_event_intelligence(landmarks, registry)
        materialized_registry = result.registry.copy()
        materialized_registry["materialization_run_id"] = run_id
        materialized_context = result.context.copy()
        materialized_context["materialization_run_id"] = run_id

        _update_progress(
            run_id,
            "MATERIALIZING_EVENT_REGISTRY_AND_CONTEXT",
            registry_events=len(materialized_registry),
            context_rows=len(materialized_context),
            event_features=len(result.feature_columns),
        )
        registry_rows = _materialize_frame(
            materialized_registry,
            REGISTRY_RELATION,
            "registry_version",
            REGISTRY_VERSION,
            "b60ch_registry_stage",
            "b60ch_event_registry_version_event_uidx",
            ("registry_version", "event_id"),
        )
        context_rows = _materialize_frame(
            materialized_context,
            CONTEXT_RELATION,
            "event_context_version",
            CONTEXT_VERSION,
            "b60ch_context_stage",
            "b60ch_event_context_version_call_time_uidx",
            ("event_context_version", "port_call_id", "landmark_at"),
        )

        _update_progress(
            run_id,
            "WRITING_AUDIT_ARTIFACTS",
            association_rows=len(result.reports["04_call_event_associations.csv"]),
            latent_candidates=len(result.reports["06_latent_period_candidates.csv"]),
        )
        client = _s3_client()
        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b60ch-") as temporary:
            directory = Path(temporary)
            registry_path = directory / "business_event_history_v1.parquet"
            context_path = directory / "maritime_port_call_event_context_v1.parquet"
            materialized_registry.to_parquet(registry_path, index=False, compression="zstd")
            materialized_context.to_parquet(context_path, index=False, compression="zstd")
            outputs[registry_path.name] = _upload(
                client,
                registry_path,
                f"datasets/b60ch/{OUTPUT_PREFIX}/{registry_path.name}",
            )
            outputs[context_path.name] = _upload(
                client,
                context_path,
                f"datasets/b60ch/{OUTPUT_PREFIX}/{context_path.name}",
            )
            for name, report in result.reports.items():
                path = directory / name
                report.to_csv(path, index=False)
                outputs[name] = _upload(
                    client, path, f"reports/b60ch/{OUTPUT_PREFIX}/{name}"
                )
            decision_path = directory / "09_b60ch_final_decision.json"
            decision_path.write_text(
                json.dumps(clean_json(result.decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[decision_path.name] = _upload(
                client,
                decision_path,
                f"configs/b60ch/{OUTPUT_PREFIX}/{decision_path.name}",
            )
            features_path = directory / "10_event_feature_columns.json"
            features_path.write_text(
                json.dumps(result.feature_columns, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[features_path.name] = _upload(
                client,
                features_path,
                f"configs/b60ch/{OUTPUT_PREFIX}/{features_path.name}",
            )
            readme_path = directory / "README_B60CH.md"
            readme_path.write_text(_readme(result.decision), encoding="utf-8")
            outputs[readme_path.name] = _upload(
                client,
                readme_path,
                f"reports/b60ch/{OUTPUT_PREFIX}/{readme_path.name}",
            )
            manifest = []
            for path in sorted(directory.iterdir()):
                manifest.append(
                    {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                )
            manifest_path = directory / "11_artifact_manifest.csv"
            pd.DataFrame(manifest).to_csv(manifest_path, index=False)
            outputs[manifest_path.name] = _upload(
                client,
                manifest_path,
                f"reports/b60ch/{OUTPUT_PREFIX}/{manifest_path.name}",
            )

        metadata = {
            **result.decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b60ch-historical-event-intelligence",
            "registry_relation": REGISTRY_RELATION,
            "context_relation": CONTEXT_RELATION,
            "registry_rows": registry_rows,
            "context_rows": context_rows,
            "registry_uri": outputs["business_event_history_v1.parquet"],
            "context_uri": outputs["maritime_port_call_event_context_v1.parquet"],
            "report_prefix": f"s3://{OUTPUT_BUCKET}/reports/b60ch/{OUTPUT_PREFIX}/",
            "outputs": outputs,
        }
        _finish_run(run_id, "SUCCESS", context_rows, metadata)
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
                "registry_version": REGISTRY_VERSION,
                "context_version": CONTEXT_VERSION,
                "source_dataset_version": SOURCE_DATASET_VERSION,
                "orchestrator": "PREFECT",
            },
            str(exc),
        )
        raise


def verify_b60ch_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B60C-H status: {result.get('status')}")
    metadata = result.get("results") or {}
    expected = "READY_FOR_B60C_V2_EVENT_ENRICHMENT"
    if metadata.get("decision") != expected:
        raise RuntimeError(f"B60C-H quality gates failed: {metadata.get('decision')}")
    for field in (
        "synthetic_event_rows",
        "synthetic_target_rows",
        "latent_candidates_inserted_as_events",
    ):
        if metadata.get(field) not in (0, "0"):
            raise RuntimeError(f"B60C-H safety violation: {field}")
    for field in (
        "test_used_for_selection",
        "causal_claim_allowed",
        "production_promotion_allowed",
    ):
        if metadata.get(field) not in (False, "false"):
            raise RuntimeError(f"B60C-H scientific contract violation: {field}")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "registry_events": metadata.get("registry_events"),
        "event_families": metadata.get("event_families"),
        "context_rows": metadata.get("event_context_rows"),
        "event_feature_count": metadata.get("event_feature_count"),
        "association_rows": metadata.get("association_rows"),
        "latent_period_candidates": metadata.get("latent_period_candidates"),
        "quality_gates_passed": metadata.get("quality_gates_passed"),
        "next_block": metadata.get("next_block"),
        "safety_contract": "PASS",
    }
