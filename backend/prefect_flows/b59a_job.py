from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json

from prefect_flows.b59a_core import AUDIT_VERSION, build_reports, clean_json


SOURCE_NAME = "b59a_dynamic_port_call_data_audit"
DATASET_NAME = "dynamic_port_call_target_and_availability_audit"
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


def load_port_calls() -> pd.DataFrame:
    if not _relation_exists("core.port_call"):
        raise RuntimeError("Required source does not exist: core.port_call")
    frame = _query_frame(
        """
        SELECT
            port_call_id::text AS port_call_id,
            port_code,
            terminal_code,
            mmsi,
            imo,
            vessel_name,
            voyage_id,
            planned_eta,
            planned_etb,
            planned_etd,
            actual_ata,
            actual_atb,
            actual_atd,
            cargo_type,
            vessel_type,
            source,
            source_record_id,
            created_at,
            updated_at
        FROM core.port_call
        ORDER BY actual_ata NULLS LAST, port_call_id
        """
    )
    if frame.empty:
        raise RuntimeError("core.port_call is empty")
    return frame


def source_schema_report() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT
            ordinal_position,
            column_name,
            data_type,
            is_nullable,
            CASE
                WHEN column_name IN ('actual_ata', 'actual_atd') THEN 'ACTUAL_EVENT_TIME'
                WHEN column_name IN ('planned_eta', 'planned_etd') THEN 'PLANNED_EVENT_TIME'
                WHEN column_name IN ('created_at', 'updated_at') THEN 'WAREHOUSE_METADATA_TIME'
                ELSE 'ATTRIBUTE'
            END AS semantic_role
        FROM information_schema.columns
        WHERE table_schema='core' AND table_name='port_call'
        ORDER BY ordinal_position
        """
    )


def relation_inventory_report() -> pd.DataFrame:
    relations = [
        "core.port_call",
        "features.port_hourly_state_v1",
        "core.maritime_observation",
        "features.maritime_external_weather_hourly_v1",
        "features.maritime_issue_time_weather_forecast_v1",
        "reference.business_event",
        "lineage.source_availability_event",
    ]
    rows = []
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            for relation in relations:
                schema, name = relation.split(".", 1)
                cursor.execute("SELECT to_regclass(%s)", (relation,))
                exists = cursor.fetchone()[0] is not None
                cursor.execute(
                    """
                    SELECT
                        bool_or(column_name='available_at'),
                        bool_or(column_name='issue_at'),
                        bool_or(column_name IN ('observed_at', 'actual_ata', 'event_time'))
                    FROM information_schema.columns
                    WHERE table_schema=%s AND table_name=%s
                    """,
                    (schema, name),
                )
                available_at, issue_at, event_time = cursor.fetchone()
                rows.append(
                    {
                        "relation": relation,
                        "exists": bool(exists),
                        "has_event_time": bool(event_time),
                        "has_issue_at": bool(issue_at),
                        "has_available_at": bool(available_at),
                    }
                )
    return pd.DataFrame(rows)


def _source_signature(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(AUDIT_VERSION.encode("ascii"))
    selected = frame[
        [
            "port_call_id",
            "imo",
            "planned_eta",
            "planned_etd",
            "actual_ata",
            "actual_atd",
            "cargo_type",
            "source",
        ]
    ].copy()
    hashed = pd.util.hash_pandas_object(selected, index=False)
    digest.update(hashed.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _start_run(checksum: str) -> str:
    metadata = {
        "audit_version": AUDIT_VERSION,
        "orchestrator": "PREFECT",
        "source_modified": False,
        "synthetic_rows_created": 0,
        "landmark_rows_materialized": 0,
        "training_executed": False,
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
                    "postgresql://maritime/core.port_call",
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
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upload_file(client, source: Path, bucket: str, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
    }.get(source.suffix.lower(), "application/octet-stream")
    client.upload_file(
        str(source), bucket, key, ExtraArgs={"ContentType": content_type}
    )
    client.head_object(Bucket=bucket, Key=key)
    return f"s3://{bucket}/{key}"


def _write_readme(path: Path, decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# B59A dynamic port-call data audit",
                "",
                f"Decision: {decision['decision']}",
                "",
                "## Target contract",
                "",
                "Primary target: P(actual_atd > planned_etd + 3 hours | data available at landmark).",
                "Secondary target: remaining hours until actual_atd.",
                "Historical missing ATD values are not treated as right-censoring.",
                "",
                "## Safety",
                "",
                "This audit modifies no Bronze/Core row, materializes no landmark, and trains no model.",
                "Historical ETA/ETD available_at is not captured, so formal live promotion remains blocked.",
                "",
                "## Next block",
                "",
                str(decision["next_block"]),
            ]
        ),
        encoding="utf-8",
    )


def run_b59a_data_audit(force: bool = False) -> dict[str, Any]:
    source = load_port_calls()
    checksum = _source_signature(source)
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
        relation_inventory = relation_inventory_report()
        calls, reports, decision = build_reports(source, relation_inventory)
        reports = {
            "00_port_call_schema.csv": source_schema_report(),
            **reports,
        }
        client = _s3_client()
        uploaded: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b59a-") as temporary:
            output_dir = Path(temporary)
            for name, report in reports.items():
                report.to_csv(output_dir / name, index=False)

            decision_path = output_dir / "13_b59a_final_decision.json"
            decision_path.write_text(
                json.dumps(
                    clean_json(decision),
                    indent=2,
                    ensure_ascii=True,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            _write_readme(output_dir / "README_B59A.md", decision)

            manifest_rows = []
            for path in sorted(output_dir.iterdir()):
                manifest_rows.append(
                    {
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            pd.DataFrame(manifest_rows).to_csv(
                output_dir / "14_artifact_manifest.csv", index=False
            )

            for path in sorted(output_dir.iterdir()):
                prefix = "configs/b59a" if path == decision_path else "reports/b59a"
                key = f"{prefix}/{OUTPUT_PREFIX}/{path.name}"
                uploaded[path.name] = _upload_file(client, path, OUTPUT_BUCKET, key)

        metadata = {
            **decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b59a-dynamic-port-call-data-audit",
            "outputs": uploaded,
            "report_prefix": f"s3://{OUTPUT_BUCKET}/reports/b59a/{OUTPUT_PREFIX}/",
            "config_uri": uploaded["13_b59a_final_decision.json"],
            "landmark_rows_materialized": 0,
        }
        _finish_run(run_id, "SUCCESS", len(calls), metadata)
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
            {"audit_version": AUDIT_VERSION, "orchestrator": "PREFECT"},
            str(exc),
        )
        raise


def verify_b59a_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B59A status: {result.get('status')}")
    metadata = result.get("results") or {}
    if metadata.get("synthetic_rows_created") not in (0, "0", None):
        raise RuntimeError("B59A safety violation: synthetic rows were created")
    if metadata.get("source_modified") not in (False, "false", None):
        raise RuntimeError("B59A safety violation: a source was modified")
    if metadata.get("training_executed") not in (False, "false", None):
        raise RuntimeError("B59A safety violation: training was executed")
    if metadata.get("landmark_rows_materialized") not in (0, "0", None):
        raise RuntimeError("B59A safety violation: landmarks were materialized")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "audit_gates_passed": metadata.get("audit_gates_passed"),
        "retrospective_benchmark_allowed": metadata.get(
            "retrospective_benchmark_allowed"
        ),
        "formal_production_promotion": metadata.get("formal_production_promotion"),
        "next_block": metadata.get("next_block"),
        "safety_contract": "PASS",
    }
