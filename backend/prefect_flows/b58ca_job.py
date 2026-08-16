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

from prefect_flows.b58ca_core import (
    AUDIT_VERSION,
    WEATHER_VARIABLES,
    build_reports,
    clean_json,
)


SOURCE_NAME = "b58ca_prefect_missingness_audit"
DATASET_NAME = "maritime_weather_missingness_diagnostics"
SOURCE_TABLE = "core.maritime_observation"
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


def _relation_exists(schema: str, relation: str) -> bool:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (f"{schema}.{relation}",))
            return cursor.fetchone()[0] is not None


def load_source() -> pd.DataFrame:
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


def _source_signature(frame: pd.DataFrame) -> str:
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
        "orchestrator": "PREFECT",
        "synthetic_rows_created": 0,
        "source_modified": False,
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
    payload = clean_json(metadata)
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at=now(), status=%s, row_count=%s,
                    metadata=metadata || %s, error_message=%s
                WHERE run_id=%s
                """,
                (status, row_count, Json(payload), error_message, run_id),
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
    absent = ", ".join(decision["structurally_absent_variables"]) or "none"
    partial = ", ".join(decision["partially_missing_variables"]) or "none"
    path.write_text(
        "\n".join(
            [
                "# B58C-A Prefect weather missingness audit",
                "",
                f"Decision: {decision['decision']}",
                "",
                "## Scientific interpretation",
                "",
                f"- Structurally absent variables: {absent}",
                f"- Partial-gap candidates: {partial}",
                f"- Wave track complete: {decision['wave_track_complete']}",
                "- Thresholds are integrity constraints, never measurement generators.",
                "- MAR versus MNAR cannot be proven from observed data alone.",
                "- Future imputation must fit on TRAIN and be scored on real held-out values.",
                "",
                "## Safety contract",
                "",
                "This flow is read-only for Bronze/Core, creates zero synthetic rows, and does no training.",
                "Historical replay and production promotion remain blocked without trustworthy available_at.",
                "",
                "## Next block",
                "",
                str(decision["next_block"]),
            ]
        ),
        encoding="utf-8",
    )


def run_missingness_audit(
    output_bucket: str = OUTPUT_BUCKET,
    output_prefix: str = OUTPUT_PREFIX,
    force: bool = False,
) -> dict[str, Any]:
    source = load_source()
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
        hourly, reports, decision = build_reports(source)
        reports = {"00_source_schema.csv": source_schema_report(), **reports}
        client = _s3_client()
        uploaded: dict[str, str] = {}

        with tempfile.TemporaryDirectory(prefix="b58ca-") as temporary:
            output_dir = Path(temporary)
            for name, report in reports.items():
                report.to_csv(output_dir / name, index=False)

            decision_path = output_dir / "12_b58ca_final_decision.json"
            decision_path.write_text(
                json.dumps(
                    clean_json(decision),
                    indent=2,
                    ensure_ascii=True,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            readme_path = output_dir / "README_B58CA.md"
            _write_readme(readme_path, decision)

            manifest_rows = []
            for path in sorted(output_dir.iterdir()):
                manifest_rows.append(
                    {
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            manifest_path = output_dir / "13_artifact_manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

            for path in sorted(output_dir.iterdir()):
                if path == decision_path:
                    key = f"configs/b58ca/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b58ca/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload_file(client, path, output_bucket, key)

        metadata = {
            **decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b58ca-weather-missingness-audit",
            "outputs": uploaded,
            "report_prefix": f"s3://{output_bucket}/reports/b58ca/{output_prefix}/",
            "config_uri": uploaded["12_b58ca_final_decision.json"],
        }
        _finish_run(run_id, "SUCCESS", len(hourly), metadata)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "results": clean_json(metadata),
            "outputs": uploaded,
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {"audit_version": AUDIT_VERSION, "orchestrator": "PREFECT"},
            error_message=str(exc),
        )
        raise


def verify_audit_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected audit status: {result.get('status')}")
    metadata = result.get("results") or {}
    if metadata.get("synthetic_rows_created") not in (0, "0", None):
        raise RuntimeError("B58C-A safety violation: synthetic rows were created")
    if metadata.get("source_modified") not in (False, "false", None):
        raise RuntimeError("B58C-A safety violation: source was modified")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "next_block": metadata.get("next_block"),
        "safety_contract": "PASS",
    }
