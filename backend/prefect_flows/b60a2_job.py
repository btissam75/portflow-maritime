from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import psycopg2
from psycopg2.extras import Json

from prefect_flows.b60a2_core import (
    AUDIT_VERSION,
    audit_predictive_signal,
    clean_json,
    content_checksum,
    decision_json,
)


SOURCE_NAME = "b60a2_predictive_signal_audit"
DATASET_NAME = "maritime_predictive_signal_audit_v2"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=2"
SOURCE_DATASET_KEY = "datasets/b60a/version=1/maritime_multitask_hourly_v1.parquet"
SOURCE_FEATURE_SETS_KEY = "configs/b60a/version=1/11_feature_sets.json"
SOURCE_REPRESENTATIONS_KEY = "configs/b60a1/version=1/representation_feature_sets.json"


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


def _read_sources(client):
    dataset_bytes = client.get_object(
        Bucket=OUTPUT_BUCKET, Key=SOURCE_DATASET_KEY
    )["Body"].read()
    feature_sets_bytes = client.get_object(
        Bucket=OUTPUT_BUCKET, Key=SOURCE_FEATURE_SETS_KEY
    )["Body"].read()
    representation_sets_bytes = client.get_object(
        Bucket=OUTPUT_BUCKET, Key=SOURCE_REPRESENTATIONS_KEY
    )["Body"].read()
    return (
        dataset_bytes,
        feature_sets_bytes,
        representation_sets_bytes,
        pd.read_parquet(io.BytesIO(dataset_bytes)),
        json.loads(feature_sets_bytes.decode("utf-8")),
        json.loads(representation_sets_bytes.decode("utf-8")),
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


def _start_run(checksum: str) -> str:
    metadata = {
        "audit_version": AUDIT_VERSION,
        "orchestrator": "PREFECT",
        "source_dataset": SOURCE_DATASET_KEY,
        "selection_used_valid": False,
        "selection_used_test": False,
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
                    f"s3://{OUTPUT_BUCKET}/{SOURCE_DATASET_KEY}",
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


def _upload(client, path: Path, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(
        str(path), OUTPUT_BUCKET, key, ExtraArgs={"ContentType": content_type}
    )
    client.head_object(Bucket=OUTPUT_BUCKET, Key=key)
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _readme(decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B60A.2 predictive signal audit",
            "",
            f"Decision: {decision['decision']}",
            "",
            "Signal selection uses four purged rolling-origin folds inside TRAIN.",
            "VALID is a regime-shift confirmation only and TEST remains untouched.",
            "",
            "Ridge probes marginal and conditional linear value. ExtraTrees probes",
            "nonlinear value and block permutation loss. Safe naive baselines use",
            "only target shifts whose lag is greater than or equal to the horizon.",
            "The shuffled-target placebo is judged against a TRAIN-only mean control.",
            "",
            "These are diagnostic probe models, not production candidates. External",
            "retrospective weather remains RESEARCH_ONLY regardless of measured lift.",
        ]
    )


def run_b60a2_predictive_signal_audit(force: bool = False) -> dict[str, Any]:
    client = _s3_client()
    (
        dataset_bytes,
        feature_sets_bytes,
        representation_sets_bytes,
        dataset,
        feature_sets,
        representation_sets,
    ) = _read_sources(client)
    checksum = content_checksum(
        dataset_bytes, feature_sets_bytes, representation_sets_bytes
    )
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
            "ROLLING_ORIGIN_SIGNAL_PROBES",
            rows=len(dataset),
            columns=len(dataset.columns),
            folds=4,
        )
        result = audit_predictive_signal(dataset, feature_sets, representation_sets)
        _update_progress(
            run_id,
            "WRITING_SIGNAL_REPORTS",
            reports=len(result.reports),
            decision=result.decision["decision"],
        )
        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b60a2-") as temporary:
            directory = Path(temporary)
            for name, report in result.reports.items():
                path = directory / name
                report.to_csv(path, index=False)
                outputs[name] = _upload(
                    client, path, f"reports/b60a2/{OUTPUT_PREFIX}/{name}"
                )

            decision_path = directory / "b60a2_final_decision.json"
            decision_path.write_text(
                decision_json(result.decision), encoding="utf-8"
            )
            outputs[decision_path.name] = _upload(
                client,
                decision_path,
                f"configs/b60a2/{OUTPUT_PREFIX}/{decision_path.name}",
            )

            readme_path = directory / "README_B60A2.md"
            readme_path.write_text(_readme(result.decision), encoding="utf-8")
            outputs[readme_path.name] = _upload(
                client,
                readme_path,
                f"reports/b60a2/{OUTPUT_PREFIX}/{readme_path.name}",
            )

        metadata = {
            **result.decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b60a2-predictive-signal-audit",
            "report_prefix": f"s3://{OUTPUT_BUCKET}/reports/b60a2/{OUTPUT_PREFIX}/",
            "decision_uri": outputs["b60a2_final_decision.json"],
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
                "audit_version": AUDIT_VERSION,
                "orchestrator": "PREFECT",
                "selection_used_valid": False,
                "selection_used_test": False,
            },
            str(exc),
        )
        raise


def verify_b60a2_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B60A.2 status: {result.get('status')}")
    metadata = result.get("results") or {}
    if metadata.get("decision") == "BLOCKED_PREDICTIVE_SIGNAL_AUDIT_REPAIR_REQUIRED":
        raise RuntimeError("B60A.2 critical quality gate failed")
    if metadata.get("selection_used_valid") not in (False, "false"):
        raise RuntimeError("B60A.2 leakage violation: VALID used for selection")
    if metadata.get("selection_used_test") not in (False, "false"):
        raise RuntimeError("B60A.2 leakage violation: TEST used for selection")
    if metadata.get("production_promotion_allowed") not in (False, "false"):
        raise RuntimeError("B60A.2 cannot promote diagnostic probes")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "core_primary_targets": metadata.get("core_primary_targets"),
        "core_targets_with_stable_signal": metadata.get(
            "core_targets_with_stable_signal"
        ),
        "core_stable_signal_pct": metadata.get("core_stable_signal_pct"),
        "tasks_with_stable_core_signal": metadata.get(
            "tasks_with_stable_core_signal"
        ),
        "next_block": metadata.get("next_block"),
    }
