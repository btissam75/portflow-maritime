from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import boto3
import joblib
import pandas as pd
import psycopg2
from psycopg2.extras import Json

from prefect_flows.b60a1_core import (
    AUDIT_VERSION,
    audit_feature_representations,
    clean_json,
    content_checksum,
    representation_sets_json,
)


SOURCE_NAME = "b60a1_feature_representation_audit"
DATASET_NAME = "maritime_feature_representations_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"
SOURCE_DATASET_KEY = "datasets/b60a/version=1/maritime_multitask_hourly_v1.parquet"
SOURCE_FEATURE_SETS_KEY = "configs/b60a/version=1/11_feature_sets.json"


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


def _read_source(client) -> tuple[bytes, bytes, pd.DataFrame, dict[str, list[str]]]:
    dataset_bytes = client.get_object(
        Bucket=OUTPUT_BUCKET, Key=SOURCE_DATASET_KEY
    )["Body"].read()
    feature_sets_bytes = client.get_object(
        Bucket=OUTPUT_BUCKET, Key=SOURCE_FEATURE_SETS_KEY
    )["Body"].read()
    dataset = pd.read_parquet(io.BytesIO(dataset_bytes))
    feature_sets = json.loads(feature_sets_bytes.decode("utf-8"))
    return dataset_bytes, feature_sets_bytes, dataset, feature_sets


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
        "selection_used_test": False,
        "predictive_training_executed": False,
        "source_modified": False,
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
        ".parquet": "application/octet-stream",
        ".joblib": "application/octet-stream",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(
        str(path), OUTPUT_BUCKET, key, ExtraArgs={"ContentType": content_type}
    )
    client.head_object(Bucket=OUTPUT_BUCKET, Key=key)
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _readme(decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B60A.1 feature representation audit",
            "",
            f"Decision: {decision['decision']}",
            "",
            "Pearson, Spearman, mutual information, approximate VIF, temporal",
            "stability, ACF and PACF are evaluated without using TEST for selection.",
            "",
            "Four representations are prepared for every task/track: RAW, PRUNED,",
            "COMPACT and BLOCK_PCA. PCA scalers and components are fitted on TRAIN",
            "only and separately by semantic block.",
            "",
            "High correlation does not automatically remove intentional temporal lags.",
            "PPCA is not selected because the external-weather missingness is structural.",
            "No predictive model is trained in this block.",
        ]
    )


def run_b60a1_feature_audit(force: bool = False) -> dict[str, Any]:
    client = _s3_client()
    dataset_bytes, feature_sets_bytes, dataset, feature_sets = _read_source(client)
    checksum = content_checksum(dataset_bytes, feature_sets_bytes)
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
            "CORRELATION_AND_STABILITY_AUDIT",
            rows=len(dataset),
            columns=len(dataset.columns),
        )
        result = audit_feature_representations(dataset, feature_sets)
        _update_progress(
            run_id,
            "WRITING_REPRESENTATIONS",
            representations=len(result.representation_sets),
        )
        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b60a1-") as temporary:
            directory = Path(temporary)
            for name, report in result.reports.items():
                path = directory / name
                report.to_csv(path, index=False)
                outputs[name] = _upload(
                    client, path, f"reports/b60a1/{OUTPUT_PREFIX}/{name}"
                )

            sets_path = directory / "representation_feature_sets.json"
            sets_path.write_text(
                representation_sets_json(result.representation_sets),
                encoding="utf-8",
            )
            outputs[sets_path.name] = _upload(
                client,
                sets_path,
                f"configs/b60a1/{OUTPUT_PREFIX}/{sets_path.name}",
            )

            decision_path = directory / "b60a1_final_decision.json"
            decision_path.write_text(
                json.dumps(clean_json(result.decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[decision_path.name] = _upload(
                client,
                decision_path,
                f"configs/b60a1/{OUTPUT_PREFIX}/{decision_path.name}",
            )

            for track, pca_frame in result.pca_frames.items():
                frame_path = directory / f"{track}_block_pca_v1.parquet"
                pca_frame.to_parquet(frame_path, index=False, compression="zstd")
                outputs[frame_path.name] = _upload(
                    client,
                    frame_path,
                    f"datasets/b60a1/{OUTPUT_PREFIX}/{frame_path.name}",
                )
                transformer_path = directory / f"{track}_block_pca_v1.joblib"
                joblib.dump(
                    {
                        "audit_version": AUDIT_VERSION,
                        "track": track,
                        "fit_split": "TRAIN",
                        "transformers": result.transformers[track],
                    },
                    transformer_path,
                    compress=3,
                )
                outputs[transformer_path.name] = _upload(
                    client,
                    transformer_path,
                    f"transformers/b60a1/{OUTPUT_PREFIX}/{transformer_path.name}",
                )

            readme_path = directory / "README_B60A1.md"
            readme_path.write_text(_readme(result.decision), encoding="utf-8")
            outputs[readme_path.name] = _upload(
                client,
                readme_path,
                f"reports/b60a1/{OUTPUT_PREFIX}/{readme_path.name}",
            )

        metadata = {
            **result.decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b60a1-feature-representation-audit",
            "report_prefix": f"s3://{OUTPUT_BUCKET}/reports/b60a1/{OUTPUT_PREFIX}/",
            "config_uri": outputs["representation_feature_sets.json"],
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
                "selection_used_test": False,
            },
            str(exc),
        )
        raise


def verify_b60a1_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B60A.1 status: {result.get('status')}")
    metadata = result.get("results") or {}
    if metadata.get("decision") != "READY_FOR_B60B_FEATURE_REPRESENTATION_BENCHMARK":
        raise RuntimeError(f"B60A.1 gates failed: {metadata.get('decision')}")
    if metadata.get("selection_used_test") not in (False, "false"):
        raise RuntimeError("B60A.1 leakage violation: TEST used for selection")
    if metadata.get("predictive_training_executed") not in (False, "false"):
        raise RuntimeError("B60A.1 scope violation: predictive model was trained")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "source_rows": metadata.get("source_rows"),
        "representation_count": metadata.get("representation_count"),
        "quality_gates_passed": metadata.get("quality_gates_passed"),
        "next_block": metadata.get("next_block"),
        "safety_contract": "PASS",
    }
