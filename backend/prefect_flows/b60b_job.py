from __future__ import annotations

import hashlib
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

from prefect_flows.b60b_core import (
    BENCHMARK_VERSION,
    clean_json,
    run_benchmark_core,
)
from prefect_flows.b60b_sequence import run_sequence_models, sequence_runtime


SOURCE_NAME = "b60b_advanced_timeseries_benchmark"
DATASET_NAME = "maritime_advanced_timeseries_benchmark_v1"
BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"
SOURCE_DATASET_KEY = "datasets/b60a/version=1/maritime_multitask_hourly_v1.parquet"
REPRESENTATIONS_KEY = "configs/b60a1/version=1/representation_feature_sets.json"
B60A2_DECISION_KEY = "configs/b60a2/version=2/b60a2_final_decision.json"


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


def _read(client, key: str) -> bytes:
    return client.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def _sequence_checkpoint_keys(checksum: str) -> dict[str, str]:
    prefix = f"checkpoints/b60b/{checksum}"
    return {
        "predictions": f"{prefix}/sequence_predictions.parquet",
        "inventory": f"{prefix}/sequence_inventory.parquet",
        "runtime": f"{prefix}/sequence_runtime.json",
    }


def _load_sequence_checkpoint(client, checksum: str):
    keys = _sequence_checkpoint_keys(checksum)
    try:
        predictions = pd.read_parquet(io.BytesIO(_read(client, keys["predictions"])))
        inventory = pd.read_parquet(io.BytesIO(_read(client, keys["inventory"])))
        runtime = json.loads(_read(client, keys["runtime"]).decode("utf-8"))
    except Exception:
        return None
    if predictions.empty or not runtime.get("ready"):
        return None
    return predictions, inventory, runtime


def _save_sequence_checkpoint(
    client,
    checksum: str,
    predictions: pd.DataFrame,
    inventory: pd.DataFrame,
    runtime: dict[str, Any],
) -> dict[str, str]:
    keys = _sequence_checkpoint_keys(checksum)
    prediction_buffer = io.BytesIO()
    predictions.to_parquet(prediction_buffer, index=False)
    client.put_object(
        Bucket=BUCKET,
        Key=keys["predictions"],
        Body=prediction_buffer.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    inventory_buffer = io.BytesIO()
    inventory.to_parquet(inventory_buffer, index=False)
    client.put_object(
        Bucket=BUCKET,
        Key=keys["inventory"],
        Body=inventory_buffer.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    client.put_object(
        Bucket=BUCKET,
        Key=keys["runtime"],
        Body=json.dumps(clean_json(runtime), ensure_ascii=True).encode("utf-8"),
        ContentType="application/json",
    )
    return {name: f"s3://{BUCKET}/{key}" for name, key in keys.items()}


def _load_inputs(client):
    dataset_bytes = _read(client, SOURCE_DATASET_KEY)
    representation_bytes = _read(client, REPRESENTATIONS_KEY)
    decision_bytes = _read(client, B60A2_DECISION_KEY)
    frame = pd.read_parquet(io.BytesIO(dataset_bytes))
    representations = json.loads(representation_bytes.decode("utf-8"))
    upstream = json.loads(decision_bytes.decode("utf-8"))
    if upstream.get("decision") != "PREDICTIVE_SIGNAL_CONFIRMED_FOR_B60B":
        raise RuntimeError(f"B60A.2 contract is not ready: {upstream.get('decision')}")
    if upstream.get("quality_gates_passed") is not True:
        raise RuntimeError("B60A.2 quality gates did not pass")
    if upstream.get("selection_used_test") is not False:
        raise RuntimeError("B60A.2 unexpectedly used TEST")
    return (
        dataset_bytes,
        representation_bytes,
        decision_bytes,
        frame,
        representations,
        upstream,
    )


def _checksum(parts: list[bytes], max_steps: int) -> str:
    digest = hashlib.sha256(BENCHMARK_VERSION.encode("ascii"))
    digest.update(str(max_steps).encode("ascii"))
    for value in parts:
        digest.update(hashlib.sha256(value).digest())
    return digest.hexdigest()


def _previous_success(checksum: str):
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


def _start_run(checksum: str, max_steps: int) -> str:
    metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "orchestrator": "PREFECT",
        "selection_used_test": False,
        "research_weather_used": False,
        "production_promotion_allowed": False,
        "sequence_max_steps": max_steps,
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
                    f"s3://{BUCKET}/{SOURCE_DATASET_KEY}",
                    checksum,
                    Json(metadata),
                ),
            )
            return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    progress = {"stage": stage, **clean_json(details)}
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
                SET status=%s, row_count=%s, finished_at=NOW(),
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
        ".parquet": "application/vnd.apache.parquet",
        ".md": "text/markdown",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(
        str(path), BUCKET, key, ExtraArgs={"ContentType": content_type}
    )
    client.head_object(Bucket=BUCKET, Key=key)
    return f"s3://{BUCKET}/{key}"


def _log_mlflow(result) -> str:
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("maritime-b60b-advanced-timeseries")
        with mlflow.start_run(run_name="b60b-v1"):
            mlflow.log_params(
                {
                    "benchmark_version": BENCHMARK_VERSION,
                    "selection_split": "VALID_ONLY",
                    "research_weather_used": False,
                    "target_count": result.decision["target_count"],
                }
            )
            valid = result.reports["06_valid_metrics.csv"]
            test = result.reports["08_test_metrics.csv"]
            selection = result.reports["07_model_selection.csv"]
            for row in selection.itertuples(index=False):
                safe = "".join(character if character.isalnum() else "_" for character in row.target)
                mlflow.log_metric(f"valid_mae_{safe}", float(row.selected_valid_mae))
            for row in test.itertuples(index=False):
                safe = "".join(character if character.isalnum() else "_" for character in row.target)
                mlflow.log_metric(f"test_mae_{safe}", float(row.MAE))
            mlflow.log_metric("accepted_challengers", result.decision["accepted_challengers"])
            mlflow.log_metric("valid_candidate_rows", len(valid))
        return "LOGGED"
    except Exception as exc:  # benchmark artifacts remain authoritative
        return f"SKIPPED:{type(exc).__name__}:{exc}"


def _readme(decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B60B advanced time-series benchmark",
            "",
            f"Decision: {decision['decision']}",
            "",
            "Model selection uses VALID only. TEST is evaluated after the selection",
            "contract is frozen and is never used to choose a model.",
            "",
            "Counts: Dynamic Negative Binomial, HGB Poisson, CatBoost Poisson,",
            "N-HiTS and PatchTST. Wait time: discrete-time hazard, HGB and CatBoost.",
            "Wave period: HGB, CatBoost, N-HiTS and PatchTST.",
            "",
            "P10/P50/P90 use asymmetric split conformal calibration on VALID.",
            "External retrospective weather is excluded from every production track.",
        ]
    )


def run_b60b_benchmark(
    force: bool = False,
    sequence_max_steps: int = 250,
) -> dict[str, Any]:
    client = _s3_client()
    (
        dataset_bytes,
        representation_bytes,
        decision_bytes,
        frame,
        representations,
        upstream,
    ) = _load_inputs(client)
    checksum = _checksum(
        [dataset_bytes, representation_bytes, decision_bytes], sequence_max_steps
    )
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "results": previous[1],
        }

    run_id = _start_run(checksum, sequence_max_steps)
    try:
        runtime = sequence_runtime()
        if not runtime["ready"]:
            raise RuntimeError(f"Neural sequence runtime is unavailable: {runtime['error']}")
        checkpoint = None if force else _load_sequence_checkpoint(client, checksum)
        if checkpoint is None:
            _update_progress(
                run_id,
                "TRAINING_NHITS_AND_PATCHTST",
                rows=len(frame),
                max_steps=sequence_max_steps,
            )
            sequence_predictions, sequence_inventory, runtime = run_sequence_models(
                frame, max_steps=sequence_max_steps
            )
            checkpoint_uris = _save_sequence_checkpoint(
                client,
                checksum,
                sequence_predictions,
                sequence_inventory,
                runtime,
            )
        else:
            sequence_predictions, sequence_inventory, runtime = checkpoint
            checkpoint_uris = {
                name: f"s3://{BUCKET}/{key}"
                for name, key in _sequence_checkpoint_keys(checksum).items()
            }
            _update_progress(
                run_id,
                "REUSING_SEQUENCE_CHECKPOINT",
                sequence_prediction_rows=len(sequence_predictions),
            )
        _update_progress(
            run_id,
            "FITTING_TARGET_SPECIFIC_MODELS",
            sequence_prediction_rows=len(sequence_predictions),
        )
        result = run_benchmark_core(
            frame,
            representations,
            sequence_predictions,
            sequence_inventory,
            sequence_runtime_ready=bool(runtime["ready"]),
            enable_catboost=True,
        )
        mlflow_status = _log_mlflow(result)
        result.decision["mlflow_status"] = mlflow_status
        result.decision["sequence_runtime"] = clean_json(runtime)
        result.decision["sequence_checkpoint_uris"] = checkpoint_uris
        result.decision["upstream_b60a2_decision"] = upstream["decision"]

        _update_progress(
            run_id,
            "WRITING_BENCHMARK_ARTIFACTS",
            decision=result.decision["decision"],
            reports=len(result.reports),
        )
        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b60b-") as temporary:
            directory = Path(temporary)
            for name, report in result.reports.items():
                path = directory / name
                report.to_csv(path, index=False)
                outputs[name] = _upload(
                    client, path, f"reports/b60b/{OUTPUT_PREFIX}/{name}"
                )
            for name, predictions in result.predictions.items():
                path = directory / name
                predictions.to_parquet(path, index=False)
                outputs[name] = _upload(
                    client, path, f"predictions/b60b/{OUTPUT_PREFIX}/{name}"
                )
            decision_path = directory / "b60b_final_decision.json"
            decision_path.write_text(
                json.dumps(clean_json(result.decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[decision_path.name] = _upload(
                client,
                decision_path,
                f"configs/b60b/{OUTPUT_PREFIX}/{decision_path.name}",
            )
            runtime_path = directory / "sequence_runtime.json"
            runtime_path.write_text(
                json.dumps(clean_json(runtime), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[runtime_path.name] = _upload(
                client,
                runtime_path,
                f"reports/b60b/{OUTPUT_PREFIX}/{runtime_path.name}",
            )
            readme_path = directory / "README_B60B.md"
            readme_path.write_text(_readme(result.decision), encoding="utf-8")
            outputs[readme_path.name] = _upload(
                client,
                readme_path,
                f"reports/b60b/{OUTPUT_PREFIX}/{readme_path.name}",
            )

        metadata = {
            **result.decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b60b-advanced-timeseries-benchmark",
            "report_prefix": f"s3://{BUCKET}/reports/b60b/{OUTPUT_PREFIX}/",
            "prediction_prefix": f"s3://{BUCKET}/predictions/b60b/{OUTPUT_PREFIX}/",
            "outputs": outputs,
        }
        _finish_run(run_id, "SUCCESS", len(frame), metadata)
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
                "benchmark_version": BENCHMARK_VERSION,
                "selection_used_test": False,
                "production_promotion_allowed": False,
            },
            str(exc),
        )
        raise


def verify_b60b_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B60B status: {result.get('status')}")
    metadata = result.get("results") or {}
    if metadata.get("quality_gates_passed") not in (True, "true"):
        raise RuntimeError("B60B critical quality gate failed")
    if metadata.get("selection_used_test") not in (False, "false"):
        raise RuntimeError("B60B leakage violation: TEST used for selection")
    if metadata.get("production_promotion_allowed") not in (False, "false"):
        raise RuntimeError("B60B benchmark cannot promote directly")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "selected_models": metadata.get("selected_models"),
        "accepted_challengers": metadata.get("accepted_challengers"),
        "next_block": metadata.get("next_block"),
    }
