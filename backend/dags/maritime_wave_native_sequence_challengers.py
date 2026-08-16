from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import boto3
from airflow.sdk import dag, task


MODEL_TRAINER_URL = os.getenv(
    "SMART_PORT_MODEL_TRAINER_URL", "http://model-trainer:8091"
).rstrip("/")
S3_ENDPOINT = os.getenv("SMART_PORT_S3_ENDPOINT", "http://minio:9000")
GOLD_BUCKET = os.getenv("SMART_PORT_GOLD_BUCKET", "gold-maritime")

CRITICAL_KEYS = (
    "reports/b58b1/version=1/01_source_and_champion_contract.csv",
    "reports/b58b1/version=1/02_temporal_protocol.csv",
    "reports/b58b1/version=1/03_training_inventory.csv",
    "reports/b58b1/version=1/04_validation_comparison.csv",
    "reports/b58b1/version=1/05_model_selection.csv",
    "reports/b58b1/version=1/06_test_comparison.csv",
    "reports/b58b1/version=1/07_selected_test_metrics.csv",
    "reports/b58b1/version=1/08_quarterly_stability.csv",
    "reports/b58b1/version=1/09_probabilistic_calibration.csv",
    "reports/b58b1/version=1/10_resource_usage.csv",
    "reports/b58b1/version=1/11_anti_leakage.csv",
    "reports/b58b1/version=1/README_B58B1.md",
    "configs/b58b1/version=1/12_b58b1_decision.json",
    "predictions/b58b1/version=1/sequence_comparison_predictions.parquet",
)


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def request_json(
    url: str, method: str = "GET", payload: dict | None = None
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=43200) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_wave_native_sequence_challengers",
    description=(
        "B58B.1: compare native N-HiTS and PatchTST sequence models with "
        "the frozen B58B champion under the same temporal protocol."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "waves",
        "nhits",
        "patchtst",
        "time-series",
        "challenger",
    ],
)
def maritime_wave_native_sequence_challengers():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if config.get("wave_sequence_challenger_version") != (
            "b58b1-native-wave-sequence-challengers-v1"
        ):
            raise RuntimeError("Model Trainer does not expose B58B.1 v1")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=12))
    def run_challengers(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/train/b58b1",
            method="POST",
            payload={
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "force": False,
            },
        )

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_outputs(result: dict) -> dict:
        client = s3_client()
        objects = []
        for key in CRITICAL_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            size = int(metadata["ContentLength"])
            if size <= 0:
                raise RuntimeError(
                    f"Empty B58B.1 output: s3://{GOLD_BUCKET}/{key}"
                )
            objects.append({"key": key, "size": size})
        model_objects = client.list_objects_v2(
            Bucket=GOLD_BUCKET,
            Prefix="models/b58b1/version=1/neuralforecast_checkpoints/",
        ).get("Contents", [])
        if not model_objects:
            raise RuntimeError("B58B.1 NeuralForecast checkpoints are missing")
        return {
            "run_id": result.get("run_id"),
            "objects": objects,
            "model_objects": len(model_objects),
        }

    @task
    def enforce(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B58B.1 service failed: {result}")
        findings = result.get("results", {})
        for field in (
            "selection_used_test",
            "b58b_modified",
            "bronze_modified",
            "core_modified",
            "historical_replay_allowed",
            "production_promotion_allowed",
        ):
            if findings.get(field) is not False:
                raise RuntimeError(f"B58B.1 invariant failed: {field}")
        if int(findings.get("critical_leakage_violations", -1)) != 0:
            raise RuntimeError("B58B.1 found critical temporal leakage")
        allowed = {
            "SEQUENCE_CHALLENGER_ACCEPTED",
            "KEEP_B58B_CHAMPION",
            "NEED_SEQUENCE_CHALLENGER_REPAIR",
        }
        if findings.get("decision") not in allowed:
            raise RuntimeError(
                f"Unknown B58B.1 decision: {findings.get('decision')}"
            )
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("decision"),
            "sequence_acceptances": findings.get("sequence_acceptances"),
            "verified_objects": len(outputs.get("objects", [])),
            "model_objects": outputs.get("model_objects"),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = run_challengers(trainer)
    outputs = verify_outputs(result)
    enforce(result, outputs)


maritime_wave_native_sequence_challengers()
