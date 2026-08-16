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

CRITICAL_OUTPUT_KEYS = (
    "reports/b56e/version=1/00_upstream_and_integrity_contract.csv",
    "reports/b56e/version=1/01_metrics_valid_selection.csv",
    "reports/b56e/version=1/02_metrics_test_locked.csv",
    "reports/b56e/version=1/03_online_weight_summary.csv",
    "reports/b56e/version=1/04_probabilistic_calibration_valid.csv",
    "reports/b56e/version=1/05_probabilistic_calibration_test.csv",
    "reports/b56e/version=1/06_paired_day_bootstrap_test.csv",
    "reports/b56e/version=1/07_source_freshness_and_serving_gate.csv",
    "reports/b56e/version=1/README_B56E.md",
    "configs/b56e/version=1/08_b56e_decision.json",
    "predictions/b56e/version=1/valid_probabilistic_predictions.parquet",
    "predictions/b56e/version=1/test_probabilistic_predictions.parquet",
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
    url: str,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=21600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_arrival_flow_probabilistic_ensemble",
    description=(
        "B56E: validate past-only adaptive arrival-flow policies and rolling "
        "P10/P50/P90 intervals without retraining frozen B56C models."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "arrival-flow",
        "probabilistic",
        "adaptive-ensemble",
        "historical-replay",
        "no-retraining",
    ],
)
def maritime_arrival_flow_probabilistic_ensemble():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            config.get("arrival_flow_probabilistic_ensemble_version")
            != "b56e-arrival-flow-probabilistic-ensemble-v1"
        ):
            raise RuntimeError("Model Trainer does not expose B56E v1")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=2))
    def validate_and_package(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/audit/b56e",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "materialize_timescale": True,
                "force": False,
            },
        )

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_outputs(result: dict) -> dict:
        client = s3_client()
        objects = []
        for key in CRITICAL_OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            size = int(metadata["ContentLength"])
            if size <= 0:
                raise RuntimeError(f"Empty B56E output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B56E service failed: {result}")
        findings = result.get("results", {})
        if findings.get("training_executed") is not False:
            raise RuntimeError("B56E unexpectedly retrained a model")
        if findings.get("selection_used_test") is not False:
            raise RuntimeError("B56E used locked TEST for selection")
        if findings.get("integrity_gates_passed") is not True:
            raise RuntimeError("B56E integrity gates failed")
        allowed = {
            "READY_FOR_HISTORICAL_REPLAY_NOT_LIVE",
            "READY_FOR_CONTROLLED_LIVE_SHADOW",
            "NEED_INTERVAL_RECALIBRATION",
            "NEED_DATA_REPAIR",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(f"Unknown B56E decision: {findings.get('status')}")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "selected_models": findings.get("selected_models"),
            "historical_replay_allowed": findings.get(
                "historical_replay_allowed"
            ),
            "live_serving_allowed": findings.get("live_serving_allowed"),
            "source_status": findings.get("source_status"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = validate_and_package(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_arrival_flow_probabilistic_ensemble()
