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

OUTPUT_KEYS = (
    "configs/b57d/version=1/b57d_calibration.json",
    "configs/b57d/version=1/b57d_api_manifest.json",
    "configs/b57d/version=1/b57d_decision.json",
    "reports/b57d/version=1/01_conformal_calibration.csv",
    "reports/b57d/version=1/02_final_test_reliability.csv",
    "reports/b57d/version=1/03_artifact_and_feature_contract.csv",
    "reports/b57d/version=1/README_B57D.md",
)


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def request_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_tir_probabilistic_forecast_api",
    description=(
        "B57D: promote B57C models, calibrate conformal uncertainty and expose "
        "a guarded daily replay/operational forecast API."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "tir",
        "probabilistic",
        "conformal",
        "api",
        "anti-leakage",
        "no-retraining",
    ],
)
def maritime_tir_probabilistic_forecast_api():
    @task(retries=2, retry_delay=timedelta(seconds=20))
    def ensure_model_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        expected = "b57d-probabilistic-forecast-api-v1"
        if config.get("probabilistic_forecast_api_version") != expected:
            raise RuntimeError(f"Model Trainer does not expose {expected}")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(minutes=30))
    def promote(dependencies: dict) -> dict:
        del dependencies
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/promote/b57d",
            method="POST",
            payload={
                "artifact_bucket": GOLD_BUCKET,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "force": False,
            },
        )

    @task(retries=2, retry_delay=timedelta(seconds=10))
    def verify_outputs(result: dict) -> dict:
        client = s3_client()
        objects = []
        for key in OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            size = int(metadata["ContentLength"])
            if size <= 0:
                raise RuntimeError(f"Empty B57D object: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        runtime = request_json(f"{MODEL_TRAINER_URL}/v1/forecast/b57d/status")
        if runtime.get("status") != "READY":
            raise RuntimeError(f"B57D runtime did not load: {runtime}")
        return {
            "run_id": result.get("run_id"),
            "objects": objects,
            "runtime": runtime,
        }

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B57D service failed: {result}")
        decision = result.get("results", {})
        allowed = {
            "READY_FOR_OPERATIONAL_PROBABILISTIC_API",
            "READY_FOR_HISTORICAL_REPLAY_API_NEED_LIVE_DATA",
            "READY_FOR_HISTORICAL_REPLAY_API_WITH_DRIFT_GUARD_WARNING",
        }
        if decision.get("status") not in allowed:
            raise RuntimeError(f"B57D promotion failed: {decision}")
        if decision.get("gates_passed") is not True:
            raise RuntimeError("B57D structural and drift-guard gates failed")
        if decision.get("training_executed") is not False:
            raise RuntimeError("B57D must not retrain models")
        if decision.get("target_or_actual_exposed") is not False:
            raise RuntimeError("B57D exposes a forbidden realized target")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": decision.get("status"),
            "operating_mode": decision.get("operating_mode"),
            "models_promoted": decision.get("models_promoted"),
            "source_last_date": decision.get("source_last_date"),
            "source_freshness_days": decision.get("source_freshness_days"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": decision.get("next_block"),
        }

    dependencies = ensure_model_trainer()
    result = promote(dependencies)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_tir_probabilistic_forecast_api()
