from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import boto3
from airflow.sdk import dag, task


S3_ENDPOINT = os.getenv("SMART_PORT_S3_ENDPOINT", "http://minio:9000")
SILVER_BUCKET = os.getenv("SMART_PORT_SILVER_BUCKET", "silver-maritime")
GOLD_BUCKET = os.getenv("SMART_PORT_GOLD_BUCKET", "gold-maritime")
MODEL_TRAINER_URL = os.getenv(
    "SMART_PORT_MODEL_TRAINER_URL", "http://model-trainer:8091"
).rstrip("/")

SOURCE_KEY = (
    "wave_features_model_ready/version=1/"
    "arrival_multi_horizon_wave_model_ready_v1.parquet"
)
SPLIT_KEY = "datasets/b54d/version=1/b54d_temporal_split_assignment.parquet"
FEATURE_CONFIG_KEY = "configs/b54d/version=1/b54d_feature_config.json"
DECISION_KEY = "configs/b54d/version=1/b54d_decision.json"
VALID_PREDICTIONS_KEY = "predictions/b54d/version=1/b54d_valid_predictions.parquet"
TEST_PREDICTIONS_KEY = "predictions/b54d/version=1/b54d_test_predictions.parquet"


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
        with urllib.request.urlopen(request, timeout=21600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Model Trainer HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Model Trainer unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_arrival_delay_dependency_audit",
    description=(
        "Leakage-safe mixed-type dependency, sea-state, temporal stability, "
        "and official-model residual audit."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "arrival-delay",
        "dependency",
        "correlation",
        "mutual-information",
        "sea-state",
        "anti-leakage",
    ],
)
def maritime_arrival_delay_dependency_audit():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_dependencies() -> dict:
        readiness = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if config.get("dependency_audit_version") != "b54ea-dependency-weather-v1":
            raise RuntimeError("Model Trainer does not expose B54E-A")
        client = s3_client()
        objects = [
            (SILVER_BUCKET, SOURCE_KEY),
            (GOLD_BUCKET, SPLIT_KEY),
            (GOLD_BUCKET, FEATURE_CONFIG_KEY),
            (GOLD_BUCKET, DECISION_KEY),
            (GOLD_BUCKET, VALID_PREDICTIONS_KEY),
            (GOLD_BUCKET, TEST_PREDICTIONS_KEY),
        ]
        audit = []
        for bucket, key in objects:
            metadata = client.head_object(Bucket=bucket, Key=key)
            audit.append(
                {
                    "uri": f"s3://{bucket}/{key}",
                    "size": int(metadata["ContentLength"]),
                    "etag": metadata["ETag"].strip('"'),
                }
            )
        return {"readiness": readiness, "config": config, "objects": audit}

    @task(retries=0, execution_timeout=timedelta(hours=4))
    def run_dependency_audit(dependencies: dict) -> dict:
        del dependencies
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/audit/b54ea",
            method="POST",
            payload={
                "source_bucket": SILVER_BUCKET,
                "source_key": SOURCE_KEY,
                "artifacts_bucket": GOLD_BUCKET,
                "split_key": SPLIT_KEY,
                "feature_config_key": FEATURE_CONFIG_KEY,
                "decision_key": DECISION_KEY,
                "valid_predictions_key": VALID_PREDICTIONS_KEY,
                "test_predictions_key": TEST_PREDICTIONS_KEY,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "force": False,
            },
        )

    @task
    def enforce_and_summarize(result: dict) -> dict:
        results = result.get("results", {})
        leakage = results.get("leakage_gate", {})
        decision = results.get("decision", {})
        if not leakage.get("passed", False):
            raise RuntimeError(f"B54E-A leakage gate failed: {leakage}")
        if decision.get("status") != "READY_FOR_B54E":
            raise RuntimeError(f"B54E-A decision is not ready: {decision}")
        return {
            "status": result.get("status"),
            "run_id": result.get("run_id"),
            "rows": results.get("source_rows"),
            "port_calls": results.get("source_calls"),
            "features": results.get("feature_count"),
            "numeric_features": results.get("numeric_feature_count"),
            "categorical_features": results.get("categorical_feature_count"),
            "wave_features": results.get("wave_feature_count"),
            "weather_signal": decision.get("weather_signal"),
            "next_block": results.get("next_block"),
            "outputs": result.get("outputs", {}),
        }

    dependencies = ensure_dependencies()
    result = run_dependency_audit(dependencies)
    enforce_and_summarize(result)


maritime_arrival_delay_dependency_audit()
