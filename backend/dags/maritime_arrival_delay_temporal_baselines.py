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
REPORT_KEY = "audits/version=1/b54c_wave_feature_report_v1.json"


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
    dag_id="maritime_arrival_delay_temporal_baselines",
    description=(
        "Temporal port-call split and controlled CatBoost no-wave versus wave uplift test."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["arrival-delay", "catboost", "temporal-split", "mlflow", "wave-uplift"],
)
def maritime_arrival_delay_temporal_baselines():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_dependencies() -> dict:
        readiness = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if config.get("training_version") != "b54d-temporal-catboost-v1":
            raise RuntimeError("Model Trainer does not expose B54D")
        client = s3_client()
        source = client.head_object(Bucket=SILVER_BUCKET, Key=SOURCE_KEY)
        report = client.head_object(Bucket=SILVER_BUCKET, Key=REPORT_KEY)
        return {
            "readiness": readiness,
            "config": config,
            "source_size": int(source["ContentLength"]),
            "source_etag": source["ETag"].strip('"'),
            "report_etag": report["ETag"].strip('"'),
        }

    @task(
        retries=0,
        execution_timeout=timedelta(hours=6),
    )
    def train_and_compare(dependencies: dict) -> dict:
        del dependencies
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/train/b54d",
            method="POST",
            payload={
                "source_bucket": SILVER_BUCKET,
                "source_key": SOURCE_KEY,
                "report_key": REPORT_KEY,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "train_fraction": 0.70,
                "valid_fraction": 0.15,
                "force": False,
            },
        )

    @task
    def enforce_and_summarize(result: dict) -> dict:
        results = result.get("results", {})
        split_report = results.get("split_report", {})
        if int(split_report.get("temporal_leakage_violations", -1)) != 0:
            raise RuntimeError(f"B54D temporal leakage gate failed: {split_report}")
        decision = results.get("decision", {})
        if not decision.get("official_model"):
            raise RuntimeError(f"B54D did not select an official model: {decision}")
        return {
            "status": result.get("status"),
            "run_id": result.get("run_id"),
            "source_rows": results.get("source_rows"),
            "source_calls": results.get("source_calls"),
            "feature_count_no_wave": results.get("no_wave_feature_count"),
            "feature_count_with_wave": results.get("usable_feature_count"),
            "weather_status": decision.get("weather_status"),
            "official_model": decision.get("official_model"),
            "valid_mae_gain_pct": decision.get("valid_mae_gain_pct"),
            "official_test_metrics": decision.get("official_test_metrics"),
            "mlflow": results.get("mlflow"),
            "next_block": results.get("next_block"),
            "outputs": result.get("outputs", {}),
        }

    dependencies = ensure_dependencies()
    result = train_and_compare(dependencies)
    enforce_and_summarize(result)


maritime_arrival_delay_temporal_baselines()
