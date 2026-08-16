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
MODEL_READY_KEY = (
    "datasets/b54f/version=1/port_call_one_row_model_ready_no_split_v1.parquet"
)

CRITICAL_OUTPUT_KEYS = (
    "reports/b54g/version=1/00_schema_semantic_audit.csv",
    "reports/b54g/version=1/01_target_distribution_and_drift.csv",
    "reports/b54g/version=1/02_all_numeric_target_associations.csv",
    "reports/b54g/version=1/03_categorical_associations.csv",
    "reports/b54g/version=1/04_adjusted_weather_associations.csv",
    "reports/b54g/version=1/05_adjusted_delay_risk_differences.csv",
    "reports/b54g/version=1/06_wave_threshold_dose_response.csv",
    "reports/b54g/version=1/07_within_vessel_matched_summary.csv",
    "reports/b54g/version=1/09_effect_heterogeneity.csv",
    "reports/b54g/version=1/10_temporal_stability.csv",
    "reports/b54g/version=1/11_required_operational_data.csv",
    "reports/b54g/version=1/B54G_EXECUTIVE_REPORT.md",
    "configs/b54g/version=1/b54g_influence_decision_v1.json",
    "research/b54g/version=1/b54g_explanatory_weather_tracks_v1.parquet",
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
        with urllib.request.urlopen(request, timeout=21600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_delay_influence_study",
    description=(
        "B54G: adjusted maritime delay influence study with SAFE_T24, "
        "explanatory Oracle weather, within-vessel matching and negative controls."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "influence",
        "weather",
        "within-vessel",
        "negative-control",
        "not-causal",
    ],
)
def maritime_delay_influence_study():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if config.get("influence_study_version") != "b54g-maritime-influence-v1":
            raise RuntimeError("Model Trainer does not expose B54G v1")
        return {"ready": ready, "config": config}

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_input(trainer: dict) -> dict:
        del trainer
        metadata = s3_client().head_object(Bucket=GOLD_BUCKET, Key=MODEL_READY_KEY)
        return {
            "uri": f"s3://{GOLD_BUCKET}/{MODEL_READY_KEY}",
            "size": int(metadata["ContentLength"]),
            "etag": metadata["ETag"].strip('"'),
        }

    @task(retries=0, execution_timeout=timedelta(hours=6))
    def run_study(source: dict) -> dict:
        del source
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/audit/b54g",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "model_ready_key": MODEL_READY_KEY,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
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
                raise RuntimeError(f"Empty B54G output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B54G service failed: {result}")
        findings = result.get("results", {})
        if findings.get("causal_identification") != "NOT_IDENTIFIED":
            raise RuntimeError("B54G must not claim causal identification")
        if findings.get("predictive_oracle_contamination") is not False:
            raise RuntimeError("Oracle explanatory weather contaminated prediction data")
        if findings.get("training_executed") is not False:
            raise RuntimeError("B54G must not train a predictive model")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "claim_allowed": findings.get("claim_allowed"),
            "source_rows": findings.get("source_rows"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    source = verify_input(trainer)
    result = run_study(source)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_delay_influence_study()
