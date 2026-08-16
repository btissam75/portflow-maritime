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
    "reports/b56b/version=1/00_source_completeness_by_month.csv",
    "reports/b56b/version=1/01_temporal_split_audit.csv",
    "reports/b56b/version=1/02_feature_contract.csv",
    "reports/b56b/version=1/03_metrics_valid.csv",
    "reports/b56b/version=1/04_metrics_test.csv",
    "reports/b56b/version=1/05_weather_ablation_bootstrap.csv",
    "reports/b56b/version=1/06_high_pressure_metrics.csv",
    "reports/b56b/version=1/07_residual_stability_by_year.csv",
    "reports/b56b/version=1/09_target_stability_valid_test.csv",
    "reports/b56b/version=1/README_B56B.md",
    "configs/b56b/version=1/08_b56b_decision.json",
    "datasets/b56b/version=1/temporal_split_assignments.parquet",
    "predictions/b56b/version=1/valid_predictions.parquet",
    "predictions/b56b/version=1/test_predictions.parquet",
    "models/b56b/version=1/hgb_poisson_core_6h.pkl",
    "models/b56b/version=1/hgb_poisson_core_wave_6h.pkl",
    "models/b56b/version=1/hgb_poisson_core_12h.pkl",
    "models/b56b/version=1/hgb_poisson_core_wave_12h.pkl",
    "models/b56b/version=1/hgb_poisson_core_24h.pkl",
    "models/b56b/version=1/hgb_poisson_core_wave_24h.pkl",
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
    dag_id="maritime_arrival_flow_temporal_baselines",
    description=(
        "B56B: strict temporal arrival-count baselines for 6h, 12h and 24h "
        "with 24h purge and block-bootstrap wave ablation."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "arrival-flow",
        "time-series",
        "poisson",
        "temporal-purged",
        "wave-ablation",
    ],
)
def maritime_arrival_flow_temporal_baselines():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if config.get("arrival_flow_baseline_version") != "b56b-arrival-flow-temporal-v1.1":
            raise RuntimeError("Model Trainer does not expose B56B v1.1")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=6))
    def train_and_evaluate(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/train/b56b",
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
        for key in CRITICAL_OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            size = int(metadata["ContentLength"])
            if size <= 0:
                raise RuntimeError(f"Empty B56B output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B56B service failed: {result}")
        findings = result.get("results", {})
        if int(findings.get("temporal_leakage_violations", -1)) != 0:
            raise RuntimeError("B56B found temporal leakage")
        if findings.get("selection_used_test") is not False:
            raise RuntimeError("B56B used test data for model selection")
        if findings.get("official_protocol") != "TEMPORAL_70_15_15_PURGED_24H":
            raise RuntimeError("B56B temporal protocol is not the frozen contract")
        allowed = {
            "READY_FOR_ARRIVAL_FLOW_MVP",
            "BASELINES_ONLY_NO_ML_UPLIFT",
            "NEED_DATA_REPAIR",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(f"Unknown B56B decision: {findings.get('status')}")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "selected_models": findings.get("selected_models"),
            "validation_uplift": findings.get(
                "validation_uplift_vs_best_baseline_pct"
            ),
            "wave_decisions": findings.get("wave_decisions"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = train_and_evaluate(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_arrival_flow_temporal_baselines()
