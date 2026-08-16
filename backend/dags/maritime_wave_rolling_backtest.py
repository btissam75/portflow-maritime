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
    "reports/b58b/version=1/01_upstream_and_data_contract.csv",
    "reports/b58b/version=1/02_temporal_split_audit.csv",
    "reports/b58b/version=1/03_feature_inventory.csv",
    "reports/b58b/version=1/04_latency_baseline_stress.csv",
    "reports/b58b/version=1/05_validation_candidate_metrics.csv",
    "reports/b58b/version=1/06_test_candidate_metrics.csv",
    "reports/b58b/version=1/07_selected_models.csv",
    "reports/b58b/version=1/08_rolling_origin_stability.csv",
    "reports/b58b/version=1/09_probabilistic_calibration.csv",
    "reports/b58b/version=1/10_extreme_state_metrics.csv",
    "reports/b58b/version=1/11_anti_leakage_audit.csv",
    "reports/b58b/version=1/12_latest_shadow_forecast.csv",
    "reports/b58b/version=1/13_model_inventory.csv",
    "reports/b58b/version=1/README_B58B.md",
    "configs/b58b/version=1/14_b58b_decision.json",
    "configs/b58b/version=1/selected_models.json",
    "models/b58b/version=1/wave_model_bank.pkl",
    "predictions/b58b/version=1/selected_point_predictions.parquet",
    "predictions/b58b/version=1/test_probabilistic_predictions.parquet",
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
        with urllib.request.urlopen(request, timeout=21600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_wave_rolling_backtest",
    description=(
        "B58B: leakage-safe multihorizon wave forecasting with latency "
        "stress, rolling-origin evaluation and adaptive conformal intervals."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "waves",
        "time-series",
        "rolling-backtest",
        "probabilistic",
        "anti-leakage",
    ],
)
def maritime_wave_rolling_backtest():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if config.get("wave_rolling_backtest_version") != (
            "b58b-wave-rolling-backtest-v1"
        ):
            raise RuntimeError("Model Trainer does not expose B58B v1")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=6))
    def run_backtest(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/train/b58b",
            method="POST",
            payload={
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
                raise RuntimeError(f"Empty B58B output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B58B service failed: {result}")
        findings = result.get("results", {})
        required_false = (
            "selection_used_test",
            "bronze_modified",
            "core_modified",
            "historical_replay_allowed",
            "production_promotion_allowed",
        )
        for field in required_false:
            if findings.get(field) is not False:
                raise RuntimeError(f"B58B invariant failed: {field} must be false")
        if findings.get("training_executed") is not True:
            raise RuntimeError("B58B did not execute the declared training")
        if int(findings.get("critical_leakage_violations", -1)) != 0:
            raise RuntimeError("B58B found critical temporal leakage")
        allowed = {
            "READY_FOR_IBI_HYBRID_ENRICHMENT",
            "KEEP_PERSISTENCE_AS_WAVE_BASELINE",
            "NEED_WAVE_MODEL_REPAIR",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(
                f"Unknown B58B decision: {findings.get('status')}"
            )
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "selected_models": findings.get("selected_models"),
            "shadow_serving_rows": findings.get("shadow_serving_rows"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = run_backtest(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_wave_rolling_backtest()
