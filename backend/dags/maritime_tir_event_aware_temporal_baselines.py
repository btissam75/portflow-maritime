from __future__ import annotations

import json
import os
import time
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
    "configs/b57c/version=1/b57c_decision.json",
    "reports/b57c/version=1/00_feature_contract.csv",
    "reports/b57c/version=1/01_anti_leakage_audit.csv",
    "reports/b57c/version=1/02_walk_forward_split_audit.csv",
    "reports/b57c/version=1/03_cv_fold_metrics.csv",
    "reports/b57c/version=1/04_cv_model_summary.csv",
    "reports/b57c/version=1/05_final_test_metrics.csv",
    "reports/b57c/version=1/06_model_selection.csv",
    "reports/b57c/version=1/07_feature_family_ablation_bootstrap.csv",
    "reports/b57c/version=1/08_target_stability_by_year.csv",
    "reports/b57c/version=1/09_feature_usage_and_zero_variance.csv",
    "datasets/b57c/version=1/temporal_split_audit.parquet",
    "predictions/b57c/version=1/cv_predictions.parquet",
    "predictions/b57c/version=1/test_predictions.parquet",
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
    dag_id="maritime_tir_event_aware_temporal_baselines",
    description=(
        "B57C: strict walk-forward TIR volume, duration and long-stay baselines "
        "with weather/event/port ablations and a post-break official track."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "tir",
        "event-aware",
        "walk-forward",
        "anti-leakage",
        "baselines",
        "weather-ablation",
    ],
)
def maritime_tir_event_aware_temporal_baselines():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_model_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        expected = "b57c-event-aware-temporal-baselines-v1.1"
        if config.get("event_aware_baseline_version") != expected:
            raise RuntimeError(f"Model Trainer does not expose {expected}")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(minutes=5))
    def start_training(dependencies: dict) -> dict:
        del dependencies
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/train/b57c/start",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "source_key": (
                    "datasets/b57b/version=1/"
                    "tir_daily_predictive_gold_v1.parquet"
                ),
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "force": False,
            },
        )

    @task(retries=0, execution_timeout=timedelta(hours=12))
    def wait_for_training(started: dict) -> dict:
        if started.get("state") not in {"STARTING", "RUNNING"}:
            raise RuntimeError(f"B57C did not start: {started}")
        deadline = time.monotonic() + 11.5 * 3600
        poll = 0
        while time.monotonic() < deadline:
            status = request_json(f"{MODEL_TRAINER_URL}/v1/train/b57c/status")
            state = status.get("state")
            if state == "SUCCESS":
                result = status.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(f"B57C returned no result: {status}")
                return result
            if state == "FAILED":
                raise RuntimeError(f"B57C background job failed: {status.get('error')}")
            if state not in {"STARTING", "RUNNING"}:
                raise RuntimeError(f"Unexpected B57C background state: {status}")
            poll += 1
            if poll % 10 == 0:
                print(f"B57C is {state}; polling continues (poll={poll})")
            time.sleep(30)
        raise RuntimeError("B57C exceeded the 11.5-hour polling deadline")

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_outputs(result: dict) -> dict:
        client = s3_client()
        objects = []
        for key in OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            size = int(metadata["ContentLength"])
            if size <= 0:
                raise RuntimeError(f"Empty B57C object: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B57C service failed: {result}")
        decision = result.get("results", {})
        allowed = {
            "READY_FOR_EVENT_AWARE_MVP",
            "BASELINES_ONLY_NO_STABLE_ML_UPLIFT",
        }
        if decision.get("status") not in allowed:
            raise RuntimeError(f"B57C decision is not deployable: {decision}")
        if decision.get("gates_passed") is not True:
            raise RuntimeError("B57C quality gates failed")
        if int(decision.get("critical_leakage_violations", -1)) != 0:
            raise RuntimeError("B57C anti-leakage gate failed")
        if decision.get("selection_used_test") is not False:
            raise RuntimeError("B57C improperly used final test for model selection")
        if decision.get("official_track") != "FULL_NO_PORT":
            raise RuntimeError("B57C official track must exclude broken port history")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": decision.get("status"),
            "official_track": decision.get("official_track"),
            "selected_models": decision.get("selected_models"),
            "stable_official_models": decision.get("stable_official_models"),
            "weather_ablation": decision.get("weather_ablation"),
            "known_event_ablation": decision.get("known_event_ablation"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": decision.get("next_block"),
        }

    dependencies = ensure_model_trainer()
    started = start_training(dependencies)
    result = wait_for_training(started)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_tir_event_aware_temporal_baselines()
