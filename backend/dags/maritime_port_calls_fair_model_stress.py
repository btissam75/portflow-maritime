from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, task


MODEL_TRAINER_URL = os.getenv(
    "SMART_PORT_MODEL_TRAINER_URL", "http://model-trainer:8091"
).rstrip("/")
S3_ENDPOINT = os.getenv("SMART_PORT_S3_ENDPOINT", "http://minio:9000")
GOLD_BUCKET = os.getenv("SMART_PORT_GOLD_BUCKET", "gold-maritime")

MODEL_READY_KEY = (
    "datasets/b54f/version=1/port_call_one_row_model_ready_no_split_v1.parquet"
)
SPLIT_ASSIGNMENTS_KEY = "splits/b54fc/version=1/all_protocol_assignments_v1.parquet"
SPLIT_DECISION_KEY = "configs/b54fc/version=1/b54fc_split_decision_v1.json"
READINESS_CONFIG_KEY = (
    "configs/b54fd0/version=1/b54fd0_frozen_feature_config_v1.json"
)
READINESS_DECISION_KEY = (
    "configs/b54fd0/version=1/b54fd0_train_readiness_decision_v1.json"
)

CRITICAL_OUTPUT_KEYS = (
    "reports/b54fd1/version=2/01_protocol_capability_and_integrity.csv",
    "reports/b54fd1/version=2/02_metrics_by_protocol_model_split_segment.csv",
    "reports/b54fd1/version=2/03_protocol_scorecard.csv",
    "reports/b54fd1/version=2/03a_zero_median_history_baselines.csv",
    "reports/b54fd1/version=2/04_baseline_gains_bootstrap_ci.csv",
    "reports/b54fd1/version=2/05_ordered_family_ablations.csv",
    "reports/b54fd1/version=2/06_feature_usage_by_protocol_track.csv",
    "reports/b54fd1/version=2/07_target_distribution_by_protocol.csv",
    "predictions/b54fd1/version=2/08_valid_test_predictions.parquet",
    "reports/b54fd1/version=2/09_native_test_mae_comparison.png",
    "configs/b54fd1/version=2/b54fd1_model_stress_config_v2.json",
    "configs/b54fd1/version=2/b54fd1_model_stress_decision_v2.json",
)


def s3_client():
    import boto3

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
        with urllib.request.urlopen(request, timeout=43200) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_port_calls_fair_model_stress",
    description=(
        "B54F-D1: fair CatBoost and baseline comparison across Random IID, "
        "Random-by-IMO and official Temporal Purged validation protocols."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "port-calls",
        "catboost",
        "random-iid",
        "random-by-imo",
        "temporal-purged",
        "model-stress",
    ],
)
def maritime_port_calls_fair_model_stress():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if config.get("model_stress_version") != "b54fd1-ordered-ablation-model-stress-v2":
            raise RuntimeError("Model Trainer does not expose B54F-D1 v2")
        if config.get("train_readiness_version") != "b54fd0-independent-train-readiness-v2":
            raise RuntimeError("B54F-D1 requires the corrected B54F-D0 v2")
        return {"ready": ready, "config": config}

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_inputs(trainer: dict) -> dict:
        del trainer
        client = s3_client()
        objects = []
        for key in (
            MODEL_READY_KEY,
            SPLIT_ASSIGNMENTS_KEY,
            SPLIT_DECISION_KEY,
            READINESS_CONFIG_KEY,
            READINESS_DECISION_KEY,
        ):
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            objects.append(
                {
                    "uri": f"s3://{GOLD_BUCKET}/{key}",
                    "size": int(metadata["ContentLength"]),
                    "etag": metadata["ETag"].strip('"'),
                }
            )
        return {"objects": objects}

    @task(retries=0, execution_timeout=timedelta(hours=12))
    def run_model_stress(inputs: dict) -> dict:
        del inputs
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/train/b54fd1",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "model_ready_key": MODEL_READY_KEY,
                "split_assignments_key": SPLIT_ASSIGNMENTS_KEY,
                "split_decision_key": SPLIT_DECISION_KEY,
                "readiness_config_key": READINESS_CONFIG_KEY,
                "readiness_decision_key": READINESS_DECISION_KEY,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=2",
                "force": False,
            },
        )

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_outputs(stress_result: dict) -> dict:
        client = s3_client()
        objects = []
        for key in CRITICAL_OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            if int(metadata["ContentLength"]) <= 0:
                raise RuntimeError(f"Empty B54F-D1 output: s3://{GOLD_BUCKET}/{key}")
            objects.append(
                {
                    "uri": f"s3://{GOLD_BUCKET}/{key}",
                    "size": int(metadata["ContentLength"]),
                    "etag": metadata["ETag"].strip('"'),
                }
            )
        return {"run_id": stress_result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(stress_result: dict, objects: dict) -> dict:
        if stress_result.get("status") != "SUCCESS":
            raise RuntimeError(f"B54F-D1 service failed: {stress_result}")
        results = stress_result.get("results", {})
        gates = results.get("gates", {})
        decision = results.get("decision", {})
        if results.get("training_executed") is not True:
            raise RuntimeError("B54F-D1 must execute the declared model stress training")
        if decision.get("official_protocol") != "TEMPORAL_PURGED":
            raise RuntimeError("B54F-D1 official protocol must remain TEMPORAL_PURGED")
        if gates.get("test_selection_leakage") != 0:
            raise RuntimeError("B54F-D1 detected TEST-based model selection")
        if gates.get("random_results_official") is not False:
            raise RuntimeError("Random protocol results must remain diagnostic")
        if not gates.get("all_protocol_gates_passed", False):
            raise RuntimeError(f"B54F-D1 protocol gate failed: {decision}")
        if decision.get("status") == "NEED_PROTOCOL_REPAIR":
            raise RuntimeError(f"B54F-D1 requires protocol repair: {decision}")
        return {
            "status": "SUCCESS",
            "run_id": stress_result.get("run_id"),
            "source_rows": results.get("source_rows"),
            "trained_models": results.get("trained_model_count"),
            "decision": decision,
            "official_protocol": "TEMPORAL_PURGED",
            "verified_objects": len(objects.get("objects", [])),
        }

    trainer = ensure_trainer()
    inputs = verify_inputs(trainer)
    stress_result = run_model_stress(inputs)
    objects = verify_outputs(stress_result)
    enforce_and_summarize(stress_result, objects)


maritime_port_calls_fair_model_stress()
