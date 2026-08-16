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
FEATURE_CONFIG_KEY = "configs/b54f/version=1/b54f_feature_config_v1.json"
SPLIT_ASSIGNMENTS_KEY = "splits/b54fc/version=1/all_protocol_assignments_v1.parquet"
SPLIT_DECISION_KEY = "configs/b54fc/version=1/b54fc_split_decision_v1.json"
BUILD_REPORT_KEY = "reports/b54f/version=1/b54fa_build_report_v1.json"

CRITICAL_OUTPUT_KEYS = (
    "reports/b54fd0/version=1/01_schema_audit.csv",
    "reports/b54fd0/version=1/02_recalculation_sample_manifest.csv",
    "reports/b54fd0/version=1/03_independent_recalculation_checks.parquet",
    "reports/b54fd0/version=1/04_recalculation_summary.csv",
    "reports/b54fd0/version=1/05_split_missingness.csv",
    "reports/b54fd0/version=1/06_missingness_drift.csv",
    "reports/b54fd0/version=1/07_target_distribution_by_split.csv",
    "reports/b54fd0/version=1/08_train_pearson_feature_matrix.csv",
    "reports/b54fd0/version=1/09_train_spearman_feature_matrix.csv",
    "reports/b54fd0/version=1/12_high_correlation_pairs_ge_0p95.csv",
    "reports/b54fd0/version=1/13_train_feature_target_associations.csv",
    "reports/b54fd0/version=1/14_categorical_associations.csv",
    "reports/b54fd0/version=1/15_imo_name_consistency.csv",
    "reports/b54fd0/version=1/16_rolling_train_correlation_stability.csv",
    "reports/b54fd0/version=1/17_train_missingness_target_associations.csv",
    "reports/b54fd0/version=1/18_split_and_leakage_recheck.csv",
    "reports/b54fd0/version=1/20_train_pearson_heatmap.png",
    "reports/b54fd0/version=1/21_train_spearman_heatmap.png",
    "reports/b54fd0/version=1/22_train_pearson_clustered_heatmap.png",
    "reports/b54fd0/version=1/23_train_family_correlation_heatmap.png",
    "configs/b54fd0/version=1/b54fd0_frozen_feature_config_v1.json",
    "configs/b54fd0/version=1/b54fd0_train_readiness_decision_v1.json",
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
    dag_id="maritime_port_calls_train_readiness_audit",
    description=(
        "B54F-D0: independently recompute leakage-safe T-24 features, audit "
        "missingness/correlations/stability and freeze TRAIN-only features."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "port-calls",
        "anti-leakage",
        "correlation",
        "missingness",
        "train-readiness",
        "no-training",
    ],
)
def maritime_port_calls_train_readiness_audit():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            config.get("train_readiness_version")
            != "b54fd0-independent-train-readiness-v2"
        ):
            raise RuntimeError("Model Trainer does not expose B54F-D0")
        return {"ready": ready, "config": config}

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_inputs(trainer: dict) -> dict:
        del trainer
        client = s3_client()
        objects = []
        for key in (
            MODEL_READY_KEY,
            FEATURE_CONFIG_KEY,
            SPLIT_ASSIGNMENTS_KEY,
            SPLIT_DECISION_KEY,
            BUILD_REPORT_KEY,
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

    @task(retries=0, execution_timeout=timedelta(hours=6))
    def run_train_readiness(inputs: dict) -> dict:
        del inputs
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/audit/b54fd0",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "model_ready_key": MODEL_READY_KEY,
                "feature_config_key": FEATURE_CONFIG_KEY,
                "split_assignments_key": SPLIT_ASSIGNMENTS_KEY,
                "split_decision_key": SPLIT_DECISION_KEY,
                "build_report_key": BUILD_REPORT_KEY,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "sample_size": 600,
                "numeric_atol": 0.002,
                "numeric_rtol": 0.0001,
                "force": False,
            },
        )

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_outputs(audit_result: dict) -> dict:
        client = s3_client()
        objects = []
        for key in CRITICAL_OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            if int(metadata["ContentLength"]) <= 0:
                raise RuntimeError(f"Empty B54F-D0 output: s3://{GOLD_BUCKET}/{key}")
            objects.append(
                {
                    "uri": f"s3://{GOLD_BUCKET}/{key}",
                    "size": int(metadata["ContentLength"]),
                    "etag": metadata["ETag"].strip('"'),
                }
            )
        return {"run_id": audit_result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(audit_result: dict, objects: dict) -> dict:
        results = audit_result.get("results", {})
        decision = results.get("decision", {})
        gates = results.get("gates", {})
        if audit_result.get("status") != "SUCCESS":
            raise RuntimeError(f"B54F-D0 service failed: {audit_result}")
        if results.get("training_executed") is not False:
            raise RuntimeError("B54F-D0 must not train a target model")
        if decision.get("official_protocol") != "TEMPORAL_PURGED":
            raise RuntimeError("B54F-D0 official protocol must remain TEMPORAL_PURGED")
        if not gates.get("all_critical_gates_passed", False):
            raise RuntimeError(f"B54F-D0 critical gates failed: {decision}")
        if decision.get("status") != "READY_FOR_MODEL_STRESS":
            raise RuntimeError(f"B54F-D0 rejected model stress: {decision}")
        return {
            "status": "SUCCESS",
            "run_id": audit_result.get("run_id"),
            "source_rows": results.get("source_rows"),
            "participating_rows": results.get("participating_rows"),
            "features": results.get("feature_count"),
            "frozen_features": results.get("frozen_feature_count"),
            "recalculation_failures": results.get("recalculation_failures"),
            "future_source_violations": results.get("future_source_violations"),
            "decision": decision,
            "training_executed": False,
            "verified_objects": len(objects.get("objects", [])),
        }

    trainer = ensure_trainer()
    inputs = verify_inputs(trainer)
    audit_result = run_train_readiness(inputs)
    objects = verify_outputs(audit_result)
    enforce_and_summarize(audit_result, objects)


maritime_port_calls_train_readiness_audit()
