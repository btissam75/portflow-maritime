from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import boto3
from airflow.sdk import dag, task


FEATURE_BUILDER_URL = os.getenv(
    "SMART_PORT_FEATURE_BUILDER_URL", "http://feature-builder:8090"
).rstrip("/")
MODEL_TRAINER_URL = os.getenv(
    "SMART_PORT_MODEL_TRAINER_URL", "http://model-trainer:8091"
).rstrip("/")
S3_ENDPOINT = os.getenv("SMART_PORT_S3_ENDPOINT", "http://minio:9000")
GOLD_BUCKET = os.getenv("SMART_PORT_GOLD_BUCKET", "gold-maritime")

FULL_KEY = "datasets/b54f/version=1/port_call_one_row_full_no_split_v1.parquet"
MODEL_READY_KEY = (
    "datasets/b54f/version=1/port_call_one_row_model_ready_no_split_v1.parquet"
)
QUARANTINE_KEY = "quarantine/b54f/version=1/port_call_one_row_quarantine_v1.parquet"
FEATURE_CONFIG_KEY = "configs/b54f/version=1/b54f_feature_config_v1.json"
BUILD_REPORT_KEY = "reports/b54f/version=1/b54fa_build_report_v1.json"


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
    dag_id="maritime_port_calls_one_row_dataset_audit",
    description=(
        "B54F-A/B: build one Gold row per port call at ETA-24h, then audit "
        "structure, dependencies, temporal stability and leakage without split or training."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "port-calls",
        "one-row",
        "no-split",
        "correlation",
        "mutual-information",
        "anti-leakage",
    ],
)
def maritime_port_calls_one_row_dataset_audit():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_services() -> dict:
        feature_ready = request_json(f"{FEATURE_BUILDER_URL}/ready")
        feature_config = request_json(f"{FEATURE_BUILDER_URL}/config")
        trainer_ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        trainer_config = request_json(f"{MODEL_TRAINER_URL}/config")
        if feature_config.get("one_row_feature_version") != "b54f-one-row-24h-v1":
            raise RuntimeError("Feature Builder does not expose B54F-A")
        if trainer_config.get("one_row_audit_version") != "b54fb-one-row-audit-v1":
            raise RuntimeError("Model Trainer does not expose B54F-B")
        return {
            "feature_ready": feature_ready,
            "feature_config": feature_config,
            "trainer_ready": trainer_ready,
            "trainer_config": trainer_config,
        }

    @task(
        retries=0,
        execution_timeout=timedelta(hours=3),
    )
    def build_one_row_dataset(dependencies: dict) -> dict:
        del dependencies
        result = request_json(
            f"{FEATURE_BUILDER_URL}/v1/port-calls/one-row-features",
            method="POST",
            payload={
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "force": False,
            },
        )
        quality = result.get("quality", {})
        decision = quality.get("decision", {})
        if int(quality.get("duplicate_port_call_rows", -1)) != 0:
            raise RuntimeError(f"B54F-A grain gate failed: {quality}")
        if int(quality.get("temporal_leakage_violations", -1)) != 0:
            raise RuntimeError(f"B54F-A leakage gate failed: {quality}")
        if quality.get("split_created") is not False:
            raise RuntimeError("B54F-A must not create a split")
        if quality.get("training_executed") is not False:
            raise RuntimeError("B54F-A must not train a model")
        if decision.get("status") != "READY_FOR_B54F_B_AUDIT":
            raise RuntimeError(f"B54F-A rejected the dataset: {decision}")
        return result

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_gold_objects(build_result: dict) -> dict:
        client = s3_client()
        objects = []
        for key in (
            FULL_KEY,
            MODEL_READY_KEY,
            QUARANTINE_KEY,
            FEATURE_CONFIG_KEY,
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
        return {
            "build_run_id": build_result.get("run_id"),
            "objects": objects,
        }

    @task(retries=0, execution_timeout=timedelta(hours=4))
    def run_structure_dependency_audit(objects: dict) -> dict:
        del objects
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/audit/b54fb",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "full_key": FULL_KEY,
                "model_ready_key": MODEL_READY_KEY,
                "feature_config_key": FEATURE_CONFIG_KEY,
                "build_report_key": BUILD_REPORT_KEY,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "force": False,
            },
        )

    @task
    def enforce_and_summarize(build_result: dict, audit_result: dict) -> dict:
        build = build_result.get("quality", {})
        audit = audit_result.get("results", {})
        leakage = audit.get("leakage_gate", {})
        decision = audit.get("decision", {})
        if not leakage.get("passed", False):
            raise RuntimeError(f"B54F-B leakage gate failed: {leakage}")
        if decision.get("status") not in {
            "READY_FOR_TEMPORAL_SPLIT",
            "TARGET_NOT_PREDICTABLE",
        }:
            raise RuntimeError(f"B54F-B requires data repair: {decision}")
        return {
            "status": audit_result.get("status"),
            "build_run_id": build_result.get("run_id"),
            "audit_run_id": audit_result.get("run_id"),
            "full_rows": build.get("full_rows"),
            "model_ready_rows": build.get("model_ready_rows"),
            "quarantine_rows": build.get("quarantine_rows"),
            "unique_port_calls": build.get("unique_port_calls"),
            "arrived_before_cutoff_rows": build.get("arrived_before_cutoff_rows"),
            "feature_count": audit.get("feature_count"),
            "wave_feature_count": audit.get("wave_feature_count"),
            "leakage_violations": leakage.get("violations"),
            "decision": decision,
            "split_created": False,
            "training_executed": False,
            "outputs": audit_result.get("outputs", {}),
        }

    dependencies = ensure_services()
    build_result = build_one_row_dataset(dependencies)
    objects = verify_gold_objects(build_result)
    audit_result = run_structure_dependency_audit(objects)
    enforce_and_summarize(build_result, audit_result)


maritime_port_calls_one_row_dataset_audit()
