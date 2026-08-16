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
UPSTREAM_DECISION_KEY = "configs/b54f/version=1/b54fb_decision_v1.json"

OUTPUT_KEYS = (
    "splits/b54fc/version=1/all_protocol_assignments_v1.parquet",
    "splits/b54fc/version=1/random_iid_assignments_v1.parquet",
    "splits/b54fc/version=1/random_by_imo_assignments_v1.parquet",
    "splits/b54fc/version=1/temporal_purged_assignments_v1.parquet",
    "splits/b54fc/version=1/rolling_temporal_folds_v1.parquet",
    "reports/b54fc/version=1/01_protocol_summary.csv",
    "reports/b54fc/version=1/02_target_distribution.csv",
    "reports/b54fc/version=1/03_numeric_distribution_shift.csv",
    "reports/b54fc/version=1/04_categorical_distribution_shift.csv",
    "reports/b54fc/version=1/05_temporal_boundaries.csv",
    "reports/b54fc/version=1/06_overlap_and_purge_audit.csv",
    "configs/b54fc/version=1/b54fc_split_decision_v1.json",
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
    dag_id="maritime_port_calls_split_stress_audit",
    description=(
        "B54F-C: freeze and audit random IID, random-by-IMO, purged temporal "
        "and rolling temporal splits without target-model training."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "port-calls",
        "random-vs-temporal",
        "purged-split",
        "group-split",
        "distribution-shift",
        "no-training",
    ],
)
def maritime_port_calls_split_stress_audit():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if config.get("split_stress_version") != "b54fc-random-temporal-stress-v1":
            raise RuntimeError("Model Trainer does not expose B54F-C")
        return {"ready": ready, "config": config}

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_inputs(trainer: dict) -> dict:
        del trainer
        client = s3_client()
        objects = []
        for key in (MODEL_READY_KEY, FEATURE_CONFIG_KEY, UPSTREAM_DECISION_KEY):
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            objects.append(
                {
                    "uri": f"s3://{GOLD_BUCKET}/{key}",
                    "size": int(metadata["ContentLength"]),
                    "etag": metadata["ETag"].strip('"'),
                }
            )
        return {"objects": objects}

    @task(retries=0, execution_timeout=timedelta(hours=4))
    def build_and_audit_splits(inputs: dict) -> dict:
        del inputs
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/splits/b54fc",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "model_ready_key": MODEL_READY_KEY,
                "feature_config_key": FEATURE_CONFIG_KEY,
                "upstream_decision_key": UPSTREAM_DECISION_KEY,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "train_fraction": 0.70,
                "valid_fraction": 0.15,
                "purge_hours": 72,
                "random_seed": 42,
                "force": False,
            },
        )

    @task(retries=2, retry_delay=timedelta(seconds=15))
    def verify_outputs(split_result: dict) -> dict:
        client = s3_client()
        objects = []
        for key in OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            objects.append(
                {
                    "uri": f"s3://{GOLD_BUCKET}/{key}",
                    "size": int(metadata["ContentLength"]),
                    "etag": metadata["ETag"].strip('"'),
                }
            )
        return {"run_id": split_result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(split_result: dict, objects: dict) -> dict:
        results = split_result.get("results", {})
        decision = results.get("decision", {})
        overlap = results.get("overlap_gate", {})
        if split_result.get("status") != "SUCCESS":
            raise RuntimeError(f"B54F-C service failed: {split_result}")
        if decision.get("status") != "READY_FOR_SPLIT_STRESS_MODELS":
            raise RuntimeError(f"B54F-C split rejected: {decision}")
        if not overlap.get("passed", False):
            raise RuntimeError(f"B54F-C overlap gate failed: {overlap}")
        if results.get("training_executed") is not False:
            raise RuntimeError("B54F-C must not train a target model")
        if decision.get("official_protocol") != "TEMPORAL_PURGED":
            raise RuntimeError("B54F-C official protocol must remain TEMPORAL_PURGED")
        if decision.get("random_results_are_not_official") is not True:
            raise RuntimeError("B54F-C random protocols must remain diagnostic")
        return {
            "status": "SUCCESS",
            "run_id": split_result.get("run_id"),
            "source_rows": results.get("source_rows"),
            "protocols": results.get("protocol_summary"),
            "decision": decision,
            "split_created": True,
            "training_executed": False,
            "verified_objects": len(objects.get("objects", [])),
        }

    trainer = ensure_trainer()
    inputs = verify_inputs(trainer)
    split_result = build_and_audit_splits(inputs)
    objects = verify_outputs(split_result)
    enforce_and_summarize(split_result, objects)


maritime_port_calls_split_stress_audit()
