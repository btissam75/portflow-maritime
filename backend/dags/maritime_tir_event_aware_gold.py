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
S3_ENDPOINT = os.getenv("SMART_PORT_S3_ENDPOINT", "http://minio:9000")
GOLD_BUCKET = os.getenv("SMART_PORT_GOLD_BUCKET", "gold-maritime")

OUTPUT_KEYS = (
    "datasets/b57b/version=1/tir_daily_predictive_gold_v1.parquet",
    "datasets/b57b/version=1/tir_daily_explanatory_gold_v1.parquet",
    "configs/b57b/version=1/b57b_feature_registry_v1.json",
    "configs/b57b/version=1/b57b_build_decision_v1.json",
    "reports/b57b/version=1/01_schema_and_roles.csv",
    "reports/b57b/version=1/02_missingness.csv",
    "reports/b57b/version=1/03_anti_leakage_audit.csv",
    "reports/b57b/version=1/04_event_coverage.csv",
    "reports/b57b/version=1/05_business_event_calendar.csv",
    "reports/b57b/version=1/README_B57B.md",
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
    dag_id="maritime_tir_event_aware_gold",
    description=(
        "B57B: build daily predictive and explanatory TIR Gold datasets with "
        "past-only weather, known calendar events and source-break guards."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "tir",
        "event-aware",
        "gold",
        "past-only",
        "anti-leakage",
        "no-training",
    ],
)
def maritime_tir_event_aware_gold():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_feature_builder() -> dict:
        ready = request_json(f"{FEATURE_BUILDER_URL}/ready")
        config = request_json(f"{FEATURE_BUILDER_URL}/config")
        expected = "b57b-event-aware-daily-gold-v1"
        if config.get("event_aware_gold_version") != expected:
            raise RuntimeError(f"Feature Builder does not expose {expected}")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=4))
    def build_gold(dependencies: dict) -> dict:
        del dependencies
        return request_json(
            f"{FEATURE_BUILDER_URL}/v1/tir/event-aware-gold",
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
        for key in OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            size = int(metadata["ContentLength"])
            if size <= 0:
                raise RuntimeError(f"Empty B57B object: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B57B service failed: {result}")
        quality = result.get("quality", {})
        if quality.get("status") != "READY_FOR_EVENT_AWARE_BASELINES":
            raise RuntimeError(f"B57B quality gates failed: {quality}")
        if int(quality.get("duplicate_days", -1)) != 0:
            raise RuntimeError("B57B daily grain is not unique")
        if int(quality.get("critical_leakage_violations", -1)) != 0:
            raise RuntimeError("B57B anti-leakage gate failed")
        if quality.get("training_executed") is not False:
            raise RuntimeError("B57B must not train a model")
        if quality.get("split_created") is not False:
            raise RuntimeError("B57B must not create a split")
        if quality.get("bronze_modified") is not False:
            raise RuntimeError("B57B must not modify Bronze")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": quality.get("status"),
            "predictive_rows": quality.get("predictive_rows"),
            "model_ready_rows": quality.get("model_ready_rows"),
            "predictive_features": quality.get("predictive_features"),
            "targets": quality.get("targets"),
            "port_safe_end": quality.get("port_safe_end"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": quality.get("next_block"),
        }

    dependencies = ensure_feature_builder()
    result = build_gold(dependencies)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_tir_event_aware_gold()
