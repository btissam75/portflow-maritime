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
    "reports/b56a/version=1/01_source_inventory.csv",
    "reports/b56a/version=1/02_port_call_grain_audit.csv",
    "reports/b56a/version=1/03_timestamp_semantic_audit.csv",
    "reports/b56a/version=1/04_arrival_departure_coverage.csv",
    "reports/b56a/version=1/05_hourly_grid_continuity.csv",
    "reports/b56a/version=1/06_weather_coverage.csv",
    "reports/b56a/version=1/07_target_feasibility.csv",
    "reports/b56a/version=1/08_missingness_by_year.csv",
    "reports/b56a/version=1/09_missingness_by_vessel.csv",
    "reports/b56a/version=1/10_invalid_sequences.csv",
    "reports/b56a/version=1/11_join_cardinality_audit.csv",
    "reports/b56a/version=1/12_temporal_leakage_audit.csv",
    "reports/b56a/version=1/13_target_distribution_by_year.csv",
    "reports/b56a/version=1/14_vessel_concentration.csv",
    "reports/b56a/version=1/15_hourly_target_autocorrelation.csv",
    "reports/b56a/version=1/16_seasonality_diagnostics.csv",
    "reports/b56a/version=1/README_B56A.md",
    "configs/b56a/version=1/17_final_feasibility_decision.json",
    "datasets/b56a/version=1/port_hourly_state_feasibility_v1.parquet",
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
    dag_id="maritime_operational_dataset_feasibility_audit",
    description=(
        "B56A: verify whether current port-call and maritime data support "
        "hourly arrival-flow, occupancy and weather-impact forecasting."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "feasibility",
        "port-pressure",
        "time-series",
        "anti-leakage",
        "no-training",
    ],
)
def maritime_operational_dataset_feasibility_audit():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            config.get("operational_feasibility_version")
            != "b56a-operational-feasibility-v1.1"
        ):
            raise RuntimeError("Model Trainer does not expose B56A v1.1")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=6))
    def run_audit(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/audit/b56a",
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
                raise RuntimeError(f"Empty B56A output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B56A service failed: {result}")

        findings = result.get("results", {})
        if findings.get("training_executed") is not False:
            raise RuntimeError("B56A must not train a model")
        if findings.get("split_created") is not False:
            raise RuntimeError("B56A must not create a split")
        if findings.get("bronze_modified") is not False:
            raise RuntimeError("B56A must not modify Bronze")
        if int(findings.get("critical_leakage_violations", -1)) != 0:
            raise RuntimeError("B56A found critical temporal leakage")

        allowed = {
            "READY_FOR_PORT_PRESSURE_BASELINES",
            "READY_FOR_ARRIVAL_FLOW_ONLY",
            "NEED_DATA_REPAIR",
            "DATASET_NOT_SUITABLE",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(f"Unknown B56A decision: {findings.get('status')}")

        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "readiness": findings.get("readiness"),
            "source_calls": findings.get("source_calls"),
            "hourly_rows": findings.get("hourly_rows"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = run_audit(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_operational_dataset_feasibility_audit()
