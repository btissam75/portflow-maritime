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
    "reports/b56gv21-shadow/version=1/00_prospective_contract_checks.csv",
    "reports/b56gv21-shadow/version=1/01_shadow_inventory.csv",
    "reports/b56gv21-shadow/version=1/02_rolling_shadow_metrics.csv",
    "reports/b56gv21-shadow/version=1/03_matured_forecast_observations.csv",
    "reports/b56gv21-shadow/version=1/README_B56G_V21_SHADOW.md",
    "configs/b56gv21-shadow/version=1/04_b56g_v21_shadow_decision.json",
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
    url: str,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Service HTTP {exc.code} at {url}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_arrival_flow_prospective_shadow_monitor_v21",
    description=(
        "Monitor immutable prospective B56G-v2.1 forecasts against "
        "matured, availability-stamped arrival-flow observations."
    ),
    schedule="15 4 * * *",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "arrival-flow",
        "prospective-shadow",
        "available-at",
        "no-test-reuse",
    ],
)
def maritime_arrival_flow_prospective_shadow_monitor_v21():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            config.get("arrival_flow_shadow_monitor_version")
            != "b56g-v2.1-prospective-shadow-monitor-v1"
        ):
            raise RuntimeError(
                "Model Trainer does not expose the B56G-v2.1 monitor"
            )
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=1))
    def monitor(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/monitor/b56g-v2-1-shadow",
            method="POST",
            payload={
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "as_of": None,
                "auto_capture_observations": True,
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
                raise RuntimeError(
                    f"Empty shadow output: s3://{GOLD_BUCKET}/{key}"
                )
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"Shadow monitor service failed: {result}")
        findings = result.get("results", {})
        if findings.get("training_executed") is not False:
            raise RuntimeError("Shadow monitor unexpectedly trained a model")
        if findings.get("selection_used_test") is not False:
            raise RuntimeError("Shadow monitor used TEST for selection")
        if findings.get("historical_test_reused_as_prospective") is not False:
            raise RuntimeError("Historical TEST was relabeled as prospective")
        if findings.get("contract_gates_passed") is not True:
            raise RuntimeError("Prospective shadow contract failed")
        allowed = {
            "WAITING_FOR_PROSPECTIVE_FORECASTS",
            "WAITING_FOR_MATURE_LABELS",
            "COLLECTING_PROSPECTIVE_EVIDENCE",
            "PROSPECTIVE_SHADOW_WARNING",
            "READY_FOR_CONTROLLED_CANARY",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(
                f"Unknown shadow decision: {findings.get('status')}"
            )
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "forecast_rows": findings.get("forecast_rows"),
            "observed_forecasts": findings.get("observed_forecasts"),
            "quality_gates": findings.get("quality_gates_passed"),
            "controlled_canary": findings.get(
                "controlled_canary_allowed"
            ),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = monitor(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_arrival_flow_prospective_shadow_monitor_v21()
