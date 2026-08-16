from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import boto3
from airflow.decorators import dag, task


MODEL_TRAINER_URL = os.getenv(
    "SMART_PORT_MODEL_TRAINER_URL",
    "http://model-trainer:8091",
).rstrip("/")
GOLD_BUCKET = os.getenv("SMART_PORT_GOLD_BUCKET", "gold-maritime")

OUTPUT_KEYS = (
    "configs/b57e/version=1/b57e_manifest.json",
    "configs/b57e/version=1/b57e_decision.json",
    "reports/b57e/version=1/01_cv_prequential_interval_audit.csv",
    "reports/b57e/version=1/02_final_test_adaptive_diagnostic.csv",
    "reports/b57e/version=1/03_recalibration_contract.csv",
    "reports/b57e/version=1/README_B57E.md",
)


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
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
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_tir_adaptive_interval_recalibration",
    description=(
        "B57E: initialize bitemporal forecast monitoring, run past-only "
        "adaptive interval recalibration and retain drift-guard fallback."
    ),
    schedule="15 3 * * *",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "tir",
        "adaptive-conformal",
        "bitemporal",
        "drift",
        "platform-replay",
        "anti-leakage",
        "no-retraining",
    ],
)
def maritime_tir_adaptive_interval_recalibration():
    @task(retries=2, retry_delay=timedelta(seconds=20))
    def ensure_model_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        expected = "b57e-adaptive-interval-recalibration-v1"
        if config.get("adaptive_recalibration_version") != expected:
            raise RuntimeError(f"Model Trainer does not expose {expected}")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(minutes=30))
    def initialize(dependencies: dict) -> dict:
        del dependencies
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/forecast/b57e/initialize",
            method="POST",
            payload={
                "artifact_bucket": GOLD_BUCKET,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "force": False,
            },
        )

    @task(retries=0, execution_timeout=timedelta(minutes=10))
    def recalibrate(initialization: dict) -> dict:
        if initialization.get("status") != "SUCCESS":
            raise RuntimeError(f"B57E initialization failed: {initialization}")
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/forecast/b57e/recalibrate",
            method="POST",
            payload={},
        )

    @task(retries=2, retry_delay=timedelta(seconds=10))
    def verify_outputs(
        initialization: dict,
        recalibration: dict,
    ) -> dict:
        client = s3_client()
        objects = []
        for key in OUTPUT_KEYS:
            metadata = client.head_object(Bucket=GOLD_BUCKET, Key=key)
            size = int(metadata["ContentLength"])
            if size <= 0:
                raise RuntimeError(f"Empty B57E object: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        runtime = request_json(f"{MODEL_TRAINER_URL}/v1/forecast/b57e/status")
        monitoring = request_json(f"{MODEL_TRAINER_URL}/v1/forecast/b57e/monitoring")
        if runtime.get("status") != "READY":
            raise RuntimeError(f"B57E runtime did not load: {runtime}")
        return {
            "run_id": initialization.get("run_id"),
            "objects": objects,
            "runtime": runtime,
            "recalibration": recalibration,
            "monitoring_privacy": monitoring.get("privacy"),
        }

    @task
    def enforce_and_summarize(
        initialization: dict,
        recalibration: dict,
        outputs: dict,
    ) -> dict:
        decision = initialization.get("results", {})
        allowed = {
            "READY_FOR_PLATFORM_REPLAY_WAITING_LIVE_DATA",
            "READY_FOR_OPERATIONAL_POINT_FORECAST_WITH_DRIFT_GUARD",
            "READY_FOR_OPERATIONAL_ADAPTIVE_INTERVALS",
        }
        if decision.get("status") not in allowed:
            raise RuntimeError(f"B57E initialization failed: {decision}")
        if decision.get("gates_passed") is not True:
            raise RuntimeError("B57E structural gates failed")
        if decision.get("training_executed") is not False:
            raise RuntimeError("B57E must not retrain models")
        if decision.get("test_used_for_tuning") is not False:
            raise RuntimeError("B57E must not tune intervals on final test")
        if decision.get("target_or_actual_exposed") is not False:
            raise RuntimeError("B57E forecast response exposes actual targets")
        privacy = outputs.get("monitoring_privacy", {})
        if privacy.get("actuals_exposed_by_forecast_endpoint") is not False:
            raise RuntimeError("B57E monitoring privacy contract failed")
        return {
            "status": "SUCCESS",
            "run_id": initialization.get("run_id"),
            "decision": decision.get("status"),
            "operating_mode": decision.get("operating_mode"),
            "source_last_date": decision.get("source_last_date"),
            "source_freshness_days": decision.get("source_freshness_days"),
            "active_adaptive_targets": recalibration.get("active_targets"),
            "fallback": recalibration.get("fallback"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": decision.get("next_block"),
        }

    dependencies = ensure_model_trainer()
    initialization = initialize(dependencies)
    recalibration = recalibrate(initialization)
    outputs = verify_outputs(initialization, recalibration)
    enforce_and_summarize(initialization, recalibration, outputs)


maritime_tir_adaptive_interval_recalibration()
