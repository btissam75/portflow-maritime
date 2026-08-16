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
    "reports/b56g/version=1/00_integrity_and_anti_leakage.csv",
    "reports/b56g/version=1/01_target_count_dispersion.csv",
    "reports/b56g/version=1/02_expert_fit_summary.csv",
    "reports/b56g/version=1/03_valid_candidate_metrics.csv",
    "reports/b56g/version=1/04_valid_expert_ranking.csv",
    "reports/b56g/version=1/05_test_locked_metrics.csv",
    "reports/b56g/version=1/06_valid_calibration_search.csv",
    "reports/b56g/version=1/07_valid_probabilistic_metrics.csv",
    "reports/b56g/version=1/08_test_probabilistic_metrics.csv",
    "reports/b56g/version=1/09_test_paired_day_bootstrap.csv",
    "reports/b56g/version=1/10_coherence_audit.csv",
    "reports/b56g/version=1/README_B56G.md",
    "configs/b56g/version=1/11_b56g_decision.json",
    "predictions/b56g/version=1/valid_expert_predictions.parquet",
    "predictions/b56g/version=1/test_expert_predictions.parquet",
    "models/b56g/version=1/selected_expert_bundle.pkl",
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
        with urllib.request.urlopen(request, timeout=21600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_arrival_flow_expert_count",
    description=(
        "B56G: coherent incremental count experts, Dynamic Negative Binomial, "
        "matured online adaptation and adaptive conformal P10/P50/P90."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "arrival-flow",
        "negative-binomial",
        "adaptive-conformal",
        "temporal-reconciliation",
        "test-locked",
    ],
)
def maritime_arrival_flow_expert_count():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            config.get("arrival_flow_expert_count_version")
            != "b56g-expert-probabilistic-count-v1"
        ):
            raise RuntimeError("Model Trainer does not expose B56G v1")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=4))
    def train_validate_and_package(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/train/b56g",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "materialize_timescale": True,
                "simulation_samples": 400,
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
                raise RuntimeError(f"Empty B56G output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B56G service failed: {result}")
        findings = result.get("results", {})
        if findings.get("training_executed") is not True:
            raise RuntimeError("B56G did not record model training")
        if findings.get("selection_used_test") is not False:
            raise RuntimeError("B56G used locked TEST for selection")
        if findings.get("integrity_gates_passed") is not True:
            raise RuntimeError("B56G integrity gates failed")
        if findings.get("coherence_gates_passed") is not True:
            raise RuntimeError("B56G coherence gates failed")
        allowed = {
            "READY_FOR_B56G_SHADOW_REPLAY",
            "NEED_INTERVAL_RECALIBRATION",
            "KEEP_B56E_POINT_MODEL",
            "NEED_RECONCILIATION_REPAIR",
            "NEED_DATA_REPAIR",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(f"Unknown B56G decision: {findings.get('status')}")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "selected_model": findings.get("selected_model"),
            "coverage_gates": findings.get("coverage_gates_passed"),
            "point_non_inferior": findings.get("point_non_inferior"),
            "point_improvement_horizons": findings.get(
                "point_improvement_horizons"
            ),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = train_validate_and_package(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_arrival_flow_expert_count()
