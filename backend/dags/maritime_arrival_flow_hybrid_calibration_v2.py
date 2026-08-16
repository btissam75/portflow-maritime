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
    "reports/b56gv2/version=2/00_integrity_and_temporal_contract.csv",
    "reports/b56gv2/version=2/01_valid_calibration_search.csv",
    "reports/b56gv2/version=2/02_point_fidelity.csv",
    "reports/b56gv2/version=2/03_interval_metrics_before_after.csv",
    "reports/b56gv2/version=2/04_interval_score_paired_bootstrap.csv",
    "reports/b56gv2/version=2/05_coherence_audit.csv",
    "reports/b56gv2/version=2/README_B56G_V2.md",
    "configs/b56gv2/version=2/06_b56g_v2_decision.json",
    "predictions/b56gv2/version=2/valid_hybrid_predictions.parquet",
    "predictions/b56gv2/version=2/test_hybrid_predictions.parquet",
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
        with urllib.request.urlopen(request, timeout=7200) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Service HTTP {exc.code} at {url}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_arrival_flow_hybrid_calibration_v2",
    description=(
        "B56G-v2: preserve B56E points and recalibrate coherent adaptive "
        "P10/P50/P90 intervals for retrospective and prospective shadowing."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "arrival-flow",
        "b56e-point",
        "adaptive-conformal",
        "reconciliation",
        "shadow-only",
    ],
)
def maritime_arrival_flow_hybrid_calibration_v2():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            config.get("arrival_flow_hybrid_calibration_version")
            != "b56g-v2-b56e-adaptive-conformal-v1"
        ):
            raise RuntimeError("Model Trainer does not expose B56G-v2")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=2))
    def recalibrate(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/recalibrate/b56g-v2",
            method="POST",
            payload={
                "source_bucket": GOLD_BUCKET,
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=2",
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
                raise RuntimeError(
                    f"Empty B56G-v2 output: s3://{GOLD_BUCKET}/{key}"
                )
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B56G-v2 service failed: {result}")
        findings = result.get("results", {})
        if findings.get("training_executed") is not False:
            raise RuntimeError("B56G-v2 unexpectedly retrained a point model")
        if findings.get("selection_used_test") is not False:
            raise RuntimeError("B56G-v2 used TEST for hyperparameter selection")
        if findings.get("integrity_gates_passed") is not True:
            raise RuntimeError("B56G-v2 integrity gates failed")
        if findings.get("point_fidelity_passed") is not True:
            raise RuntimeError("B56G-v2 altered the B56E TEST point forecast")
        allowed = {
            "READY_FOR_PROSPECTIVE_SHADOW",
            "NEED_INTERVAL_RECALIBRATION",
            "NEED_RECONCILIATION_REPAIR",
            "NEED_POINT_FIDELITY_REPAIR",
            "NEED_DATA_REPAIR",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(
                f"Unknown B56G-v2 decision: {findings.get('status')}"
            )
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "coverage": findings.get("coverage_gates_passed"),
            "point_fidelity": findings.get("point_fidelity_passed"),
            "formal_promotion": findings.get("formal_promotion_allowed"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = recalibrate(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_arrival_flow_hybrid_calibration_v2()
