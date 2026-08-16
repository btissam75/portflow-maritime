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
    "reports/b57a/version=1/01_source_inventory.csv",
    "reports/b57a/version=1/02_business_event_calendar.csv",
    "reports/b57a/version=1/03_daily_port_call_metrics.csv",
    "reports/b57a/version=1/04_daily_weather_metrics.csv",
    "reports/b57a/version=1/05_daily_tir_metrics.csv",
    "reports/b57a/version=1/06_monthly_metric_panel.csv",
    "reports/b57a/version=1/07_source_completeness_diagnostics.csv",
    "reports/b57a/version=1/08_change_point_candidates.csv",
    "reports/b57a/version=1/09_event_impact_analysis.csv",
    "reports/b57a/version=1/10_numeric_drift_by_year.csv",
    "reports/b57a/version=1/11_categorical_drift_by_year.csv",
    "reports/b57a/version=1/12_seasonal_profiles.csv",
    "reports/b57a/version=1/13_regime_timeline.csv",
    "reports/b57a/version=1/14_temporal_semantics_and_leakage.csv",
    "reports/b57a/version=1/README_B57A.md",
    "configs/b57a/version=1/15_final_regime_decision.json",
    "datasets/b57a/version=1/monthly_regime_panel_v1.parquet",
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
    dag_id="maritime_temporal_regime_and_data_break_audit",
    description=(
        "B57A: distinguish seasonality, known business events, source data "
        "breaks and unexplained 2025-2026 maritime regime changes."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "time-series",
        "regime-change",
        "business-events",
        "data-quality",
        "no-training",
    ],
)
def maritime_temporal_regime_and_data_break_audit():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        expected = "b57a-temporal-regime-event-audit-v1.2"
        if config.get("temporal_regime_audit_version") != expected:
            raise RuntimeError(f"Model Trainer does not expose {expected}")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=6))
    def run_audit(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/audit/b57a",
            method="POST",
            payload={
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "materialize_event_calendar": True,
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
                raise RuntimeError(f"Empty B57A output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B57A service failed: {result}")

        findings = result.get("results", {})
        if findings.get("training_executed") is not False:
            raise RuntimeError("B57A must not train a model")
        if findings.get("split_created") is not False:
            raise RuntimeError("B57A must not create a split")
        if findings.get("bronze_modified") is not False:
            raise RuntimeError("B57A must not modify Bronze")
        if int(findings.get("critical_leakage_violations", -1)) != 0:
            raise RuntimeError("B57A found a temporal leakage policy violation")

        allowed = {
            "READY_FOR_EVENT_AWARE_PRE_BREAK_FEATURES",
            "READY_FOR_EVENT_AWARE_TEMPORAL_FEATURES",
            "NEED_TEMPORAL_DATA_REPAIR",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(f"Unknown B57A decision: {findings.get('status')}")

        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "first_sustained_breaks": findings.get("first_sustained_breaks"),
            "safe_periods": findings.get("safe_periods"),
            "material_event_effects": findings.get("material_event_effects"),
            "material_2026_numeric_drifts": findings.get(
                "material_2026_numeric_drifts"
            ),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = run_audit(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_temporal_regime_and_data_break_audit()
