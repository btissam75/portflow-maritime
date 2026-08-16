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
    "reports/b58a/version=1/00_source_schema.csv",
    "reports/b58a/version=1/01_source_inventory.csv",
    "reports/b58a/version=1/02_timestamp_semantics.csv",
    "reports/b58a/version=1/03_hourly_continuity.csv",
    "reports/b58a/version=1/04_variable_coverage.csv",
    "reports/b58a/version=1/05_missing_runs.csv",
    "reports/b58a/version=1/06_descriptive_statistics.csv",
    "reports/b58a/version=1/07_seasonality_profiles.csv",
    "reports/b58a/version=1/08_autocorrelation_by_lag.csv",
    "reports/b58a/version=1/09_cross_correlation.csv",
    "reports/b58a/version=1/10_drift_by_period.csv",
    "reports/b58a/version=1/11_psi_by_year.csv",
    "reports/b58a/version=1/12_extreme_frequency.csv",
    "reports/b58a/version=1/13_forecast_target_feasibility.csv",
    "reports/b58a/version=1/14_anti_leakage_contract.csv",
    "reports/b58a/version=1/README_B58A.md",
    "configs/b58a/version=1/15_b58a_final_decision.json",
    "datasets/b58a/version=1/maritime_weather_hourly_past_only_v1.parquet",
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
    url: str, method: str = "GET", payload: dict | None = None
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
    dag_id="maritime_weather_timeseries_feasibility_audit",
    description=(
        "B58A: build and audit an hourly maritime weather time series without "
        "training, interpolation, split creation, or source mutation."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "weather",
        "waves",
        "time-series",
        "anti-leakage",
        "no-training",
    ],
)
def maritime_weather_timeseries_feasibility_audit():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            config.get("weather_timeseries_audit_version")
            != "b58a-weather-timeseries-feasibility-v1"
        ):
            raise RuntimeError("Model Trainer does not expose B58A v1")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=4))
    def run_audit(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/audit/b58a",
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
                raise RuntimeError(f"Empty B58A output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B58A service failed: {result}")

        findings = result.get("results", {})
        for field in (
            "training_executed",
            "split_created",
            "bronze_modified",
            "interpolation_executed",
        ):
            if findings.get(field) is not False:
                raise RuntimeError(f"B58A invariant failed: {field} must be false")
        if int(findings.get("critical_leakage_violations", -1)) != 0:
            raise RuntimeError("B58A found critical temporal leakage")

        allowed = {
            "READY_FOR_MULTIVARIATE_WEATHER_BASELINES",
            "READY_FOR_WAVE_ONLY_TEMPORAL_BASELINES",
            "NEED_WEATHER_DATA_REPAIR",
            "WEATHER_SERIES_NOT_FORECASTABLE",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(
                f"Unknown B58A decision: {findings.get('status')}"
            )
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "hourly_rows": findings.get("hourly_rows"),
            "wave_track_ready": findings.get("wave_track_ready"),
            "full_weather_track_ready": findings.get(
                "full_weather_track_ready"
            ),
            "historical_replay_allowed": findings.get(
                "historical_replay_allowed"
            ),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = run_audit(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_weather_timeseries_feasibility_audit()
