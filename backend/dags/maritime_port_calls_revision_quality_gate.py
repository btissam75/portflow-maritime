from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import boto3
from airflow.sdk import dag, task


S3_ENDPOINT = os.getenv("SMART_PORT_S3_ENDPOINT", "http://minio:9000")
SILVER_BUCKET = os.getenv("SMART_PORT_SILVER_BUCKET", "silver-maritime")
FEATURE_BUILDER_URL = os.getenv(
    "SMART_PORT_FEATURE_BUILDER_URL", "http://feature-builder:8090"
).rstrip("/")
SOURCE_PREFIX = os.getenv(
    "SMART_PORT_TIR_UNIT_EVENTS_PREFIX", "tir/version=1/"
)
SOURCE_FILENAME = "tir_unit_events_clean_v1.parquet"


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
        with urllib.request.urlopen(request, timeout=7200) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feature Builder HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Feature Builder unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_port_calls_revision_quality_gate",
    description=(
        "Build canonical revision-aware ETA/ETD/RTA/RTD labels and quality gates."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["tir", "port-calls", "silver", "quality", "revision-aware"],
)
def maritime_port_calls_revision_quality_gate():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_feature_builder() -> dict:
        readiness = request_json(f"{FEATURE_BUILDER_URL}/ready")
        config = request_json(f"{FEATURE_BUILDER_URL}/config")
        if config.get("quality_gate_version") != "b54b-revision-aware-v1":
            raise RuntimeError(
                "Feature Builder does not expose B54B. Rebuild compose.features.yaml."
            )
        return readiness

    @task
    def discover_unit_events(readiness: dict) -> dict:
        del readiness
        client = s3_client()
        response = client.list_objects_v2(
            Bucket=SILVER_BUCKET,
            Prefix=SOURCE_PREFIX,
        )
        candidates = sorted(
            (
                item
                for item in response.get("Contents", [])
                if item["Key"].endswith(SOURCE_FILENAME)
            ),
            key=lambda item: item["LastModified"],
        )
        if not candidates:
            raise RuntimeError(
                f"No {SOURCE_FILENAME} found under "
                f"s3://{SILVER_BUCKET}/{SOURCE_PREFIX}"
            )
        source = candidates[-1]
        return {
            "source_bucket": SILVER_BUCKET,
            "source_key": source["Key"],
            "source_size_bytes": int(source["Size"]),
            "source_etag": source["ETag"].strip('"'),
        }

    @task(
        retries=1,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(hours=2),
    )
    def run_quality_gate(source: dict) -> dict:
        payload = {
            "source_bucket": source["source_bucket"],
            "source_key": source["source_key"],
            "output_bucket": SILVER_BUCKET,
            "output_prefix": "version=1",
            "force": False,
        }
        result = request_json(
            f"{FEATURE_BUILDER_URL}/v1/port-calls/quality-gate",
            method="POST",
            payload=payload,
        )
        result["source_size_bytes"] = source["source_size_bytes"]
        result["source_etag"] = source["source_etag"]
        return result

    @task
    def enforce_and_summarize(result: dict) -> dict:
        quality = result.get("quality", {})
        decision = quality.get("decision", {})
        if not decision.get("arrival_target_ready", False):
            raise RuntimeError(
                "B54B quality gate rejected arrival modeling. "
                f"Decision: {decision}"
            )
        return {
            "status": result.get("status"),
            "run_id": result.get("run_id"),
            "port_calls": quality.get("port_calls"),
            "arrival_labeled_calls": quality.get("arrival_labeled_calls"),
            "departure_labeled_calls": quality.get("departure_labeled_calls"),
            "arrival_wave_covered_calls": quality.get(
                "arrival_wave_covered_calls"
            ),
            "multiple_eta_calls": quality.get("multiple_eta_calls"),
            "multiple_rta_calls": quality.get("multiple_rta_calls"),
            "chronology_review_calls": quality.get("chronology_review_calls"),
            "revision_risk": decision.get("revision_risk"),
            "next_block": decision.get("next_block"),
            "outputs": result.get("outputs", {}),
        }

    readiness = ensure_feature_builder()
    source = discover_unit_events(readiness)
    result = run_quality_gate(source)
    enforce_and_summarize(result)


maritime_port_calls_revision_quality_gate()
