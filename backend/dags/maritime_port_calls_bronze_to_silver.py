from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import boto3
from airflow.sdk import dag, task


S3_ENDPOINT = os.getenv("SMART_PORT_S3_ENDPOINT", "http://minio:9000")
BRONZE_BUCKET = os.getenv("SMART_PORT_BRONZE_BUCKET", "bronze-maritime")
SILVER_BUCKET = os.getenv("SMART_PORT_SILVER_BUCKET", "silver-maritime")
FEATURE_BUILDER_URL = os.getenv(
    "SMART_PORT_FEATURE_BUILDER_URL", "http://feature-builder:8090"
).rstrip("/")
SOURCE_PREFIX = os.getenv(
    "SMART_PORT_TIR_BRONZE_PREFIX", "tir/source/version=1/"
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
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feature Builder HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Feature Builder unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_port_calls_bronze_to_silver",
    description="Validate and transform TIR vessel/port-call Bronze data into Silver.",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["tir", "port-calls", "bronze", "silver", "duckdb"],
)
def maritime_port_calls_bronze_to_silver():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_feature_builder() -> dict:
        return request_json(f"{FEATURE_BUILDER_URL}/ready")

    @task
    def discover_source_bundle(readiness: dict) -> dict:
        del readiness
        client = s3_client()
        response = client.list_objects_v2(Bucket=BRONZE_BUCKET, Prefix=SOURCE_PREFIX)
        objects = response.get("Contents", [])
        parquet = sorted(
            (item for item in objects if item["Key"].lower().endswith(".parquet")),
            key=lambda item: item["LastModified"],
        )
        manifests = sorted(
            (
                item
                for item in objects
                if item["Key"].lower().endswith("manifest.json")
            ),
            key=lambda item: item["LastModified"],
        )
        if not parquet:
            raise RuntimeError(
                f"No Parquet object found in s3://{BRONZE_BUCKET}/{SOURCE_PREFIX}"
            )
        if not manifests:
            raise RuntimeError(
                f"No manifest JSON found in s3://{BRONZE_BUCKET}/{SOURCE_PREFIX}"
            )
        source = parquet[-1]
        manifest = manifests[-1]
        return {
            "source_bucket": BRONZE_BUCKET,
            "source_key": source["Key"],
            "manifest_key": manifest["Key"],
            "source_size_bytes": int(source["Size"]),
            "source_etag": source["ETag"].strip('"'),
        }

    @task(
        retries=1,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(hours=2),
    )
    def process_source_bundle(bundle: dict) -> dict:
        payload = {
            "source_bucket": bundle["source_bucket"],
            "source_key": bundle["source_key"],
            "manifest_key": bundle["manifest_key"],
            "output_bucket": SILVER_BUCKET,
            "output_prefix": "version=1",
            "force": False,
        }
        result = request_json(
            f"{FEATURE_BUILDER_URL}/v1/port-calls/process",
            method="POST",
            payload=payload,
        )
        result["source_size_bytes"] = bundle["source_size_bytes"]
        result["source_etag"] = bundle["source_etag"]
        return result

    @task
    def summarize(result: dict) -> dict:
        quality = result.get("quality", {})
        return {
            "status": result.get("status"),
            "run_id": result.get("run_id"),
            "source_uri": result.get("source_uri"),
            "source_rows": quality.get("source_rows"),
            "linked_rows": quality.get("linked_rows"),
            "quarantine_rows": quality.get("quarantine_rows"),
            "port_calls": quality.get("port_calls"),
            "arrival_label_pct": quality.get("arrival_label_pct"),
            "departure_label_pct": quality.get("departure_label_pct"),
            "outputs": result.get("outputs", {}),
        }

    readiness = ensure_feature_builder()
    bundle = discover_source_bundle(readiness)
    result = process_source_bundle(bundle)
    summarize(result)


maritime_port_calls_bronze_to_silver()
