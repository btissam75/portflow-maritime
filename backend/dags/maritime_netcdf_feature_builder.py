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
FEATURE_BUILDER_URL = os.getenv(
    "SMART_PORT_FEATURE_BUILDER_URL", "http://feature-builder:8090"
).rstrip("/")
MAX_FILES_PER_RUN = int(os.getenv("SMART_PORT_MAX_NETCDF_FILES_PER_RUN", "12"))


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
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feature Builder HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Feature Builder unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_netcdf_to_timescale_features",
    description="Build Tanger Med wave features from Bronze NetCDF into Silver Parquet and TimescaleDB.",
    schedule="20 * * * *",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["copernicus", "netcdf", "silver", "timescaledb", "xarray"],
)
def maritime_netcdf_to_timescale_features():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_feature_builder() -> dict:
        return request_json(f"{FEATURE_BUILDER_URL}/ready")

    @task
    def discover_netcdf_objects(readiness: dict) -> list[dict]:
        del readiness
        client = s3_client()
        paginator = client.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=BRONZE_BUCKET, Prefix="copernicus/"):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.lower().endswith((".nc", ".nc4")):
                    continue
                objects.append(
                    {
                        "source_bucket": BRONZE_BUCKET,
                        "source_key": key,
                        "source_last_modified": item["LastModified"].isoformat(),
                        "source_size_bytes": int(item["Size"]),
                        "source_etag": item["ETag"].strip('"'),
                    }
                )

        objects.sort(key=lambda item: item["source_last_modified"])
        return objects[-MAX_FILES_PER_RUN:]

    @task(
        retries=2,
        retry_delay=timedelta(seconds=45),
        max_active_tis_per_dag=1,
        execution_timeout=timedelta(minutes=30),
    )
    def process_netcdf_object(item: dict) -> dict:
        payload = {
            "source_bucket": item["source_bucket"],
            "source_key": item["source_key"],
            "output_bucket": "silver-maritime",
            "source_last_modified": item["source_last_modified"],
            "force": False,
        }
        result = request_json(
            f"{FEATURE_BUILDER_URL}/v1/process", method="POST", payload=payload
        )
        result["source_size_bytes"] = item["source_size_bytes"]
        result["source_etag"] = item["source_etag"]
        return result

    @task
    def summarize(results: list[dict]) -> dict:
        counts: dict[str, int] = {}
        total_rows = 0
        outputs = []
        for result in results:
            status = result.get("status", "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
            total_rows += int(result.get("row_count", 0))
            if result.get("output_uri"):
                outputs.append(result["output_uri"])
        return {
            "objects": len(results),
            "status_counts": counts,
            "total_rows": total_rows,
            "outputs": outputs,
        }

    readiness = ensure_feature_builder()
    objects = discover_netcdf_objects(readiness)
    results = process_netcdf_object.expand(item=objects)
    summarize(results)


maritime_netcdf_to_timescale_features()
