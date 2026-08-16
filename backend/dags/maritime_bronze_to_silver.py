from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import boto3
import psycopg2
from airflow.sdk import dag, get_current_context, task


S3_ENDPOINT = os.environ["SMART_PORT_S3_ENDPOINT"]
BRONZE_BUCKET = os.environ["SMART_PORT_BRONZE_BUCKET"]
SILVER_BUCKET = os.environ["SMART_PORT_SILVER_BUCKET"]


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


@dag(
    dag_id="maritime_bronze_to_silver_manifest",
    description="Audit Copernicus bronze objects and publish a traceable silver manifest.",
    schedule="0 * * * *",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["maritime", "copernicus", "bronze", "silver"],
)
def maritime_bronze_to_silver_manifest():
    @task
    def ensure_storage() -> dict:
        client = s3_client()
        existing = {item["Name"] for item in client.list_buckets().get("Buckets", [])}
        required = [BRONZE_BUCKET, SILVER_BUCKET]
        missing = [bucket for bucket in required if bucket not in existing]
        if missing:
            raise RuntimeError(f"Missing required buckets: {missing}")
        return {"endpoint": S3_ENDPOINT, "buckets": required}

    @task
    def discover_copernicus_files(storage_status: dict) -> list[dict]:
        del storage_status
        client = s3_client()
        paginator = client.get_paginator("list_objects_v2")
        files = []

        for page in paginator.paginate(Bucket=BRONZE_BUCKET, Prefix="copernicus/"):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.lower().endswith((".nc", ".nc4", ".grib", ".grib2")):
                    continue
                files.append(
                    {
                        "key": key,
                        "size_bytes": int(item["Size"]),
                        "etag": item["ETag"].strip('"'),
                        "last_modified": item["LastModified"].isoformat(),
                    }
                )

        files.sort(key=lambda item: item["key"])
        return files

    @task
    def publish_manifest(files: list[dict]) -> dict:
        context = get_current_context()
        logical_date = context["logical_date"].astimezone(timezone.utc)
        manifest_key = (
            "manifests/copernicus/"
            f"{logical_date:%Y/%m/%d/%H}/manifest_{context['run_id'].replace(':', '_')}.json"
        )
        payload = {
            "schema_version": "1.0",
            "source": "copernicus",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "airflow_run_id": context["run_id"],
            "logical_date": logical_date.isoformat(),
            "source_bucket": BRONZE_BUCKET,
            "file_count": len(files),
            "total_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
        }
        body = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
        s3_client().put_object(
            Bucket=SILVER_BUCKET,
            Key=manifest_key,
            Body=body,
            ContentType="application/json",
            Metadata={"schema-version": "1.0", "source": "copernicus"},
        )
        return {
            "manifest_uri": f"s3://{SILVER_BUCKET}/{manifest_key}",
            "file_count": payload["file_count"],
            "total_bytes": payload["total_bytes"],
            "airflow_run_id": payload["airflow_run_id"],
        }

    @task
    def record_audit(manifest: dict) -> str:
        metadata = {
            "airflow_run_id": manifest["airflow_run_id"],
            "total_bytes": manifest["total_bytes"],
            "pipeline": "maritime_bronze_to_silver_manifest",
        }
        with psycopg2.connect(
            host=os.environ["SMART_PORT_DB_HOST"],
            port=int(os.environ["SMART_PORT_DB_PORT"]),
            dbname=os.environ["SMART_PORT_DB_NAME"],
            user=os.environ["SMART_PORT_DB_USER"],
            password=os.environ["SMART_PORT_DB_PASSWORD"],
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO audit.ingestion_run (
                        source_name,
                        dataset_name,
                        finished_at,
                        status,
                        object_uri,
                        row_count,
                        metadata
                    )
                    VALUES (%s, %s, now(), %s, %s, %s, %s::jsonb)
                    RETURNING run_id::text
                    """,
                    (
                        "copernicus",
                        "bronze_object_manifest",
                        "SUCCESS",
                        manifest["manifest_uri"],
                        manifest["file_count"],
                        json.dumps(metadata),
                    ),
                )
                run_id = cursor.fetchone()[0]
        return run_id

    storage = ensure_storage()
    files = discover_copernicus_files(storage)
    manifest = publish_manifest(files)
    record_audit(manifest)


maritime_bronze_to_silver_manifest()
