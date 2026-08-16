import os
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError


ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

BUCKETS = (
    "bronze-maritime",
    "silver-maritime",
    "gold-maritime",
    "mlflow-artifacts",
)


def build_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
    )


def wait_for_storage(client, attempts=30):
    for attempt in range(1, attempts + 1):
        try:
            client.list_buckets()
            return
        except (BotoCoreError, ClientError, EndpointConnectionError) as exc:
            if attempt == attempts:
                raise RuntimeError("MinIO did not become ready") from exc
            time.sleep(2)


def main():
    client = build_client()
    wait_for_storage(client)

    existing = {item["Name"] for item in client.list_buckets().get("Buckets", [])}
    for bucket in BUCKETS:
        if bucket not in existing:
            client.create_bucket(Bucket=bucket)
            print(f"Created bucket: {bucket}")
        else:
            print(f"Bucket already exists: {bucket}")

    for bucket in ("gold-maritime", "mlflow-artifacts"):
        client.put_bucket_versioning(
            Bucket=bucket,
            VersioningConfiguration={"Status": "Enabled"},
        )
        print(f"Versioning enabled: {bucket}")


if __name__ == "__main__":
    main()
