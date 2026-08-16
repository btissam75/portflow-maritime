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
    "reports/b56c/version=1/00_source_completeness_by_month.csv",
    "reports/b56c/version=1/01_temporal_split_audit.csv",
    "reports/b56c/version=1/02_feature_contract.csv",
    "reports/b56c/version=1/03_feature_missingness.csv",
    "reports/b56c/version=1/04_anti_leakage_audit.csv",
    "reports/b56c/version=1/05_feature_redundancy.csv",
    "reports/b56c/version=1/06_metrics_valid.csv",
    "reports/b56c/version=1/07_metrics_test.csv",
    "reports/b56c/version=1/08_incremental_ablation_bootstrap.csv",
    "reports/b56c/version=1/09_target_stability.csv",
    "reports/b56c/version=1/README_B56C.md",
    "configs/b56c/version=1/10_b56c_decision.json",
    "datasets/b56c/version=1/arrival_flow_enriched_full.parquet",
    "datasets/b56c/version=1/arrival_flow_enriched_model_ready.parquet",
    "predictions/b56c/version=1/valid_predictions.parquet",
    "predictions/b56c/version=1/test_predictions.parquet",
    "models/b56c/version=1/hgb_legacy_core_6h.pkl",
    "models/b56c/version=1/hgb_enriched_history_6h.pkl",
    "models/b56c/version=1/hgb_enriched_operational_6h.pkl",
    "models/b56c/version=1/hgb_enriched_history_wave_6h.pkl",
    "models/b56c/version=1/hgb_legacy_core_12h.pkl",
    "models/b56c/version=1/hgb_enriched_history_12h.pkl",
    "models/b56c/version=1/hgb_enriched_operational_12h.pkl",
    "models/b56c/version=1/hgb_enriched_history_wave_12h.pkl",
    "models/b56c/version=1/hgb_legacy_core_24h.pkl",
    "models/b56c/version=1/hgb_enriched_history_24h.pkl",
    "models/b56c/version=1/hgb_enriched_operational_24h.pkl",
    "models/b56c/version=1/hgb_enriched_history_wave_24h.pkl",
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
    dag_id="maritime_arrival_flow_feature_enrichment",
    description=(
        "B56C: leakage-safe hourly arrival-flow enrichment, strict temporal "
        "evaluation and incremental bootstrap ablations."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "arrival-flow",
        "feature-enrichment",
        "anti-leakage",
        "temporal-purged",
    ],
)
def maritime_arrival_flow_feature_enrichment():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_trainer() -> dict:
        ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            config.get("arrival_flow_enrichment_version")
            != "b56c-arrival-flow-enrichment-v1"
        ):
            raise RuntimeError("Model Trainer does not expose B56C v1")
        return {"ready": ready, "config": config}

    @task(retries=0, execution_timeout=timedelta(hours=6))
    def enrich_train_and_evaluate(trainer: dict) -> dict:
        del trainer
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/train/b56c",
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
                raise RuntimeError(f"Empty B56C output: s3://{GOLD_BUCKET}/{key}")
            objects.append({"key": key, "size": size})
        return {"run_id": result.get("run_id"), "objects": objects}

    @task
    def enforce_and_summarize(result: dict, outputs: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B56C service failed: {result}")
        findings = result.get("results", {})
        if int(findings.get("critical_leakage_violations", -1)) != 0:
            raise RuntimeError("B56C found temporal leakage")
        if findings.get("selection_used_test") is not False:
            raise RuntimeError("B56C used test data for model selection")
        if findings.get("eta_features_used") is not False:
            raise RuntimeError("B56C unexpectedly used final ETA data")
        allowed = {
            "READY_FOR_ENRICHED_FLOW_MVP",
            "NO_STABLE_ENRICHMENT_UPLIFT",
            "NEED_DATA_REPAIR",
        }
        if findings.get("status") not in allowed:
            raise RuntimeError(f"Unknown B56C decision: {findings.get('status')}")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": findings.get("status"),
            "selected_models": findings.get("selected_models"),
            "incremental_valid_gain": findings.get(
                "incremental_valid_gain_vs_legacy_core_pct"
            ),
            "incremental_test_gain": findings.get(
                "incremental_test_gain_vs_legacy_core_pct"
            ),
            "promotion_horizons": findings.get("promotion_horizons"),
            "verified_objects": len(outputs.get("objects", [])),
            "next_block": findings.get("next_block"),
        }

    trainer = ensure_trainer()
    result = enrich_train_and_evaluate(trainer)
    outputs = verify_outputs(result)
    enforce_and_summarize(result, outputs)


maritime_arrival_flow_feature_enrichment()
