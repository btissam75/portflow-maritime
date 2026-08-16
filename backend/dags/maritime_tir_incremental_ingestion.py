from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, task


FEATURE_BUILDER_URL = os.getenv(
    "SMART_PORT_FEATURE_BUILDER_URL",
    "http://feature-builder:8090",
).rstrip("/")
MODEL_TRAINER_URL = os.getenv(
    "SMART_PORT_MODEL_TRAINER_URL",
    "http://model-trainer:8091",
).rstrip("/")
BRONZE_BUCKET = os.getenv("SMART_PORT_BRONZE_BUCKET", "bronze-maritime")
GOLD_BUCKET = os.getenv("SMART_PORT_GOLD_BUCKET", "gold-maritime")


def request_json(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: int = 21600,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_tir_incremental_ingestion",
    description=(
        "B57G: archive and merge immutable incremental TIR batches, preserve "
        "missing source days, then refresh B57B and the prequential B57F cycle."
    ),
    schedule="*/15 * * * *",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "tir",
        "incremental",
        "bronze",
        "bitemporal",
        "anti-leakage",
        "b57g",
    ],
)
def maritime_tir_incremental_ingestion():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def discover() -> dict:
        ready = request_json(f"{FEATURE_BUILDER_URL}/ready")
        config = request_json(f"{FEATURE_BUILDER_URL}/config")
        expected = "b57g-tir-incremental-ingestion-v1"
        if config.get("tir_incremental_collector_version") != expected:
            raise RuntimeError(f"Feature Builder does not expose {expected}")
        listing = request_json(f"{FEATURE_BUILDER_URL}/v1/tir/incoming?limit=20")
        return {
            "ready": ready,
            "objects": listing.get("objects", []),
            "count": int(listing.get("count", 0)),
        }

    @task(retries=0, execution_timeout=timedelta(hours=2))
    def ingest(discovery: dict) -> dict:
        results = []
        canonical_updated = False
        for item in discovery.get("objects", []):
            response = request_json(
                f"{FEATURE_BUILDER_URL}/v1/tir/incremental-ingest",
                method="POST",
                payload={
                    "source_bucket": item.get("bucket", BRONZE_BUCKET),
                    "source_key": item["key"],
                    "delete_source": True,
                    "allowed_lateness_days": 3,
                },
            )
            if response.get("status") != "SUCCESS":
                raise RuntimeError(f"B57G ingestion failed: {response}")
            result = response.get("result", {})
            canonical_updated = canonical_updated or bool(
                result.get("canonical_updated")
            )
            results.append(
                {
                    "run_id": response.get("run_id"),
                    "decision": result.get("decision"),
                    "input_rows": result.get("input_rows"),
                    "fresh_rows": result.get("fresh_rows"),
                    "source_gap_days": result.get("source_gap_days"),
                    "canonical_key": result.get("canonical_key"),
                }
            )
        return {
            "status": "SUCCESS",
            "objects_found": discovery.get("count", 0),
            "objects_processed": len(results),
            "canonical_updated": canonical_updated,
            "results": results,
        }

    @task(retries=0, execution_timeout=timedelta(hours=4))
    def refresh_gold(ingestion: dict) -> dict:
        if not ingestion.get("canonical_updated"):
            return {
                "status": "SKIPPED",
                "reason": "NO_CANONICAL_UPDATE",
                "ingestion": ingestion,
            }
        result = request_json(
            f"{FEATURE_BUILDER_URL}/v1/tir/event-aware-gold",
            method="POST",
            payload={
                "output_bucket": GOLD_BUCKET,
                "output_prefix": "version=1",
                "materialize_timescale": True,
                "force": True,
            },
        )
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B57B refresh failed: {result}")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "quality": result.get("quality", {}),
            "ingestion": ingestion,
        }

    @task(retries=0, execution_timeout=timedelta(hours=1))
    def run_operational_cycle(gold: dict) -> dict:
        if gold.get("status") == "SKIPPED":
            return {
                "status": "SUCCESS",
                "decision": "WAITING_FOR_TIR_INPUT",
                "objects_processed": gold["ingestion"].get(
                    "objects_processed", 0
                ),
            }
        result = request_json(
            f"{MODEL_TRAINER_URL}/v1/forecast/b57f/run",
            method="POST",
            payload={"artifact_bucket": GOLD_BUCKET},
        )
        details = result.get("results", {})
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B57F cycle failed: {result}")
        if details.get("historical_backfill_used") is not False:
            raise RuntimeError("B57F used forbidden historical backfill")
        return {
            "status": "SUCCESS",
            "decision": details.get("status"),
            "forecast_date": details.get("forecast_date"),
            "forecast_mode": details.get("forecast_mode"),
            "registered_days": details.get("registered_days"),
            "adaptive_targets": details.get("adaptive_targets"),
            "objects_processed": gold["ingestion"].get(
                "objects_processed", 0
            ),
        }

    discovery = discover()
    ingestion = ingest(discovery)
    gold = refresh_gold(ingestion)
    run_operational_cycle(gold)


maritime_tir_incremental_ingestion()
