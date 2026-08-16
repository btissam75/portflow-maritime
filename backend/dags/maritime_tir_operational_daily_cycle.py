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
GOLD_BUCKET = os.getenv("SMART_PORT_GOLD_BUCKET", "gold-maritime")


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
        with urllib.request.urlopen(request, timeout=21600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Service HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Service unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_tir_operational_daily_cycle",
    description=(
        "B57F: refresh one-day-ahead TIR features, issue a prequential "
        "forecast, capture stable outcomes and recalibrate B57E."
    ),
    schedule="0 2,8 * * *",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=[
        "maritime",
        "tir",
        "operational",
        "prequential",
        "bitemporal",
        "anti-leakage",
        "adaptive-conformal",
    ],
)
def maritime_tir_operational_daily_cycle():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_services() -> dict:
        feature_ready = request_json(f"{FEATURE_BUILDER_URL}/ready")
        feature_config = request_json(f"{FEATURE_BUILDER_URL}/config")
        model_ready = request_json(f"{MODEL_TRAINER_URL}/ready")
        model_config = request_json(f"{MODEL_TRAINER_URL}/config")
        if (
            feature_config.get("event_aware_gold_version")
            != "b57b-event-aware-daily-gold-v1"
        ):
            raise RuntimeError("Feature Builder does not expose B57B")
        if (
            model_config.get("operational_daily_cycle_version")
            != "b57f-operational-daily-cycle-v1"
        ):
            raise RuntimeError("Model Trainer does not expose B57F")
        return {
            "feature_ready": feature_ready,
            "model_ready": model_ready,
        }

    @task(retries=0, execution_timeout=timedelta(hours=4))
    def refresh_daily_gold(dependencies: dict) -> dict:
        del dependencies
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
        return result

    @task(retries=0, execution_timeout=timedelta(hours=1))
    def run_cycle(gold: dict) -> dict:
        del gold
        return request_json(
            f"{MODEL_TRAINER_URL}/v1/forecast/b57f/run",
            method="POST",
            payload={"artifact_bucket": GOLD_BUCKET},
        )

    @task
    def enforce_and_summarize(result: dict) -> dict:
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"B57F service failed: {result}")
        details = result.get("results", {})
        allowed = {
            "LIVE_CYCLE_ACTIVE",
            "LIVE_CYCLE_INSTALLED_WAITING_FRESH_SOURCE",
        }
        if details.get("status") not in allowed:
            raise RuntimeError(f"Unexpected B57F decision: {details}")
        if details.get("gates_passed") is not True:
            raise RuntimeError("B57F operational gates failed")
        if details.get("training_executed") is not False:
            raise RuntimeError("B57F must not retrain models")
        if details.get("historical_backfill_used") is not False:
            raise RuntimeError("B57F used forbidden historical backfill")
        if details.get("target_exposed_by_forecast") is not False:
            raise RuntimeError("B57F exposed an outcome in forecast response")
        return {
            "status": "SUCCESS",
            "run_id": result.get("run_id"),
            "decision": details.get("status"),
            "baseline_cutoff_date": details.get("baseline_cutoff_date"),
            "forecast_date": details.get("forecast_date"),
            "forecast_mode": details.get("forecast_mode"),
            "registered_days": details.get("registered_days"),
            "adaptive_targets": details.get("adaptive_targets"),
            "next_block": details.get("next_block"),
        }

    dependencies = ensure_services()
    gold = refresh_daily_gold(dependencies)
    result = run_cycle(gold)
    enforce_and_summarize(result)


maritime_tir_operational_daily_cycle()
