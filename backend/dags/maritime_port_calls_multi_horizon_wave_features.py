from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, task


FEATURE_BUILDER_URL = os.getenv(
    "SMART_PORT_FEATURE_BUILDER_URL", "http://feature-builder:8090"
).rstrip("/")
SILVER_BUCKET = os.getenv("SMART_PORT_SILVER_BUCKET", "silver-maritime")


def request_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10800) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feature Builder HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Feature Builder unavailable at {url}: {exc}") from exc


@dag(
    dag_id="maritime_port_calls_multi_horizon_wave_features",
    description=(
        "Build leakage-safe ETA-24h/-12h/-6h/-3h sea-state and vessel-history features."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["tir", "port-calls", "waves", "multi-horizon", "anti-leakage"],
)
def maritime_port_calls_multi_horizon_wave_features():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ensure_feature_builder() -> dict:
        readiness = request_json(f"{FEATURE_BUILDER_URL}/ready")
        config = request_json(f"{FEATURE_BUILDER_URL}/config")
        if config.get("wave_feature_version") != "b54c-wave-history-v1":
            raise RuntimeError(
                "Feature Builder does not expose B54C. Rebuild feature-builder first."
            )
        return {"readiness": readiness, "config": config}

    @task(
        retries=1,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(hours=3),
    )
    def build_wave_features(dependencies: dict) -> dict:
        del dependencies
        return request_json(
            f"{FEATURE_BUILDER_URL}/v1/port-calls/wave-features",
            method="POST",
            payload={
                "output_bucket": SILVER_BUCKET,
                "output_prefix": "version=1",
                "horizons_h": [24, 12, 6, 3],
                "force": False,
            },
        )

    @task
    def enforce_and_summarize(result: dict) -> dict:
        quality = result.get("quality", {})
        decision = quality.get("decision", {})
        if int(quality.get("temporal_leakage_violations", -1)) != 0:
            raise RuntimeError(f"B54C temporal leakage gate failed: {quality}")
        if not decision.get("ready_for_modeling", False):
            raise RuntimeError(f"B54C modeling gate rejected the dataset: {decision}")
        return {
            "status": result.get("status"),
            "run_id": result.get("run_id"),
            "calls_loaded": quality.get("calls_loaded"),
            "snapshots_total": quality.get("snapshots_total"),
            "snapshots_model_ready": quality.get("snapshots_model_ready"),
            "model_ready_pct": quality.get("model_ready_pct"),
            "feature_count": quality.get("model_feature_count"),
            "horizon_report": quality.get("horizon_report"),
            "next_block": decision.get("next_block"),
            "outputs": result.get("outputs", {}),
        }

    dependencies = ensure_feature_builder()
    result = build_wave_features(dependencies)
    enforce_and_summarize(result)


maritime_port_calls_multi_horizon_wave_features()
