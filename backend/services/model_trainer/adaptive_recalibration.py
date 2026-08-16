from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import mlflow
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json

from model_trainer.probabilistic_forecast import (
    CALIBRATION_LEVELS,
    DEFAULT_BUCKET,
    MODEL_KEYS,
    SOURCE_DATASET_KEY,
    SOURCE_FEATURE_VERSION,
    TARGET_UNITS,
    _clean_json,
    _clip,
    _json_default,
    _load_model_payloads,
    _parse_prediction_date,
)


API_VERSION = "b57e-adaptive-interval-recalibration-v1"
UPSTREAM_API_VERSION = "b57d-probabilistic-forecast-api-v1"
SOURCE_NAME = "b57e_adaptive_interval_recalibration"
DATASET_NAME = "tir_daily_adaptive_probabilistic_forecast"
OUTPUT_PREFIX = "version=1"

B57D_MANIFEST_KEY = "configs/b57d/version=1/b57d_api_manifest.json"
B57C_DECISION_KEY = "configs/b57c/version=1/b57c_decision.json"
CV_PREDICTIONS_KEY = "predictions/b57c/version=1/cv_predictions.parquet"
TEST_PREDICTIONS_KEY = "predictions/b57c/version=1/test_predictions.parquet"

WINDOW_DAYS = 365
HALF_LIFE_DAYS = 90.0
MIN_LIVE_LABELS = 60
MIN_PREQUENTIAL_LABELS = 30
MIN_P90_EMPIRICAL_COVERAGE = 0.80
FRESHNESS_LIMIT_DAYS = 2
TARGET_NAMES = tuple(MODEL_KEYS)

RUNTIME_LOCK = threading.RLock()
RUNTIME_CACHE: dict[str, Any] = {}


def _db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _bytes(client, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def _json_object(client, bucket: str, key: str) -> dict[str, Any]:
    return json.loads(_bytes(client, bucket, key))


def _parquet_object(client, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_bytes(client, bucket, key)))


def _json_dumps(value: Any) -> str:
    return json.dumps(
        _clean_json(value),
        default=_json_default,
        allow_nan=False,
    )


def _checksum(client, bucket: str) -> str:
    digest = hashlib.sha256(API_VERSION.encode("ascii"))
    for key in (
        B57D_MANIFEST_KEY,
        B57C_DECISION_KEY,
        CV_PREDICTIONS_KEY,
        TEST_PREDICTIONS_KEY,
        SOURCE_DATASET_KEY,
        *MODEL_KEYS.values(),
    ):
        metadata = client.head_object(Bucket=bucket, Key=key)
        digest.update(key.encode("utf-8"))
        digest.update(str(metadata.get("ETag", "")).encode("ascii"))
        digest.update(str(metadata.get("ContentLength", 0)).encode("ascii"))
    return digest.hexdigest()


def _ensure_schema() -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE SCHEMA IF NOT EXISTS serving;

                CREATE TABLE IF NOT EXISTS serving.tir_daily_forecast_ledger (
                    forecast_id UUID PRIMARY KEY,
                    api_version TEXT NOT NULL,
                    prediction_date TIMESTAMPTZ NOT NULL,
                    target_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    point_prediction DOUBLE PRECISION NOT NULL,
                    recommended_interval_kind TEXT NOT NULL,
                    recommended_lower DOUBLE PRECISION NOT NULL,
                    recommended_upper DOUBLE PRECISION NOT NULL,
                    operating_mode TEXT NOT NULL,
                    source_feature_version TEXT NOT NULL,
                    source_last_date TIMESTAMPTZ NOT NULL,
                    issued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    CONSTRAINT tir_daily_forecast_target_check
                        CHECK (target_name IN (
                            'TIR_VOLUME', 'DURATION_MEDIAN', 'LONG_24H_RATE'
                        )),
                    CONSTRAINT tir_daily_forecast_bounds_check
                        CHECK (recommended_lower <= point_prediction
                               AND point_prediction <= recommended_upper)
                );

                CREATE INDEX IF NOT EXISTS
                    ix_tir_daily_forecast_ledger_lookup
                ON serving.tir_daily_forecast_ledger
                    (prediction_date, target_name, issued_at);

                CREATE TABLE IF NOT EXISTS
                    serving.tir_daily_forecast_observation (
                    observation_id UUID PRIMARY KEY,
                    prediction_date TIMESTAMPTZ NOT NULL,
                    target_name TEXT NOT NULL,
                    actual_value DOUBLE PRECISION NOT NULL,
                    source TEXT NOT NULL,
                    available_at TIMESTAMPTZ NOT NULL,
                    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    CONSTRAINT tir_daily_observation_target_check
                        CHECK (target_name IN (
                            'TIR_VOLUME', 'DURATION_MEDIAN', 'LONG_24H_RATE'
                        )),
                    CONSTRAINT tir_daily_observation_availability_check
                        CHECK (
                            available_at >= prediction_date + interval '1 day'
                        ),
                    CONSTRAINT tir_daily_observation_revision_unique
                        UNIQUE (
                            prediction_date, target_name, source, available_at
                        )
                );

                CREATE INDEX IF NOT EXISTS
                    ix_tir_daily_forecast_observation_lookup
                ON serving.tir_daily_forecast_observation
                    (prediction_date, target_name, available_at);

                CREATE TABLE IF NOT EXISTS
                    serving.tir_interval_calibration (
                    calibration_id UUID PRIMARY KEY,
                    api_version TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    as_of TIMESTAMPTZ NOT NULL,
                    calibration_start TIMESTAMPTZ,
                    calibration_end TIMESTAMPTZ,
                    residual_rows INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT tir_interval_calibration_target_check
                        CHECK (target_name IN (
                            'TIR_VOLUME', 'DURATION_MEDIAN', 'LONG_24H_RATE'
                        ))
                );

                CREATE INDEX IF NOT EXISTS
                    ix_tir_interval_calibration_active
                ON serving.tir_interval_calibration
                    (api_version, target_name, created_at DESC);
                """
            )


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, metadata
                FROM audit.ingestion_run
                WHERE source_name=%s AND dataset_name=%s
                  AND checksum=%s AND status='SUCCESS'
                ORDER BY finished_at DESC LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _start_run(checksum: str) -> str:
    metadata = {
        "api_version": API_VERSION,
        "upstream_api_version": UPSTREAM_API_VERSION,
        "policy": (
            "PAST_ONLY_ASYMMETRIC_WEIGHTED_CONFORMAL_"
            "NO_TEST_TUNING_DRIFT_GUARD_FALLBACK"
        ),
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit.ingestion_run
                    (source_name, dataset_name, object_uri, checksum, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    SOURCE_NAME,
                    DATASET_NAME,
                    f"s3://{DEFAULT_BUCKET}/{B57D_MANIFEST_KEY}",
                    checksum,
                    Json(metadata),
                ),
            )
            return str(cursor.fetchone()[0])


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at=now(), status=%s, row_count=%s,
                    metadata=metadata || %s, error_message=%s
                WHERE run_id=%s
                """,
                (
                    status,
                    row_count,
                    Json(_clean_json(metadata), dumps=_json_dumps),
                    error_message,
                    run_id,
                ),
            )


def _validate_upstream(
    b57d_manifest: dict[str, Any],
    b57c_decision: dict[str, Any],
) -> None:
    allowed = {
        "READY_FOR_OPERATIONAL_PROBABILISTIC_API",
        "READY_FOR_HISTORICAL_REPLAY_API_NEED_LIVE_DATA",
        "READY_FOR_HISTORICAL_REPLAY_API_WITH_DRIFT_GUARD_WARNING",
    }
    if b57d_manifest.get("api_version") != UPSTREAM_API_VERSION:
        raise RuntimeError("Unexpected B57D API version")
    if b57d_manifest.get("status") not in allowed:
        raise RuntimeError("B57D manifest is not serveable")
    if b57c_decision.get("status") != "READY_FOR_EVENT_AWARE_MVP":
        raise RuntimeError("B57C official models are not approved")
    if b57c_decision.get("selection_used_test") is not False:
        raise RuntimeError("B57C final test was used for model selection")
    if int(b57c_decision.get("critical_leakage_violations", -1)) != 0:
        raise RuntimeError("B57C leakage gate did not pass")
    if set(b57d_manifest.get("models", {})) != set(TARGET_NAMES):
        raise RuntimeError("B57D does not expose the three expected targets")


def _selected_rows(
    predictions: pd.DataFrame,
    decision: dict[str, Any],
    target_name: str,
) -> pd.DataFrame:
    selected_model = decision["selected_models"][target_name]
    rows = predictions[
        predictions["track"].eq("FULL_NO_PORT")
        & predictions["target_name"].eq(target_name)
        & predictions["model"].eq(selected_model)
    ].copy()
    rows["prediction_date"] = pd.to_datetime(rows["prediction_date"], utc=True)
    rows["actual"] = pd.to_numeric(rows["actual"], errors="coerce")
    rows["prediction"] = pd.to_numeric(rows["prediction"], errors="coerce")
    rows = rows.dropna(subset=["prediction_date", "actual", "prediction"])
    return rows.sort_values("prediction_date").reset_index(drop=True)


def _weighted_quantile(
    values: np.ndarray,
    quantile: float,
    weights: np.ndarray,
) -> float:
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        raise ValueError("No finite values for weighted quantile")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    threshold = quantile * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1
    )
    return float(values[index])


def _recency_weights(
    dates: pd.Series,
    as_of: pd.Timestamp,
    half_life_days: float = HALF_LIFE_DAYS,
) -> np.ndarray:
    parsed = pd.to_datetime(dates, utc=True)
    age_days = np.maximum(
        0.0,
        (as_of - parsed).dt.total_seconds().to_numpy(dtype="float64") / 86400.0,
    )
    return np.exp(-math.log(2.0) * age_days / half_life_days)


def _asymmetric_offsets(
    residuals: np.ndarray,
    weights: np.ndarray,
    level: float,
) -> tuple[float, float]:
    alpha = 1.0 - level
    lower = _weighted_quantile(residuals, alpha / 2.0, weights)
    upper = _weighted_quantile(residuals, 1.0 - alpha / 2.0, weights)
    return min(lower, 0.0), max(upper, 0.0)


def _interval(
    point: np.ndarray | float,
    lower_offset: float,
    upper_offset: float,
    kind: str,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    lower = _clip(np.asarray(point, dtype="float64") + lower_offset, kind)
    upper = _clip(np.asarray(point, dtype="float64") + upper_offset, kind)
    return lower, upper


def _prequential_audit(
    rows: pd.DataFrame,
    kind: str,
    level: float = 0.90,
) -> dict[str, Any]:
    rows = rows.sort_values("prediction_date").reset_index(drop=True).copy()
    rows["residual"] = rows["actual"] - rows["prediction"]
    records = []
    for index in range(MIN_PREQUENTIAL_LABELS, len(rows)):
        current = rows.iloc[index]
        cutoff = current["prediction_date"]
        history = rows.iloc[:index]
        history = history[
            history["prediction_date"].ge(cutoff - pd.Timedelta(days=WINDOW_DAYS))
        ]
        if len(history) < MIN_PREQUENTIAL_LABELS:
            continue
        weights = _recency_weights(history["prediction_date"], cutoff)
        lower_offset, upper_offset = _asymmetric_offsets(
            history["residual"].to_numpy(dtype="float64"),
            weights,
            level,
        )
        lower, upper = _interval(
            float(current["prediction"]),
            lower_offset,
            upper_offset,
            kind,
        )
        actual = float(current["actual"])
        records.append(
            {
                "prediction_date": cutoff,
                "actual": actual,
                "prediction": float(current["prediction"]),
                "lower": float(lower),
                "upper": float(upper),
                "covered": lower <= actual <= upper,
                "actual_below": actual < lower,
                "actual_above": actual > upper,
                "history_rows": len(history),
            }
        )
    if not records:
        return {
            "rows": 0,
            "coverage": None,
            "mean_width": None,
            "below_rate": None,
            "above_rate": None,
        }
    report = pd.DataFrame(records)
    return {
        "rows": len(report),
        "coverage": float(report["covered"].mean()),
        "mean_width": float((report["upper"] - report["lower"]).mean()),
        "below_rate": float(report["actual_below"].mean()),
        "above_rate": float(report["actual_above"].mean()),
    }


def _seed_offsets(rows: pd.DataFrame) -> dict[str, Any]:
    as_of = rows["prediction_date"].max() + pd.Timedelta(seconds=1)
    history = rows[
        rows["prediction_date"].ge(as_of - pd.Timedelta(days=WINDOW_DAYS))
    ].copy()
    residuals = history["actual"].to_numpy(dtype="float64") - history[
        "prediction"
    ].to_numpy(dtype="float64")
    weights = _recency_weights(history["prediction_date"], as_of)
    offsets = {}
    for level in CALIBRATION_LEVELS:
        name = f"p{int(level * 100)}"
        lower, upper = _asymmetric_offsets(residuals, weights, level)
        offsets[name] = {
            "level": level,
            "lower_offset": lower,
            "upper_offset": upper,
        }
    return {
        "method": "PAST_ONLY_WEIGHTED_ASYMMETRIC_SIGNED_RESIDUAL",
        "window_days": WINDOW_DAYS,
        "half_life_days": HALF_LIFE_DAYS,
        "rows": len(history),
        "start": history["prediction_date"].min(),
        "end": history["prediction_date"].max(),
        "offsets": offsets,
        "deployment_eligible": False,
        "reason": "CV seed is audit-only after observed 2026 regime shift",
    }


def _final_test_diagnostic(
    test_rows: pd.DataFrame,
    seed: dict[str, Any],
    kind: str,
    target_name: str,
) -> list[dict[str, Any]]:
    actual = test_rows["actual"].to_numpy(dtype="float64")
    point = test_rows["prediction"].to_numpy(dtype="float64")
    output = []
    for name, values in seed["offsets"].items():
        lower, upper = _interval(
            point,
            float(values["lower_offset"]),
            float(values["upper_offset"]),
            kind,
        )
        covered = (actual >= lower) & (actual <= upper)
        output.append(
            {
                "target_name": target_name,
                "interval_name": name,
                "nominal_coverage": values["level"],
                "test_rows": len(test_rows),
                "empirical_coverage": float(covered.mean()),
                "coverage_gap_pp": 100.0
                * (float(covered.mean()) - float(values["level"])),
                "mean_interval_width": float(np.mean(upper - lower)),
                "actual_below_rate": float((actual < lower).mean()),
                "actual_above_rate": float((actual > upper).mean()),
                "used_for_tuning": False,
            }
        )
    return output


def _audit_intervals(
    cv_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    decision: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    seeds = {}
    prequential_rows = []
    final_rows = []
    for target_name, payload in payloads.items():
        cv_rows = _selected_rows(cv_predictions, decision, target_name)
        test_rows = _selected_rows(test_predictions, decision, target_name)
        if len(cv_rows) < MIN_LIVE_LABELS or len(test_rows) < 30:
            raise RuntimeError(f"Insufficient audit rows for {target_name}")
        seeds[target_name] = _seed_offsets(cv_rows)
        audit = _prequential_audit(cv_rows, payload["kind"])
        prequential_rows.append(
            {
                "target_name": target_name,
                "selected_model": decision["selected_models"][target_name],
                "level": 0.90,
                **audit,
                "policy": (
                    "PAST_ONLY_WINDOW_365D_EXPONENTIAL_HALF_LIFE_90D_ASYMMETRIC"
                ),
            }
        )
        final_rows.extend(
            _final_test_diagnostic(
                test_rows,
                seeds[target_name],
                payload["kind"],
                target_name,
            )
        )
    return seeds, pd.DataFrame(prequential_rows), pd.DataFrame(final_rows)


def _upload(client, path: Path, bucket: str, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
    return f"s3://{bucket}/{key}"


def _log_mlflow(output_dir: Path, decision: dict[str, Any]) -> str:
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("smart-port-tir-event-aware")
        with mlflow.start_run(run_name=API_VERSION):
            mlflow.log_params(
                {
                    "api_version": API_VERSION,
                    "window_days": WINDOW_DAYS,
                    "half_life_days": HALF_LIFE_DAYS,
                    "minimum_live_labels": MIN_LIVE_LABELS,
                    "test_used_for_tuning": False,
                    "decision": decision["status"],
                }
            )
            mlflow.log_artifacts(str(output_dir), artifact_path="b57e")
        return "LOGGED"
    except Exception as exc:
        return f"ERROR: {exc}"


def initialize_b57e(
    artifact_bucket: str = DEFAULT_BUCKET,
    output_bucket: str = DEFAULT_BUCKET,
    output_prefix: str = OUTPUT_PREFIX,
    force: bool = False,
) -> dict[str, Any]:
    _ensure_schema()
    client = _s3_client()
    checksum = _checksum(client, artifact_bucket)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        reload_b57e_runtime(output_bucket, output_prefix)
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "results": previous[1],
        }

    run_id = _start_run(checksum)
    try:
        b57d = _json_object(client, artifact_bucket, B57D_MANIFEST_KEY)
        b57c = _json_object(client, artifact_bucket, B57C_DECISION_KEY)
        _validate_upstream(b57d, b57c)
        payloads = _load_model_payloads(client, artifact_bucket)
        cv_predictions = _parquet_object(client, artifact_bucket, CV_PREDICTIONS_KEY)
        test_predictions = _parquet_object(
            client, artifact_bucket, TEST_PREDICTIONS_KEY
        )
        source = _parquet_object(client, artifact_bucket, SOURCE_DATASET_KEY)
        source["prediction_date"] = pd.to_datetime(source["prediction_date"], utc=True)
        if source["prediction_date"].duplicated().any():
            raise RuntimeError("B57B source has duplicate prediction dates")

        seeds, prequential, final_diagnostic = _audit_intervals(
            cv_predictions,
            test_predictions,
            b57c,
            payloads,
        )

        source_last_date = source["prediction_date"].max()
        today = pd.Timestamp.now(tz="UTC").floor("D")
        freshness_days = max(0, int((today - source_last_date).days))
        live_source_ready = freshness_days <= FRESHNESS_LIMIT_DAYS
        status = (
            "READY_FOR_OPERATIONAL_POINT_FORECAST_WITH_DRIFT_GUARD"
            if live_source_ready
            else "READY_FOR_PLATFORM_REPLAY_WAITING_LIVE_DATA"
        )
        operating_mode = (
            "OPERATIONAL_GUARDED" if live_source_ready else "HISTORICAL_REPLAY"
        )

        manifest = {
            "api_version": API_VERSION,
            "upstream_api_version": UPSTREAM_API_VERSION,
            "status": status,
            "operating_mode": operating_mode,
            "artifact_bucket": artifact_bucket,
            "source_dataset_key": SOURCE_DATASET_KEY,
            "source_feature_version": SOURCE_FEATURE_VERSION,
            "source_first_date": source["prediction_date"].min(),
            "source_last_date": source_last_date,
            "source_freshness_days": freshness_days,
            "freshness_limit_days": FRESHNESS_LIMIT_DAYS,
            "models": b57d["models"],
            "b57d_calibration": b57d["calibration"],
            "seed_calibration": seeds,
            "adaptive_policy": {
                "method": (
                    "PAST_ONLY_ROLLING_EXPONENTIALLY_WEIGHTED_"
                    "ASYMMETRIC_SIGNED_RESIDUAL"
                ),
                "window_days": WINDOW_DAYS,
                "half_life_days": HALF_LIFE_DAYS,
                "minimum_live_labels": MIN_LIVE_LABELS,
                "minimum_prequential_labels": MIN_PREQUENTIAL_LABELS,
                "minimum_p90_empirical_coverage": MIN_P90_EMPIRICAL_COVERAGE,
                "test_used_for_tuning": False,
                "cv_seed_deployment_eligible": False,
            },
            "guardrails": {
                "future_without_feature_row": "REJECT",
                "stale_source": "REPLAY_ONLY",
                "insufficient_live_labels": "USE_B57D_DRIFT_GUARD",
                "failed_prequential_coverage": "USE_B57D_DRIFT_GUARD",
                "forecast_must_precede_observation_availability": True,
                "target_or_actual_in_forecast_response": False,
                "causal_claim": False,
            },
        }
        decision = {
            "status": status,
            "api_version": API_VERSION,
            "operating_mode": operating_mode,
            "source_rows": len(source),
            "source_last_date": source_last_date,
            "source_freshness_days": freshness_days,
            "models": len(payloads),
            "schema_ready": True,
            "live_recalibration_active": False,
            "training_executed": False,
            "test_used_for_tuning": False,
            "target_or_actual_exposed": False,
            "gates_passed": True,
            "next_block": (
                "B57E_LIVE_LABEL_COLLECTION_AND_PLATFORM_REPLAY"
                if not live_source_ready
                else "B57E_ACCUMULATE_LIVE_LABELS_FOR_ADAPTIVE_INTERVALS"
            ),
        }

        with tempfile.TemporaryDirectory(prefix="b57e-") as temporary:
            root = Path(temporary)
            prequential.to_csv(
                root / "01_cv_prequential_interval_audit.csv", index=False
            )
            final_diagnostic.to_csv(
                root / "02_final_test_adaptive_diagnostic.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "policy": "forecast_precedes_available_at",
                        "value": True,
                    },
                    {
                        "policy": "window_days",
                        "value": WINDOW_DAYS,
                    },
                    {
                        "policy": "half_life_days",
                        "value": HALF_LIFE_DAYS,
                    },
                    {
                        "policy": "minimum_live_labels_per_target",
                        "value": MIN_LIVE_LABELS,
                    },
                    {
                        "policy": "test_used_for_tuning",
                        "value": False,
                    },
                    {
                        "policy": "fallback",
                        "value": "B57D_DRIFT_GUARD",
                    },
                ]
            ).to_csv(root / "03_recalibration_contract.csv", index=False)
            (root / "b57e_manifest.json").write_text(
                _json_dumps(manifest),
                encoding="utf-8",
            )
            (root / "b57e_decision.json").write_text(
                _json_dumps(decision),
                encoding="utf-8",
            )
            (root / "README_B57E.md").write_text(
                "\n".join(
                    [
                        "# B57E Adaptive Recalibration",
                        "",
                        f"Decision: **{status}**",
                        f"Operating mode: **{operating_mode}**",
                        "",
                        "The final 2026 test is diagnostic only.",
                        "Historical replay never feeds live recalibration.",
                        "Adaptive intervals need real operational forecasts and "
                        "labels with known available_at timestamps.",
                        "Until then, the recommended interval is the B57D drift guard.",
                    ]
                ),
                encoding="utf-8",
            )
            mlflow_status = _log_mlflow(root, decision)
            uploaded = {}
            for path in sorted(root.iterdir()):
                if path.suffix == ".json":
                    key = f"configs/b57e/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b57e/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload(client, path, output_bucket, key)

        metadata = {
            **decision,
            "checksum": checksum,
            "mlflow_status": mlflow_status,
            "outputs": uploaded,
        }
        _finish_run(run_id, "SUCCESS", len(source), metadata)
        reload_b57e_runtime(output_bucket, output_prefix)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "results": metadata,
            "outputs": uploaded,
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {"api_version": API_VERSION},
            error_message=str(exc),
        )
        raise


def _latest_calibrations() -> dict[str, dict[str, Any]]:
    _ensure_schema()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT ON (target_name)
                    target_name, status, as_of, residual_rows, payload
                FROM serving.tir_interval_calibration
                WHERE api_version=%s
                ORDER BY target_name, created_at DESC
                """,
                (API_VERSION,),
            )
            rows = cursor.fetchall()
    return {
        str(target): {
            "status": str(status),
            "as_of": as_of,
            "residual_rows": int(residual_rows),
            "payload": dict(payload or {}),
        }
        for target, status, as_of, residual_rows, payload in rows
    }


def reload_b57e_runtime(
    bucket: str = DEFAULT_BUCKET,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    _ensure_schema()
    client = _s3_client()
    manifest_key = f"configs/b57e/{output_prefix}/b57e_manifest.json"
    manifest = _json_object(client, bucket, manifest_key)
    allowed = {
        "READY_FOR_PLATFORM_REPLAY_WAITING_LIVE_DATA",
        "READY_FOR_OPERATIONAL_POINT_FORECAST_WITH_DRIFT_GUARD",
        "READY_FOR_OPERATIONAL_ADAPTIVE_INTERVALS",
    }
    if manifest.get("status") not in allowed:
        raise RuntimeError(f"B57E manifest is not serveable: {manifest.get('status')}")
    models = _load_model_payloads(client, manifest["artifact_bucket"])
    source = _parquet_object(
        client,
        manifest["artifact_bucket"],
        manifest["source_dataset_key"],
    )
    source["prediction_date"] = pd.to_datetime(source["prediction_date"], utc=True)
    source = source.sort_values("prediction_date").set_index(
        "prediction_date", drop=False
    )
    with RUNTIME_LOCK:
        RUNTIME_CACHE.clear()
        RUNTIME_CACHE.update(
            {
                "manifest": manifest,
                "models": models,
                "source": source,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "bucket": bucket,
                "manifest_key": manifest_key,
            }
        )
    return runtime_status_b57e()


def _runtime() -> dict[str, Any]:
    with RUNTIME_LOCK:
        if not RUNTIME_CACHE:
            reload_b57e_runtime()
        return dict(RUNTIME_CACHE)


def _current_freshness_days(manifest: dict[str, Any]) -> int:
    source_last = pd.Timestamp(manifest["source_last_date"])
    if source_last.tzinfo is None:
        source_last = source_last.tz_localize("UTC")
    else:
        source_last = source_last.tz_convert("UTC")
    today = pd.Timestamp.now(tz="UTC").floor("D")
    return max(0, int((today - source_last.floor("D")).days))


def runtime_status_b57e() -> dict[str, Any]:
    with RUNTIME_LOCK:
        if not RUNTIME_CACHE:
            return {
                "status": "NOT_LOADED",
                "api_version": API_VERSION,
                "loaded_at": None,
            }
        manifest = RUNTIME_CACHE["manifest"]
        loaded_at = RUNTIME_CACHE["loaded_at"]
    calibrations = _latest_calibrations()
    active = [
        target for target, item in calibrations.items() if item["status"] == "ACTIVE"
    ]
    freshness_days = _current_freshness_days(manifest)
    live_source_ready = freshness_days <= FRESHNESS_LIMIT_DAYS
    if not live_source_ready:
        effective_decision = "READY_FOR_PLATFORM_REPLAY_WAITING_LIVE_DATA"
        effective_mode = "HISTORICAL_REPLAY"
    elif len(active) == len(TARGET_NAMES):
        effective_decision = "READY_FOR_OPERATIONAL_ADAPTIVE_INTERVALS"
        effective_mode = "OPERATIONAL_ADAPTIVE"
    else:
        effective_decision = "READY_FOR_OPERATIONAL_POINT_FORECAST_WITH_DRIFT_GUARD"
        effective_mode = "OPERATIONAL_GUARDED"
    return {
        "status": "READY",
        "api_version": API_VERSION,
        "decision": effective_decision,
        "manifest_decision": manifest["status"],
        "operating_mode": effective_mode,
        "source_first_date": manifest["source_first_date"],
        "source_last_date": manifest["source_last_date"],
        "source_freshness_days": freshness_days,
        "models": list(manifest["models"]),
        "adaptive_targets": active,
        "adaptive_target_count": len(active),
        "default_interval": (
            "ADAPTIVE_P90" if len(active) == len(TARGET_NAMES) else "B57D_DRIFT_GUARD"
        ),
        "loaded_at": loaded_at,
    }


def _quality_payload(row: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "tir_history_available_flag",
        "weather_history_available_flag",
        "weather_history_stale_flag",
        "weather_forecast_available_flag",
        "port_source_break_flag",
    ]
    return {column: _clean_json(row.iloc[0].get(column)) for column in columns}


def _adaptive_interval_for_target(
    target_name: str,
    point: float,
    kind: str,
    calibrations: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    calibration = calibrations.get(target_name)
    if not calibration or calibration["status"] != "ACTIVE":
        return None
    values = calibration["payload"].get("offsets", {}).get("p90")
    if not values:
        return None
    lower, upper = _interval(
        point,
        float(values["lower_offset"]),
        float(values["upper_offset"]),
        kind,
    )
    return {
        "kind": "ADAPTIVE_ASYMMETRIC_P90",
        "nominal_level": 0.90,
        "lower": float(lower),
        "upper": float(upper),
        "calibration_as_of": _clean_json(calibration["as_of"]),
        "residual_rows": calibration["residual_rows"],
        "empirical_prequential_coverage": calibration["payload"].get(
            "prequential_p90_coverage"
        ),
    }


def _guard_interval(
    manifest: dict[str, Any],
    target_name: str,
    point: float,
    kind: str,
) -> dict[str, Any]:
    guard = manifest["b57d_calibration"][target_name]["drift_guard"]
    radius = float(guard["radius"])
    lower, upper = _interval(point, -radius, radius, kind)
    return {
        "kind": "B57D_DRIFT_GUARD",
        "nominal_level": None,
        "nominal_coverage_claimed": False,
        "lower": float(lower),
        "upper": float(upper),
        "radius": radius,
        "warning": (
            "Conservative diagnostic guard; it is not a calibrated 90% interval."
        ),
    }


def _diagnostic_intervals(
    manifest: dict[str, Any],
    target_name: str,
    point: float,
    kind: str,
) -> dict[str, Any]:
    result = {}
    calibration = manifest["b57d_calibration"][target_name]
    for level in CALIBRATION_LEVELS:
        name = f"p{int(level * 100)}"
        radius = float(calibration["quantiles"][name])
        lower, upper = _interval(point, -radius, radius, kind)
        result[name] = {
            "nominal_level": level,
            "lower": float(lower),
            "upper": float(upper),
            "production_recommended": False,
            "warning": "Known undercoverage on the untouched 2026 test.",
        }
    return result


def _record_forecast(
    forecast: dict[str, Any],
    response: dict[str, Any],
) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO serving.tir_daily_forecast_ledger (
                    forecast_id, api_version, prediction_date, target_name,
                    model_name, point_prediction,
                    recommended_interval_kind, recommended_lower,
                    recommended_upper, operating_mode,
                    source_feature_version, source_last_date, payload
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    str(uuid.uuid4()),
                    API_VERSION,
                    response["prediction_date"],
                    forecast["target_name"],
                    forecast["model"],
                    forecast["point"],
                    forecast["recommended_interval"]["kind"],
                    forecast["recommended_interval"]["lower"],
                    forecast["recommended_interval"]["upper"],
                    response["mode"],
                    response["source_feature_version"],
                    response["source_last_date"],
                    Json(_clean_json(forecast), dumps=_json_dumps),
                ),
            )


def forecast_b57e_daily(prediction_date: str) -> dict[str, Any]:
    runtime = _runtime()
    manifest = runtime["manifest"]
    source = runtime["source"]
    date = _parse_prediction_date(prediction_date)
    source_last = pd.Timestamp(manifest["source_last_date"])
    if source_last.tzinfo is None:
        source_last = source_last.tz_localize("UTC")
    else:
        source_last = source_last.tz_convert("UTC")
    if date > source_last:
        raise LookupError(
            f"NO_FEATURE_ROW: requested {date.date()}, latest canonical feature "
            f"date is {source_last.date()}. Refresh B57B before forecasting."
        )
    if date not in source.index:
        raise LookupError(f"NO_FEATURE_ROW: no canonical B57B row for {date.date()}")

    row = source.loc[[date]]
    current_freshness_days = _current_freshness_days(manifest)
    response_mode = (
        "OPERATIONAL"
        if date == source_last and current_freshness_days <= FRESHNESS_LIMIT_DAYS
        else "HISTORICAL_REPLAY"
    )
    calibrations = _latest_calibrations()
    forecasts = []
    warnings = []
    if response_mode != "OPERATIONAL":
        warnings.append("Historical replay only; source is not current.")
    quality = _quality_payload(row)
    if quality.get("weather_history_available_flag") != 1:
        warnings.append("Weather history is unavailable for this feature row.")
    if quality.get("weather_history_stale_flag") == 1:
        warnings.append("Weather history is stale.")
    if quality.get("weather_forecast_available_flag") != 1:
        warnings.append("Weather forecast is unavailable.")
    if quality.get("port_source_break_flag") == 1:
        warnings.append("A source-regime break is active.")

    for target_name, payload in runtime["models"].items():
        features = payload["features"]
        missing = sorted(set(features) - set(row.columns))
        if missing:
            raise RuntimeError(f"Runtime feature contract failed: {missing}")
        point = float(
            _clip(
                payload["model"].predict(row[features])[0],
                payload["kind"],
            )
        )
        guard = _guard_interval(manifest, target_name, point, payload["kind"])
        adaptive = (
            _adaptive_interval_for_target(
                target_name,
                point,
                payload["kind"],
                calibrations,
            )
            if response_mode == "OPERATIONAL"
            else None
        )
        recommended = adaptive or guard
        forecasts.append(
            {
                "target_name": target_name,
                "unit": TARGET_UNITS[target_name],
                "point": point,
                "model": manifest["models"][target_name]["selected_model"],
                "recommended_interval": recommended,
                "fallback_interval": guard,
                "diagnostic_intervals": _diagnostic_intervals(
                    manifest, target_name, point, payload["kind"]
                ),
                "adaptive_interval_active": adaptive is not None,
            }
        )

    response = {
        "api_version": API_VERSION,
        "prediction_date": date.isoformat(),
        "mode": response_mode,
        "source_feature_version": manifest["source_feature_version"],
        "source_last_date": manifest["source_last_date"],
        "source_freshness_days": current_freshness_days,
        "forecasts": forecasts,
        "quality": quality,
        "warnings": sorted(set(warnings)),
        "actual_target_exposed": False,
        "causal_interpretation_allowed": False,
    }
    for forecast in forecasts:
        _record_forecast(forecast, response)
    return response


def _parse_available_at(value: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid available_at: {value}") from exc
    if timestamp.tzinfo is None:
        raise ValueError("available_at must include a timezone")
    return timestamp.tz_convert("UTC")


def _validate_actual(target_name: str, value: float) -> float:
    if target_name not in TARGET_NAMES:
        raise ValueError(f"Unsupported target_name: {target_name}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite actual value for {target_name}")
    if target_name in {"TIR_VOLUME", "DURATION_MEDIAN"} and number < 0:
        raise ValueError(f"Negative actual value for {target_name}")
    if target_name == "LONG_24H_RATE" and not 0.0 <= number <= 1.0:
        raise ValueError("LONG_24H_RATE must be between 0 and 1")
    return number


def register_b57e_observations(
    prediction_date: str,
    available_at: str,
    source: str,
    values: dict[str, float],
) -> dict[str, Any]:
    _ensure_schema()
    date = _parse_prediction_date(prediction_date)
    availability = _parse_available_at(available_at)
    if availability < date + pd.Timedelta(days=1):
        raise ValueError(
            "Daily outcomes cannot be available before the end of prediction_date"
        )
    now = pd.Timestamp.now(tz="UTC")
    if availability > now + pd.Timedelta(minutes=5):
        raise ValueError("available_at cannot be in the future")
    if not source or len(source.strip()) < 3:
        raise ValueError("source must contain at least three characters")
    if not values:
        raise ValueError("values cannot be empty")
    unknown = set(values) - set(TARGET_NAMES)
    if unknown:
        raise ValueError(f"Unsupported targets: {sorted(unknown)}")

    inserted = 0
    duplicates = 0
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            for target_name, raw_value in values.items():
                actual = _validate_actual(target_name, raw_value)
                cursor.execute(
                    """
                    INSERT INTO serving.tir_daily_forecast_observation (
                        observation_id, prediction_date, target_name,
                        actual_value, source, available_at, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (
                        prediction_date, target_name, source, available_at
                    ) DO NOTHING
                    """,
                    (
                        str(uuid.uuid4()),
                        date.to_pydatetime(),
                        target_name,
                        actual,
                        source.strip(),
                        availability.to_pydatetime(),
                        Json(
                            {
                                "registered_via": API_VERSION,
                                "received_at": now.isoformat(),
                            }
                        ),
                    ),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    duplicates += 1
    return {
        "status": "SUCCESS",
        "prediction_date": date.isoformat(),
        "available_at": availability.isoformat(),
        "inserted": inserted,
        "duplicates": duplicates,
        "revision_preserved": True,
    }


def _eligible_live_residuals(
    target_name: str,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    with _db_connection() as connection:
        return pd.read_sql_query(
            """
            WITH observations AS (
                SELECT DISTINCT ON (prediction_date, target_name)
                    prediction_date, target_name, actual_value,
                    available_at, received_at
                FROM serving.tir_daily_forecast_observation
                WHERE target_name=%s
                  AND available_at <= %s
                  AND received_at <= %s
                ORDER BY prediction_date, target_name,
                         available_at DESC, received_at DESC
            ),
            forecasts AS (
                SELECT DISTINCT ON (prediction_date, target_name)
                    prediction_date, target_name, point_prediction,
                    issued_at
                FROM serving.tir_daily_forecast_ledger
                WHERE api_version=%s
                  AND target_name=%s
                  AND operating_mode='OPERATIONAL'
                ORDER BY prediction_date, target_name, issued_at ASC
            )
            SELECT
                f.prediction_date,
                f.point_prediction AS prediction,
                f.issued_at,
                o.actual_value AS actual,
                o.available_at,
                o.received_at
            FROM forecasts f
            JOIN observations o USING (prediction_date, target_name)
            WHERE f.issued_at < o.available_at
              AND f.prediction_date >= %s
            ORDER BY f.prediction_date
            """,
            connection,
            params=(
                target_name,
                as_of.to_pydatetime(),
                as_of.to_pydatetime(),
                API_VERSION,
                target_name,
                (as_of - pd.Timedelta(days=WINDOW_DAYS)).to_pydatetime(),
            ),
        )


def _live_calibration_payload(
    rows: pd.DataFrame,
    kind: str,
    as_of: pd.Timestamp,
) -> tuple[str, dict[str, Any]]:
    if len(rows) < MIN_LIVE_LABELS:
        return "WAITING_FOR_LABELS", {
            "reason": (
                f"Need {MIN_LIVE_LABELS} eligible live residuals; found {len(rows)}"
            ),
            "offsets": {},
            "prequential_p90_coverage": None,
        }
    rows = rows.copy()
    rows["prediction_date"] = pd.to_datetime(rows["prediction_date"], utc=True)
    rows["residual"] = rows["actual"] - rows["prediction"]
    weights = _recency_weights(rows["prediction_date"], as_of)
    offsets = {}
    for level in CALIBRATION_LEVELS:
        name = f"p{int(level * 100)}"
        lower, upper = _asymmetric_offsets(
            rows["residual"].to_numpy(dtype="float64"),
            weights,
            level,
        )
        offsets[name] = {
            "level": level,
            "lower_offset": lower,
            "upper_offset": upper,
        }
    audit = _prequential_audit(rows, kind)
    coverage = audit["coverage"]
    status = (
        "ACTIVE"
        if coverage is not None
        and audit["rows"] >= MIN_PREQUENTIAL_LABELS
        and coverage >= MIN_P90_EMPIRICAL_COVERAGE
        else "FALLBACK_DRIFT_GUARD"
    )
    return status, {
        "method": (
            "PAST_ONLY_ROLLING_EXPONENTIALLY_WEIGHTED_ASYMMETRIC_SIGNED_RESIDUAL"
        ),
        "window_days": WINDOW_DAYS,
        "half_life_days": HALF_LIFE_DAYS,
        "offsets": offsets,
        "prequential_p90_coverage": coverage,
        "prequential_rows": audit["rows"],
        "prequential_mean_width": audit["mean_width"],
        "actual_below_rate": audit["below_rate"],
        "actual_above_rate": audit["above_rate"],
        "activation_threshold": MIN_P90_EMPIRICAL_COVERAGE,
        "test_used_for_tuning": False,
    }


def recalibrate_b57e(as_of: str | None = None) -> dict[str, Any]:
    _ensure_schema()
    timestamp = (
        pd.Timestamp.now(tz="UTC") if as_of is None else _parse_available_at(as_of)
    )
    results = {}
    runtime = _runtime()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            for target_name, model_payload in runtime["models"].items():
                rows = _eligible_live_residuals(target_name, timestamp)
                status, payload = _live_calibration_payload(
                    rows, model_payload["kind"], timestamp
                )
                start = (
                    pd.to_datetime(rows["prediction_date"], utc=True).min()
                    if len(rows)
                    else None
                )
                end = (
                    pd.to_datetime(rows["prediction_date"], utc=True).max()
                    if len(rows)
                    else None
                )
                cursor.execute(
                    """
                    INSERT INTO serving.tir_interval_calibration (
                        calibration_id, api_version, target_name, status,
                        as_of, calibration_start, calibration_end,
                        residual_rows, method, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        API_VERSION,
                        target_name,
                        status,
                        timestamp.to_pydatetime(),
                        None if start is None else start.to_pydatetime(),
                        None if end is None else end.to_pydatetime(),
                        len(rows),
                        payload.get("method", "WAITING_FOR_LIVE_LABELS"),
                        Json(_clean_json(payload), dumps=_json_dumps),
                    ),
                )
                results[target_name] = {
                    "status": status,
                    "eligible_residuals": len(rows),
                    "prequential_p90_coverage": payload.get("prequential_p90_coverage"),
                }
    active = sum(item["status"] == "ACTIVE" for item in results.values())
    return {
        "status": "SUCCESS",
        "api_version": API_VERSION,
        "as_of": timestamp.isoformat(),
        "targets": results,
        "active_targets": active,
        "all_targets_active": active == len(TARGET_NAMES),
        "fallback": (None if active == len(TARGET_NAMES) else "B57D_DRIFT_GUARD"),
    }


def monitoring_b57e() -> dict[str, Any]:
    _ensure_schema()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    operating_mode,
                    target_name,
                    count(*) AS forecasts,
                    min(prediction_date) AS first_date,
                    max(prediction_date) AS last_date,
                    max(issued_at) AS last_issued_at
                FROM serving.tir_daily_forecast_ledger
                WHERE api_version=%s
                GROUP BY operating_mode, target_name
                ORDER BY operating_mode, target_name
                """,
                (API_VERSION,),
            )
            forecast_rows = cursor.fetchall()
            cursor.execute(
                """
                SELECT
                    target_name,
                    count(*) AS observations,
                    count(DISTINCT prediction_date) AS days,
                    min(prediction_date) AS first_date,
                    max(prediction_date) AS last_date,
                    max(available_at) AS last_available_at
                FROM serving.tir_daily_forecast_observation
                GROUP BY target_name
                ORDER BY target_name
                """
            )
            observation_rows = cursor.fetchall()
    calibrations = _latest_calibrations()
    return {
        "api_version": API_VERSION,
        "runtime": runtime_status_b57e(),
        "forecasts": [
            {
                "operating_mode": row[0],
                "target_name": row[1],
                "forecasts": int(row[2]),
                "first_date": _clean_json(row[3]),
                "last_date": _clean_json(row[4]),
                "last_issued_at": _clean_json(row[5]),
            }
            for row in forecast_rows
        ],
        "observations": [
            {
                "target_name": row[0],
                "observations": int(row[1]),
                "days": int(row[2]),
                "first_date": _clean_json(row[3]),
                "last_date": _clean_json(row[4]),
                "last_available_at": _clean_json(row[5]),
            }
            for row in observation_rows
        ],
        "calibrations": _clean_json(calibrations),
        "privacy": {
            "actuals_exposed_by_forecast_endpoint": False,
            "observation_values_returned_by_monitoring": False,
        },
    }
