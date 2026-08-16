from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import mlflow
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json


API_VERSION = "b57d-probabilistic-forecast-api-v1"
UPSTREAM_VERSION = "b57c-event-aware-temporal-baselines-v1.1"
SOURCE_FEATURE_VERSION = "b57b-event-aware-daily-gold-v1"
SOURCE_NAME = "b57d_probabilistic_forecast_api"
DATASET_NAME = "tir_daily_probabilistic_forecast_api"
DEFAULT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"
FRESHNESS_LIMIT_DAYS = 2
CALIBRATION_LEVELS = (0.80, 0.90, 0.95)
DRIFT_GUARD_SOURCE_QUANTILE = 0.975

SOURCE_DATASET_KEY = (
    "datasets/b57b/version=1/tir_daily_predictive_gold_v1.parquet"
)
B57C_DECISION_KEY = "configs/b57c/version=1/b57c_decision.json"
CV_PREDICTIONS_KEY = "predictions/b57c/version=1/cv_predictions.parquet"
TEST_PREDICTIONS_KEY = "predictions/b57c/version=1/test_predictions.parquet"

MODEL_KEYS = {
    "TIR_VOLUME": (
        "models/b57c/version=1/"
        "full_no_port-tir_volume-hgb_calendar_tir.pkl"
    ),
    "DURATION_MEDIAN": (
        "models/b57c/version=1/"
        "full_no_port-duration_median-catboost_quantile_maximal.pkl"
    ),
    "LONG_24H_RATE": (
        "models/b57c/version=1/"
        "full_no_port-long_24h_rate-hgb_calendar_tir_weather.pkl"
    ),
}

TARGET_UNITS = {
    "TIR_VOLUME": "rows_per_day",
    "DURATION_MEDIAN": "hours",
    "LONG_24H_RATE": "proportion",
}

RUNTIME_LOCK = threading.RLock()
RUNTIME_CACHE: dict[str, Any] = {}


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return _json_default(value)


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


def _checksum(client, bucket: str) -> str:
    digest = hashlib.sha256(API_VERSION.encode("ascii"))
    for key in [
        B57C_DECISION_KEY,
        CV_PREDICTIONS_KEY,
        TEST_PREDICTIONS_KEY,
        SOURCE_DATASET_KEY,
        *MODEL_KEYS.values(),
    ]:
        metadata = client.head_object(Bucket=bucket, Key=key)
        digest.update(key.encode("utf-8"))
        digest.update(str(metadata.get("ETag", "")).encode("ascii"))
        digest.update(str(metadata.get("ContentLength", 0)).encode("ascii"))
    return digest.hexdigest()


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
        "upstream_version": UPSTREAM_VERSION,
        "policy": "OOF_CONFORMAL_CALIBRATION_TEST_RELIABILITY_NO_RETRAINING",
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
                    f"s3://{DEFAULT_BUCKET}/{B57C_DECISION_KEY}",
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
                    Json(
                        _clean_json(metadata),
                        dumps=lambda value: json.dumps(
                            value, default=_json_default, allow_nan=False
                        ),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _clip(values: np.ndarray | float, kind: str):
    result = np.asarray(values, dtype="float64")
    if kind == "COUNT":
        clipped = np.clip(result, 0.0, None)
    elif kind == "DURATION":
        clipped = np.clip(result, 0.0, 720.0)
    elif kind == "RATE":
        clipped = np.clip(result, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported target kind: {kind}")
    return float(clipped) if clipped.ndim == 0 else clipped


def _validate_upstream(decision: dict[str, Any]) -> None:
    if decision.get("status") != "READY_FOR_EVENT_AWARE_MVP":
        raise RuntimeError("B57C did not approve the event-aware MVP")
    if decision.get("training_version") != UPSTREAM_VERSION:
        raise RuntimeError("B57C training version is not the approved v1.1")
    if decision.get("official_track") != "FULL_NO_PORT":
        raise RuntimeError("B57C official track is not FULL_NO_PORT")
    if decision.get("selection_used_test") is not False:
        raise RuntimeError("B57C used final test for selection")
    if int(decision.get("critical_leakage_violations", -1)) != 0:
        raise RuntimeError("B57C leakage gate did not pass")
    if int(decision.get("stable_official_models", 0)) != len(MODEL_KEYS):
        raise RuntimeError("Not all three B57C models are stable")


def _load_model_payloads(client, bucket: str) -> dict[str, dict[str, Any]]:
    payloads = {}
    for target_name, key in MODEL_KEYS.items():
        payload = pickle.loads(_bytes(client, bucket, key))
        required = {"model", "features", "target", "target_name", "kind"}
        missing = required - set(payload)
        if missing:
            raise RuntimeError(f"Model {target_name} misses payload fields: {missing}")
        if payload["target_name"] != target_name:
            raise RuntimeError(f"Model key and payload target disagree for {target_name}")
        if not payload["features"]:
            raise RuntimeError(f"Model {target_name} has no features")
        payloads[target_name] = payload
    return payloads


def _selected_prediction_rows(
    predictions: pd.DataFrame,
    decision: dict[str, Any],
    target_name: str,
) -> pd.DataFrame:
    selected = decision["selected_models"][target_name]
    rows = predictions[
        predictions["track"].eq("FULL_NO_PORT")
        & predictions["target_name"].eq(target_name)
        & predictions["model"].eq(selected)
    ].copy()
    rows["actual"] = pd.to_numeric(rows["actual"], errors="coerce")
    rows["prediction"] = pd.to_numeric(rows["prediction"], errors="coerce")
    rows = rows.dropna(subset=["actual", "prediction"])
    if len(rows) < 100:
        raise RuntimeError(f"Insufficient calibration predictions for {target_name}")
    return rows


def _conformal_calibration(
    cv_predictions: pd.DataFrame,
    decision: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    calibration = {}
    rows = []
    for target_name in MODEL_KEYS:
        part = _selected_prediction_rows(cv_predictions, decision, target_name)
        residual = np.abs(part["actual"].to_numpy() - part["prediction"].to_numpy())
        quantiles = {}
        fold_quantiles = {}
        for level in CALIBRATION_LEVELS:
            value = float(np.quantile(residual, level, method="higher"))
            key = f"p{int(level * 100)}"
            quantiles[key] = value
            by_fold = {
                str(fold): float(
                    np.quantile(
                        np.abs(group["actual"] - group["prediction"]),
                        level,
                        method="higher",
                    )
                )
                for fold, group in part.groupby("fold", observed=True)
            }
            fold_quantiles[key] = by_fold
            rows.append(
                {
                    "target_name": target_name,
                    "selected_model": decision["selected_models"][target_name],
                    "level": level,
                    "calibration_rows": len(part),
                    "absolute_residual_quantile": value,
                    "max_fold_residual_quantile": max(by_fold.values()),
                    "calibration_source": "WALK_FORWARD_CV_OUT_OF_FOLD",
                }
            )
        guard_by_fold = {
            str(fold): float(
                np.quantile(
                    np.abs(group["actual"] - group["prediction"]),
                    DRIFT_GUARD_SOURCE_QUANTILE,
                    method="higher",
                )
            )
            for fold, group in part.groupby("fold", observed=True)
        }
        calibration[target_name] = {
            "method": "SYMMETRIC_ABSOLUTE_RESIDUAL_CONFORMAL",
            "calibration_source": "WALK_FORWARD_CV_OUT_OF_FOLD",
            "rows": len(part),
            "quantiles": quantiles,
            "fold_quantiles": fold_quantiles,
            "drift_guard": {
                "method": "MAX_FOLD_RESIDUAL_QUANTILE",
                "source_quantile": DRIFT_GUARD_SOURCE_QUANTILE,
                "radius": max(guard_by_fold.values()),
                "fold_radii": guard_by_fold,
                "nominal_coverage_claimed": False,
            },
        }
    return calibration, pd.DataFrame(rows)


def _reliability_report(
    test_predictions: pd.DataFrame,
    decision: dict[str, Any],
    calibration: dict[str, Any],
    model_payloads: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for target_name, payload in model_payloads.items():
        part = _selected_prediction_rows(test_predictions, decision, target_name)
        actual = part["actual"].to_numpy(dtype="float64")
        point = part["prediction"].to_numpy(dtype="float64")
        for level in CALIBRATION_LEVELS:
            key = f"p{int(level * 100)}"
            radius = float(calibration[target_name]["quantiles"][key])
            lower = _clip(point - radius, payload["kind"])
            upper = _clip(point + radius, payload["kind"])
            covered = (actual >= lower) & (actual <= upper)
            rows.append(
                {
                    "target_name": target_name,
                    "selected_model": decision["selected_models"][target_name],
                    "interval_name": key,
                    "nominal_coverage": level,
                    "test_rows": len(part),
                    "empirical_coverage": float(covered.mean()),
                    "coverage_gap_pp": 100.0 * (covered.mean() - level),
                    "mean_interval_width": float(np.mean(upper - lower)),
                    "under_interval_rate": float((actual < lower).mean()),
                    "over_interval_rate": float((actual > upper).mean()),
                }
            )
        guard = calibration[target_name]["drift_guard"]
        radius = float(guard["radius"])
        lower = _clip(point - radius, payload["kind"])
        upper = _clip(point + radius, payload["kind"])
        covered = (actual >= lower) & (actual <= upper)
        rows.append(
            {
                "target_name": target_name,
                "selected_model": decision["selected_models"][target_name],
                "interval_name": "drift_guard",
                "nominal_coverage": np.nan,
                "test_rows": len(part),
                "empirical_coverage": float(covered.mean()),
                "coverage_gap_pp": np.nan,
                "mean_interval_width": float(np.mean(upper - lower)),
                "under_interval_rate": float((actual < lower).mean()),
                "over_interval_rate": float((actual > upper).mean()),
            }
        )
    return pd.DataFrame(rows)


def _artifact_contract(
    decision: dict[str, Any], payloads: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for target_name, payload in payloads.items():
        for feature in payload["features"]:
            rows.append(
                {
                    "target_name": target_name,
                    "target": payload["target"],
                    "kind": payload["kind"],
                    "unit": TARGET_UNITS[target_name],
                    "selected_model": decision["selected_models"][target_name],
                    "model_key": MODEL_KEYS[target_name],
                    "feature": feature,
                    "target_or_actual_exposed": False,
                }
            )
    return pd.DataFrame(rows)


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
                    "upstream_version": UPSTREAM_VERSION,
                    "calibration": "OOF_SYMMETRIC_CONFORMAL",
                    "operating_mode": decision["operating_mode"],
                }
            )
            mlflow.log_artifacts(str(output_dir), artifact_path="b57d")
        return "LOGGED"
    except Exception as exc:
        return f"ERROR: {exc}"


def promote_b57d_probabilistic_forecaster(
    artifact_bucket: str = DEFAULT_BUCKET,
    output_bucket: str = DEFAULT_BUCKET,
    output_prefix: str = OUTPUT_PREFIX,
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    checksum = _checksum(client, artifact_bucket)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        reload_b57d_runtime(output_bucket, output_prefix)
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "results": previous[1],
        }

    run_id = _start_run(checksum)
    try:
        upstream = _json_object(client, artifact_bucket, B57C_DECISION_KEY)
        _validate_upstream(upstream)
        payloads = _load_model_payloads(client, artifact_bucket)
        cv_predictions = _parquet_object(client, artifact_bucket, CV_PREDICTIONS_KEY)
        test_predictions = _parquet_object(
            client, artifact_bucket, TEST_PREDICTIONS_KEY
        )
        source = _parquet_object(client, artifact_bucket, SOURCE_DATASET_KEY)
        source["prediction_date"] = pd.to_datetime(source["prediction_date"], utc=True)
        if source["prediction_date"].duplicated().any():
            raise RuntimeError("Canonical B57B source contains duplicate prediction dates")
        versions = set(source["feature_version"].dropna().astype(str).unique())
        if versions != {SOURCE_FEATURE_VERSION}:
            raise RuntimeError(f"Unexpected source feature versions: {versions}")

        missing_features = {
            target_name: sorted(set(payload["features"]) - set(source.columns))
            for target_name, payload in payloads.items()
        }
        missing_features = {
            key: value for key, value in missing_features.items() if value
        }
        if missing_features:
            raise RuntimeError(f"Source misses model features: {missing_features}")

        calibration, calibration_report = _conformal_calibration(
            cv_predictions, upstream
        )
        reliability = _reliability_report(
            test_predictions, upstream, calibration, payloads
        )
        contract = _artifact_contract(upstream, payloads)

        source_last_date = source["prediction_date"].max()
        now_date = pd.Timestamp.now(tz="UTC").floor("D")
        freshness_days = max(0, int((now_date - source_last_date).days))
        live_ready = freshness_days <= FRESHNESS_LIMIT_DAYS
        operating_mode = "OPERATIONAL" if live_ready else "HISTORICAL_REPLAY"
        decision_status = (
            "READY_FOR_OPERATIONAL_PROBABILISTIC_API"
            if live_ready
            else "READY_FOR_HISTORICAL_REPLAY_API_NEED_LIVE_DATA"
        )

        p90_reliability = reliability[reliability["interval_name"].eq("p90")]
        reliability_passed = bool(
            len(p90_reliability) == len(MODEL_KEYS)
            and p90_reliability["empirical_coverage"].ge(0.80).all()
        )
        guard_reliability = reliability[
            reliability["interval_name"].eq("drift_guard")
        ]
        drift_guard_passed = bool(
            len(guard_reliability) == len(MODEL_KEYS)
            and guard_reliability["empirical_coverage"].ge(0.80).all()
        )
        uncertainty_status = (
            "STANDARD_P90_ACCEPTABLE"
            if reliability_passed
            else "STANDARD_INTERVAL_UNDERCOVERAGE_USE_DRIFT_GUARD"
        )
        if not reliability_passed and drift_guard_passed:
            decision_status = (
                "READY_FOR_HISTORICAL_REPLAY_API_WITH_DRIFT_GUARD_WARNING"
            )
            operating_mode = "HISTORICAL_REPLAY"
        elif not reliability_passed:
            decision_status = "NEED_PROBABILISTIC_CALIBRATION_REPAIR"

        manifest = {
            "api_version": API_VERSION,
            "status": decision_status,
            "operating_mode": operating_mode,
            "artifact_bucket": artifact_bucket,
            "source_dataset_key": SOURCE_DATASET_KEY,
            "source_feature_version": SOURCE_FEATURE_VERSION,
            "source_first_date": source["prediction_date"].min(),
            "source_last_date": source_last_date,
            "source_freshness_days": freshness_days,
            "freshness_limit_days": FRESHNESS_LIMIT_DAYS,
            "models": {
                target_name: {
                    "key": MODEL_KEYS[target_name],
                    "selected_model": upstream["selected_models"][target_name],
                    "target": payload["target"],
                    "kind": payload["kind"],
                    "unit": TARGET_UNITS[target_name],
                    "feature_count": len(payload["features"]),
                }
                for target_name, payload in payloads.items()
            },
            "calibration": calibration,
            "uncertainty_status": uncertainty_status,
            "guardrails": {
                "future_without_feature_row": "REJECT",
                "target_or_actual_in_response": False,
                "post_break_port_features": False,
                "prediction_date_grain": "ONE_UTC_DAY",
                "causal_claim": False,
            },
        }
        decision = {
            "status": decision_status,
            "api_version": API_VERSION,
            "operating_mode": operating_mode,
            "source_rows": len(source),
            "source_last_date": source_last_date,
            "source_freshness_days": freshness_days,
            "reliability_passed": reliability_passed,
            "drift_guard_passed": drift_guard_passed,
            "uncertainty_status": uncertainty_status,
            "models_promoted": len(payloads),
            "calibration_levels": list(CALIBRATION_LEVELS),
            "training_executed": False,
            "target_or_actual_exposed": False,
            "future_without_features_allowed": False,
            "gates_passed": reliability_passed or drift_guard_passed,
            "next_block": (
                "B57E_PLATFORM_INTEGRATION"
                if decision_status == "READY_FOR_OPERATIONAL_PROBABILISTIC_API"
                else "B57E_LIVE_DATA_INTERVAL_RECALIBRATION_AND_PLATFORM_REPLAY"
            ),
        }

        with tempfile.TemporaryDirectory(prefix="b57d-") as temporary:
            root = Path(temporary)
            calibration_report.to_csv(
                root / "01_conformal_calibration.csv", index=False
            )
            reliability.to_csv(root / "02_final_test_reliability.csv", index=False)
            contract.to_csv(root / "03_artifact_and_feature_contract.csv", index=False)
            calibration_path = root / "b57d_calibration.json"
            manifest_path = root / "b57d_api_manifest.json"
            decision_path = root / "b57d_decision.json"
            calibration_path.write_text(
                json.dumps(_clean_json(calibration), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(_clean_json(manifest), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            decision_path.write_text(
                json.dumps(_clean_json(decision), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            (root / "README_B57D.md").write_text(
                "\n".join(
                    [
                        "# B57D Probabilistic Forecast API",
                        "",
                        f"Decision: **{decision_status}**",
                        f"Operating mode: **{operating_mode}**",
                        "",
                        "Intervals use absolute-residual conformal calibration from B57C walk-forward CV.",
                        "Final-test coverage is diagnostic and never tunes interval radii.",
                        "The API rejects dates without a canonical B57B feature row.",
                        "Targets and realized outcomes are never returned by the forecast endpoint.",
                    ]
                ),
                encoding="utf-8",
            )
            mlflow_status = _log_mlflow(root, decision)
            uploaded = {}
            for path in sorted(root.iterdir()):
                if path.suffix == ".json":
                    key = f"configs/b57d/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b57d/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload(client, path, output_bucket, key)

        metadata = {
            **decision,
            "checksum": checksum,
            "mlflow_status": mlflow_status,
            "outputs": uploaded,
            "manifest": manifest,
        }
        _finish_run(run_id, "SUCCESS", len(source), metadata)
        reload_b57d_runtime(output_bucket, output_prefix)
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


def reload_b57d_runtime(
    bucket: str = DEFAULT_BUCKET, output_prefix: str = OUTPUT_PREFIX
) -> dict[str, Any]:
    client = _s3_client()
    manifest_key = f"configs/b57d/{output_prefix}/b57d_api_manifest.json"
    manifest = _json_object(client, bucket, manifest_key)
    allowed = {
        "READY_FOR_OPERATIONAL_PROBABILISTIC_API",
        "READY_FOR_HISTORICAL_REPLAY_API_NEED_LIVE_DATA",
        "READY_FOR_HISTORICAL_REPLAY_API_WITH_DRIFT_GUARD_WARNING",
    }
    if manifest.get("status") not in allowed:
        raise RuntimeError(f"B57D manifest is not serveable: {manifest.get('status')}")
    models = _load_model_payloads(client, manifest["artifact_bucket"])
    source = _parquet_object(
        client, manifest["artifact_bucket"], manifest["source_dataset_key"]
    )
    source["prediction_date"] = pd.to_datetime(source["prediction_date"], utc=True)
    source = source.sort_values("prediction_date").set_index("prediction_date", drop=False)
    loaded_at = datetime.now(timezone.utc).isoformat()
    with RUNTIME_LOCK:
        RUNTIME_CACHE.clear()
        RUNTIME_CACHE.update(
            {
                "manifest": manifest,
                "models": models,
                "source": source,
                "loaded_at": loaded_at,
                "bucket": bucket,
                "manifest_key": manifest_key,
            }
        )
    return runtime_status_b57d()


def _runtime() -> dict[str, Any]:
    with RUNTIME_LOCK:
        if not RUNTIME_CACHE:
            reload_b57d_runtime()
        return dict(RUNTIME_CACHE)


def runtime_status_b57d() -> dict[str, Any]:
    with RUNTIME_LOCK:
        if not RUNTIME_CACHE:
            return {
                "status": "NOT_LOADED",
                "api_version": API_VERSION,
                "loaded_at": None,
            }
        manifest = RUNTIME_CACHE["manifest"]
        return {
            "status": "READY",
            "api_version": API_VERSION,
            "decision": manifest["status"],
            "operating_mode": manifest["operating_mode"],
            "source_first_date": manifest["source_first_date"],
            "source_last_date": manifest["source_last_date"],
            "source_freshness_days": manifest["source_freshness_days"],
            "models": list(manifest["models"]),
            "uncertainty_status": manifest["uncertainty_status"],
            "loaded_at": RUNTIME_CACHE["loaded_at"],
        }


def _parse_prediction_date(value: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid prediction_date: {value}") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("D"):
        raise ValueError("prediction_date must be a UTC calendar date")
    return timestamp


def forecast_b57d_daily(prediction_date: str) -> dict[str, Any]:
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
            f"NO_FEATURE_ROW: requested {date.date()}, latest canonical feature date "
            f"is {source_last.date()}. Collect and build new B57B daily features first."
        )
    if date not in source.index:
        raise LookupError(f"NO_FEATURE_ROW: no canonical B57B row for {date.date()}")
    row = source.loc[[date]]
    forecasts = []
    for target_name, payload in runtime["models"].items():
        features = payload["features"]
        missing = sorted(set(features) - set(row.columns))
        if missing:
            raise RuntimeError(f"Runtime feature contract failed: {missing}")
        point = _clip(payload["model"].predict(row[features])[0], payload["kind"])
        intervals = {}
        for level in CALIBRATION_LEVELS:
            key = f"p{int(level * 100)}"
            radius = float(manifest["calibration"][target_name]["quantiles"][key])
            intervals[key] = {
                "level": level,
                "lower": _clip(point - radius, payload["kind"]),
                "upper": _clip(point + radius, payload["kind"]),
                "radius": radius,
            }
        guard = manifest["calibration"][target_name]["drift_guard"]
        guard_radius = float(guard["radius"])
        intervals["drift_guard"] = {
            "level": None,
            "lower": _clip(point - guard_radius, payload["kind"]),
            "upper": _clip(point + guard_radius, payload["kind"]),
            "radius": guard_radius,
            "nominal_coverage_claimed": False,
            "warning": "Use during regime drift; interval is conservative and diagnostic.",
        }
        forecasts.append(
            {
                "target_name": target_name,
                "unit": TARGET_UNITS[target_name],
                "point": point,
                "intervals": intervals,
                "model": manifest["models"][target_name]["selected_model"],
            }
        )
    quality_columns = [
        "tir_history_available_flag",
        "weather_history_available_flag",
        "weather_history_stale_flag",
        "weather_forecast_available_flag",
        "port_source_break_flag",
    ]
    quality = {
        column: _clean_json(row.iloc[0].get(column)) for column in quality_columns
    }
    response_mode = (
        "OPERATIONAL"
        if manifest["operating_mode"] == "OPERATIONAL" and date == source_last
        else "HISTORICAL_REPLAY"
    )
    return {
        "api_version": API_VERSION,
        "prediction_date": date.isoformat(),
        "mode": response_mode,
        "manifest_operating_mode": manifest["operating_mode"],
        "uncertainty_status": manifest["uncertainty_status"],
        "source_feature_version": manifest["source_feature_version"],
        "source_last_date": manifest["source_last_date"],
        "forecasts": forecasts,
        "quality": quality,
        "actual_target_exposed": False,
        "causal_interpretation_allowed": False,
    }
