from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import mlflow
import numpy as np
import pandas as pd
import psycopg2
from catboost import CatBoostRegressor, Pool
from psycopg2.extras import Json
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)


SOURCE_NAME = "b54c_wave_model_ready"
DATASET_NAME = "b54d_temporal_split_and_baselines"
TRAINING_VERSION = "b54d-temporal-catboost-v1"
TARGET = "target_arrival_delay_h"
RANDOM_SEED = int(os.getenv("B54D_RANDOM_SEED", "42"))
THREAD_COUNT = int(os.getenv("B54D_THREAD_COUNT", "2"))
TASK_TYPE = os.getenv("B54D_TASK_TYPE", "CPU").upper()
ITERATIONS = int(os.getenv("B54D_ITERATIONS", "700"))

WAVE_TOKENS = (
    "wave",
    "sea_",
    "wind_",
    "surface_current",
    "visibility",
    "pressure",
    "high_wave",
    "severe_wave",
)

BANNED_EXACT = {
    "port_call_id",
    "source_record_id",
    "prediction_time",
    "planned_eta",
    "model_ready_flag",
    "target_arrival_delay_h",
    "target_departure_delay_h",
}


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def check_dependencies() -> dict[str, Any]:
    client = s3_client()
    buckets = sorted(item["Name"] for item in client.list_buckets().get("Buckets", []))
    required = {"silver-maritime", "gold-maritime", "mlflow-artifacts"}
    missing = sorted(required.difference(buckets))
    if missing:
        raise RuntimeError(f"Missing S3 buckets: {missing}")
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), now()")
            database, database_time = cursor.fetchone()
    mlflow_status = "UNKNOWN"
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.search_experiments(max_results=1)
        mlflow_status = "READY"
    except Exception as exc:
        mlflow_status = f"UNAVAILABLE: {exc}"
    return {
        "buckets": buckets,
        "database": database,
        "database_time": database_time.isoformat(),
        "mlflow": mlflow_status,
        "task_type": TASK_TYPE,
        "thread_count": THREAD_COUNT,
    }


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _download(client, bucket: str, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target))


def _upload(client, source: Path, bucket: str, key: str, content_type: str) -> str:
    client.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{bucket}/{key}"


def _object_etag(client, bucket: str, key: str) -> str:
    return str(client.head_object(Bucket=bucket, Key=key)["ETag"]).strip('"')


def _signature(
    source_etag: str,
    report_etag: str,
    train_fraction: float,
    valid_fraction: float,
) -> str:
    payload = {
        "training_version": TRAINING_VERSION,
        "source_etag": source_etag,
        "report_etag": report_etag,
        "train_fraction": train_fraction,
        "valid_fraction": valid_fraction,
        "iterations": ITERATIONS,
        "random_seed": RANDOM_SEED,
        "task_type": TASK_TYPE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _start_run(source_uri: str, checksum: str, metadata: dict[str, Any]) -> str:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit.ingestion_run
                    (source_name, dataset_name, object_uri, checksum, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (SOURCE_NAME, DATASET_NAME, source_uri, checksum, Json(metadata)),
            )
            return str(cursor.fetchone()[0])


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, metadata
                FROM audit.ingestion_run
                WHERE source_name = %s
                  AND dataset_name = %s
                  AND checksum = %s
                  AND status = 'SUCCESS'
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    if row is None:
        return None
    return str(row[0]), dict(row[1] or {})


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at = now(), status = %s, row_count = %s,
                    metadata = metadata || %s, error_message = %s
                WHERE run_id = %s
                """,
                (
                    status,
                    row_count,
                    Json(
                        metadata or {},
                        dumps=lambda obj: json.dumps(obj, default=_json_default),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _is_safe_feature(column: str) -> bool:
    if column in BANNED_EXACT:
        return False
    lowered = column.lower()
    if lowered.startswith(("actual_", "target_", "has_arrival", "has_departure")):
        return False
    if "delay" in lowered and not lowered.startswith(
        ("vessel_hist_", "global_hist_")
    ):
        return False
    if any(token in lowered for token in ("label", "outlier", "quarantine")):
        return False
    return True


def _load_dataset_and_config(
    source_path: Path, report_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(source_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {"port_call_id", "planned_eta", "prediction_time", "horizon_h", TARGET}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"B54C model-ready dataset is missing columns: {missing}")
    frame["port_call_id"] = frame["port_call_id"].astype("string")
    frame["planned_eta"] = pd.to_datetime(frame["planned_eta"], errors="coerce", utc=True)
    frame["prediction_time"] = pd.to_datetime(
        frame["prediction_time"], errors="coerce", utc=True
    )
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=["port_call_id", "planned_eta", "prediction_time", TARGET])

    configured = report.get("model_feature_columns", [])
    feature_columns = [
        column
        for column in configured
        if column in frame.columns and _is_safe_feature(column)
    ]
    if not feature_columns:
        raise RuntimeError("No safe model feature survived the B54D feature gate")
    leak_candidates = [
        column for column in feature_columns if not _is_safe_feature(column)
    ]
    if leak_candidates:
        raise RuntimeError(f"Target leakage columns survived: {leak_candidates}")
    return frame.reset_index(drop=True), feature_columns


def _temporal_split(
    frame: pd.DataFrame,
    train_fraction: float,
    valid_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if train_fraction + valid_fraction >= 0.95:
        raise RuntimeError("train_fraction + valid_fraction must be below 0.95")
    calls = (
        frame.groupby("port_call_id", as_index=False)
        .agg(planned_eta=("planned_eta", "min"), n_snapshots=("horizon_h", "size"))
        .sort_values(["planned_eta", "port_call_id"])
        .reset_index(drop=True)
    )
    if len(calls) < 100:
        raise RuntimeError("Too few port calls for a reliable temporal split")
    train_position = max(1, min(len(calls) - 3, math.floor(len(calls) * train_fraction)))
    valid_position = max(
        train_position + 1,
        min(len(calls) - 2, math.floor(len(calls) * (train_fraction + valid_fraction))),
    )
    train_cutoff = calls.iloc[train_position - 1]["planned_eta"]
    valid_cutoff = calls.iloc[valid_position - 1]["planned_eta"]
    calls["split"] = np.select(
        [calls["planned_eta"] <= train_cutoff, calls["planned_eta"] <= valid_cutoff],
        ["TRAIN", "VALID"],
        default="TEST",
    )
    split_map = calls.set_index("port_call_id")["split"]
    result = frame.copy()
    result["split"] = result["port_call_id"].map(split_map)

    call_sets = {
        split: set(calls.loc[calls["split"] == split, "port_call_id"])
        for split in ("TRAIN", "VALID", "TEST")
    }
    overlaps = {
        "train_valid": len(call_sets["TRAIN"] & call_sets["VALID"]),
        "train_test": len(call_sets["TRAIN"] & call_sets["TEST"]),
        "valid_test": len(call_sets["VALID"] & call_sets["TEST"]),
    }
    leakage = sum(overlaps.values())
    if leakage:
        raise RuntimeError(f"Port-call leakage across temporal splits: {overlaps}")

    dates = result.groupby("split")["planned_eta"].agg(["min", "max"])
    if not (
        dates.loc["TRAIN", "max"] < dates.loc["VALID", "min"]
        and dates.loc["VALID", "max"] < dates.loc["TEST", "min"]
    ):
        raise RuntimeError(f"Temporal order violation: {dates.to_dict()}")

    report_rows = []
    for split in ("TRAIN", "VALID", "TEST"):
        subset = result.loc[result["split"] == split]
        report_rows.append(
            {
                "split": split,
                "rows": int(len(subset)),
                "port_calls": int(subset["port_call_id"].nunique()),
                "planned_eta_min": subset["planned_eta"].min(),
                "planned_eta_max": subset["planned_eta"].max(),
                "target_mean_h": float(subset[TARGET].mean()),
                "target_std_h": float(subset[TARGET].std()),
                "target_p50_h": float(subset[TARGET].median()),
                "target_p95_h": float(subset[TARGET].quantile(0.95)),
            }
        )
    split_report = {
        "train_cutoff": train_cutoff,
        "valid_cutoff": valid_cutoff,
        "overlaps": overlaps,
        "temporal_leakage_violations": leakage,
        "splits": report_rows,
    }
    return result, split_report


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("port_call_id")["port_call_id"].transform("size").astype(float)
    weights = 1.0 / counts
    return (weights / weights.mean()).to_numpy(dtype=float)


def _drop_unusable_features(
    train: pd.DataFrame, feature_columns: list[str]
) -> tuple[list[str], list[str]]:
    usable = []
    dropped = []
    for column in feature_columns:
        non_missing = train[column].notna().sum()
        unique = train[column].nunique(dropna=True)
        if non_missing == 0 or unique <= 1:
            dropped.append(column)
        else:
            usable.append(column)
    return usable, dropped


def _is_wave_feature(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in WAVE_TOKENS)


def _categorical_columns(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    explicit = {
        "imo",
        "vessel_name",
        "vessel_type",
        "cargo_type",
        "port_code",
        "source",
        "sea_source",
    }
    categorical = []
    for column in feature_columns:
        dtype = frame[column].dtype
        if (
            column in explicit
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            categorical.append(column)
    return categorical


def _matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    matrix = frame[feature_columns].copy()
    categorical_set = set(categorical_columns)
    for column in feature_columns:
        if column in categorical_set:
            matrix[column] = matrix[column].astype("string").fillna("MISSING").astype(str)
        else:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
            matrix[column] = matrix[column].replace([np.inf, -np.inf], np.nan).astype("float32")
    return matrix


def _pool(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    with_weights: bool,
) -> Pool:
    matrix = _matrix(frame, feature_columns, categorical_columns)
    weights = _sample_weights(frame) if with_weights else None
    return Pool(
        matrix,
        label=frame[TARGET].to_numpy(dtype=float),
        weight=weights,
        cat_features=categorical_columns,
        feature_names=feature_columns,
    )


def _model_parameters() -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": ITERATIONS,
        "learning_rate": 0.05,
        "depth": 7,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "l2_leaf_reg": 10.0,
        "random_strength": 0.4,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.6,
        "random_seed": RANDOM_SEED,
        "thread_count": THREAD_COUNT,
        "task_type": TASK_TYPE,
        "od_type": "Iter",
        "od_wait": 60,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if TASK_TYPE == "CPU":
        parameters["rsm"] = 0.90
    return parameters


def _fit_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> tuple[CatBoostRegressor, list[str]]:
    categorical = _categorical_columns(train, feature_columns)
    train_pool = _pool(train, feature_columns, categorical, with_weights=True)
    valid_pool = _pool(valid, feature_columns, categorical, with_weights=True)
    model = CatBoostRegressor(**_model_parameters())
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    model.save_model(str(output_path))
    return model, categorical


def _predict(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> np.ndarray:
    pool = _pool(frame, feature_columns, categorical_columns, with_weights=False)
    return np.asarray(model.predict(pool), dtype=float)


def _metric_row(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    split: str,
    model_name: str,
    segment: str = "ALL",
) -> dict[str, Any]:
    y = frame[TARGET].to_numpy(dtype=float)
    weights = _sample_weights(frame)
    error = prediction - y
    row = {
        "split": split,
        "model": model_name,
        "segment": segment,
        "n_rows": int(len(frame)),
        "n_calls": int(frame["port_call_id"].nunique()),
        "MAE": float(mean_absolute_error(y, prediction, sample_weight=weights)),
        "RMSE": float(math.sqrt(mean_squared_error(y, prediction, sample_weight=weights))),
        "R2": float(r2_score(y, prediction, sample_weight=weights)),
        "MEDAE": float(median_absolute_error(y, prediction, sample_weight=weights)),
        "BIAS": float(np.average(error, weights=weights)),
    }
    for threshold in (1, 6, 12, 24):
        truth = y > threshold
        key = f"AUC_GT_{threshold}H"
        row[key] = (
            float(roc_auc_score(truth, prediction, sample_weight=weights))
            if np.unique(truth).size == 2
            else None
        )
    return row


def _evaluate(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    split: str,
    model_name: str,
) -> list[dict[str, Any]]:
    rows = [_metric_row(frame, prediction, split, model_name)]
    for horizon in sorted(frame["horizon_h"].dropna().unique(), reverse=True):
        mask = frame["horizon_h"] == horizon
        rows.append(
            _metric_row(
                frame.loc[mask],
                prediction[mask.to_numpy()],
                split,
                model_name,
                segment=f"HORIZON_{int(horizon)}H",
            )
        )
    segments = {
        "EARLY_LT_NEG1H": frame[TARGET] < -1,
        "ON_TIME_ABS_LE1H": frame[TARGET].abs() <= 1,
        "LATE_1_6H": (frame[TARGET] > 1) & (frame[TARGET] <= 6),
        "LATE_6_12H": (frame[TARGET] > 6) & (frame[TARGET] <= 12),
        "LATE_GE12H": frame[TARGET] > 12,
        "LATE_GE24H": frame[TARGET] > 24,
    }
    for segment, mask in segments.items():
        if int(mask.sum()) >= 20:
            rows.append(
                _metric_row(
                    frame.loc[mask],
                    prediction[mask.to_numpy()],
                    split,
                    model_name,
                    segment=segment,
                )
            )
    return rows


def _baseline_predictions(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> dict[str, np.ndarray]:
    global_median = float(train[TARGET].median())
    medians = train.groupby("horizon_h")[TARGET].median().to_dict()
    median_prediction = (
        evaluation["horizon_h"].map(medians).fillna(global_median).to_numpy(dtype=float)
    )
    if "vessel_hist_mean_delay_h" in evaluation.columns:
        history = pd.to_numeric(
            evaluation["vessel_hist_mean_delay_h"], errors="coerce"
        )
    else:
        history = pd.Series(np.nan, index=evaluation.index)
    if "global_hist_mean_delay_h" in evaluation.columns:
        global_history = pd.to_numeric(
            evaluation["global_hist_mean_delay_h"], errors="coerce"
        )
    else:
        global_history = pd.Series(np.nan, index=evaluation.index)
    history_prediction = history.fillna(global_history).fillna(
        pd.Series(median_prediction, index=evaluation.index)
    )
    return {
        "BASELINE_MEDIAN_BY_HORIZON": median_prediction,
        "BASELINE_SAFE_VESSEL_HISTORY": history_prediction.to_numpy(dtype=float),
    }


def _global_metric(metrics: pd.DataFrame, split: str, model: str) -> pd.Series:
    matched = metrics.loc[
        (metrics["split"] == split)
        & (metrics["model"] == model)
        & (metrics["segment"] == "ALL")
    ]
    if len(matched) != 1:
        raise RuntimeError(f"Global metric not found for {split}/{model}")
    return matched.iloc[0]


def _weather_decision(metrics: pd.DataFrame) -> dict[str, Any]:
    no_wave = _global_metric(metrics, "VALID", "CATBOOST_NO_WAVE")
    with_wave = _global_metric(metrics, "VALID", "CATBOOST_WITH_WAVE")
    gain_mae = float(no_wave["MAE"] - with_wave["MAE"])
    gain_pct = 100.0 * gain_mae / max(1e-12, float(no_wave["MAE"]))
    delta_r2 = float(with_wave["R2"] - no_wave["R2"])
    horizon_degradations = []
    for horizon in (24, 12, 6, 3):
        segment = f"HORIZON_{horizon}H"
        left = metrics.loc[
            (metrics["split"] == "VALID")
            & (metrics["model"] == "CATBOOST_NO_WAVE")
            & (metrics["segment"] == segment),
            "MAE",
        ]
        right = metrics.loc[
            (metrics["split"] == "VALID")
            & (metrics["model"] == "CATBOOST_WITH_WAVE")
            & (metrics["segment"] == segment),
            "MAE",
        ]
        if len(left) and len(right):
            degradation_pct = 100.0 * (float(right.iloc[0]) - float(left.iloc[0])) / max(
                1e-12, float(left.iloc[0])
            )
            horizon_degradations.append(
                {"horizon_h": horizon, "mae_degradation_pct": degradation_pct}
            )
    max_degradation = max(
        (row["mae_degradation_pct"] for row in horizon_degradations), default=0.0
    )
    if gain_pct >= 0.5 and max_degradation <= 2.0:
        weather_status = "KEEP_WAVE_FEATURES"
        official_model = "CATBOOST_WITH_WAVE"
    elif gain_mae > 0:
        weather_status = "WEAK_OR_INCONSISTENT_WAVE_GAIN"
        official_model = "CATBOOST_NO_WAVE"
    else:
        weather_status = "NO_VALID_WAVE_UPLIFT"
        official_model = "CATBOOST_NO_WAVE"
    return {
        "selection_split": "VALID",
        "weather_status": weather_status,
        "official_model": official_model,
        "valid_mae_gain_h": gain_mae,
        "valid_mae_gain_pct": gain_pct,
        "valid_r2_delta": delta_r2,
        "max_horizon_mae_degradation_pct": max_degradation,
        "horizon_details": horizon_degradations,
    }


def _prediction_frame(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    columns = [
        "port_call_id",
        "source_record_id",
        "prediction_time",
        "planned_eta",
        "horizon_h",
        TARGET,
        "split",
    ]
    output = frame[[column for column in columns if column in frame.columns]].copy()
    for name, values in predictions.items():
        output[f"PRED_{name}"] = values
    return output


def _log_mlflow(
    run_id: str,
    metrics: pd.DataFrame,
    decision: dict[str, Any],
    artifact_paths: list[Path],
    params: dict[str, Any],
) -> dict[str, Any]:
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("smart-port-maritime-arrival-delay")
        with mlflow.start_run(run_name=f"B54D-{run_id[:8]}") as active:
            mlflow.log_params({key: str(value) for key, value in params.items()})
            mlflow.log_param("official_model", decision["official_model"])
            for model in ("CATBOOST_NO_WAVE", "CATBOOST_WITH_WAVE"):
                for split in ("VALID", "TEST"):
                    row = _global_metric(metrics, split, model)
                    mlflow.log_metric(f"{model.lower()}_{split.lower()}_mae", float(row["MAE"]))
                    mlflow.log_metric(f"{model.lower()}_{split.lower()}_r2", float(row["R2"]))
            mlflow.log_metric("valid_wave_mae_gain_pct", decision["valid_mae_gain_pct"])
            for artifact in artifact_paths:
                mlflow.log_artifact(str(artifact), artifact_path="b54d")
            return {
                "status": "LOGGED",
                "experiment": "smart-port-maritime-arrival-delay",
                "mlflow_run_id": active.info.run_id,
            }
    except Exception as exc:
        return {"status": "FAILED_NON_BLOCKING", "error": str(exc)}


def train_b54d(
    source_bucket: str,
    source_key: str,
    report_key: str,
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    train_fraction: float = 0.70,
    valid_fraction: float = 0.15,
    force: bool = False,
) -> dict[str, Any]:
    client = s3_client()
    source_etag = _object_etag(client, source_bucket, source_key)
    report_etag = _object_etag(client, source_bucket, report_key)
    checksum = _signature(source_etag, report_etag, train_fraction, valid_fraction)
    source_uri = f"s3://{source_bucket}/{source_key}"
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            previous_run_id, metadata = previous
            return {
                "status": "SKIPPED_ALREADY_PROCESSED",
                "run_id": previous_run_id,
                "checksum": checksum,
                "results": metadata,
                "outputs": metadata.get("output_uris", {}),
            }

    run_metadata = {
        "training_version": TRAINING_VERSION,
        "source_etag": source_etag,
        "report_etag": report_etag,
        "train_fraction": train_fraction,
        "valid_fraction": valid_fraction,
        "test_fraction": 1.0 - train_fraction - valid_fraction,
    }
    run_id = _start_run(source_uri, checksum, run_metadata)
    try:
        with tempfile.TemporaryDirectory(prefix="b54d-") as temporary:
            work_dir = Path(temporary)
            source_path = work_dir / "b54c_model_ready.parquet"
            report_path = work_dir / "b54c_report.json"
            _download(client, source_bucket, source_key, source_path)
            _download(client, source_bucket, report_key, report_path)
            frame, configured_features = _load_dataset_and_config(source_path, report_path)
            frame, split_report = _temporal_split(
                frame, train_fraction, valid_fraction
            )
            train = frame.loc[frame["split"] == "TRAIN"].copy()
            valid = frame.loc[frame["split"] == "VALID"].copy()
            test = frame.loc[frame["split"] == "TEST"].copy()

            usable_features, dropped_features = _drop_unusable_features(
                train, configured_features
            )
            wave_features = [column for column in usable_features if _is_wave_feature(column)]
            no_wave_features = [
                column for column in usable_features if column not in wave_features
            ]
            if not wave_features:
                raise RuntimeError("No wave feature found; B54D cannot run the uplift test")
            if len(no_wave_features) < 5:
                raise RuntimeError("Too few non-wave features for the control model")

            no_wave_path = work_dir / "b54d_catboost_no_wave.cbm"
            with_wave_path = work_dir / "b54d_catboost_with_wave.cbm"
            no_wave_model, no_wave_cats = _fit_model(
                train, valid, no_wave_features, no_wave_path
            )
            with_wave_model, with_wave_cats = _fit_model(
                train, valid, usable_features, with_wave_path
            )

            metrics_rows = []
            prediction_outputs: dict[str, pd.DataFrame] = {}
            for split_name, subset in (("VALID", valid), ("TEST", test)):
                predictions = _baseline_predictions(train, subset)
                predictions["CATBOOST_NO_WAVE"] = _predict(
                    no_wave_model, subset, no_wave_features, no_wave_cats
                )
                predictions["CATBOOST_WITH_WAVE"] = _predict(
                    with_wave_model, subset, usable_features, with_wave_cats
                )
                for model_name, values in predictions.items():
                    metrics_rows.extend(
                        _evaluate(subset, values, split_name, model_name)
                    )
                prediction_outputs[split_name] = _prediction_frame(subset, predictions)

            metrics = pd.DataFrame(metrics_rows)
            decision = _weather_decision(metrics)
            official_test = _global_metric(
                metrics, "TEST", decision["official_model"]
            ).to_dict()
            decision["official_test_metrics"] = official_test

            metrics_path = work_dir / "b54d_metrics.csv"
            valid_path = work_dir / "b54d_valid_predictions.parquet"
            test_path = work_dir / "b54d_test_predictions.parquet"
            split_path = work_dir / "b54d_temporal_split_assignment.parquet"
            config_path = work_dir / "b54d_feature_config.json"
            decision_path = work_dir / "b54d_decision.json"
            metrics.to_csv(metrics_path, index=False)
            prediction_outputs["VALID"].to_parquet(valid_path, index=False, compression="zstd")
            prediction_outputs["TEST"].to_parquet(test_path, index=False, compression="zstd")
            frame[["port_call_id", "planned_eta", "split"]].drop_duplicates(
                "port_call_id"
            ).to_parquet(split_path, index=False, compression="zstd")

            feature_config = {
                "training_version": TRAINING_VERSION,
                "target": TARGET,
                "all_features": usable_features,
                "no_wave_features": no_wave_features,
                "wave_features": wave_features,
                "dropped_constant_or_empty": dropped_features,
                "no_wave_categorical": no_wave_cats,
                "with_wave_categorical": with_wave_cats,
                "model_parameters": _model_parameters(),
                "sample_weight_policy": "1/N_SNAPSHOTS_PER_PORT_CALL_NORMALIZED",
            }
            config_path.write_text(
                json.dumps(feature_config, indent=2, default=_json_default),
                encoding="utf-8",
            )
            decision_path.write_text(
                json.dumps(decision, indent=2, default=_json_default),
                encoding="utf-8",
            )

            mlflow_result = _log_mlflow(
                run_id,
                metrics,
                decision,
                [
                    no_wave_path,
                    with_wave_path,
                    metrics_path,
                    config_path,
                    decision_path,
                ],
                {
                    "train_fraction": train_fraction,
                    "valid_fraction": valid_fraction,
                    "iterations": ITERATIONS,
                    "task_type": TASK_TYPE,
                    "feature_count_no_wave": len(no_wave_features),
                    "feature_count_with_wave": len(usable_features),
                },
            )

            prefix = output_prefix.strip("/") or "version=1"
            uploads = {
                "model_no_wave": (
                    no_wave_path,
                    f"models/b54d/{prefix}/b54d_catboost_no_wave.cbm",
                    "application/octet-stream",
                ),
                "model_with_wave": (
                    with_wave_path,
                    f"models/b54d/{prefix}/b54d_catboost_with_wave.cbm",
                    "application/octet-stream",
                ),
                "metrics": (
                    metrics_path,
                    f"reports/b54d/{prefix}/b54d_metrics.csv",
                    "text/csv",
                ),
                "valid_predictions": (
                    valid_path,
                    f"predictions/b54d/{prefix}/b54d_valid_predictions.parquet",
                    "application/x-parquet",
                ),
                "test_predictions": (
                    test_path,
                    f"predictions/b54d/{prefix}/b54d_test_predictions.parquet",
                    "application/x-parquet",
                ),
                "split_assignment": (
                    split_path,
                    f"datasets/b54d/{prefix}/b54d_temporal_split_assignment.parquet",
                    "application/x-parquet",
                ),
                "feature_config": (
                    config_path,
                    f"configs/b54d/{prefix}/b54d_feature_config.json",
                    "application/json",
                ),
                "decision": (
                    decision_path,
                    f"configs/b54d/{prefix}/b54d_decision.json",
                    "application/json",
                ),
            }
            output_uris = {
                name: _upload(client, path, output_bucket, key, content_type)
                for name, (path, key, content_type) in uploads.items()
            }

        results = {
            "training_version": TRAINING_VERSION,
            "source_uri": source_uri,
            "source_rows": int(len(frame)),
            "source_calls": int(frame["port_call_id"].nunique()),
            "split_report": split_report,
            "configured_feature_count": len(configured_features),
            "usable_feature_count": len(usable_features),
            "no_wave_feature_count": len(no_wave_features),
            "wave_feature_count": len(wave_features),
            "decision": decision,
            "mlflow": mlflow_result,
            "output_uris": output_uris,
            "next_block": "B54E_HORIZON_EXPERTS_OR_PROBABILISTIC_ETA",
            "generated_at_utc": datetime.now(timezone.utc),
        }
        _finish_run(
            run_id,
            "SUCCESS",
            row_count=int(len(frame)),
            metadata=results,
        )
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "checksum": checksum,
            "results": results,
            "outputs": output_uris,
        }
    except Exception as exc:
        _finish_run(run_id, "FAILED", error_message=str(exc)[:4000])
        raise
