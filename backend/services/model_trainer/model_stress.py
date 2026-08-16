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


SOURCE_NAME = "b54fd0_train_readiness"
DATASET_NAME = "b54fd1_fair_split_model_stress"
STRESS_VERSION = "b54fd1-ordered-ablation-model-stress-v2"
TARGET = "target_arrival_delay_h"
CALL_COLUMN = "port_call_id"
TIME_COLUMN = "prediction_at"
OFFICIAL_PROTOCOL = "TEMPORAL_PURGED"
PROTOCOLS = ("RANDOM_IID", "RANDOM_BY_IMO", OFFICIAL_PROTOCOL)
RANDOM_SEED = int(os.getenv("B54FD1_RANDOM_SEED", "42"))
THREAD_COUNT = int(os.getenv("B54D_THREAD_COUNT", "2"))
TASK_TYPE = os.getenv("B54D_TASK_TYPE", "CPU").upper()
ITERATIONS = int(os.getenv("B54FD1_ITERATIONS", "600"))
BOOTSTRAP_REPEATS = int(os.getenv("B54FD1_BOOTSTRAP_REPEATS", "500"))

BASELINE_ZERO = "BASELINE_ZERO_DELAY"
BASELINE_MEDIAN = "BASELINE_TRAIN_MEDIAN"
BASELINE_HISTORY = "BASELINE_SAFE_VESSEL_HISTORY"
BASELINE_NAMES = (BASELINE_ZERO, BASELINE_MEDIAN, BASELINE_HISTORY)

TRACK_CALENDAR = "CATBOOST_CALENDAR_ONLY"
TRACK_GLOBAL = "CATBOOST_CALENDAR_PLUS_GLOBAL_HISTORY"
TRACK_VESSEL = "CATBOOST_CALENDAR_PLUS_GLOBAL_AND_VESSEL_HISTORY"
TRACK_WEATHER = "CATBOOST_HISTORY_PLUS_PAST_WEATHER"
TRACK_IDENTITY = "CATBOOST_HISTORY_PLUS_PAST_WEATHER_PLUS_IMO"
MODEL_TRACKS = (
    TRACK_CALENDAR,
    TRACK_GLOBAL,
    TRACK_VESSEL,
    TRACK_WEATHER,
    TRACK_IDENTITY,
)

CALENDAR_PREFIXES = ("cutoff_", "eta_")
GLOBAL_HISTORY_PREFIX = "global_hist_"
VESSEL_HISTORY_PREFIX = "vessel_hist_"
WEATHER_MIN_GAIN_PCT = float(os.getenv("B54FD1_WEATHER_MIN_GAIN_PCT", "2.0"))

WEATHER_TOKENS = (
    "wave",
    "sea_",
    "wind_",
    "surface_current",
    "visibility",
    "pressure",
    "high_wave",
    "severe_wave",
)
IDENTITY_FEATURES = {"imo"}
BANNED_EXACT = {
    CALL_COLUMN,
    "source_record_id",
    TIME_COLUMN,
    "planned_eta",
    "actual_ata",
    "actual_atd",
    TARGET,
    "target_departure_delay_h",
    "arrived_before_cutoff_flag",
    "model_ready_flag",
}


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _start_run(source_uri: str, checksum: str, metadata: dict[str, Any]) -> str:
    with _db_connection() as connection:
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


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection:
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
                        _json_safe(metadata or {}),
                        dumps=lambda obj: json.dumps(
                            obj, default=_json_default, allow_nan=False
                        ),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection:
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


def _download(client, bucket: str, key: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))
    return destination


def _upload(
    client,
    source: Path,
    bucket: str,
    key: str,
    content_type: str,
) -> str:
    client.upload_file(
        str(source), bucket, key, ExtraArgs={"ContentType": content_type}
    )
    return f"s3://{bucket}/{key}"


def _source_checksum(
    client,
    bucket: str,
    keys: list[str],
    parameters: dict[str, Any],
) -> str:
    payload = []
    for key in keys:
        head = client.head_object(Bucket=bucket, Key=key)
        payload.append(
            {
                "bucket": bucket,
                "key": key,
                "etag": str(head["ETag"]).strip('"'),
                "size": int(head["ContentLength"]),
            }
        )
    payload.append({"parameters": parameters, "stress_version": STRESS_VERSION})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _is_safe_feature(column: str) -> bool:
    lowered = column.lower()
    if column in BANNED_EXACT:
        return False
    if lowered.startswith(("actual_", "target_", "has_arrival", "has_departure")):
        return False
    if "delay" in lowered and not lowered.startswith(
        ("vessel_hist_", "global_hist_")
    ):
        return False
    return not any(
        token in lowered
        for token in ("label", "outlier", "quarantine", "split")
    )


def _is_weather_feature(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in WEATHER_TOKENS)


def _prepare_source(frame: pd.DataFrame) -> pd.DataFrame:
    required = {CALL_COLUMN, TIME_COLUMN, "planned_eta", TARGET, "imo"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"B54F-D1 source columns missing: {missing}")
    result = frame.copy().reset_index(drop=True)
    result[CALL_COLUMN] = result[CALL_COLUMN].astype("string")
    result["imo"] = result["imo"].astype("string")
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], errors="coerce", utc=True)
    result["planned_eta"] = pd.to_datetime(
        result["planned_eta"], errors="coerce", utc=True
    )
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    if result[CALL_COLUMN].isna().any() or result[CALL_COLUMN].duplicated().any():
        raise RuntimeError("B54F-D1 requires exactly one row per port_call_id")
    if result[[TIME_COLUMN, "planned_eta", TARGET]].isna().any().any():
        raise RuntimeError("B54F-D1 source contains missing time or target values")
    if not np.isfinite(result[TARGET].to_numpy(dtype="float64")).all():
        raise RuntimeError("B54F-D1 target contains non-finite values")
    cutoff_delta = (result["planned_eta"] - result[TIME_COLUMN]).dt.total_seconds() / 3600
    if not np.allclose(cutoff_delta, 24.0, atol=1e-6):
        raise RuntimeError("B54F-D1 source violates PLANNED_ETA_MINUS_24H cutoff")
    result["label_available_at"] = result["planned_eta"] + pd.to_timedelta(
        result[TARGET], unit="h"
    )
    return result


def _prepare_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    required = {CALL_COLUMN, "protocol", "fold", "split", "diagnostic_only"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise RuntimeError(f"B54F-D1 assignment columns missing: {missing}")
    result = assignments[
        assignments["protocol"].isin(PROTOCOLS) & assignments["fold"].isna()
    ].copy()
    result[CALL_COLUMN] = result[CALL_COLUMN].astype("string")
    if set(result["protocol"].unique()) != set(PROTOCOLS):
        raise RuntimeError("B54F-D1 did not find all three frozen protocols")
    duplicates = result.duplicated(["protocol", CALL_COLUMN]).sum()
    if duplicates:
        raise RuntimeError(f"B54F-D1 duplicate protocol assignments: {duplicates}")
    return result


def _feature_tracks(config: dict[str, Any]) -> dict[str, list[str]]:
    frozen = list(dict.fromkeys(config.get("frozen_feature_columns", [])))
    if not frozen:
        raise RuntimeError("B54F-D1 frozen feature configuration is empty")
    unsafe = [column for column in frozen if not _is_safe_feature(column)]
    if unsafe:
        raise RuntimeError(f"Unsafe frozen features detected: {unsafe}")
    weather = {column for column in frozen if _is_weather_feature(column)}
    identity = set(frozen).intersection(IDENTITY_FEATURES)
    calendar = [
        column for column in frozen if column.startswith(CALENDAR_PREFIXES)
    ]
    global_history = [
        column for column in frozen if column.startswith(GLOBAL_HISTORY_PREFIX)
    ]
    vessel_history = [
        column for column in frozen if column.startswith(VESSEL_HISTORY_PREFIX)
    ]
    classified = (
        set(calendar)
        | set(global_history)
        | set(vessel_history)
        | weather
        | identity
    )
    unclassified = sorted(set(frozen) - classified)
    if unclassified:
        raise RuntimeError(
            "B54F-D1 frozen features have no ordered family: "
            f"{unclassified}"
        )
    calendar_global = calendar + global_history
    calendar_global_vessel = calendar_global + vessel_history
    tracks = {
        TRACK_CALENDAR: calendar,
        TRACK_GLOBAL: calendar_global,
        TRACK_VESSEL: calendar_global_vessel,
        TRACK_WEATHER: calendar_global_vessel
        + [column for column in frozen if column in weather],
        TRACK_IDENTITY: calendar_global_vessel
        + [column for column in frozen if column in weather]
        + [column for column in frozen if column in identity],
    }
    if not weather:
        raise RuntimeError("B54F-D1 needs weather features for the ablation")
    if not identity:
        raise RuntimeError("B54F-D1 needs IMO for the identity stress track")
    if len(calendar) < 5:
        raise RuntimeError("B54F-D1 calendar track contains too few features")
    if not global_history:
        raise RuntimeError("B54F-D1 needs global history features")
    if not vessel_history:
        raise RuntimeError("B54F-D1 needs vessel history features")
    ordered = list(tracks.values())
    if any(not set(left).issubset(right) for left, right in zip(ordered, ordered[1:])):
        raise RuntimeError("B54F-D1 ordered feature tracks are not nested")
    return tracks


def _protocol_frame(
    source: pd.DataFrame,
    assignments: pd.DataFrame,
    protocol: str,
) -> pd.DataFrame:
    assigned = assignments.loc[
        assignments["protocol"] == protocol,
        [CALL_COLUMN, "split", "diagnostic_only"],
    ]
    result = source.merge(assigned, on=CALL_COLUMN, how="inner", validate="one_to_one")
    if len(result) != len(source):
        raise RuntimeError(
            f"{protocol}: assignment coverage {len(result)} != source {len(source)}"
        )
    return result


def build_protocol_integrity_report(
    source: pd.DataFrame,
    assignments: pd.DataFrame,
    purge_hours: int = 72,
) -> pd.DataFrame:
    rows = []
    for protocol in PROTOCOLS:
        frame = _protocol_frame(source, assignments, protocol)
        active = frame[frame["split"].isin(["TRAIN", "VALID", "TEST"])]
        sets = {
            split: set(active.loc[active["split"] == split, CALL_COLUMN])
            for split in ("TRAIN", "VALID", "TEST")
        }
        id_overlap = sum(
            len(sets[left] & sets[right])
            for left, right in (
                ("TRAIN", "VALID"),
                ("TRAIN", "TEST"),
                ("VALID", "TEST"),
            )
        )
        vessel_sets = {
            split: set(
                active.loc[active["split"] == split, "imo"].dropna().astype(str)
            )
            for split in ("TRAIN", "VALID", "TEST")
        }
        vessel_overlap = sum(
            len(vessel_sets[left] & vessel_sets[right])
            for left, right in (
                ("TRAIN", "VALID"),
                ("TRAIN", "TEST"),
                ("VALID", "TEST"),
            )
        )
        train = active[active["split"] == "TRAIN"]
        valid = active[active["split"] == "VALID"]
        test = active[active["split"] == "TEST"]
        chronological = False
        label_available = False
        train_valid_gap = np.nan
        valid_test_gap = np.nan
        if not train.empty and not valid.empty and not test.empty:
            train_valid_gap = (
                valid[TIME_COLUMN].min() - train[TIME_COLUMN].max()
            ).total_seconds() / 3600.0
            valid_test_gap = (
                test[TIME_COLUMN].min() - valid[TIME_COLUMN].max()
            ).total_seconds() / 3600.0
            chronological = bool(
                train[TIME_COLUMN].max() < valid[TIME_COLUMN].min()
                and valid[TIME_COLUMN].max() < test[TIME_COLUMN].min()
            )
            label_available = bool(
                train["label_available_at"].max() <= valid[TIME_COLUMN].min()
                and valid["label_available_at"].max() <= test[TIME_COLUMN].min()
            )
        protocol_passed = id_overlap == 0 and min(len(train), len(valid), len(test)) >= 500
        if protocol == "RANDOM_BY_IMO":
            protocol_passed = protocol_passed and vessel_overlap == 0
        if protocol == OFFICIAL_PROTOCOL:
            protocol_passed = bool(
                protocol_passed
                and chronological
                and label_available
                and train_valid_gap >= purge_hours
                and valid_test_gap >= purge_hours
            )
        train_imo = set(train["imo"].dropna().astype(str))
        test_imo = test["imo"].astype("string")
        unseen_test = ~test_imo.isin(train_imo)
        rows.append(
            {
                "protocol": protocol,
                "diagnostic_only": protocol != OFFICIAL_PROTOCOL,
                "train_rows": len(train),
                "valid_rows": len(valid),
                "test_rows": len(test),
                "purged_rows": int((frame["split"] == "PURGED").sum()),
                "port_call_overlap": id_overlap,
                "vessel_group_overlap": vessel_overlap,
                "test_unseen_imo_rows": int(unseen_test.sum()),
                "test_unseen_imo_pct": 100.0 * float(unseen_test.mean()),
                "chronological_order_passed": chronological,
                "label_availability_passed": label_available,
                "train_valid_gap_h": train_valid_gap,
                "valid_test_gap_h": valid_test_gap,
                "protocol_gate_passed": protocol_passed,
            }
        )
    return pd.DataFrame(rows)


def _drop_unusable(
    train: pd.DataFrame, feature_columns: list[str]
) -> tuple[list[str], list[str]]:
    usable = []
    dropped = []
    for column in feature_columns:
        if column not in train.columns:
            dropped.append(column)
            continue
        if train[column].notna().sum() == 0 or train[column].nunique(dropna=True) <= 1:
            dropped.append(column)
        else:
            usable.append(column)
    return usable, dropped


def _matched_train_sample(train: pd.DataFrame, budget: int) -> pd.DataFrame:
    if len(train) < budget:
        raise RuntimeError(f"TRAIN rows {len(train)} are below matched budget {budget}")
    if len(train) == budget:
        return train.copy()
    return (
        train.sample(n=budget, replace=False, random_state=RANDOM_SEED)
        .sort_values([TIME_COLUMN, CALL_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def _matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    matrix = frame[feature_columns].copy()
    categorical = set(categorical_columns)
    for column in feature_columns:
        if column in categorical:
            matrix[column] = (
                matrix[column].astype("string").fillna("__MISSING__").astype(str)
            )
        else:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
            matrix[column] = matrix[column].replace([np.inf, -np.inf], np.nan)
            matrix[column] = matrix[column].astype("float32")
    return matrix


def _pool(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> Pool:
    return Pool(
        _matrix(frame, feature_columns, categorical_columns),
        label=frame[TARGET].to_numpy(dtype="float64"),
        cat_features=categorical_columns,
        feature_names=feature_columns,
    )


def _model_parameters() -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": ITERATIONS,
        "learning_rate": 0.04,
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
        "od_wait": 75,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if TASK_TYPE == "CPU":
        parameters["rsm"] = 0.90
    return parameters


def _fit_track(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    categorical_candidates: list[str],
    model_path: Path,
) -> tuple[CatBoostRegressor, list[str], int | None]:
    categorical = [
        column for column in categorical_candidates if column in feature_columns
    ]
    model = CatBoostRegressor(**_model_parameters())
    model.fit(
        _pool(train, feature_columns, categorical),
        eval_set=_pool(valid, feature_columns, categorical),
        use_best_model=True,
    )
    model.save_model(str(model_path))
    best_iteration = model.get_best_iteration()
    return model, categorical, None if best_iteration is None else int(best_iteration)


def _predict(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> np.ndarray:
    return np.asarray(
        model.predict(_pool(frame, feature_columns, categorical_columns)),
        dtype="float64",
    )


def baseline_predictions(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> dict[str, np.ndarray]:
    median = float(train[TARGET].median())
    vessel = (
        pd.to_numeric(evaluation["vessel_hist_mean_delay_h"], errors="coerce")
        if "vessel_hist_mean_delay_h" in evaluation.columns
        else pd.Series(np.nan, index=evaluation.index, dtype="float64")
    )
    global_history = (
        pd.to_numeric(evaluation["global_hist_mean_delay_h"], errors="coerce")
        if "global_hist_mean_delay_h" in evaluation.columns
        else pd.Series(np.nan, index=evaluation.index, dtype="float64")
    )
    history = vessel.fillna(global_history).fillna(median).to_numpy(dtype="float64")
    return {
        BASELINE_ZERO: np.zeros(len(evaluation), dtype="float64"),
        BASELINE_MEDIAN: np.full(len(evaluation), median, dtype="float64"),
        BASELINE_HISTORY: history,
    }


def _safe_auc(y: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    truth = y > threshold
    if np.unique(truth).size != 2:
        return float("nan")
    return float(roc_auc_score(truth, prediction))


def _metric_row(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    protocol: str,
    split: str,
    model: str,
    segment: str,
) -> dict[str, Any]:
    y = frame[TARGET].to_numpy(dtype="float64")
    error = prediction - y
    rmse = math.sqrt(mean_squared_error(y, prediction))
    r2 = float(r2_score(y, prediction)) if len(y) >= 2 else float("nan")
    return {
        "protocol": protocol,
        "diagnostic_only": protocol != OFFICIAL_PROTOCOL,
        "split": split,
        "model": model,
        "segment": segment,
        "n": len(frame),
        "MAE": float(mean_absolute_error(y, prediction)),
        "RMSE": float(rmse),
        "R2": r2,
        "MEDAE": float(median_absolute_error(y, prediction)),
        "BIAS": float(np.mean(error)),
        "WITHIN_1H_PCT": 100.0 * float(np.mean(np.abs(error) <= 1.0)),
        "WITHIN_2H_PCT": 100.0 * float(np.mean(np.abs(error) <= 2.0)),
        "WITHIN_3H_PCT": 100.0 * float(np.mean(np.abs(error) <= 3.0)),
        "AUC_DELAY_GT1H": _safe_auc(y, prediction, 1.0),
        "AUC_DELAY_GT6H": _safe_auc(y, prediction, 6.0),
    }


def evaluate_predictions(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    protocol: str,
    split: str,
    model: str,
) -> list[dict[str, Any]]:
    masks = {
        "ALL": np.ones(len(frame), dtype=bool),
        "EARLY_LT_NEG1H": (frame[TARGET] < -1).to_numpy(),
        "ON_TIME_ABS_LE1H": (frame[TARGET].abs() <= 1).to_numpy(),
        "LATE_1_6H": ((frame[TARGET] > 1) & (frame[TARGET] <= 6)).to_numpy(),
        "LATE_GT6H": (frame[TARGET] > 6).to_numpy(),
    }
    if "vessel_hist_count" in frame.columns:
        history_count = pd.to_numeric(
            frame["vessel_hist_count"], errors="coerce"
        ).fillna(0)
        masks["VESSEL_COLD_START"] = (history_count <= 0).to_numpy()
        masks["VESSEL_HISTORY_AVAILABLE"] = (history_count > 0).to_numpy()
    rows = []
    for segment, mask in masks.items():
        if int(mask.sum()) < (1 if segment == "ALL" else 20):
            continue
        rows.append(
            _metric_row(
                frame.loc[mask],
                prediction[mask],
                protocol,
                split,
                model,
                segment,
            )
        )
    return rows


def _global_metric(
    metrics: pd.DataFrame, protocol: str, split: str, model: str
) -> pd.Series:
    rows = metrics[
        (metrics["protocol"] == protocol)
        & (metrics["split"] == split)
        & (metrics["model"] == model)
        & (metrics["segment"] == "ALL")
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Missing global metric for {protocol}/{split}/{model}")
    return rows.iloc[0]


def _select_model(metrics: pd.DataFrame, protocol: str) -> str:
    candidates = metrics[
        (metrics["protocol"] == protocol)
        & (metrics["split"] == "VALID")
        & (metrics["segment"] == "ALL")
        & (metrics["model"].isin(MODEL_TRACKS))
    ].sort_values(["MAE", "RMSE", "model"], kind="mergesort")
    if candidates.empty:
        raise RuntimeError(f"No VALID CatBoost candidate for {protocol}")
    return str(candidates.iloc[0]["model"])


def _bootstrap_groups(frame: pd.DataFrame, protocol: str) -> tuple[pd.Series, str]:
    if protocol == OFFICIAL_PROTOCOL:
        timestamps = pd.to_datetime(frame[TIME_COLUMN], errors="coerce", utc=True)
        groups = timestamps.dt.strftime("%G-W%V").astype("string")
        fallback = frame[CALL_COLUMN].astype("string")
        groups = groups.fillna(fallback)
        return groups, "CALENDAR_WEEK_BLOCK"
    if protocol == "RANDOM_BY_IMO":
        groups = frame["imo"].astype("string").fillna(
            frame[CALL_COLUMN].astype("string")
        )
        return groups, "IMO_CLUSTER"
    return frame[CALL_COLUMN].astype("string"), "PORT_CALL_IID"


def _bootstrap_gain(
    frame: pd.DataFrame,
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    protocol: str,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    y = frame[TARGET].to_numpy(dtype="float64")
    model_error = np.abs(model_prediction - y)
    baseline_error = np.abs(baseline_prediction - y)
    groups, bootstrap_unit = _bootstrap_groups(frame, protocol)
    unique_groups = groups.unique()
    positions = {
        group: np.flatnonzero((groups == group).to_numpy()) for group in unique_groups
    }
    rng = np.random.default_rng(RANDOM_SEED + 911)
    gains = np.empty(repeats, dtype="float64")
    for index in range(repeats):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected = np.concatenate([positions[group] for group in sampled])
        gains[index] = float(
            baseline_error[selected].mean() - model_error[selected].mean()
        )
    observed = float(baseline_error.mean() - model_error.mean())
    return {
        "mae_gain_h": observed,
        "mae_gain_ci_low_h": float(np.quantile(gains, 0.025)),
        "mae_gain_ci_high_h": float(np.quantile(gains, 0.975)),
        "bootstrap_repeats": repeats,
        "bootstrap_unit": bootstrap_unit,
    }


def build_scorecards(
    metrics: pd.DataFrame,
    prediction_lookup: dict[tuple[str, str, str], np.ndarray],
    protocol_frames: dict[str, pd.DataFrame],
    integrity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows = []
    gain_rows = []
    ablation_rows = []
    for protocol in PROTOCOLS:
        selected = _select_model(metrics, protocol)
        capability = integrity[integrity["protocol"] == protocol].iloc[0]
        row: dict[str, Any] = {
            "protocol": protocol,
            "diagnostic_only": protocol != OFFICIAL_PROTOCOL,
            "capability": {
                "RANDOM_IID": "INTERPOLATION_MIXED_CALLS_OPTIMISTIC",
                "RANDOM_BY_IMO": "UNSEEN_VESSEL_COLD_START",
                OFFICIAL_PROTOCOL: "FUTURE_OPERATIONAL_GENERALIZATION",
            }[protocol],
            "selected_model_by_valid_mae": selected,
            "train_rows": int(capability["train_rows"]),
            "train_fit_rows": int(capability["train_fit_rows"]),
            "valid_rows": int(capability["valid_rows"]),
            "test_rows": int(capability["test_rows"]),
            "test_unseen_imo_pct": float(capability["test_unseen_imo_pct"]),
        }
        for split in ("VALID", "TEST"):
            model_metric = _global_metric(metrics, protocol, split, selected)
            baseline_metric = _global_metric(metrics, protocol, split, BASELINE_HISTORY)
            gain_h = float(baseline_metric["MAE"] - model_metric["MAE"])
            gain_pct = 100.0 * gain_h / max(1e-12, float(baseline_metric["MAE"]))
            prefix = split.lower()
            row[f"{prefix}_model_mae"] = float(model_metric["MAE"])
            row[f"{prefix}_baseline_mae"] = float(baseline_metric["MAE"])
            row[f"{prefix}_mae_gain_h"] = gain_h
            row[f"{prefix}_mae_gain_pct"] = gain_pct
            row[f"{prefix}_model_rmse"] = float(model_metric["RMSE"])
            row[f"{prefix}_model_r2"] = float(model_metric["R2"])
            row[f"{prefix}_model_bias"] = float(model_metric["BIAS"])
        test = protocol_frames[protocol]
        test = test[test["split"] == "TEST"].reset_index(drop=True)
        bootstrap = _bootstrap_gain(
            test,
            prediction_lookup[(protocol, "TEST", selected)],
            prediction_lookup[(protocol, "TEST", BASELINE_HISTORY)],
            protocol,
        )
        gain_rows.append({"protocol": protocol, "selected_model": selected, **bootstrap})
        row.update(bootstrap)
        score_rows.append(row)

        comparisons = (
            ("GLOBAL_HISTORY_UPLIFT", TRACK_CALENDAR, TRACK_GLOBAL),
            ("VESSEL_HISTORY_SYNERGY", TRACK_GLOBAL, TRACK_VESSEL),
            ("PAST_WEATHER_STRICT", TRACK_VESSEL, TRACK_WEATHER),
            ("IMO_IDENTITY_INCREMENT", TRACK_WEATHER, TRACK_IDENTITY),
        )
        for split in ("VALID", "TEST"):
            evaluation = protocol_frames[protocol]
            evaluation = evaluation[evaluation["split"] == split].reset_index(
                drop=True
            )
            for name, control, treatment in comparisons:
                left = _global_metric(metrics, protocol, split, control)
                right = _global_metric(metrics, protocol, split, treatment)
                delta = float(left["MAE"] - right["MAE"])
                row = {
                    "protocol": protocol,
                    "split": split,
                    "comparison": name,
                    "control_model": control,
                    "treatment_model": treatment,
                    "control_mae": float(left["MAE"]),
                    "treatment_mae": float(right["MAE"]),
                    "mae_gain_h": delta,
                    "mae_gain_pct": 100.0
                    * delta
                    / max(1e-12, float(left["MAE"])),
                    "r2_delta": float(right["R2"] - left["R2"]),
                    "mae_gain_ci_low_h": np.nan,
                    "mae_gain_ci_high_h": np.nan,
                    "bootstrap_repeats": 0,
                    "bootstrap_unit": "NOT_RUN_ON_VALID",
                }
                if split == "TEST":
                    uncertainty = _bootstrap_gain(
                        evaluation,
                        prediction_lookup[(protocol, split, treatment)],
                        prediction_lookup[(protocol, split, control)],
                        protocol,
                    )
                    row.update(uncertainty)
                ablation_rows.append(row)
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(gain_rows),
        pd.DataFrame(ablation_rows),
    )


def build_baseline_scorecard(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in PROTOCOLS:
        for split in ("VALID", "TEST"):
            history = _global_metric(
                metrics, protocol, split, BASELINE_HISTORY
            )
            history_mae = float(history["MAE"])
            for baseline in BASELINE_NAMES:
                metric = _global_metric(metrics, protocol, split, baseline)
                mae = float(metric["MAE"])
                gain_h = mae - history_mae
                rows.append(
                    {
                        "protocol": protocol,
                        "diagnostic_only": protocol != OFFICIAL_PROTOCOL,
                        "split": split,
                        "baseline": baseline,
                        "MAE": mae,
                        "RMSE": float(metric["RMSE"]),
                        "R2": float(metric["R2"]),
                        "BIAS": float(metric["BIAS"]),
                        "history_baseline_mae": history_mae,
                        "history_gain_vs_baseline_h": gain_h,
                        "history_gain_vs_baseline_pct": 100.0
                        * gain_h
                        / max(1e-12, mae),
                    }
                )
    return pd.DataFrame(rows)


def _decision(
    integrity: pd.DataFrame,
    scorecard: pd.DataFrame,
    ablations: pd.DataFrame,
    upstream_readiness: dict[str, Any],
) -> dict[str, Any]:
    fatal = []
    readiness_decision = upstream_readiness.get("decision", {}).get("status")
    readiness_gates = upstream_readiness.get("gates", {})
    if readiness_decision != "READY_FOR_MODEL_STRESS":
        fatal.append(f"UPSTREAM_READINESS={readiness_decision}")
    if not readiness_gates.get("all_critical_gates_passed", False):
        fatal.append("UPSTREAM_CRITICAL_GATES_FAILED")
    if not integrity["protocol_gate_passed"].all():
        failed = integrity.loc[
            ~integrity["protocol_gate_passed"], "protocol"
        ].tolist()
        fatal.append(f"PROTOCOL_GATES_FAILED={failed}")
    temporal = scorecard[scorecard["protocol"] == OFFICIAL_PROTOCOL]
    if len(temporal) != 1:
        fatal.append("TEMPORAL_SCORECARD_MISSING")
        return {
            "status": "NEED_PROTOCOL_REPAIR",
            "fatal_reasons": fatal,
            "official_protocol": OFFICIAL_PROTOCOL,
        }

    official = temporal.iloc[0]
    valid_gain = float(official["valid_mae_gain_pct"])
    test_gain = float(official["test_mae_gain_pct"])
    ci_low = float(official["mae_gain_ci_low_h"])
    test_r2 = float(official["test_model_r2"])
    if fatal:
        status = "NEED_PROTOCOL_REPAIR"
        next_block = "B54F_D1_REPAIR"
    elif valid_gain <= 0 or test_gain <= 0:
        status = "KEEP_BASELINE_NEED_NEW_SIGNALS"
        next_block = "B54F_SIGNAL_ACQUISITION_PORT_PRESSURE_AIS"
    elif test_gain >= 5.0 and ci_low > 0 and test_r2 > 0:
        status = "READY_FOR_B54F_D2_TEMPORAL_TUNING"
        next_block = "B54F_D2_TEMPORAL_TUNING_AND_CALIBRATION"
    else:
        status = "MODEST_UPLIFT_CONTINUE_WITH_CAUTION"
        next_block = "B54F_D2_LIMITED_TEMPORAL_TUNING"

    temporal_ablation = ablations[
        (ablations["protocol"] == OFFICIAL_PROTOCOL)
        & (ablations["split"].isin(["VALID", "TEST"]))
    ]
    weather = temporal_ablation[
        temporal_ablation["comparison"] == "PAST_WEATHER_STRICT"
    ]
    identity = temporal_ablation[
        temporal_ablation["comparison"] == "IMO_IDENTITY_INCREMENT"
    ]
    weather_valid = weather[weather["split"] == "VALID"]
    weather_test = weather[weather["split"] == "TEST"]
    weather_keep = bool(
        len(weather_valid) == 1
        and len(weather_test) == 1
        and float(weather_valid.iloc[0]["mae_gain_pct"])
        >= WEATHER_MIN_GAIN_PCT
        and float(weather_test.iloc[0]["mae_gain_pct"])
        >= WEATHER_MIN_GAIN_PCT
        and float(weather_test.iloc[0]["mae_gain_ci_low_h"]) > 0
    )
    identity_keep = bool(
        len(identity) == 2 and (identity["mae_gain_h"] > 0).all()
    )
    random_iid = scorecard[scorecard["protocol"] == "RANDOM_IID"].iloc[0]
    random_group = scorecard[scorecard["protocol"] == "RANDOM_BY_IMO"].iloc[0]
    temporal_mae = float(official["test_model_mae"])
    iid_mae = float(random_iid["test_model_mae"])
    group_mae = float(random_group["test_model_mae"])
    return {
        "status": status,
        "fatal_reasons": fatal,
        "official_protocol": OFFICIAL_PROTOCOL,
        "official_selected_model": official["selected_model_by_valid_mae"],
        "random_protocols_are_diagnostic_only": True,
        "selection_rule": "MIN_VALID_MAE_WITHIN_EACH_PROTOCOL; TEST_NEVER_SELECTS",
        "official_test_metrics": {
            "MAE": temporal_mae,
            "RMSE": float(official["test_model_rmse"]),
            "R2": test_r2,
            "BIAS": float(official["test_model_bias"]),
            "baseline_history_MAE": float(official["test_baseline_mae"]),
            "MAE_gain_h": float(official["test_mae_gain_h"]),
            "MAE_gain_pct": test_gain,
            "MAE_gain_ci_low_h": ci_low,
            "MAE_gain_ci_high_h": float(official["mae_gain_ci_high_h"]),
        },
        "protocol_diagnostics": {
            "random_iid_test_mae": iid_mae,
            "random_by_imo_test_mae": group_mae,
            "temporal_purged_test_mae": temporal_mae,
            "iid_apparent_optimism_pct_vs_temporal": 100.0
            * (temporal_mae - iid_mae)
            / max(1e-12, temporal_mae),
            "unseen_vessel_penalty_pct_vs_iid": 100.0
            * (group_mae - iid_mae)
            / max(1e-12, iid_mae),
            "warning": "RAW_PROTOCOL_MAE_VALUES_USE_DIFFERENT_TEST_POPULATIONS",
        },
        "past_weather_assessment": {
            "decision": (
                "KEEP_PAST_WEATHER"
                if weather_keep
                else "REJECT_CURRENT_PAST_WEATHER_TRACK"
            ),
            "minimum_required_gain_pct": WEATHER_MIN_GAIN_PCT,
            "valid_gain_pct": (
                None
                if weather_valid.empty
                else float(weather_valid.iloc[0]["mae_gain_pct"])
            ),
            "test_gain_pct": (
                None
                if weather_test.empty
                else float(weather_test.iloc[0]["mae_gain_pct"])
            ),
            "test_gain_ci_low_h": (
                None
                if weather_test.empty
                else float(weather_test.iloc[0]["mae_gain_ci_low_h"])
            ),
            "scope": (
                "This decision concerns the current past-only single-point "
                "weather representation, not future as-of forecasts or the "
                "physical relevance of weather."
            ),
        },
        "weather_decision": (
            "KEEP_PAST_WEATHER"
            if weather_keep
            else "REJECT_CURRENT_PAST_WEATHER_TRACK"
        ),
        "identity_decision": (
            "KEEP_IMO" if identity_keep else "IMO_DIAGNOSTIC_ONLY"
        ),
        "next_block": next_block,
    }


def _target_distribution(
    protocol_frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for protocol, frame in protocol_frames.items():
        for split in ("TRAIN", "VALID", "TEST"):
            values = frame.loc[frame["split"] == split, TARGET]
            rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "p05": values.quantile(0.05),
                    "p50": values.quantile(0.50),
                    "p95": values.quantile(0.95),
                    "late_gt1h_pct": 100.0 * (values > 1).mean(),
                    "late_gt6h_pct": 100.0 * (values > 6).mean(),
                }
            )
    return pd.DataFrame(rows)


def _save_comparison_plot(scorecard: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = scorecard.set_index("protocol").reindex(PROTOCOLS)
    positions = np.arange(len(ordered))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        positions - width / 2,
        ordered["test_baseline_mae"],
        width,
        label="Safe vessel-history baseline",
        color="#6b7280",
    )
    axis.bar(
        positions + width / 2,
        ordered["test_model_mae"],
        width,
        label="VALID-selected CatBoost",
        color="#087f8c",
    )
    axis.set_xticks(positions, ["Random IID", "Random by IMO", "Temporal purged"])
    axis.set_ylabel("TEST MAE (hours, lower is better)")
    axis.set_title("B54F-D1: native-test performance by validation capability")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _log_mlflow(
    run_id: str,
    scorecard: pd.DataFrame,
    decision: dict[str, Any],
    artifacts: list[Path],
) -> dict[str, Any]:
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("smart-port-maritime-b54fd1-split-stress")
        with mlflow.start_run(run_name=f"B54F-D1-{run_id[:8]}") as active:
            mlflow.log_params(
                {
                    "stress_version": STRESS_VERSION,
                    "official_protocol": OFFICIAL_PROTOCOL,
                    "iterations": ITERATIONS,
                    "random_seed": RANDOM_SEED,
                    "task_type": TASK_TYPE,
                    "bootstrap_repeats": BOOTSTRAP_REPEATS,
                    "training_tracks_per_protocol": len(MODEL_TRACKS),
                }
            )
            for _, row in scorecard.iterrows():
                slug = str(row["protocol"]).lower()
                mlflow.log_metric(f"{slug}_test_mae", float(row["test_model_mae"]))
                mlflow.log_metric(
                    f"{slug}_test_baseline_mae", float(row["test_baseline_mae"])
                )
                mlflow.log_metric(
                    f"{slug}_test_gain_pct", float(row["test_mae_gain_pct"])
                )
            mlflow.set_tag("decision", decision["status"])
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact), artifact_path="b54fd1")
            return {"status": "LOGGED", "mlflow_run_id": active.info.run_id}
    except Exception as exc:
        return {"status": "FAILED_NON_BLOCKING", "error": str(exc)}


def run_b54fd1_model_stress(
    source_bucket: str,
    model_ready_key: str,
    split_assignments_key: str,
    split_decision_key: str,
    readiness_config_key: str,
    readiness_decision_key: str,
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=2",
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    input_keys = [
        model_ready_key,
        split_assignments_key,
        split_decision_key,
        readiness_config_key,
        readiness_decision_key,
    ]
    parameters = {
        "iterations": ITERATIONS,
        "random_seed": RANDOM_SEED,
        "thread_count": THREAD_COUNT,
        "task_type": TASK_TYPE,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_policy": {
            "TEMPORAL_PURGED": "CALENDAR_WEEK_BLOCK",
            "RANDOM_BY_IMO": "IMO_CLUSTER",
            "RANDOM_IID": "PORT_CALL_IID",
        },
        "weather_min_gain_pct": WEATHER_MIN_GAIN_PCT,
        "protocols": list(PROTOCOLS),
        "model_tracks": list(MODEL_TRACKS),
        "selection_split": "VALID",
    }
    checksum = _source_checksum(client, source_bucket, input_keys, parameters)
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            previous_run_id, metadata = previous
            return {
                "status": "SUCCESS",
                "cached": True,
                "run_id": previous_run_id,
                "checksum": checksum,
                "results": metadata,
                "outputs": metadata.get("output_uris", {}),
            }

    source_uri = f"s3://{source_bucket}/{model_ready_key}"
    run_id = _start_run(
        source_uri,
        checksum,
        {
            "stress_version": STRESS_VERSION,
            "parameters": parameters,
            "training_executed": False,
        },
    )
    outputs: dict[str, str] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="b54fd1-") as temp_dir:
            work = Path(temp_dir)
            paths = {
                "source": _download(
                    client, source_bucket, model_ready_key, work / "model_ready.parquet"
                ),
                "assignments": _download(
                    client,
                    source_bucket,
                    split_assignments_key,
                    work / "assignments.parquet",
                ),
                "split_decision": _download(
                    client,
                    source_bucket,
                    split_decision_key,
                    work / "split_decision.json",
                ),
                "readiness_config": _download(
                    client,
                    source_bucket,
                    readiness_config_key,
                    work / "readiness_config.json",
                ),
                "readiness_decision": _download(
                    client,
                    source_bucket,
                    readiness_decision_key,
                    work / "readiness_decision.json",
                ),
            }
            source = _prepare_source(pd.read_parquet(paths["source"]))
            assignments = _prepare_assignments(pd.read_parquet(paths["assignments"]))
            split_decision = json.loads(
                paths["split_decision"].read_text(encoding="utf-8")
            )
            readiness_config = json.loads(
                paths["readiness_config"].read_text(encoding="utf-8")
            )
            readiness_decision = json.loads(
                paths["readiness_decision"].read_text(encoding="utf-8")
            )
            if (
                split_decision.get("decision", {}).get("status")
                != "READY_FOR_SPLIT_STRESS_MODELS"
            ):
                raise RuntimeError("B54F-D1 upstream split decision is not ready")
            tracks = _feature_tracks(readiness_config)
            missing_features = sorted(
                set(column for values in tracks.values() for column in values)
                - set(source.columns)
            )
            if missing_features:
                raise RuntimeError(f"B54F-D1 frozen source features missing: {missing_features}")
            categorical_candidates = [
                column
                for column in readiness_config.get("frozen_categorical_features", [])
                if column in source.columns
            ]
            integrity = build_protocol_integrity_report(source, assignments)
            if not integrity["protocol_gate_passed"].all():
                raise RuntimeError(
                    "B54F-D1 protocol integrity failed: "
                    + integrity.loc[
                        ~integrity["protocol_gate_passed"], "protocol"
                    ].to_json(orient="values")
                )
            train_budget = int(integrity["train_rows"].min())
            integrity["train_fit_rows"] = train_budget
            integrity["train_budget_matched"] = True

            metrics_rows: list[dict[str, Any]] = []
            usage_rows: list[dict[str, Any]] = []
            protocol_frames: dict[str, pd.DataFrame] = {}
            prediction_lookup: dict[tuple[str, str, str], np.ndarray] = {}
            prediction_outputs = []
            model_uploads: list[tuple[str, Path]] = []

            for protocol in PROTOCOLS:
                frame = _protocol_frame(source, assignments, protocol)
                protocol_frames[protocol] = frame
                assigned_train = frame[frame["split"] == "TRAIN"].copy()
                train = _matched_train_sample(assigned_train, train_budget)
                valid = frame[frame["split"] == "VALID"].copy()
                test = frame[frame["split"] == "TEST"].copy()
                model_records = {}
                for track, candidates in tracks.items():
                    usable, dropped = _drop_unusable(train, candidates)
                    if len(usable) < 5:
                        raise RuntimeError(
                            f"{protocol}/{track}: fewer than five usable features"
                        )
                    model_path = work / f"{protocol.lower()}__{track.lower()}.cbm"
                    model, categoricals, best_iteration = _fit_track(
                        train, valid, usable, categorical_candidates, model_path
                    )
                    model_records[track] = (model, usable, categoricals)
                    usage_rows.append(
                        {
                            "protocol": protocol,
                            "model": track,
                            "candidate_features": len(candidates),
                            "assigned_train_rows": len(assigned_train),
                            "fit_train_rows": len(train),
                            "used_features": len(usable),
                            "categorical_features": len(categoricals),
                            "dropped_features": json.dumps(dropped),
                            "best_iteration": best_iteration,
                        }
                    )
                    model_uploads.append(
                        (
                            f"model_{protocol.lower()}_{track.lower()}",
                            model_path,
                        )
                    )

                for split, evaluation in (("VALID", valid), ("TEST", test)):
                    predictions = baseline_predictions(train, evaluation)
                    for track, (model, usable, categoricals) in model_records.items():
                        predictions[track] = _predict(
                            model, evaluation, usable, categoricals
                        )
                    prediction_frame = evaluation[
                        [CALL_COLUMN, "imo", TIME_COLUMN, "planned_eta", TARGET, "split"]
                    ].copy()
                    prediction_frame["protocol"] = protocol
                    for model_name, values in predictions.items():
                        prediction_lookup[(protocol, split, model_name)] = values
                        metrics_rows.extend(
                            evaluate_predictions(
                                evaluation,
                                values,
                                protocol,
                                split,
                                model_name,
                            )
                        )
                        prediction_frame[f"PRED_{model_name}"] = values
                    prediction_outputs.append(prediction_frame)

            metrics = pd.DataFrame(metrics_rows)
            usage = pd.DataFrame(usage_rows)
            scorecard, baseline_gains, ablations = build_scorecards(
                metrics, prediction_lookup, protocol_frames, integrity
            )
            baseline_scorecard = build_baseline_scorecard(metrics)
            decision = _decision(
                integrity, scorecard, ablations, readiness_decision
            )
            target_distribution = _target_distribution(protocol_frames)
            predictions = pd.concat(prediction_outputs, ignore_index=True)

            report_files = {
                "protocol_capability": ("01_protocol_capability_and_integrity.csv", integrity),
                "metrics": ("02_metrics_by_protocol_model_split_segment.csv", metrics),
                "scorecard": ("03_protocol_scorecard.csv", scorecard),
                "baseline_scorecard": (
                    "03a_zero_median_history_baselines.csv",
                    baseline_scorecard,
                ),
                "baseline_gains": ("04_baseline_gains_bootstrap_ci.csv", baseline_gains),
                "ablations": ("05_ordered_family_ablations.csv", ablations),
                "feature_usage": ("06_feature_usage_by_protocol_track.csv", usage),
                "target_distribution": ("07_target_distribution_by_protocol.csv", target_distribution),
            }
            artifact_paths = []
            prefix = output_prefix.strip("/") or "version=2"
            for label, (name, data) in report_files.items():
                path = work / name
                data.to_csv(path, index=False)
                artifact_paths.append(path)
                outputs[label] = _upload(
                    client,
                    path,
                    output_bucket,
                    f"reports/b54fd1/{prefix}/{name}",
                    "text/csv",
                )

            predictions_path = work / "08_valid_test_predictions.parquet"
            predictions.to_parquet(predictions_path, index=False, compression="zstd")
            artifact_paths.append(predictions_path)
            outputs["predictions"] = _upload(
                client,
                predictions_path,
                output_bucket,
                f"predictions/b54fd1/{prefix}/08_valid_test_predictions.parquet",
                "application/x-parquet",
            )
            plot_path = work / "09_native_test_mae_comparison.png"
            _save_comparison_plot(scorecard, plot_path)
            artifact_paths.append(plot_path)
            outputs["comparison_plot"] = _upload(
                client,
                plot_path,
                output_bucket,
                f"reports/b54fd1/{prefix}/09_native_test_mae_comparison.png",
                "image/png",
            )

            for label, model_path in model_uploads:
                protocol, track = model_path.stem.split("__", 1)
                outputs[label] = _upload(
                    client,
                    model_path,
                    output_bucket,
                    f"models/b54fd1/{prefix}/{protocol}/{track}.cbm",
                    "application/octet-stream",
                )

            config_payload = {
                "stress_version": STRESS_VERSION,
                "target": TARGET,
                "grain": "ONE_ROW_PER_PORT_CALL",
                "prediction_cutoff": "PLANNED_ETA_MINUS_24H",
                "official_protocol": OFFICIAL_PROTOCOL,
                "diagnostic_protocols": ["RANDOM_IID", "RANDOM_BY_IMO"],
                "model_tracks": tracks,
                "categorical_candidates": categorical_candidates,
                "model_parameters": _model_parameters(),
                "baseline_models": list(BASELINE_NAMES),
                "selection_policy": "SELECT_MODEL_ON_VALID_ONLY_TEST_ONCE",
                "comparison_policy": (
                    "ORDERED_NESTED_ABLATION_CALENDAR_GLOBAL_VESSEL_WEATHER_IMO; "
                    "COMPARE_UPLIFT_VS_SAME_PROTOCOL_HISTORY_BASELINE; "
                    "DO_NOT_RANK_RAW_MAE_ACROSS_DIFFERENT_TEST_POPULATIONS"
                ),
                "weather_retention_policy": (
                    "KEEP_ONLY_IF_TEMPORAL_VALID_AND_TEST_GAIN_AT_LEAST_2PCT_"
                    "AND_TEST_WEEK_BLOCK_BOOTSTRAP_CI_LOW_GT_ZERO"
                ),
                "train_budget_policy": (
                    "MATCH_ALL_PROTOCOLS_TO_MINIMUM_ASSIGNED_TRAIN_ROWS"
                ),
                "train_fit_rows_per_protocol": train_budget,
            }
            config_path = work / "b54fd1_model_stress_config_v2.json"
            config_path.write_text(
                json.dumps(
                    _json_safe(config_payload), indent=2, allow_nan=False
                ),
                encoding="utf-8",
            )
            outputs["config"] = _upload(
                client,
                config_path,
                output_bucket,
                f"configs/b54fd1/{prefix}/b54fd1_model_stress_config_v2.json",
                "application/json",
            )

            summary = _json_safe(
                {
                    "stress_version": STRESS_VERSION,
                    "source_rows": len(source),
                    "source_calls": source[CALL_COLUMN].nunique(),
                    "frozen_feature_count": len(
                        readiness_config.get("frozen_feature_columns", [])
                    ),
                    "protocol_count": len(PROTOCOLS),
                    "train_fit_rows_per_protocol": train_budget,
                    "trained_model_count": len(model_uploads),
                    "baseline_count": len(BASELINE_NAMES),
                    "parameters": parameters,
                    "gates": {
                        "upstream_readiness_passed": (
                            readiness_decision.get("decision", {}).get("status")
                            == "READY_FOR_MODEL_STRESS"
                        ),
                        "all_protocol_gates_passed": bool(
                            integrity["protocol_gate_passed"].all()
                        ),
                        "test_selection_leakage": 0,
                        "random_results_official": False,
                    },
                    "decision": decision,
                    "scorecard": scorecard.to_dict(orient="records"),
                    "training_executed": True,
                    "output_uris": outputs,
                    "generated_at_utc": datetime.now(timezone.utc),
                }
            )
            decision_path = work / "b54fd1_model_stress_decision_v2.json"
            decision_path.write_text(
                json.dumps(summary, indent=2, default=_json_default, allow_nan=False),
                encoding="utf-8",
            )
            outputs["decision"] = _upload(
                client,
                decision_path,
                output_bucket,
                f"configs/b54fd1/{prefix}/b54fd1_model_stress_decision_v2.json",
                "application/json",
            )
            summary["output_uris"] = outputs
            mlflow_result = _log_mlflow(
                run_id,
                scorecard,
                decision,
                artifact_paths + [config_path, decision_path],
            )
            summary["mlflow"] = mlflow_result

        _finish_run(run_id, "SUCCESS", len(source), summary)
        return {
            "status": "SUCCESS",
            "cached": False,
            "run_id": run_id,
            "checksum": checksum,
            "results": summary,
            "outputs": outputs,
        }
    except Exception as exc:
        _finish_run(run_id, "FAILED", error_message=str(exc)[:4000])
        raise
