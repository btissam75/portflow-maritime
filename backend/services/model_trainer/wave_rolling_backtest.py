from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import boto3
import mlflow
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - reported as an unavailable challenger
    CatBoostRegressor = None


MODEL_VERSION = "b58b-wave-rolling-backtest-v1"
DATASET_VERSION = "b58b-wave-multihorizon-v1"
SOURCE_NAME = "b58b_wave_temporal_backtest"
DATASET_NAME = "maritime_wave_multihorizon_probabilistic"
SOURCE_BUCKET = "gold-maritime"
SOURCE_KEY = (
    "datasets/b58a/version=1/"
    "maritime_weather_hourly_past_only_v1.parquet"
)
SHADOW_TABLE = "features.maritime_wave_forecast_shadow_v1"

HORIZONS_H = (6, 12, 24, 48, 72)
LATENCY_STRESS_H = (1, 3, 6)
OFFICIAL_LATENCY_H = 3
PURGE_H = 72
CONFORMAL_WINDOW_H = 24 * 90
RANDOM_SEED = 20260724

TARGET_COMPONENTS = {
    "wave_height_m": "target_wave_height_m",
    "wave_period_s": "target_wave_period_s",
    "direction_sin": "target_direction_sin",
    "direction_cos": "target_direction_cos",
}
TARGETS = ("wave_height_m", "wave_period_s", "wave_direction_deg")
BASELINES = (
    "PERSISTENCE",
    "SEASONAL_SAFE_24H",
    "SEASONAL_SAFE_168H",
)
MODEL_FAMILIES = ("HGB_GLOBAL", "CATBOOST_GLOBAL")


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
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


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


def _query_frame(
    query: str, parameters: tuple[Any, ...] | None = None
) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _load_upstream_decision() -> dict[str, Any]:
    frame = _query_frame(
        """
        SELECT status, metadata
        FROM audit.ingestion_run
        WHERE source_name='b58a_weather_timeseries_audit'
          AND dataset_name='maritime_weather_hourly_multivariate'
          AND status='SUCCESS'
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    if frame.empty:
        raise RuntimeError("B58A SUCCESS audit was not found")
    metadata = frame.iloc[0]["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    decision = metadata.get("status") or metadata.get("decision")
    if decision != "READY_FOR_WAVE_ONLY_TEMPORAL_BASELINES":
        raise RuntimeError(f"B58A wave-only contract is not ready: {decision}")
    if int(metadata.get("critical_leakage_violations", -1)) != 0:
        raise RuntimeError("B58A reported critical leakage violations")
    return metadata


def _load_source(client, bucket: str) -> pd.DataFrame:
    payload = client.get_object(Bucket=bucket, Key=SOURCE_KEY)["Body"].read()
    frame = pd.read_parquet(io.BytesIO(payload))
    required = {
        "observed_at",
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "wave_direction_deg_sin",
        "wave_direction_deg_cos",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"B58A Gold is missing columns: {missing}")
    frame["observed_at"] = pd.to_datetime(
        frame["observed_at"], errors="coerce", utc=True
    )
    if frame["observed_at"].isna().any():
        raise RuntimeError("B58A Gold contains invalid observed_at values")
    frame = frame.sort_values("observed_at").reset_index(drop=True)
    if frame["observed_at"].duplicated().any():
        raise RuntimeError("B58A Gold is not one row per hour")
    expected = pd.date_range(
        frame["observed_at"].min(),
        frame["observed_at"].max(),
        freq="h",
        tz="UTC",
    )
    if len(expected) != len(frame):
        raise RuntimeError("B58A Gold is not an uninterrupted hourly grid")
    for column in (
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "wave_direction_deg_sin",
        "wave_direction_deg_cos",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[
        ["wave_height_m", "wave_period_s", "wave_direction_deg"]
    ].isna().any().any():
        raise RuntimeError("B58A wave targets contain missing values")
    return frame


def _source_signature(frame: pd.DataFrame, upstream: dict[str, Any]) -> str:
    digest = hashlib.sha256(MODEL_VERSION.encode("ascii"))
    digest.update(SOURCE_KEY.encode("ascii"))
    digest.update(
        str(upstream.get("checksum", "missing-upstream-checksum")).encode()
    )
    columns = [
        "observed_at",
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
    ]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False)
    digest.update(hashed.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _start_run(checksum: str) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "training_executed": False,
        "selection_used_test": False,
        "bronze_modified": False,
        "core_modified": False,
        "production_promotion_allowed": False,
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
                    f"s3://{SOURCE_BUCKET}/{SOURCE_KEY}",
                    checksum,
                    Json(metadata),
                ),
            )
            return str(cursor.fetchone()[0])


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    frame = _query_frame(
        """
        SELECT run_id::text, metadata
        FROM audit.ingestion_run
        WHERE source_name=%s
          AND dataset_name=%s
          AND checksum=%s
          AND status='SUCCESS'
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (SOURCE_NAME, DATASET_NAME, checksum),
    )
    if frame.empty:
        return None
    metadata = frame.iloc[0]["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return str(frame.iloc[0]["run_id"]), metadata


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    payload = _clean_json(metadata)
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
                        payload,
                        dumps=lambda x: json.dumps(
                            x, default=_json_default, allow_nan=False
                        ),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    progress = {
        "stage": stage,
        "updated_at": pd.Timestamp.now(tz="UTC"),
        **details,
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET metadata=metadata || %s
                WHERE run_id=%s
                """,
                (
                    Json(
                        {"progress": _clean_json(progress)},
                        dumps=lambda x: json.dumps(x, allow_nan=False),
                    ),
                    run_id,
                ),
            )


def _upload_file(client, path: Path, bucket: str, key: str) -> str:
    client.upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key}"


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    allowed = [
        "observation_count",
        "source_count",
        "latitude",
        "longitude",
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg_sin",
        "wave_direction_deg_cos",
        "wave_family_available_flag",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "weekend_flag",
    ]
    allowed.extend(
        column
        for column in frame.columns
        if column.startswith("past_wave_")
    )
    allowed = [column for column in allowed if column in frame.columns]
    if not any(column.startswith("past_wave_") for column in allowed):
        raise RuntimeError("B58A past-only wave features were not found")
    forbidden_tokens = (
        "target_",
        "future_",
        "actual_",
        "wind_",
        "surface_current",
        "visibility",
        "pressure",
    )
    violations = [
        column
        for column in allowed
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if violations:
        raise RuntimeError(f"Forbidden B58B features selected: {violations}")
    return allowed


def _calendar_features(times: pd.Series, prefix: str) -> pd.DataFrame:
    hour = times.dt.hour.to_numpy(dtype="float64")
    day = times.dt.dayofyear.to_numpy(dtype="float64")
    return pd.DataFrame(
        {
            f"{prefix}_hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            f"{prefix}_hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            f"{prefix}_day_sin": np.sin(2.0 * np.pi * day / 366.0),
            f"{prefix}_day_cos": np.cos(2.0 * np.pi * day / 366.0),
            f"{prefix}_weekend": (times.dt.dayofweek >= 5).astype("float64"),
        }
    )


def _split_boundaries(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    train_boundary = frame.loc[int(len(frame) * 0.70), "observed_at"]
    valid_boundary = frame.loc[int(len(frame) * 0.85), "observed_at"]
    return train_boundary, valid_boundary


def _assign_split(
    examples: pd.DataFrame,
    train_boundary: pd.Timestamp,
    valid_boundary: pd.Timestamp,
) -> pd.Series:
    purge = pd.Timedelta(hours=PURGE_H)
    split = np.full(len(examples), "EXCLUDED_PURGE", dtype=object)
    train_mask = examples["target_at"] < train_boundary
    valid_mask = (
        (examples["issue_at"] >= train_boundary + purge)
        & (examples["target_at"] < valid_boundary)
    )
    test_mask = examples["issue_at"] >= valid_boundary + purge
    split[train_mask] = "TRAIN"
    split[valid_mask] = "VALID"
    split[test_mask] = "TEST"
    return pd.Series(split, index=examples.index)


def _safe_reference_index(
    issue_indices: np.ndarray,
    horizon_h: int,
    latency_h: int,
    period_h: int,
) -> np.ndarray:
    cycles = int(math.ceil((horizon_h + latency_h) / period_h))
    return issue_indices + horizon_h - cycles * period_h


def _build_examples(
    frame: pd.DataFrame,
    latency_h: int,
    base_features: list[str],
    assign_splits: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    train_boundary, valid_boundary = _split_boundaries(frame)
    blocks: list[pd.DataFrame] = []
    n_rows = len(frame)
    for horizon_h in HORIZONS_H:
        issue_indices = np.arange(latency_h, n_rows - horizon_h)
        source_indices = issue_indices - latency_h
        target_indices = issue_indices + horizon_h
        daily_indices = _safe_reference_index(
            issue_indices, horizon_h, latency_h, 24
        )
        weekly_indices = _safe_reference_index(
            issue_indices, horizon_h, latency_h, 168
        )

        source = frame.iloc[source_indices][base_features].reset_index(drop=True)
        source = source.apply(pd.to_numeric, errors="coerce")
        issue_times = frame.iloc[issue_indices]["observed_at"].reset_index(
            drop=True
        )
        target_times = frame.iloc[target_indices]["observed_at"].reset_index(
            drop=True
        )
        block = source.copy()
        block.insert(0, "issue_index", issue_indices)
        block.insert(1, "source_index", source_indices)
        block.insert(2, "target_index", target_indices)
        block.insert(3, "issue_at", issue_times)
        block.insert(
            4,
            "source_at",
            frame.iloc[source_indices]["observed_at"].to_numpy(),
        )
        block.insert(5, "target_at", target_times)
        block["horizon_h"] = float(horizon_h)
        block["latency_h"] = float(latency_h)
        for calendar in (
            _calendar_features(issue_times, "issue"),
            _calendar_features(target_times, "target"),
        ):
            for column in calendar.columns:
                block[column] = calendar[column].to_numpy()

        block["target_wave_height_m"] = frame.iloc[target_indices][
            "wave_height_m"
        ].to_numpy()
        block["target_wave_period_s"] = frame.iloc[target_indices][
            "wave_period_s"
        ].to_numpy()
        block["target_wave_direction_deg"] = frame.iloc[target_indices][
            "wave_direction_deg"
        ].to_numpy()
        radians = np.deg2rad(block["target_wave_direction_deg"].to_numpy())
        block["target_direction_sin"] = np.sin(radians)
        block["target_direction_cos"] = np.cos(radians)

        reference_specs = {
            "persistence": source_indices,
            "seasonal_24h": daily_indices,
            "seasonal_168h": weekly_indices,
        }
        for prefix, indices in reference_specs.items():
            valid = indices >= 0
            safe_indices = np.maximum(indices, 0)
            for target in TARGETS:
                values = frame.iloc[safe_indices][target].to_numpy(
                    dtype="float64"
                )
                values[~valid] = np.nan
                block[f"{prefix}_{target}"] = values

        block["split"] = (
            _assign_split(block, train_boundary, valid_boundary)
            if assign_splits
            else "UNASSIGNED"
        )
        blocks.append(block)

    examples = pd.concat(blocks, ignore_index=True)
    engineered = [
        "horizon_h",
        "latency_h",
        "issue_hour_sin",
        "issue_hour_cos",
        "issue_day_sin",
        "issue_day_cos",
        "issue_weekend",
        "target_hour_sin",
        "target_hour_cos",
        "target_day_sin",
        "target_day_cos",
        "target_weekend",
    ]
    model_features = [*base_features, *engineered]
    examples[model_features] = examples[model_features].replace(
        [np.inf, -np.inf], np.nan
    )
    examples[model_features] = examples[model_features].astype("float32")
    return examples, model_features


def _fit_model_bank(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    families: Iterable[str] = MODEL_FAMILIES,
    fast: bool = False,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    x_train = train[features]
    x_valid = valid[features]
    bank: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    thread_count = int(os.getenv("B54D_THREAD_COUNT", "2"))
    for family in families:
        if family == "CATBOOST_GLOBAL" and CatBoostRegressor is None:
            inventory.append(
                {
                    "family": family,
                    "component": "*",
                    "status": "UNAVAILABLE",
                    "reason": "catboost is not installed",
                }
            )
            continue
        bank[family] = {}
        for component, target_column in TARGET_COMPONENTS.items():
            try:
                if family == "HGB_GLOBAL":
                    model = HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=0.055,
                        max_iter=140 if fast else 260,
                        max_leaf_nodes=25,
                        min_samples_leaf=48,
                        l2_regularization=1.5,
                        early_stopping=False,
                        random_state=RANDOM_SEED,
                    )
                    model.fit(x_train, train[target_column])
                    iterations = int(model.n_iter_)
                elif family == "CATBOOST_GLOBAL":
                    model = CatBoostRegressor(
                        loss_function="MAE",
                        eval_metric="MAE",
                        iterations=180 if fast else 420,
                        depth=7,
                        learning_rate=0.045,
                        l2_leaf_reg=5.0,
                        random_seed=RANDOM_SEED,
                        thread_count=thread_count,
                        allow_writing_files=False,
                        verbose=False,
                        od_type="Iter",
                        od_wait=35 if fast else 60,
                    )
                    model.fit(
                        x_train,
                        train[target_column],
                        eval_set=(x_valid, valid[target_column]),
                        use_best_model=True,
                        verbose=False,
                    )
                    iterations = int(model.get_best_iteration()) + 1
                else:
                    raise ValueError(f"Unknown family: {family}")
                bank[family][component] = model
                inventory.append(
                    {
                        "family": family,
                        "component": component,
                        "status": "FITTED",
                        "features": len(features),
                        "train_rows": len(train),
                        "valid_rows": len(valid),
                        "iterations": iterations,
                        "reason": None,
                    }
                )
            except Exception as exc:
                bank.pop(family, None)
                inventory.append(
                    {
                        "family": family,
                        "component": component,
                        "status": "FAILED",
                        "reason": str(exc),
                    }
                )
                break
    return bank, pd.DataFrame(inventory)


def _direction_from_components(
    sine: np.ndarray, cosine: np.ndarray
) -> np.ndarray:
    norm = np.sqrt(np.square(sine) + np.square(cosine))
    norm = np.where(norm < 1e-8, 1.0, norm)
    angle = np.rad2deg(np.arctan2(sine / norm, cosine / norm))
    return np.mod(angle, 360.0)


def _add_predictions(
    examples: pd.DataFrame,
    features: list[str],
    bank: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    result = examples.copy()
    models = list(BASELINES)
    baseline_prefixes = {
        "PERSISTENCE": "persistence",
        "SEASONAL_SAFE_24H": "seasonal_24h",
        "SEASONAL_SAFE_168H": "seasonal_168h",
    }
    for model_name, prefix in baseline_prefixes.items():
        for target in TARGETS:
            result[f"pred__{model_name}__{target}"] = result[
                f"{prefix}_{target}"
            ].to_numpy()

    matrix = result[features]
    for family, component_models in bank.items():
        predictions = {
            component: np.asarray(model.predict(matrix), dtype="float64")
            for component, model in component_models.items()
        }
        result[f"pred__{family}__wave_height_m"] = np.clip(
            predictions["wave_height_m"], 0.0, None
        )
        result[f"pred__{family}__wave_period_s"] = np.clip(
            predictions["wave_period_s"], 0.0, None
        )
        result[f"pred__{family}__wave_direction_deg"] = (
            _direction_from_components(
                predictions["direction_sin"],
                predictions["direction_cos"],
            )
        )
        models.append(family)
    return result, models


def _angular_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return (predicted - actual + 180.0) % 360.0 - 180.0


def _metric_row(
    actual: np.ndarray,
    predicted: np.ndarray,
    target: str,
) -> dict[str, Any]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if not len(actual):
        return {
            "n": 0,
            "MAE": None,
            "RMSE": None,
            "BIAS": None,
            "R2": None,
            "SMAPE_PCT": None,
            "WITHIN_15_DEG_PCT": None,
            "WITHIN_30_DEG_PCT": None,
        }
    if target == "wave_direction_deg":
        error = _angular_error(actual, predicted)
        absolute = np.abs(error)
        return {
            "n": len(actual),
            "MAE": float(np.mean(absolute)),
            "RMSE": float(np.sqrt(np.mean(np.square(error)))),
            "BIAS": float(np.mean(error)),
            "R2": None,
            "SMAPE_PCT": None,
            "WITHIN_15_DEG_PCT": float(np.mean(absolute <= 15.0) * 100.0),
            "WITHIN_30_DEG_PCT": float(np.mean(absolute <= 30.0) * 100.0),
        }
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.where(denominator > 1e-8, 2.0 * np.abs(error) / denominator, 0)
    total = np.sum(np.square(actual - np.mean(actual)))
    return {
        "n": len(actual),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "BIAS": float(np.mean(error)),
        "R2": (
            float(1.0 - np.sum(np.square(error)) / total)
            if total > 1e-12
            else None
        ),
        "SMAPE_PCT": float(np.mean(smape) * 100.0),
        "WITHIN_15_DEG_PCT": None,
        "WITHIN_30_DEG_PCT": None,
    }


def _candidate_metrics(
    predictions: pd.DataFrame,
    split: str,
    models: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = predictions.loc[predictions["split"].eq(split)]
    for horizon_h in HORIZONS_H:
        horizon = subset.loc[subset["horizon_h"].eq(horizon_h)]
        for target in TARGETS:
            actual = horizon[f"target_{target}"].to_numpy(dtype="float64")
            for model in models:
                predicted = horizon[
                    f"pred__{model}__{target}"
                ].to_numpy(dtype="float64")
                rows.append(
                    {
                        "split": split,
                        "horizon_h": horizon_h,
                        "target": target,
                        "model": model,
                        **_metric_row(actual, predicted, target),
                    }
                )
    return pd.DataFrame(rows)


def _select_models(valid_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (horizon_h, target), group in valid_metrics.groupby(
        ["horizon_h", "target"], sort=True
    ):
        eligible = group.loc[group["MAE"].notna() & group["n"].gt(0)].copy()
        if eligible.empty:
            raise RuntimeError(
                f"No valid candidate for horizon={horizon_h}, target={target}"
            )
        best = eligible.sort_values(["MAE", "model"]).iloc[0]
        persistence = eligible.loc[eligible["model"].eq("PERSISTENCE")]
        persistence_mae = (
            float(persistence.iloc[0]["MAE"]) if not persistence.empty else None
        )
        best_mae = float(best["MAE"])
        uplift = (
            100.0 * (persistence_mae - best_mae) / persistence_mae
            if persistence_mae and persistence_mae > 0
            else None
        )
        rows.append(
            {
                "horizon_h": int(horizon_h),
                "target": target,
                "selected_model": best["model"],
                "validation_mae": best_mae,
                "persistence_mae": persistence_mae,
                "validation_uplift_pct": uplift,
                "selection_split": "VALID",
                "test_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def _apply_selection(
    predictions: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    result = predictions.copy()
    for target in TARGETS:
        result[f"selected_pred_{target}"] = np.nan
        result[f"selected_model_{target}"] = ""
    for row in selection.itertuples(index=False):
        mask = result["horizon_h"].eq(row.horizon_h)
        result.loc[mask, f"selected_pred_{row.target}"] = result.loc[
            mask, f"pred__{row.selected_model}__{row.target}"
        ]
        result.loc[mask, f"selected_model_{row.target}"] = row.selected_model
    return result


def _selected_test_metrics(
    selected: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test = selected.loc[selected["split"].eq("TEST")]
    for row in selection.itertuples(index=False):
        subset = test.loc[test["horizon_h"].eq(row.horizon_h)]
        metric = _metric_row(
            subset[f"target_{row.target}"].to_numpy(dtype="float64"),
            subset[f"selected_pred_{row.target}"].to_numpy(dtype="float64"),
            row.target,
        )
        rows.append(
            {
                "split": "TEST",
                "horizon_h": row.horizon_h,
                "target": row.target,
                "selected_model": row.selected_model,
                **metric,
                "test_role": "FINAL_DIAGNOSTIC_ONLY",
            }
        )
    return pd.DataFrame(rows)


def _latency_stress(
    frame: pd.DataFrame, base_features: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for latency_h in LATENCY_STRESS_H:
        examples, _ = _build_examples(frame, latency_h, base_features)
        predictions, models = _add_predictions(examples, [], {})
        metrics = _candidate_metrics(
            predictions, "TEST", [model for model in models if model in BASELINES]
        )
        metrics.insert(1, "latency_h", latency_h)
        rows.extend(metrics.to_dict("records"))
    return pd.DataFrame(rows)


def _rolling_origin_stability(
    examples: pd.DataFrame,
    features: list[str],
    selected_families: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = [
        family for family in MODEL_FAMILIES if family in selected_families
    ]
    if not families:
        families = ["HGB_GLOBAL"]
    for year in (2023, 2024):
        fold_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        fold_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        train = examples.loc[examples["target_at"] < fold_start].copy()
        valid = examples.loc[
            (examples["issue_at"] >= fold_start + pd.Timedelta(hours=PURGE_H))
            & (examples["target_at"] < fold_end)
        ].copy()
        if len(train) < 20_000 or len(valid) < 10_000:
            continue
        bank, _ = _fit_model_bank(
            train, valid, features, families=families, fast=True
        )
        predicted, models = _add_predictions(valid, features, bank)
        predicted["split"] = "ROLLING_VALID"
        metrics = _candidate_metrics(
            predicted, "ROLLING_VALID", [*BASELINES, *bank.keys()]
        )
        metrics.insert(0, "fold_year", year)
        metrics["train_rows"] = len(train)
        metrics["valid_rows"] = len(valid)
        metrics["train_target_max"] = train["target_at"].max()
        metrics["valid_issue_min"] = valid["issue_at"].min()
        rows.extend(metrics.to_dict("records"))
    return pd.DataFrame(rows)


def _conformal_predictions(
    selected: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    valid_all = selected.loc[selected["split"].eq("VALID")]
    test_all = selected.loc[selected["split"].eq("TEST")]
    for choice in selection.itertuples(index=False):
        valid = valid_all.loc[
            valid_all["horizon_h"].eq(choice.horizon_h)
        ].sort_values("issue_at")
        test = test_all.loc[
            test_all["horizon_h"].eq(choice.horizon_h)
        ].sort_values("issue_at")
        actual_valid = valid[f"target_{choice.target}"].to_numpy(
            dtype="float64"
        )
        pred_valid = valid[f"selected_pred_{choice.target}"].to_numpy(
            dtype="float64"
        )
        actual_test = test[f"target_{choice.target}"].to_numpy(dtype="float64")
        pred_test = test[f"selected_pred_{choice.target}"].to_numpy(
            dtype="float64"
        )
        if choice.target == "wave_direction_deg":
            initial = np.abs(_angular_error(actual_valid, pred_valid))
            calibration = deque(
                initial[-CONFORMAL_WINDOW_H:].tolist(),
                maxlen=CONFORMAL_WINDOW_H,
            )
            realized: list[float] = []
            covered: list[bool] = []
            widths: list[float] = []
            for index, record in enumerate(test.itertuples(index=False)):
                if index >= choice.horizon_h:
                    calibration.append(realized[index - choice.horizon_h])
                half_width = float(np.quantile(calibration, 0.80))
                error = abs(
                    float(
                        _angular_error(
                            np.array([actual_test[index]]),
                            np.array([pred_test[index]]),
                        )[0]
                    )
                )
                realized.append(error)
                covered.append(error <= half_width)
                widths.append(2.0 * half_width)
                rows.append(
                    {
                        "issue_at": record.issue_at,
                        "target_at": record.target_at,
                        "horizon_h": choice.horizon_h,
                        "target": choice.target,
                        "model": choice.selected_model,
                        "actual": actual_test[index],
                        "p10": None,
                        "p50": pred_test[index],
                        "p90": None,
                        "circular_half_width_deg": half_width,
                        "covered": error <= half_width,
                    }
                )
        else:
            residual_valid = actual_valid - pred_valid
            calibration = deque(
                residual_valid[-CONFORMAL_WINDOW_H:].tolist(),
                maxlen=CONFORMAL_WINDOW_H,
            )
            realized = []
            covered = []
            widths = []
            for index, record in enumerate(test.itertuples(index=False)):
                if index >= choice.horizon_h:
                    calibration.append(realized[index - choice.horizon_h])
                low_offset = float(np.quantile(calibration, 0.10))
                high_offset = float(np.quantile(calibration, 0.90))
                low = max(0.0, pred_test[index] + low_offset)
                high = max(low, pred_test[index] + high_offset)
                residual = actual_test[index] - pred_test[index]
                realized.append(residual)
                is_covered = low <= actual_test[index] <= high
                covered.append(is_covered)
                widths.append(high - low)
                rows.append(
                    {
                        "issue_at": record.issue_at,
                        "target_at": record.target_at,
                        "horizon_h": choice.horizon_h,
                        "target": choice.target,
                        "model": choice.selected_model,
                        "actual": actual_test[index],
                        "p10": low,
                        "p50": pred_test[index],
                        "p90": high,
                        "circular_half_width_deg": None,
                        "covered": is_covered,
                    }
                )
        reports.append(
            {
                "split": "TEST",
                "horizon_h": choice.horizon_h,
                "target": choice.target,
                "model": choice.selected_model,
                "nominal_coverage_pct": 80.0,
                "empirical_coverage_pct": float(np.mean(covered) * 100.0),
                "mean_interval_width": float(np.mean(widths)),
                "calibration_mode": "VALID_PLUS_MATURED_TEST_ROLLING",
                "calibration_window_h": CONFORMAL_WINDOW_H,
                "test_labels_used_before_maturity": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(reports)


def _extreme_metrics(
    selected: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    train = selected.loc[selected["split"].eq("TRAIN")]
    test = selected.loc[selected["split"].eq("TEST")]
    rows: list[dict[str, Any]] = []
    for choice in selection.itertuples(index=False):
        if choice.target == "wave_direction_deg":
            continue
        train_h = train.loc[train["horizon_h"].eq(choice.horizon_h)]
        test_h = test.loc[test["horizon_h"].eq(choice.horizon_h)]
        for quantile in (0.90, 0.95):
            threshold = float(
                train_h[f"target_{choice.target}"].quantile(quantile)
            )
            actual = test_h[f"target_{choice.target}"].to_numpy(
                dtype="float64"
            )
            predicted = test_h[f"selected_pred_{choice.target}"].to_numpy(
                dtype="float64"
            )
            actual_event = actual >= threshold
            predicted_event = predicted >= threshold
            true_positive = int(np.sum(actual_event & predicted_event))
            rows.append(
                {
                    "horizon_h": choice.horizon_h,
                    "target": choice.target,
                    "model": choice.selected_model,
                    "train_quantile": quantile,
                    "threshold": threshold,
                    "test_extreme_rows": int(np.sum(actual_event)),
                    "extreme_mae": (
                        float(np.mean(np.abs(predicted[actual_event] - actual[actual_event])))
                        if actual_event.any()
                        else None
                    ),
                    "event_recall_pct": (
                        100.0 * true_positive / int(np.sum(actual_event))
                        if actual_event.any()
                        else None
                    ),
                    "event_precision_pct": (
                        100.0 * true_positive / int(np.sum(predicted_event))
                        if predicted_event.any()
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def _split_audit(examples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in examples.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "rows": len(group),
                "issue_min": group["issue_at"].min(),
                "issue_max": group["issue_at"].max(),
                "target_min": group["target_at"].min(),
                "target_max": group["target_at"].max(),
                "horizons": ",".join(
                    str(int(value))
                    for value in sorted(group["horizon_h"].unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def _feature_inventory(
    frame: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in features:
        if column in frame.columns:
            source = "B58A_SHIFTED_TO_SOURCE_AT"
            missing_pct = float(frame[column].isna().mean() * 100.0)
        else:
            source = "KNOWN_CALENDAR_OR_HORIZON"
            missing_pct = 0.0
        rows.append(
            {
                "feature": column,
                "source": source,
                "known_at_issue": True,
                "minimum_effective_lag_h": (
                    OFFICIAL_LATENCY_H
                    if not column.startswith("past_")
                    else OFFICIAL_LATENCY_H + 1
                ),
                "missing_pct": missing_pct,
                "included": True,
            }
        )
    return pd.DataFrame(rows)


def _anti_leakage_audit(
    examples: pd.DataFrame,
    features: list[str],
    selection: pd.DataFrame,
    upstream: dict[str, Any],
) -> pd.DataFrame:
    train = examples.loc[examples["split"].eq("TRAIN")]
    valid = examples.loc[examples["split"].eq("VALID")]
    test = examples.loc[examples["split"].eq("TEST")]
    forbidden = [
        column
        for column in features
        if any(
            token in column.lower()
            for token in (
                "target_wave_",
                "target_direction_",
                "future_",
                "actual_",
                "available_at",
            )
        )
    ]
    checks = [
        (
            "UPSTREAM_B58A_READY",
            upstream.get("status")
            == "READY_FOR_WAVE_ONLY_TEMPORAL_BASELINES",
            "CRITICAL",
            upstream.get("status"),
        ),
        (
            "UPSTREAM_CRITICAL_LEAKAGE_ZERO",
            int(upstream.get("critical_leakage_violations", -1)) == 0,
            "CRITICAL",
            upstream.get("critical_leakage_violations"),
        ),
        (
            "NO_FORBIDDEN_FEATURE_COLUMN",
            not forbidden,
            "CRITICAL",
            forbidden,
        ),
        (
            "SOURCE_PRECEDES_ISSUE_BY_LATENCY",
            bool(
                (
                    examples["source_at"]
                    <= examples["issue_at"]
                    - pd.Timedelta(hours=OFFICIAL_LATENCY_H)
                ).all()
            ),
            "CRITICAL",
            f"latency={OFFICIAL_LATENCY_H}h",
        ),
        (
            "TRAIN_TARGET_BEFORE_VALID_ISSUE",
            bool(train["target_at"].max() < valid["issue_at"].min()),
            "CRITICAL",
            (
                f"train_target_max={train['target_at'].max()}, "
                f"valid_issue_min={valid['issue_at'].min()}"
            ),
        ),
        (
            "VALID_TARGET_BEFORE_TEST_ISSUE",
            bool(valid["target_at"].max() < test["issue_at"].min()),
            "CRITICAL",
            (
                f"valid_target_max={valid['target_at'].max()}, "
                f"test_issue_min={test['issue_at'].min()}"
            ),
        ),
        (
            "SELECTION_USES_VALID_ONLY",
            bool(
                selection["selection_split"].eq("VALID").all()
                and ~selection["test_used_for_selection"].any()
            ),
            "CRITICAL",
            "TEST is final diagnostic only",
        ),
        (
            "AVAILABLE_AT_PRESENT",
            False,
            "WARNING",
            "Historical source has observed_at only; production replay blocked",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "passed": passed,
                "severity": severity,
                "details": json.dumps(
                    _clean_json(details), ensure_ascii=True
                )
                if not isinstance(details, str)
                else details,
                "violations": 0 if passed or severity == "WARNING" else 1,
            }
            for name, passed, severity, details in checks
        ]
    )


def _latest_feature_rows(
    frame: pd.DataFrame, base_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    issue_index = len(frame) - 1
    source_index = issue_index - OFFICIAL_LATENCY_H
    blocks: list[pd.DataFrame] = []
    for horizon_h in HORIZONS_H:
        issue_at = frame.loc[issue_index, "observed_at"]
        target_at = issue_at + pd.Timedelta(hours=horizon_h)
        block = frame.loc[[source_index], base_features].reset_index(drop=True)
        block = block.apply(pd.to_numeric, errors="coerce")
        block["issue_index"] = issue_index
        block["source_index"] = source_index
        block["target_index"] = np.nan
        block["issue_at"] = issue_at
        block["source_at"] = frame.loc[source_index, "observed_at"]
        block["target_at"] = target_at
        block["horizon_h"] = float(horizon_h)
        block["latency_h"] = float(OFFICIAL_LATENCY_H)
        issue_series = pd.Series([issue_at])
        target_series = pd.Series([target_at])
        for calendar in (
            _calendar_features(issue_series, "issue"),
            _calendar_features(target_series, "target"),
        ):
            for column in calendar.columns:
                block[column] = calendar[column].to_numpy()
        reference_specs = {
            "persistence": source_index,
            "seasonal_24h": int(
                _safe_reference_index(
                    np.array([issue_index]),
                    horizon_h,
                    OFFICIAL_LATENCY_H,
                    24,
                )[0]
            ),
            "seasonal_168h": int(
                _safe_reference_index(
                    np.array([issue_index]),
                    horizon_h,
                    OFFICIAL_LATENCY_H,
                    168,
                )[0]
            ),
        }
        for prefix, index in reference_specs.items():
            for target in TARGETS:
                block[f"{prefix}_{target}"] = float(frame.loc[index, target])
        for target in TARGETS:
            block[f"target_{target}"] = np.nan
        block["target_direction_sin"] = np.nan
        block["target_direction_cos"] = np.nan
        block["split"] = "SHADOW"
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    engineered = [
        "horizon_h",
        "latency_h",
        "issue_hour_sin",
        "issue_hour_cos",
        "issue_day_sin",
        "issue_day_cos",
        "issue_weekend",
        "target_hour_sin",
        "target_hour_cos",
        "target_day_sin",
        "target_day_cos",
        "target_weekend",
    ]
    features = [*base_features, *engineered]
    result[features] = result[features].astype("float32")
    return result, features


def _latest_shadow_forecast(
    frame: pd.DataFrame,
    base_features: list[str],
    bank: dict[str, dict[str, Any]],
    selection: pd.DataFrame,
    selected_valid: pd.DataFrame,
) -> pd.DataFrame:
    future, features = _latest_feature_rows(frame, base_features)
    predicted, _ = _add_predictions(future, features, bank)
    selected = _apply_selection(predicted, selection)
    rows: list[dict[str, Any]] = []
    valid = selected_valid.loc[selected_valid["split"].eq("VALID")]
    for choice in selection.itertuples(index=False):
        item = selected.loc[selected["horizon_h"].eq(choice.horizon_h)].iloc[0]
        valid_h = valid.loc[valid["horizon_h"].eq(choice.horizon_h)]
        actual = valid_h[f"target_{choice.target}"].to_numpy(dtype="float64")
        predicted_valid = valid_h[
            f"selected_pred_{choice.target}"
        ].to_numpy(dtype="float64")
        point = float(item[f"selected_pred_{choice.target}"])
        if choice.target == "wave_direction_deg":
            half_width = float(
                np.quantile(
                    np.abs(_angular_error(actual, predicted_valid)), 0.80
                )
            )
            low = None
            high = None
        else:
            residual = actual - predicted_valid
            low = max(0.0, point + float(np.quantile(residual, 0.10)))
            high = max(low, point + float(np.quantile(residual, 0.90)))
            half_width = None
        rows.append(
            {
                "issue_at": item["issue_at"],
                "valid_at": item["target_at"],
                "horizon_h": choice.horizon_h,
                "variable": choice.target,
                "p10": low,
                "p50": point,
                "p90": high,
                "circular_half_width_deg": half_width,
                "model": choice.selected_model,
                "latency_assumption_h": OFFICIAL_LATENCY_H,
                "status": "RESEARCH_ONLY",
            }
        )
    return pd.DataFrame(rows)


def _materialize_shadow(shadow: pd.DataFrame, run_id: str) -> int:
    ddl = """
        CREATE SCHEMA IF NOT EXISTS features;
        CREATE TABLE IF NOT EXISTS features.maritime_wave_forecast_shadow_v1 (
            issue_at timestamptz NOT NULL,
            valid_at timestamptz NOT NULL,
            horizon_h integer NOT NULL,
            variable text NOT NULL,
            p10 double precision,
            p50 double precision NOT NULL,
            p90 double precision,
            circular_half_width_deg double precision,
            model text NOT NULL,
            latency_assumption_h integer NOT NULL,
            status text NOT NULL,
            model_version text NOT NULL,
            run_id uuid NOT NULL,
            materialized_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (issue_at, valid_at, variable, model_version)
        );
    """
    sql = f"""
        INSERT INTO {SHADOW_TABLE} (
            issue_at, valid_at, horizon_h, variable, p10, p50, p90,
            circular_half_width_deg, model, latency_assumption_h, status,
            model_version, run_id
        ) VALUES %s
        ON CONFLICT (issue_at, valid_at, variable, model_version)
        DO UPDATE SET
            p10=EXCLUDED.p10,
            p50=EXCLUDED.p50,
            p90=EXCLUDED.p90,
            circular_half_width_deg=EXCLUDED.circular_half_width_deg,
            model=EXCLUDED.model,
            latency_assumption_h=EXCLUDED.latency_assumption_h,
            status=EXCLUDED.status,
            run_id=EXCLUDED.run_id,
            materialized_at=now()
    """
    values = [
        (
            row.issue_at.to_pydatetime(),
            row.valid_at.to_pydatetime(),
            int(row.horizon_h),
            row.variable,
            None if pd.isna(row.p10) else float(row.p10),
            float(row.p50),
            None if pd.isna(row.p90) else float(row.p90),
            (
                None
                if pd.isna(row.circular_half_width_deg)
                else float(row.circular_half_width_deg)
            ),
            row.model,
            int(row.latency_assumption_h),
            row.status,
            MODEL_VERSION,
            run_id,
        )
        for row in shadow.itertuples(index=False)
    ]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            execute_values(cursor, sql, values, page_size=500)
    return len(values)


def _log_mlflow(
    output_dir: Path,
    decision: dict[str, Any],
    selected_test_metrics: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment("smart-port-wave-forecast")
        with mlflow.start_run(run_name=MODEL_VERSION):
            mlflow.log_params(
                {
                    "model_version": MODEL_VERSION,
                    "horizons_h": ",".join(map(str, HORIZONS_H)),
                    "official_latency_h": OFFICIAL_LATENCY_H,
                    "purge_h": PURGE_H,
                    "selection_split": "VALID",
                    "test_role": "FINAL_DIAGNOSTIC_ONLY",
                    "production_promotion_allowed": False,
                }
            )
            for row in selected_test_metrics.itertuples(index=False):
                target = row.target.replace("wave_", "").replace("_", "-")
                mlflow.log_metric(
                    f"test_mae_{target}_{int(row.horizon_h)}h",
                    float(row.MAE),
                )
            mlflow.set_tag("decision", decision["status"])
            mlflow.log_artifacts(str(output_dir), artifact_path="b58b")
        return "LOGGED"
    except Exception as exc:
        return f"FAILED: {exc}"


def _write_readme(path: Path, decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# B58B Wave Rolling Backtest",
                "",
                f"Decision: **{decision['status']}**",
                "",
                "## Scope",
                "",
                "B58B predicts observed significant wave height, mean wave period "
                "and circular wave direction at 6/12/24/48/72 hours. It does not "
                "predict vessel arrivals and it is not a numerical weather model.",
                "",
                "## Models",
                "",
                "- Persistence and leakage-safe 24 h / 168 h seasonal baselines.",
                "- Global direct HistGradientBoosting model with horizon context.",
                "- Global direct CatBoost model with horizon context.",
                "- Selection uses VALID only; TEST remains final diagnostic only.",
                "- Adaptive asymmetric conformal P10/P50/P90 uses only matured labels.",
                "",
                "## Temporal controls",
                "",
                f"- Official simulated source latency: {OFFICIAL_LATENCY_H} h.",
                f"- Boundary purge: {PURGE_H} h.",
                "- Direction is learned as sine/cosine and scored with angular error.",
                "- Bronze and Core are read-only.",
                "- Historical production replay remains blocked without available_at.",
                "",
                "## Interpretation",
                "",
                "The shadow forecast is research-only. The next improvement is to "
                "ingest archived Copernicus IBI forecasts as they would have been "
                "available, then learn a local residual/bias correction against this "
                "wave series.",
                "",
                "## Next block",
                "",
                decision["next_block"],
            ]
        ),
        encoding="utf-8",
    )


def run_b58b_wave_rolling_backtest(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    materialize_timescale: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    upstream = _load_upstream_decision()
    source = _load_source(client, SOURCE_BUCKET)
    checksum = _source_signature(source, upstream)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "results": previous[1],
        }

    run_id = _start_run(checksum)
    try:
        _update_progress(run_id, "BUILDING_TEMPORAL_EXAMPLES")
        base_features = _feature_columns(source)
        examples, features = _build_examples(
            source, OFFICIAL_LATENCY_H, base_features
        )
        active = examples.loc[
            examples["split"].isin(["TRAIN", "VALID", "TEST"])
        ].copy()
        train = active.loc[active["split"].eq("TRAIN")].copy()
        valid = active.loc[active["split"].eq("VALID")].copy()
        if min(len(train), len(valid)) <= 0:
            raise RuntimeError("Temporal split produced an empty partition")

        _update_progress(
            run_id,
            "FITTING_CANDIDATES",
            train_rows=len(train),
            valid_rows=len(valid),
        )
        bank, model_inventory = _fit_model_bank(
            train, valid, features, families=MODEL_FAMILIES, fast=False
        )
        if "HGB_GLOBAL" not in bank:
            raise RuntimeError("Required HGB_GLOBAL candidate failed")

        predicted, models = _add_predictions(active, features, bank)
        valid_metrics = _candidate_metrics(predicted, "VALID", models)
        test_candidate_metrics = _candidate_metrics(predicted, "TEST", models)
        selection = _select_models(valid_metrics)
        selected = _apply_selection(predicted, selection)
        selected_test = _selected_test_metrics(selected, selection)

        _update_progress(
            run_id,
            "ROLLING_ORIGIN_AND_CALIBRATION",
            selected_models=selection["selected_model"].value_counts().to_dict(),
        )
        selected_families = set(selection["selected_model"]).intersection(
            MODEL_FAMILIES
        )
        rolling = _rolling_origin_stability(
            examples, features, selected_families
        )
        latency = _latency_stress(source, base_features)
        conformal, calibration = _conformal_predictions(selected, selection)
        extremes = _extreme_metrics(selected, selection)
        leakage = _anti_leakage_audit(
            examples, features, selection, upstream
        )
        critical_violations = int(
            leakage.loc[
                leakage["severity"].eq("CRITICAL") & ~leakage["passed"],
                "violations",
            ].sum()
        )

        shadow = _latest_shadow_forecast(
            source, base_features, bank, selection, selected
        )
        materialized_rows = (
            _materialize_shadow(shadow, run_id)
            if materialize_timescale and critical_violations == 0
            else 0
        )

        advanced_selected = selection["selected_model"].isin(
            MODEL_FAMILIES
        )
        positive_uplift = (
            pd.to_numeric(
                selection["validation_uplift_pct"], errors="coerce"
            ).fillna(0)
            > 0
        )
        improved_targets = int((advanced_selected & positive_uplift).sum())
        integrity_passed = critical_violations == 0
        if not integrity_passed:
            status = "NEED_WAVE_MODEL_REPAIR"
            next_block = "B58B_TEMPORAL_OR_FEATURE_REPAIR"
        elif improved_targets == 0:
            status = "KEEP_PERSISTENCE_AS_WAVE_BASELINE"
            next_block = "B58C_COPERNICUS_IBI_ARCHIVE_AND_HYBRID_CORRECTION"
        else:
            status = "READY_FOR_IBI_HYBRID_ENRICHMENT"
            next_block = "B58C_COPERNICUS_IBI_ARCHIVE_AND_HYBRID_CORRECTION"

        decision = {
            "status": status,
            "decision": status,
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "source_rows": len(source),
            "participating_rows": len(active),
            "train_rows": int(active["split"].eq("TRAIN").sum()),
            "valid_rows": int(active["split"].eq("VALID").sum()),
            "test_rows": int(active["split"].eq("TEST").sum()),
            "purged_rows": int(examples["split"].eq("EXCLUDED_PURGE").sum()),
            "horizons_h": list(HORIZONS_H),
            "official_latency_h": OFFICIAL_LATENCY_H,
            "latency_stress_h": list(LATENCY_STRESS_H),
            "selected_models": {
                f"{int(row.horizon_h)}h:{row.target}": row.selected_model
                for row in selection.itertuples(index=False)
            },
            "advanced_model_selections_with_positive_uplift": improved_targets,
            "integrity_passed": integrity_passed,
            "critical_leakage_violations": critical_violations,
            "training_executed": True,
            "split_created": True,
            "selection_split": "VALID",
            "test_role": "FINAL_DIAGNOSTIC_ONLY",
            "selection_used_test": False,
            "bronze_modified": False,
            "core_modified": False,
            "available_at_present": False,
            "historical_replay_allowed": False,
            "production_promotion_allowed": False,
            "shadow_serving_rows": materialized_rows,
            "timescale_table": (
                SHADOW_TABLE if materialized_rows else None
            ),
            "scope": (
                "observed-wave multihorizon forecasting; not vessel-flow "
                "forecasting and not numerical weather prediction"
            ),
            "next_block": next_block,
        }

        reports = {
            "01_upstream_and_data_contract.csv": pd.DataFrame(
                [
                    {
                        "source_key": SOURCE_KEY,
                        "source_rows": len(source),
                        "first_observed_at": source["observed_at"].min(),
                        "last_observed_at": source["observed_at"].max(),
                        "upstream_decision": upstream.get("status"),
                        "available_at_present": False,
                        "hourly_continuity": True,
                    }
                ]
            ),
            "02_temporal_split_audit.csv": _split_audit(examples),
            "03_feature_inventory.csv": _feature_inventory(
                source, features
            ),
            "04_latency_baseline_stress.csv": latency,
            "05_validation_candidate_metrics.csv": valid_metrics,
            "06_test_candidate_metrics.csv": test_candidate_metrics,
            "07_selected_models.csv": selection.merge(
                selected_test[
                    [
                        "horizon_h",
                        "target",
                        "MAE",
                        "RMSE",
                        "BIAS",
                        "R2",
                    ]
                ].rename(
                    columns={
                        "MAE": "test_mae",
                        "RMSE": "test_rmse",
                        "BIAS": "test_bias",
                        "R2": "test_r2",
                    }
                ),
                on=["horizon_h", "target"],
                how="left",
            ),
            "08_rolling_origin_stability.csv": rolling,
            "09_probabilistic_calibration.csv": calibration,
            "10_extreme_state_metrics.csv": extremes,
            "11_anti_leakage_audit.csv": leakage,
            "12_latest_shadow_forecast.csv": shadow,
            "13_model_inventory.csv": model_inventory,
        }

        with tempfile.TemporaryDirectory(prefix="b58b-") as temporary:
            output_dir = Path(temporary)
            for name, report in reports.items():
                report.to_csv(output_dir / name, index=False)
            selection.to_json(
                output_dir / "selected_models.json",
                orient="records",
                indent=2,
                date_format="iso",
            )
            with (output_dir / "wave_model_bank.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "model_version": MODEL_VERSION,
                        "features": features,
                        "bank": bank,
                        "selection": selection,
                        "latency_h": OFFICIAL_LATENCY_H,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            selected_export = selected[
                [
                    "issue_at",
                    "target_at",
                    "horizon_h",
                    "split",
                    *[f"target_{target}" for target in TARGETS],
                    *[f"selected_pred_{target}" for target in TARGETS],
                    *[f"selected_model_{target}" for target in TARGETS],
                ]
            ]
            selected_export.to_parquet(
                output_dir / "selected_point_predictions.parquet",
                index=False,
            )
            conformal.to_parquet(
                output_dir / "test_probabilistic_predictions.parquet",
                index=False,
            )
            decision_path = output_dir / "14_b58b_decision.json"
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            readme_path = output_dir / "README_B58B.md"
            _write_readme(readme_path, decision)
            mlflow_status = _log_mlflow(
                output_dir, decision, selected_test
            )
            decision["mlflow_status"] = mlflow_status
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            uploaded: dict[str, str] = {}
            for path in sorted(output_dir.iterdir()):
                if path.name == "wave_model_bank.pkl":
                    key = f"models/b58b/{output_prefix}/{path.name}"
                elif path.suffix == ".parquet":
                    key = f"predictions/b58b/{output_prefix}/{path.name}"
                elif path == decision_path or path.name == "selected_models.json":
                    key = f"configs/b58b/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b58b/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload_file(
                    client, path, output_bucket, key
                )

        metadata = {
            **decision,
            "checksum": checksum,
            "features": len(features),
            "candidate_models": models,
            "outputs": uploaded,
            "output_prefix": (
                f"s3://{output_bucket}/reports/b58b/{output_prefix}/"
            ),
        }
        _finish_run(run_id, "SUCCESS", len(active), metadata)
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
            {"model_version": MODEL_VERSION},
            error_message=str(exc),
        )
        raise
