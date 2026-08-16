from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import mlflow
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)


TRAINING_VERSION = "b56b-arrival-flow-temporal-v1.1"
SOURCE_TABLE = "features.port_hourly_state_v1"
SOURCE_FEATURE_VERSION = "b56a-port-hourly-state-v1"
SOURCE_AUDIT_VERSION = "b56a-operational-feasibility-v1.1"
SOURCE_NAME = "b56b_arrival_flow_temporal_baselines"
DATASET_NAME = "port_arrival_flow_6h_12h_24h"
PURGE_HOURS = 24
HORIZONS = (6, 12, 24)
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 1000
COMPLETENESS_RATIO_THRESHOLD = 0.70
COMPLETENESS_SUSTAINED_MONTHS = 3

TARGET_BY_HORIZON = {
    6: "target_arrivals_next_6h",
    12: "target_arrivals_next_12h",
    24: "target_arrivals_next_24h",
}

CORE_FEATURES = (
    "arrivals_prev_1h",
    "arrivals_last_6h",
    "arrivals_last_24h",
    "arrivals_last_168h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "weekend_flag",
)
WAVE_FEATURES = (
    "wave_height_lag_1h_m",
    "wave_period_lag_1h_s",
    "wave_direction_sin",
    "wave_direction_cos",
    "weather_available_flag",
)
MODEL_FEATURES = {
    "HGB_POISSON_CORE": CORE_FEATURES,
    "HGB_POISSON_CORE_WAVE": (*CORE_FEATURES, *WAVE_FEATURES),
}
BASELINE_NAMES = (
    "ZERO",
    "TRAIN_MEAN",
    "RECENT_24H_RATE",
    "SEASONAL_NAIVE_24H",
    "SEASONAL_NAIVE_168H",
)


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
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


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


def _query_frame(query: str, params: tuple | None = None) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _load_upstream_decision() -> dict[str, Any]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, metadata
                FROM audit.ingestion_run
                WHERE source_name='b56a_operational_feasibility'
                  AND dataset_name='port_hourly_state_feasibility'
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("B56A decision is missing")
    status, metadata = row
    metadata = dict(metadata or {})
    if status != "SUCCESS":
        raise RuntimeError(f"Latest B56A status is {status}")
    if metadata.get("audit_version") != SOURCE_AUDIT_VERSION:
        raise RuntimeError("Latest B56A decision is not v1.1")
    if metadata.get("readiness", {}).get("arrival_flow") is not True:
        raise RuntimeError("B56A did not approve arrival-flow forecasting")
    if int(metadata.get("critical_leakage_violations", -1)) != 0:
        raise RuntimeError("B56A leakage gate did not pass")
    return metadata


def _load_source() -> pd.DataFrame:
    columns = [
        "as_of_time",
        "arrivals_prev_1h",
        "arrivals_last_6h",
        "arrivals_last_24h",
        "arrivals_last_168h",
        "wave_height_lag_1h_m",
        "wave_period_lag_1h_s",
        "wave_direction_lag_1h_deg",
        "weather_available_flag",
        "hour_of_day",
        "day_of_week",
        "month",
        "weekend_flag",
        *TARGET_BY_HORIZON.values(),
    ]
    frame = _query_frame(
        f"""
        SELECT {', '.join(columns)}
        FROM {SOURCE_TABLE}
        WHERE feature_version=%s
        ORDER BY as_of_time
        """,
        (SOURCE_FEATURE_VERSION,),
    )
    if frame.empty:
        raise RuntimeError("B56A hourly source table is empty")
    frame["as_of_time"] = pd.to_datetime(frame["as_of_time"], utc=True)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour_of_day"] / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour_of_day"] / 24.0)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12.0)
    radians = np.deg2rad(frame["wave_direction_lag_1h_deg"])
    frame["wave_direction_sin"] = np.sin(radians)
    frame["wave_direction_cos"] = np.cos(radians)
    return frame


def _detect_source_completeness(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    monthly = (
        frame.assign(
            month=(
                frame["as_of_time"]
                .dt.tz_convert(None)
                .dt.to_period("M")
                .dt.to_timestamp()
                .dt.tz_localize("UTC")
            )
        )
        .groupby("month", observed=True)
        .agg(
            observed_hours=("as_of_time", "size"),
            observed_arrivals=("arrivals_prev_1h", "sum"),
        )
        .reset_index()
        .sort_values("month")
    )
    next_month = monthly["month"] + pd.offsets.MonthBegin(1)
    monthly["expected_hours"] = (
        (next_month - monthly["month"]).dt.total_seconds() / 3600
    ).astype("int64")
    monthly["hour_grid_coverage_pct"] = (
        100 * monthly["observed_hours"] / monthly["expected_hours"]
    )
    monthly["prior_12m_median_arrivals"] = (
        monthly["observed_arrivals"]
        .shift(1)
        .rolling(12, min_periods=6)
        .median()
    )
    monthly["arrival_ratio_to_prior_median"] = monthly[
        "observed_arrivals"
    ].div(monthly["prior_12m_median_arrivals"].replace(0, np.nan))
    low = monthly["arrival_ratio_to_prior_median"].lt(
        COMPLETENESS_RATIO_THRESHOLD
    )
    sustained = low.copy()
    for offset in range(1, COMPLETENESS_SUSTAINED_MONTHS):
        sustained &= low.shift(-offset, fill_value=False)
    monthly["sustained_low_coverage_break"] = sustained
    candidates = monthly.loc[sustained, "month"]
    break_start = None if candidates.empty else pd.Timestamp(candidates.iloc[0])
    monthly["included_before_break"] = (
        True if break_start is None else monthly["month"] < break_start
    )
    return monthly, break_start


def _target_stability(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for horizon, target in TARGET_BY_HORIZON.items():
        valid = frame.loc[frame["split"] == "VALID", target]
        test = frame.loc[frame["split"] == "TEST", target]
        valid_mean = float(valid.mean())
        test_mean = float(test.mean())
        ratio = test_mean / max(valid_mean, 1e-12)
        rows.append(
            {
                "horizon_h": horizon,
                "target": target,
                "valid_mean": valid_mean,
                "test_mean": test_mean,
                "test_valid_mean_ratio": ratio,
                "stable_70_to_150_pct": 0.70 <= ratio <= 1.50,
            }
        )
    return pd.DataFrame(rows)


def _checksum(frame: pd.DataFrame) -> str:
    columns = [
        "as_of_time",
        *CORE_FEATURES,
        *WAVE_FEATURES,
        *TARGET_BY_HORIZON.values(),
    ]
    digest = hashlib.sha256(TRAINING_VERSION.encode("ascii"))
    hashed = pd.util.hash_pandas_object(frame[columns], index=False)
    digest.update(hashed.to_numpy(dtype="uint64").tobytes())
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
        "training_version": TRAINING_VERSION,
        "source_feature_version": SOURCE_FEATURE_VERSION,
        "split_policy": "TEMPORAL_70_15_15_PURGED_24H",
        "selection_policy": "SELECT_VALID_TEST_FINAL_ONLY",
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
                    f"postgresql://maritime/{SOURCE_TABLE}",
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


def _temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = frame.dropna(subset=list(TARGET_BY_HORIZON.values())).copy()
    if len(eligible) < 24 * 365 * 3:
        raise RuntimeError("Fewer than three complete years are available")
    valid_position = int(len(eligible) * 0.70)
    test_position = int(len(eligible) * 0.85)
    valid_start = eligible.iloc[valid_position]["as_of_time"]
    test_start = eligible.iloc[test_position]["as_of_time"]
    purge = pd.Timedelta(hours=PURGE_HOURS)

    split = pd.Series("PURGED", index=eligible.index, dtype="string")
    split.loc[eligible["as_of_time"] < valid_start - purge] = "TRAIN"
    split.loc[
        (eligible["as_of_time"] >= valid_start)
        & (eligible["as_of_time"] < test_start - purge)
    ] = "VALID"
    split.loc[eligible["as_of_time"] >= test_start] = "TEST"
    eligible["split"] = split

    audit_rows = []
    for name in ("TRAIN", "VALID", "TEST", "PURGED"):
        part = eligible[eligible["split"] == name]
        audit_rows.append(
            {
                "split": name,
                "rows": len(part),
                "first_time": part["as_of_time"].min(),
                "last_time": part["as_of_time"].max(),
                "purge_hours": PURGE_HOURS,
            }
        )
    audit = pd.DataFrame(audit_rows)

    train_max = eligible.loc[eligible["split"] == "TRAIN", "as_of_time"].max()
    valid_min = eligible.loc[eligible["split"] == "VALID", "as_of_time"].min()
    valid_max = eligible.loc[eligible["split"] == "VALID", "as_of_time"].max()
    test_min = eligible.loc[eligible["split"] == "TEST", "as_of_time"].min()
    if valid_min - train_max <= purge or test_min - valid_max <= purge:
        raise RuntimeError("Temporal purge is not strictly greater than 24 hours")
    return eligible, audit


def _feature_contract() -> pd.DataFrame:
    rows = []
    for feature in CORE_FEATURES:
        rows.append(
            {
                "feature": feature,
                "family": "ARRIVAL_HISTORY" if feature.startswith("arrivals_") else "CALENDAR",
                "available_at_as_of": True,
                "target_derived": False,
                "official_model": True,
            }
        )
    for feature in WAVE_FEATURES:
        rows.append(
            {
                "feature": feature,
                "family": "PAST_WAVE",
                "available_at_as_of": True,
                "target_derived": False,
                "official_model": True,
            }
        )
    for horizon, target in TARGET_BY_HORIZON.items():
        rows.append(
            {
                "feature": target,
                "family": f"FUTURE_{horizon}H_TARGET",
                "available_at_as_of": False,
                "target_derived": True,
                "official_model": False,
            }
        )
    rows.extend(
        [
            {
                "feature": "SEASONAL_NAIVE_24H",
                "family": "SAFE_PAST_TARGET_BASELINE",
                "available_at_as_of": True,
                "target_derived": True,
                "official_model": False,
            },
            {
                "feature": "SEASONAL_NAIVE_168H",
                "family": "SAFE_PAST_TARGET_BASELINE",
                "available_at_as_of": True,
                "target_derived": True,
                "official_model": False,
            },
        ]
    )
    return pd.DataFrame(rows)


def _build_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype="float64"), 0.0, None)
    y_true = np.asarray(y_true, dtype="float64")
    mae = mean_absolute_error(y_true, prediction)
    rmse = math.sqrt(mean_squared_error(y_true, prediction))
    denominator = np.abs(y_true).sum()
    smape_denominator = np.abs(y_true) + np.abs(prediction)
    smape = np.mean(
        np.divide(
            2 * np.abs(prediction - y_true),
            smape_denominator,
            out=np.zeros_like(y_true),
            where=smape_denominator > 0,
        )
    )
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2_score(y_true, prediction),
        "BIAS": float(np.mean(prediction - y_true)),
        "WAPE_PCT": 100 * np.abs(prediction - y_true).sum() / max(denominator, 1e-12),
        "SMAPE_PCT": 100 * smape,
        "POISSON_DEVIANCE": mean_poisson_deviance(
            y_true, np.clip(prediction, 1e-6, None)
        ),
    }


def _baseline_predictions(
    frame: pd.DataFrame,
    horizon: int,
    train_mean: float,
) -> dict[str, np.ndarray]:
    target = TARGET_BY_HORIZON[horizon]
    return {
        "ZERO": np.zeros(len(frame), dtype="float64"),
        "TRAIN_MEAN": np.full(len(frame), train_mean, dtype="float64"),
        "RECENT_24H_RATE": (
            frame["arrivals_last_24h"].to_numpy(dtype="float64") * horizon / 24.0
        ),
        "SEASONAL_NAIVE_24H": frame[target].shift(24).to_numpy(dtype="float64"),
        "SEASONAL_NAIVE_168H": frame[target].shift(168).to_numpy(dtype="float64"),
    }


def _evaluate_part(
    frame: pd.DataFrame,
    split_name: str,
    horizon: int,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = TARGET_BY_HORIZON[horizon]
    mask = frame["split"].eq(split_name).to_numpy()
    y_all = frame[target].to_numpy(dtype="float64")
    rows = []
    prediction_frame = pd.DataFrame(
        {
            "as_of_time": frame.loc[mask, "as_of_time"].to_numpy(),
            "split": split_name,
            "horizon_h": horizon,
            "target": target,
            "actual": y_all[mask],
        }
    )
    for model_name, values in predictions.items():
        values = np.asarray(values, dtype="float64")
        valid = mask & np.isfinite(values) & np.isfinite(y_all)
        metrics = _metrics(y_all[valid], values[valid])
        rows.append(
            {
                "split": split_name,
                "horizon_h": horizon,
                "model": model_name,
                "n": int(valid.sum()),
                **metrics,
            }
        )
        prediction_frame[model_name] = np.clip(values[mask], 0.0, None)
    return pd.DataFrame(rows), prediction_frame


def _bootstrap_wave_ablation(
    prediction_frame: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    frame = prediction_frame.dropna(
        subset=["actual", "HGB_POISSON_CORE", "HGB_POISSON_CORE_WAVE"]
    ).copy()
    frame["day"] = pd.to_datetime(frame["as_of_time"], utc=True).dt.floor("D")
    frame["delta_abs"] = (
        (frame["actual"] - frame["HGB_POISSON_CORE_WAVE"]).abs()
        - (frame["actual"] - frame["HGB_POISSON_CORE"]).abs()
    )
    daily = frame.groupby("day", observed=True)["delta_abs"].agg(["sum", "count"])
    rng = np.random.default_rng(RANDOM_SEED + horizon)
    draws = np.empty(BOOTSTRAP_ITERATIONS, dtype="float64")
    positions = np.arange(len(daily))
    for iteration in range(BOOTSTRAP_ITERATIONS):
        selected = rng.choice(positions, size=len(positions), replace=True)
        sample = daily.iloc[selected]
        draws[iteration] = sample["sum"].sum() / sample["count"].sum()
    core_mae = float((frame["actual"] - frame["HGB_POISSON_CORE"]).abs().mean())
    wave_mae = float(
        (frame["actual"] - frame["HGB_POISSON_CORE_WAVE"]).abs().mean()
    )
    gain = 100 * (core_mae - wave_mae) / max(core_mae, 1e-12)
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    keep = bool(gain >= 2.0 and ci_high < 0)
    return {
        "horizon_h": horizon,
        "n": len(frame),
        "days": len(daily),
        "core_mae": core_mae,
        "core_wave_mae": wave_mae,
        "wave_gain_pct": gain,
        "delta_mae_ci95_low": ci_low,
        "delta_mae_ci95_high": ci_high,
        "decision": "KEEP_WAVE" if keep else "WAVE_UPLIFT_NOT_PROVEN",
    }


def _high_pressure_metrics(
    train: pd.DataFrame,
    test_predictions: pd.DataFrame,
    selected_model: str,
    horizon: int,
) -> dict[str, Any]:
    target = TARGET_BY_HORIZON[horizon]
    threshold = float(train[target].quantile(0.80))
    high = test_predictions[test_predictions["actual"] >= threshold]
    metrics = _metrics(
        high["actual"].to_numpy(), high[selected_model].to_numpy()
    )
    return {
        "horizon_h": horizon,
        "selected_model": selected_model,
        "train_p80_threshold": threshold,
        "test_high_pressure_rows": len(high),
        **metrics,
    }


def _upload(client, path: Path, bucket: str, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/octet-stream",
        ".pkl": "application/octet-stream",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(
        str(path), bucket, key, ExtraArgs={"ContentType": content_type}
    )
    return f"s3://{bucket}/{key}"


def _log_mlflow(
    output_dir: Path,
    decision: dict[str, Any],
    metrics_valid: pd.DataFrame,
    metrics_test: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("smart-port-arrival-flow")
        with mlflow.start_run(run_name=TRAINING_VERSION):
            mlflow.log_params(
                {
                    "training_version": TRAINING_VERSION,
                    "split_policy": "TEMPORAL_70_15_15_PURGED_24H",
                    "source_feature_version": SOURCE_FEATURE_VERSION,
                    "horizons": "6,12,24",
                }
            )
            for horizon in HORIZONS:
                selected = decision["selected_models"][str(horizon)]
                row = metrics_test[
                    (metrics_test["horizon_h"] == horizon)
                    & (metrics_test["model"] == selected)
                ].iloc[0]
                mlflow.log_metric(f"test_mae_{horizon}h", float(row["MAE"]))
                mlflow.log_metric(f"test_rmse_{horizon}h", float(row["RMSE"]))
                mlflow.log_metric(f"test_wape_{horizon}h", float(row["WAPE_PCT"]))
            mlflow.log_artifacts(str(output_dir), artifact_path="b56b")
        return "LOGGED"
    except Exception as exc:
        return f"ERROR: {exc}"


def run_b56b_arrival_flow_baselines(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    upstream = _load_upstream_decision()
    full_frame = _load_source()
    checksum = _checksum(full_frame)
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
        completeness_audit, break_start = _detect_source_completeness(full_frame)
        if break_start is None:
            model_frame = full_frame.copy()
            safe_label_end = full_frame["as_of_time"].max() + pd.Timedelta(hours=1)
        else:
            # The 24h target at t consumes t..t+23, so stop 23h before the break.
            safe_label_end = break_start - pd.Timedelta(hours=23)
            model_frame = full_frame[full_frame["as_of_time"] < safe_label_end].copy()
        excluded_incomplete_rows = len(full_frame) - len(model_frame)
        if len(model_frame) < 24 * 365 * 3:
            raise RuntimeError("Completeness cutoff leaves fewer than three years")

        frame = model_frame
        frame, split_audit = _temporal_split(frame)
        train = frame[frame["split"] == "TRAIN"]
        valid = frame[frame["split"] == "VALID"]
        test = frame[frame["split"] == "TEST"]
        if min(len(train), len(valid), len(test)) < 1000:
            raise RuntimeError("At least one temporal split is too small")

        all_metrics_valid = []
        all_metrics_test = []
        all_valid_predictions = []
        all_test_predictions = []
        models: dict[str, Any] = {}
        selected_models: dict[str, str] = {}
        validation_uplift: dict[str, float] = {}

        for horizon in HORIZONS:
            target = TARGET_BY_HORIZON[horizon]
            predictions = _baseline_predictions(
                frame, horizon, float(train[target].mean())
            )
            for model_name, features in MODEL_FEATURES.items():
                model = _build_model()
                model.fit(train[list(features)], train[target])
                predictions[model_name] = np.clip(
                    model.predict(frame[list(features)]), 0.0, None
                )
                models[f"{model_name}_{horizon}H"] = {
                    "model": model,
                    "features": list(features),
                    "target": target,
                }

            valid_metrics, valid_predictions = _evaluate_part(
                frame, "VALID", horizon, predictions
            )
            test_metrics, test_predictions = _evaluate_part(
                frame, "TEST", horizon, predictions
            )
            best_row = valid_metrics.sort_values(["MAE", "RMSE"]).iloc[0]
            selected = str(best_row["model"])
            best_baseline = valid_metrics[
                valid_metrics["model"].isin(BASELINE_NAMES)
            ].sort_values(["MAE", "RMSE"]).iloc[0]
            uplift = 100 * (best_baseline["MAE"] - best_row["MAE"]) / max(
                float(best_baseline["MAE"]), 1e-12
            )
            selected_models[str(horizon)] = selected
            validation_uplift[str(horizon)] = float(uplift)
            valid_predictions["selected_model"] = selected
            test_predictions["selected_model"] = selected

            all_metrics_valid.append(valid_metrics)
            all_metrics_test.append(test_metrics)
            all_valid_predictions.append(valid_predictions)
            all_test_predictions.append(test_predictions)

        metrics_valid = pd.concat(all_metrics_valid, ignore_index=True)
        metrics_test = pd.concat(all_metrics_test, ignore_index=True)
        valid_predictions = pd.concat(all_valid_predictions, ignore_index=True)
        test_predictions = pd.concat(all_test_predictions, ignore_index=True)

        ablation = pd.DataFrame(
            [
                _bootstrap_wave_ablation(
                    test_predictions[test_predictions["horizon_h"] == horizon],
                    horizon,
                )
                for horizon in HORIZONS
            ]
        )
        high_pressure = pd.DataFrame(
            [
                _high_pressure_metrics(
                    train,
                    test_predictions[test_predictions["horizon_h"] == horizon],
                    selected_models[str(horizon)],
                    horizon,
                )
                for horizon in HORIZONS
            ]
        )

        residual_rows = []
        for horizon in HORIZONS:
            part = test_predictions[test_predictions["horizon_h"] == horizon].copy()
            selected = selected_models[str(horizon)]
            part["year"] = pd.to_datetime(part["as_of_time"], utc=True).dt.year
            part["absolute_error"] = (part["actual"] - part[selected]).abs()
            part["signed_error"] = part[selected] - part["actual"]
            for year, year_part in part.groupby("year", observed=True):
                residual_rows.append(
                    {
                        "horizon_h": horizon,
                        "selected_model": selected,
                        "year": year,
                        "n": len(year_part),
                        "MAE": year_part["absolute_error"].mean(),
                        "BIAS": year_part["signed_error"].mean(),
                    }
                )
        residual_stability = pd.DataFrame(residual_rows)

        target_stability = _target_stability(frame)
        target_stability_passed = bool(
            target_stability["stable_70_to_150_pct"].all()
        )
        test_uplift = {}
        robust_test_horizons = 0
        for horizon in HORIZONS:
            part = metrics_test[metrics_test["horizon_h"] == horizon]
            selected = selected_models[str(horizon)]
            selected_mae = float(part.loc[part["model"] == selected, "MAE"].iloc[0])
            baseline_mae = float(
                part[part["model"].isin(BASELINE_NAMES)]["MAE"].min()
            )
            uplift = 100 * (baseline_mae - selected_mae) / max(
                baseline_mae, 1e-12
            )
            test_uplift[str(horizon)] = uplift
            robust_test_horizons += uplift >= -2.0

        meaningful_uplift_horizons = sum(
            value >= 2.0 for value in validation_uplift.values()
        )
        if not target_stability_passed:
            decision_status = "NEED_DATA_REPAIR"
        elif meaningful_uplift_horizons >= 2 and robust_test_horizons >= 2:
            decision_status = "READY_FOR_ARRIVAL_FLOW_MVP"
        else:
            decision_status = "BASELINES_ONLY_NO_ML_UPLIFT"
        decision = {
            "status": decision_status,
            "training_version": TRAINING_VERSION,
            "source_rows": len(frame),
            "train_rows": len(train),
            "valid_rows": len(valid),
            "test_rows": len(test),
            "purged_rows": int((frame["split"] == "PURGED").sum()),
            "selected_models": selected_models,
            "validation_uplift_vs_best_baseline_pct": validation_uplift,
            "test_selected_uplift_vs_best_baseline_pct": test_uplift,
            "meaningful_uplift_horizons": meaningful_uplift_horizons,
            "robust_test_horizons": robust_test_horizons,
            "wave_decisions": {
                str(row.horizon_h): row.decision for row in ablation.itertuples()
            },
            "upstream_decision": upstream.get("status"),
            "source_completeness_break_start": break_start,
            "safe_label_end_exclusive": safe_label_end,
            "full_source_rows": len(full_frame),
            "excluded_incomplete_period_rows": excluded_incomplete_rows,
            "target_stability_passed": target_stability_passed,
            "temporal_leakage_violations": 0,
            "selection_used_test": False,
            "official_protocol": "TEMPORAL_70_15_15_PURGED_24H",
            "objective": "PORT_ARRIVAL_COUNTS_NEXT_6H_12H_24H",
            "not_supported": [
                "individual_vessel_eta",
                "port_occupancy_forecast",
                "full_weather_forecast",
            ],
            "next_block": (
                "B56C_PROBABILISTIC_ARRIVAL_FLOW"
                if decision_status == "READY_FOR_ARRIVAL_FLOW_MVP"
                else (
                    "B56B_SOURCE_COMPLETENESS_REPAIR"
                    if decision_status == "NEED_DATA_REPAIR"
                    else "B56B_FEATURE_ENRICHMENT"
                )
            ),
        }

        split_assignments = frame[["as_of_time", "split"]].copy()
        feature_contract = _feature_contract()

        with tempfile.TemporaryDirectory(prefix="b56b-") as temporary:
            output_dir = Path(temporary)
            completeness_audit.to_csv(
                output_dir / "00_source_completeness_by_month.csv", index=False
            )
            split_audit.to_csv(output_dir / "01_temporal_split_audit.csv", index=False)
            feature_contract.to_csv(output_dir / "02_feature_contract.csv", index=False)
            metrics_valid.to_csv(output_dir / "03_metrics_valid.csv", index=False)
            metrics_test.to_csv(output_dir / "04_metrics_test.csv", index=False)
            ablation.to_csv(output_dir / "05_weather_ablation_bootstrap.csv", index=False)
            high_pressure.to_csv(output_dir / "06_high_pressure_metrics.csv", index=False)
            residual_stability.to_csv(output_dir / "07_residual_stability_by_year.csv", index=False)
            target_stability.to_csv(
                output_dir / "09_target_stability_valid_test.csv", index=False
            )
            decision_path = output_dir / "08_b56b_decision.json"
            decision_path.write_text(
                json.dumps(_clean_json(decision), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            split_assignments.to_parquet(
                output_dir / "temporal_split_assignments.parquet", index=False
            )
            valid_predictions.to_parquet(
                output_dir / "valid_predictions.parquet", index=False
            )
            test_predictions.to_parquet(
                output_dir / "test_predictions.parquet", index=False
            )
            for name, payload in models.items():
                with (output_dir / f"{name.lower()}.pkl").open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

            readme = output_dir / "README_B56B.md"
            readme.write_text(
                "\n".join(
                    [
                        "# B56B Arrival Flow Temporal Baselines",
                        "",
                        f"Decision: **{decision_status}**",
                        "",
                        "Strict temporal 70/15/15 evaluation with a 24-hour purge.",
                        "Model selection used validation only; test was final evaluation.",
                        "Targets are arrival counts over the next 6, 12 and 24 hours.",
                        "This block does not predict individual vessel ETA.",
                    ]
                ),
                encoding="utf-8",
            )

            mlflow_status = _log_mlflow(
                output_dir, decision, metrics_valid, metrics_test
            )
            client = _s3_client()
            uploaded = {}
            for path in sorted(output_dir.iterdir()):
                if path.name == "temporal_split_assignments.parquet":
                    key = f"datasets/b56b/{output_prefix}/{path.name}"
                elif path.name in {"valid_predictions.parquet", "test_predictions.parquet"}:
                    key = f"predictions/b56b/{output_prefix}/{path.name}"
                elif path.suffix == ".pkl":
                    key = f"models/b56b/{output_prefix}/{path.name}"
                elif path == decision_path:
                    key = f"configs/b56b/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b56b/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload(client, path, output_bucket, key)

        metadata = {
            **decision,
            "checksum": checksum,
            "mlflow_status": mlflow_status,
            "outputs": uploaded,
            "output_prefix": f"s3://{output_bucket}/reports/b56b/{output_prefix}/",
        }
        _finish_run(run_id, "SUCCESS", len(frame), metadata)
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
            {"training_version": TRAINING_VERSION},
            error_message=str(exc),
        )
        raise
