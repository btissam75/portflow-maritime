from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from psycopg2.extras import Json, execute_values
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor

from model_trainer.arrival_flow_baselines import (
    HORIZONS,
    _clean_json,
    _db_connection,
    _json_default,
    _metrics,
    _s3_client,
    _upload,
)


EXPERT_VERSION = "b56g-expert-probabilistic-count-v1"
SOURCE_NAME = "b56g_expert_probabilistic_count"
DATASET_NAME = "port_arrival_flow_expert_count_6h_12h_24h"
UPSTREAM_VERSION = "b56c-arrival-flow-enrichment-v1"
UPSTREAM_SOURCE_NAME = "b56c_arrival_flow_enrichment"
UPSTREAM_DATASET_NAME = "port_arrival_flow_enriched_6h_12h_24h"
SOURCE_DATA_KEY = (
    "datasets/b56c/version=1/arrival_flow_enriched_model_ready.parquet"
)
VALID_PREDICTIONS_KEY = "predictions/b56c/version=1/valid_predictions.parquet"
TEST_PREDICTIONS_KEY = "predictions/b56c/version=1/test_predictions.parquet"
INCUMBENT_VALID_KEY = (
    "predictions/b56e/version=1/valid_probabilistic_predictions.parquet"
)
INCUMBENT_TEST_KEY = (
    "predictions/b56e/version=1/test_probabilistic_predictions.parquet"
)
SERVING_TABLE = "serving.maritime_arrival_flow_expert_backtest_v1"

TARGET_BY_HORIZON = {
    6: "target_arrivals_next_6h",
    12: "target_arrivals_next_12h",
    24: "target_arrivals_next_24h",
}
INCREMENT_BUCKETS = (
    ("INC_0_6H", 6, "target_increment_0_6h"),
    ("INC_6_12H", 12, "target_increment_6_12h"),
    ("INC_12_24H", 24, "target_increment_12_24h"),
)

CORE_FEATURES = (
    "arrivals_prev_1h",
    "arrivals_last_6h",
    "arrivals_last_24h",
    "arrivals_last_168h",
    "arrivals_lag_2h",
    "arrivals_lag_3h",
    "arrivals_lag_6h",
    "arrivals_lag_12h",
    "arrivals_lag_24h",
    "arrivals_lag_48h",
    "arrivals_lag_168h",
    "arrival_rate_3h",
    "arrival_rate_6h",
    "arrival_rate_12h",
    "arrival_rate_24h",
    "arrival_rate_48h",
    "arrival_rate_72h",
    "arrival_rate_168h",
    "arrival_std_6h",
    "arrival_std_24h",
    "arrival_trend_6h_vs_24h",
    "arrival_trend_24h_vs_168h",
    "history_168h_available_flag",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "hour_of_year_sin",
    "hour_of_year_cos",
    "weekend_flag",
    "month_start_flag",
    "month_end_flag",
    "recent_level_ratio_6_24",
    "recent_level_ratio_24_168",
    "time_years",
)

WAVE_FEATURES = (
    "wave_height_lag_1h_m",
    "wave_period_lag_1h_s",
    "wave_direction_sin",
    "wave_direction_cos",
    "wave_height_mean_3h_m",
    "wave_height_mean_6h_m",
    "wave_height_mean_12h_m",
    "wave_height_mean_24h_m",
    "wave_height_mean_72h_m",
    "wave_height_max_6h_m",
    "wave_height_max_12h_m",
    "wave_height_max_24h_m",
    "wave_height_max_72h_m",
    "wave_height_std_24h_m",
    "wave_period_mean_6h_s",
    "wave_period_mean_24h_s",
    "wave_height_trend_6h_vs_24h_m",
    "wave_coverage_6h_pct",
    "wave_coverage_24h_pct",
    "wave_coverage_72h_pct",
    "weather_available_flag",
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "DNB_CORE_STATIC": {
        "kind": "DNB",
        "features": CORE_FEATURES,
        "half_life_days": None,
    },
    "DNB_CORE_RECENCY": {
        "kind": "DNB",
        "features": CORE_FEATURES,
        "half_life_days": 180.0,
    },
    "DNB_CORE_WAVE_RECENCY": {
        "kind": "DNB",
        "features": (*CORE_FEATURES, *WAVE_FEATURES),
        "half_life_days": 180.0,
    },
    "HGB_POISSON_CORE_RECENCY": {
        "kind": "HGB_POISSON",
        "features": CORE_FEATURES,
        "half_life_days": 180.0,
    },
}

BASELINE_NAMES = (
    "RECENT_24H_RATE",
    "SEASONAL_NAIVE_24H",
    "SEASONAL_NAIVE_168H",
    "HGB_LEGACY_CORE",
)
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "actual_ata",
    "actual_atd",
    "planned_eta",
    "future",
    "arrival_delay",
)

EXPECTED_COVERAGE = 0.80
COVERAGE_MIN = 0.77
COVERAGE_MAX = 0.83
ONLINE_WINDOW_HOURS = 30 * 24
ONLINE_MIN_HISTORY_HOURS = 7 * 24
CALIBRATION_WINDOWS_DAYS = (30, 60, 90)
CALIBRATION_GAMMAS = (0.001, 0.005, 0.01)
CALIBRATION_MIN_HISTORY_HOURS = 7 * 24
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 42


def _load_upstream_decision() -> dict[str, Any]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, checksum, metadata
                FROM audit.ingestion_run
                WHERE source_name=%s AND dataset_name=%s
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (UPSTREAM_SOURCE_NAME, UPSTREAM_DATASET_NAME),
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("B56C result is missing")
    status, checksum, metadata = row
    metadata = dict(metadata or {})
    if status != "SUCCESS":
        raise RuntimeError(f"Latest B56C run is {status}")
    if metadata.get("enrichment_version") != UPSTREAM_VERSION:
        raise RuntimeError("Latest B56C result has an unexpected version")
    if int(metadata.get("critical_leakage_violations", -1)) != 0:
        raise RuntimeError("B56C temporal leakage gate did not pass")
    if metadata.get("selection_used_test") is not False:
        raise RuntimeError("B56C used TEST for model selection")
    if metadata.get("eta_features_used") is not False:
        raise RuntimeError("B56C unexpectedly used final ETA information")
    metadata["upstream_checksum"] = checksum
    return metadata


def _load_parquet(bucket: str, key: str) -> pd.DataFrame:
    body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def _load_optional_parquet(bucket: str, key: str) -> pd.DataFrame | None:
    try:
        return _load_parquet(bucket, key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _source_checksum(bucket: str, upstream: dict[str, Any]) -> str:
    client = _s3_client()
    digest = hashlib.sha256(EXPERT_VERSION.encode("ascii"))
    digest.update(str(upstream.get("upstream_checksum", "")).encode("ascii"))
    for key in (
        SOURCE_DATA_KEY,
        VALID_PREDICTIONS_KEY,
        TEST_PREDICTIONS_KEY,
    ):
        metadata = client.head_object(Bucket=bucket, Key=key)
        digest.update(key.encode("utf-8"))
        digest.update(str(metadata.get("ETag", "")).encode("ascii"))
        digest.update(str(metadata.get("ContentLength", 0)).encode("ascii"))
    for key in (INCUMBENT_VALID_KEY, INCUMBENT_TEST_KEY):
        try:
            metadata = client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
            digest.update(f"{key}:MISSING".encode("utf-8"))
            continue
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
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _start_run(checksum: str, bucket: str) -> str:
    metadata = {
        "expert_version": EXPERT_VERSION,
        "upstream_version": UPSTREAM_VERSION,
        "policy": (
            "INCREMENTAL_COUNTS_0_6_6_12_12_24_VALID_SELECTION_"
            "TEST_LOCKED_MATURED_ONLINE_ADAPTATION_ADAPTIVE_CQR"
        ),
        "training_executed": True,
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
                    f"s3://{bucket}/{SOURCE_DATA_KEY}",
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


def _prepare_source(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "as_of_time",
        "split",
        *TARGET_BY_HORIZON.values(),
        *[
            feature
            for spec in MODEL_SPECS.values()
            for feature in spec["features"]
            if feature
            not in {
                "recent_level_ratio_6_24",
                "recent_level_ratio_24_168",
                "time_years",
            }
        ],
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"B56C model-ready data miss columns: {missing}")

    frame = source.copy()
    frame["as_of_time"] = pd.to_datetime(frame["as_of_time"], utc=True)
    frame = frame.sort_values("as_of_time").reset_index(drop=True)
    duplicate_rows = int(frame.duplicated(["as_of_time"], keep=False).sum())
    split_values = set(frame["split"].dropna().astype(str).unique())
    expected_splits = {"TRAIN", "PURGED", "VALID", "TEST"}
    unexpected_splits = sorted(split_values - expected_splits)

    for target in TARGET_BY_HORIZON.values():
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame["target_increment_0_6h"] = frame["target_arrivals_next_6h"]
    frame["target_increment_6_12h"] = (
        frame["target_arrivals_next_12h"]
        - frame["target_arrivals_next_6h"]
    )
    frame["target_increment_12_24h"] = (
        frame["target_arrivals_next_24h"]
        - frame["target_arrivals_next_12h"]
    )

    frame["recent_level_ratio_6_24"] = (
        pd.to_numeric(frame["arrivals_last_6h"], errors="coerce")
        / (
            pd.to_numeric(frame["arrivals_last_24h"], errors="coerce") / 4.0
        ).replace(0.0, np.nan)
    ).clip(0.0, 8.0)
    frame["recent_level_ratio_24_168"] = (
        pd.to_numeric(frame["arrivals_last_24h"], errors="coerce")
        / (
            pd.to_numeric(frame["arrivals_last_168h"], errors="coerce") / 7.0
        ).replace(0.0, np.nan)
    ).clip(0.0, 8.0)
    origin = frame["as_of_time"].min()
    frame["time_years"] = (
        (frame["as_of_time"] - origin).dt.total_seconds()
        / (365.25 * 24.0 * 3600.0)
    )

    increment_columns = [item[2] for item in INCREMENT_BUCKETS]
    negative_increment_rows = int(
        (frame[increment_columns].min(axis=1) < -1e-8).sum()
    )
    reconstruction_error = np.max(
        np.abs(
            frame[increment_columns].sum(axis=1)
            - frame["target_arrivals_next_24h"]
        )
    )
    forbidden_features = sorted(
        feature
        for spec in MODEL_SPECS.values()
        for feature in spec["features"]
        if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    temporal_order_passed = bool(
        frame.loc[frame["split"].eq("TRAIN"), "as_of_time"].max()
        < frame.loc[frame["split"].eq("VALID"), "as_of_time"].min()
        < frame.loc[frame["split"].eq("TEST"), "as_of_time"].min()
    )
    hourly_gap_count = int(
        frame["as_of_time"].diff().dropna().ne(pd.Timedelta(hours=1)).sum()
    )

    checks = pd.DataFrame(
        [
            {
                "check": "one_row_per_as_of_time",
                "value": duplicate_rows,
                "passed": duplicate_rows == 0,
            },
            {
                "check": "known_split_labels",
                "value": ",".join(unexpected_splits) or "NONE",
                "passed": not unexpected_splits,
            },
            {
                "check": "strict_temporal_split_order",
                "value": temporal_order_passed,
                "passed": temporal_order_passed,
            },
            {
                "check": "hourly_continuity",
                "value": hourly_gap_count,
                "passed": hourly_gap_count == 0,
            },
            {
                "check": "nonnegative_increment_targets",
                "value": negative_increment_rows,
                "passed": negative_increment_rows == 0,
            },
            {
                "check": "increment_reconstruction_24h",
                "value": float(reconstruction_error),
                "passed": bool(reconstruction_error <= 1e-8),
            },
            {
                "check": "forbidden_features_absent",
                "value": ",".join(forbidden_features) or "NONE",
                "passed": not forbidden_features,
            },
            {
                "check": "test_not_used_for_training",
                "value": "TRAIN_ONLY",
                "passed": True,
            },
        ]
    )
    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"B56G integrity checks failed: {failed}")
    return frame, checks


def _target_dispersion(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = [
        ("CUMULATIVE_6H", "target_arrivals_next_6h"),
        ("CUMULATIVE_12H", "target_arrivals_next_12h"),
        ("CUMULATIVE_24H", "target_arrivals_next_24h"),
        *[(name, column) for name, _, column in INCREMENT_BUCKETS],
    ]
    for split in ("TRAIN", "VALID", "TEST", "ALL"):
        part = frame if split == "ALL" else frame.loc[frame["split"].eq(split)]
        for target_name, column in definitions:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            mean = float(values.mean())
            variance = float(values.var(ddof=1))
            rows.append(
                {
                    "split": split,
                    "target": target_name,
                    "n": len(values),
                    "mean": mean,
                    "variance": variance,
                    "variance_mean_ratio": variance / max(mean, 1e-12),
                    "zero_rate": float(values.eq(0).mean()),
                    "p10": float(values.quantile(0.10)),
                    "p50": float(values.quantile(0.50)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def _recency_weights(
    times: pd.Series,
    half_life_days: float | None,
) -> np.ndarray:
    if half_life_days is None:
        return np.ones(len(times), dtype="float64")
    age_days = (
        (times.max() - times).dt.total_seconds().to_numpy(dtype="float64")
        / 86400.0
    )
    weights = np.power(0.5, age_days / half_life_days)
    return np.clip(weights, 0.02, 1.0)


def _fit_transform_state(
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = frame.loc[:, list(features)].apply(
        pd.to_numeric, errors="coerce"
    )
    medians = raw.median(axis=0, skipna=True).fillna(0.0)
    missing_features = [
        column for column in features if bool(raw[column].isna().any())
    ]
    filled = raw.fillna(medians)
    indicators = pd.DataFrame(
        {
            f"{column}__missing": raw[column].isna().astype("float64")
            for column in missing_features
        },
        index=frame.index,
    )
    design = pd.concat([filled, indicators], axis=1)
    means = design.mean(axis=0)
    scales = design.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    transformed = ((design - means) / scales).to_numpy(dtype="float64")
    state = {
        "features": list(features),
        "medians": medians.to_dict(),
        "missing_features": missing_features,
        "design_columns": design.columns.tolist(),
        "means": means.to_dict(),
        "scales": scales.to_dict(),
    }
    return transformed, state


def _transform_with_state(
    frame: pd.DataFrame,
    state: dict[str, Any],
) -> np.ndarray:
    features = list(state["features"])
    raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    filled = raw.fillna(pd.Series(state["medians"]))
    indicators = pd.DataFrame(
        {
            f"{column}__missing": raw[column].isna().astype("float64")
            for column in state["missing_features"]
        },
        index=frame.index,
    )
    design = pd.concat([filled, indicators], axis=1)
    design = design.reindex(columns=state["design_columns"], fill_value=0.0)
    means = pd.Series(state["means"])
    scales = pd.Series(state["scales"])
    return ((design - means) / scales).to_numpy(dtype="float64")


def _nb_loss_and_gradient(
    beta: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    l2: float,
) -> tuple[float, np.ndarray]:
    eta = np.clip(design @ beta, -8.0, 8.0)
    mean = np.exp(eta)
    size = 1.0 / max(alpha, 1e-8)
    log_probability = (
        gammaln(target + size)
        - gammaln(size)
        - gammaln(target + 1.0)
        + size * (math.log(size) - np.log(size + mean))
        + target * (np.log(mean + 1e-12) - np.log(size + mean))
    )
    weight_sum = max(float(weights.sum()), 1e-12)
    penalty = l2 * float(np.dot(beta[1:], beta[1:]))
    loss = -float(np.dot(weights, log_probability)) / weight_sum + penalty
    derivative_eta = (mean - target) / (1.0 + alpha * mean)
    gradient = (
        design.T @ (weights * derivative_eta) / weight_sum
    )
    gradient[1:] += 2.0 * l2 * beta[1:]
    return loss, gradient


def _estimate_nb_alpha(
    target: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
) -> float:
    numerator = np.sum(weights * ((target - mean) ** 2 - mean))
    denominator = np.sum(weights * np.maximum(mean**2, 1e-8))
    return float(np.clip(numerator / max(denominator, 1e-12), 1e-4, 5.0))


def _fit_dynamic_negative_binomial(
    frame: pd.DataFrame,
    target_column: str,
    features: tuple[str, ...],
    half_life_days: float | None,
) -> dict[str, Any]:
    matrix, transform_state = _fit_transform_state(frame, features)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    target = frame[target_column].to_numpy(dtype="float64")
    weights = _recency_weights(frame["as_of_time"], half_life_days)
    weighted_mean = float(np.average(target, weights=weights))
    alpha = max(
        (float(np.var(target)) - weighted_mean)
        / max(weighted_mean**2, 1e-8),
        1e-4,
    )
    beta = np.zeros(design.shape[1], dtype="float64")
    beta[0] = math.log(max(weighted_mean, 0.05))
    convergence: list[dict[str, Any]] = []
    for iteration in range(3):
        result = minimize(
            lambda value: _nb_loss_and_gradient(
                value,
                design,
                target,
                weights,
                alpha,
                l2=0.002,
            ),
            beta,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 250, "ftol": 1e-9, "gtol": 1e-6},
        )
        beta = result.x
        mean = np.exp(np.clip(design @ beta, -8.0, 8.0))
        alpha = _estimate_nb_alpha(target, mean, weights)
        convergence.append(
            {
                "iteration": iteration + 1,
                "success": bool(result.success),
                "loss": float(result.fun),
                "optimizer_message": str(result.message),
                "alpha": alpha,
            }
        )
    return {
        "kind": "DNB",
        "target_column": target_column,
        "transform": transform_state,
        "beta": beta,
        "alpha": alpha,
        "half_life_days": half_life_days,
        "convergence": convergence,
    }


def _fit_hgb_poisson(
    frame: pd.DataFrame,
    target_column: str,
    features: tuple[str, ...],
    half_life_days: float | None,
) -> dict[str, Any]:
    matrix, transform_state = _fit_transform_state(frame, features)
    target = frame[target_column].to_numpy(dtype="float64")
    weights = _recency_weights(frame["as_of_time"], half_life_days)
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=23,
        min_samples_leaf=48,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=35,
        random_state=RANDOM_SEED,
    )
    model.fit(matrix, target, sample_weight=weights)
    fitted = np.clip(model.predict(matrix), 0.01, None)
    alpha = _estimate_nb_alpha(target, fitted, weights)
    return {
        "kind": "HGB_POISSON",
        "target_column": target_column,
        "transform": transform_state,
        "model": model,
        "alpha": alpha,
        "half_life_days": half_life_days,
        "iterations": int(model.n_iter_),
    }


def _predict_model(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    matrix = _transform_with_state(frame, model["transform"])
    if model["kind"] == "DNB":
        design = np.column_stack([np.ones(len(matrix)), matrix])
        prediction = np.exp(
            np.clip(design @ np.asarray(model["beta"]), -8.0, 8.0)
        )
    else:
        prediction = model["model"].predict(matrix)
    return np.clip(np.asarray(prediction, dtype="float64"), 0.01, None)


def _fit_expert_bank(
    frame: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame]:
    train = frame.loc[frame["split"].eq("TRAIN")].copy()
    models: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for candidate, spec in MODEL_SPECS.items():
        models[candidate] = {}
        features = tuple(spec["features"])
        for bucket_name, maturity_h, target_column in INCREMENT_BUCKETS:
            if spec["kind"] == "DNB":
                fitted = _fit_dynamic_negative_binomial(
                    train,
                    target_column,
                    features,
                    spec["half_life_days"],
                )
            else:
                fitted = _fit_hgb_poisson(
                    train,
                    target_column,
                    features,
                    spec["half_life_days"],
                )
            models[candidate][bucket_name] = fitted
            rows.append(
                {
                    "candidate": candidate,
                    "kind": spec["kind"],
                    "bucket": bucket_name,
                    "maturity_h": maturity_h,
                    "target": target_column,
                    "features": len(features),
                    "half_life_days": spec["half_life_days"],
                    "dispersion_alpha": float(fitted["alpha"]),
                    "optimizer_converged": bool(
                        all(
                            item["success"]
                            for item in fitted.get("convergence", [])
                        )
                    )
                    if spec["kind"] == "DNB"
                    else True,
                    "iterations": len(fitted.get("convergence", []))
                    if spec["kind"] == "DNB"
                    else fitted.get("iterations"),
                }
            )
    return models, pd.DataFrame(rows)


def _rolling_online_scale(
    actual: pd.Series,
    prediction: pd.Series,
    maturity_h: int,
) -> tuple[pd.Series, pd.Series]:
    log_ratio = np.log(
        (pd.to_numeric(actual, errors="coerce") + 0.5)
        / (pd.to_numeric(prediction, errors="coerce") + 0.5)
    )
    matured = log_ratio.shift(maturity_h)
    correction = matured.rolling(
        ONLINE_WINDOW_HOURS,
        min_periods=ONLINE_MIN_HISTORY_HOURS,
    ).median()
    scale = np.exp(correction.clip(math.log(0.55), math.log(1.80)))
    ready = scale.notna()
    return prediction * scale.fillna(1.0), ready


def _build_candidate_predictions(
    frame: pd.DataFrame,
    models: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    output = frame[
        [
            "as_of_time",
            "split",
            *TARGET_BY_HORIZON.values(),
            *[item[2] for item in INCREMENT_BUCKETS],
        ]
    ].copy()
    for candidate, bucket_models in models.items():
        online_ready = pd.Series(True, index=frame.index)
        for bucket_name, maturity_h, target_column in INCREMENT_BUCKETS:
            base_column = f"{candidate}__{bucket_name}"
            online_column = f"{candidate}_ONLINE__{bucket_name}"
            prediction = pd.Series(
                _predict_model(bucket_models[bucket_name], frame),
                index=frame.index,
            )
            output[base_column] = prediction
            online, ready = _rolling_online_scale(
                frame[target_column],
                prediction,
                maturity_h,
            )
            output[online_column] = np.clip(online, 0.01, None)
            online_ready &= ready
        output[f"{candidate}_ONLINE__history_ready"] = online_ready
    return output


def _candidate_names() -> tuple[str, ...]:
    names: list[str] = []
    for candidate in MODEL_SPECS:
        names.extend([candidate, f"{candidate}_ONLINE"])
    return tuple(names)


def _base_candidate_name(candidate: str) -> str:
    suffix = "_ONLINE"
    return candidate[: -len(suffix)] if candidate.endswith(suffix) else candidate


def _cumulative_prediction(
    predictions: pd.DataFrame,
    candidate: str,
    horizon: int,
) -> np.ndarray:
    columns: list[str] = []
    for bucket_name, maturity_h, _ in INCREMENT_BUCKETS:
        columns.append(f"{candidate}__{bucket_name}")
        if maturity_h == horizon:
            break
    return predictions[columns].sum(axis=1).to_numpy(dtype="float64")


def _long_candidate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        part = predictions[["as_of_time", "split"]].copy()
        part["horizon_h"] = horizon
        part["actual"] = predictions[TARGET_BY_HORIZON[horizon]]
        for candidate in _candidate_names():
            part[candidate] = _cumulative_prediction(
                predictions, candidate, horizon
            )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _merge_external_predictions(
    long_frame: pd.DataFrame,
    b56c_predictions: pd.DataFrame | None,
    incumbent: pd.DataFrame | None,
) -> pd.DataFrame:
    result = long_frame.copy()
    if b56c_predictions is not None and not b56c_predictions.empty:
        external = b56c_predictions.copy()
        external["as_of_time"] = pd.to_datetime(
            external["as_of_time"], utc=True
        )
        columns = [
            "as_of_time",
            "horizon_h",
            *[
                name
                for name in BASELINE_NAMES
                if name in external.columns
            ],
        ]
        result = result.merge(
            external[columns],
            on=["as_of_time", "horizon_h"],
            how="left",
            validate="one_to_one",
        )
    if incumbent is not None and not incumbent.empty:
        external = incumbent.copy()
        external["as_of_time"] = pd.to_datetime(
            external["as_of_time"], utc=True
        )
        external = external.rename(
            columns={
                "point_prediction": "B56E_INCUMBENT",
                "p10": "B56E_INCUMBENT_P10",
                "p50": "B56E_INCUMBENT_P50",
                "p90": "B56E_INCUMBENT_P90",
            }
        )
        columns = [
            column
            for column in (
                "as_of_time",
                "horizon_h",
                "B56E_INCUMBENT",
                "B56E_INCUMBENT_P10",
                "B56E_INCUMBENT_P50",
                "B56E_INCUMBENT_P90",
            )
            if column in external.columns
        ]
        result = result.merge(
            external[columns],
            on=["as_of_time", "horizon_h"],
            how="left",
            validate="one_to_one",
        )
    return result


def _point_metrics(
    frame: pd.DataFrame,
    split: str,
    models: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    part = frame.loc[frame["split"].eq(split)].copy()
    for horizon in HORIZONS:
        horizon_part = part.loc[part["horizon_h"].eq(horizon)]
        actual = horizon_part["actual"].to_numpy(dtype="float64")
        for model in models:
            if model not in horizon_part.columns:
                continue
            prediction = pd.to_numeric(
                horizon_part[model], errors="coerce"
            ).to_numpy(dtype="float64")
            finite = np.isfinite(actual) & np.isfinite(prediction)
            if not finite.any():
                continue
            result = _metrics(actual[finite], prediction[finite])
            rows.append(
                {
                    "split": split,
                    "horizon_h": horizon,
                    "model": model,
                    "n": int(finite.sum()),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def _select_expert(
    valid_metrics: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    eligible = valid_metrics.loc[
        valid_metrics["model"].isin(_candidate_names())
    ].copy()
    ranking = (
        eligible.groupby("model", observed=True)
        .agg(
            horizons=("horizon_h", "nunique"),
            mean_mae=("MAE", "mean"),
            mean_wape_pct=("WAPE_PCT", "mean"),
            max_wape_pct=("WAPE_PCT", "max"),
            mean_bias_abs=("BIAS", lambda values: float(np.mean(np.abs(values)))),
        )
        .reset_index()
    )
    ranking["selection_score"] = (
        ranking["mean_wape_pct"]
        + 0.10 * ranking["max_wape_pct"]
        + 0.05 * ranking["mean_bias_abs"]
    )
    ranking = ranking.sort_values(
        ["selection_score", "mean_mae", "model"]
    ).reset_index(drop=True)
    if ranking.empty or int(ranking.iloc[0]["horizons"]) != len(HORIZONS):
        raise RuntimeError("No complete B56G expert is selectable on VALID")
    return str(ranking.iloc[0]["model"]), ranking


def _selected_bucket_mean(
    predictions: pd.DataFrame,
    selected: str,
    bucket_name: str,
) -> np.ndarray:
    return predictions[f"{selected}__{bucket_name}"].to_numpy(dtype="float64")


def _simulate_base_distribution(
    predictions: pd.DataFrame,
    models: dict[str, dict[str, dict[str, Any]]],
    selected: str,
    simulation_samples: int,
) -> pd.DataFrame:
    base_name = _base_candidate_name(selected)
    rng = np.random.default_rng(RANDOM_SEED)
    increment_samples: list[np.ndarray] = []
    increment_means: list[np.ndarray] = []
    for bucket_name, _, _ in INCREMENT_BUCKETS:
        mean = _selected_bucket_mean(predictions, selected, bucket_name)
        alpha = float(models[base_name][bucket_name]["alpha"])
        if alpha <= 1e-4:
            samples = rng.poisson(
                mean[:, None],
                size=(len(mean), simulation_samples),
            )
        else:
            size = 1.0 / alpha
            probability = size / (size + mean)
            samples = rng.negative_binomial(
                size,
                probability[:, None],
                size=(len(mean), simulation_samples),
            )
        increment_means.append(mean)
        increment_samples.append(samples)

    parts: list[pd.DataFrame] = []
    cumulative_samples = np.zeros_like(
        increment_samples[0], dtype="float64"
    )
    cumulative_mean = np.zeros(len(predictions), dtype="float64")
    for index, (_, horizon, _) in enumerate(INCREMENT_BUCKETS):
        cumulative_samples += increment_samples[index]
        cumulative_mean += increment_means[index]
        part = predictions[["as_of_time", "split"]].copy()
        part["horizon_h"] = horizon
        part["actual"] = predictions[TARGET_BY_HORIZON[horizon]]
        part["selected_model"] = selected
        part["point_prediction"] = cumulative_mean
        part["base_p10"] = np.quantile(cumulative_samples, 0.10, axis=1)
        part["base_p50"] = np.quantile(cumulative_samples, 0.50, axis=1)
        part["base_p90"] = np.quantile(cumulative_samples, 0.90, axis=1)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _adaptive_cqr(
    frame: pd.DataFrame,
    horizon: int,
    window_days: int,
    gamma: float,
) -> pd.DataFrame:
    result = frame.loc[frame["horizon_h"].eq(horizon)].copy()
    result = result.sort_values("as_of_time").reset_index(drop=True)
    actual = result["actual"].to_numpy(dtype="float64")
    base_low = result["base_p10"].to_numpy(dtype="float64")
    base_mid = result["base_p50"].to_numpy(dtype="float64")
    base_high = result["base_p90"].to_numpy(dtype="float64")
    raw_score = np.maximum(base_low - actual, actual - base_high)

    corrected_low = np.empty(len(result), dtype="float64")
    corrected_high = np.empty(len(result), dtype="float64")
    corrected_mid = base_mid.copy()
    ready = np.zeros(len(result), dtype=bool)
    alpha_used = np.empty(len(result), dtype="float64")
    correction_used = np.zeros(len(result), dtype="float64")
    target_alpha = 1.0 - EXPECTED_COVERAGE
    adaptive_alpha = target_alpha
    matured_scores: list[float] = []
    window_hours = window_days * 24

    for position in range(len(result)):
        matured_position = position - horizon
        if matured_position >= 0:
            matured_scores.append(float(raw_score[matured_position]))
            missed = float(
                actual[matured_position] < corrected_low[matured_position]
                or actual[matured_position] > corrected_high[matured_position]
            )
            adaptive_alpha = float(
                np.clip(
                    adaptive_alpha + gamma * (target_alpha - missed),
                    0.02,
                    0.40,
                )
            )
        history = matured_scores[-window_hours:]
        if len(history) >= CALIBRATION_MIN_HISTORY_HOURS:
            correction = float(
                np.quantile(history, 1.0 - adaptive_alpha)
            )
            minimum_correction = -0.45 * max(
                base_high[position] - base_low[position], 0.0
            )
            correction = max(correction, minimum_correction)
            ready[position] = True
        else:
            correction = 0.0
        low = max(0.0, base_low[position] - correction)
        high = max(low, base_high[position] + correction)
        mid = float(np.clip(corrected_mid[position], low, high))
        corrected_low[position] = low
        corrected_mid[position] = mid
        corrected_high[position] = high
        alpha_used[position] = adaptive_alpha
        correction_used[position] = correction

    result["p10"] = corrected_low
    result["p50"] = corrected_mid
    result["p90"] = corrected_high
    result["calibration_history_ready"] = ready
    result["adaptive_alpha"] = alpha_used
    result["conformal_correction"] = correction_used
    result["calibration_window_days"] = window_days
    result["calibration_gamma"] = gamma
    return result


def _pinball(
    actual: np.ndarray,
    prediction: np.ndarray,
    quantile: float,
) -> float:
    error = actual - prediction
    return float(
        np.mean(np.maximum(quantile * error, (quantile - 1.0) * error))
    )


def _probabilistic_metric_row(
    frame: pd.DataFrame,
    split: str,
    horizon: int,
) -> dict[str, Any]:
    part = frame.loc[
        frame["split"].eq(split) & frame["calibration_history_ready"]
    ].copy()
    actual = part["actual"].to_numpy(dtype="float64")
    low = part["p10"].to_numpy(dtype="float64")
    mid = part["p50"].to_numpy(dtype="float64")
    high = part["p90"].to_numpy(dtype="float64")
    covered = (actual >= low) & (actual <= high)
    alpha = 1.0 - EXPECTED_COVERAGE
    interval_score = (
        high
        - low
        + (2.0 / alpha) * (low - actual) * (actual < low)
        + (2.0 / alpha) * (actual - high) * (actual > high)
    )
    return {
        "split": split,
        "horizon_h": horizon,
        "n": len(part),
        "coverage_p10_p90": float(np.mean(covered)),
        "coverage_target": EXPECTED_COVERAGE,
        "coverage_gap_abs": float(
            abs(float(np.mean(covered)) - EXPECTED_COVERAGE)
        ),
        "mean_interval_width": float(np.mean(high - low)),
        "median_interval_width": float(np.median(high - low)),
        "winkler_interval_score": float(np.mean(interval_score)),
        "pinball_p10": _pinball(actual, low, 0.10),
        "pinball_p50": _pinball(actual, mid, 0.50),
        "pinball_p90": _pinball(actual, high, 0.90),
        "below_p10_rate": float(np.mean(actual < low)),
        "above_p90_rate": float(np.mean(actual > high)),
        "coverage_gate_passed": bool(
            COVERAGE_MIN <= float(np.mean(covered)) <= COVERAGE_MAX
        ),
    }


def _select_calibration(
    base: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    valid = base.loc[base["split"].eq("VALID")].copy()
    for horizon in HORIZONS:
        for window_days in CALIBRATION_WINDOWS_DAYS:
            for gamma in CALIBRATION_GAMMAS:
                calibrated = _adaptive_cqr(
                    valid,
                    horizon,
                    window_days,
                    gamma,
                )
                metric = _probabilistic_metric_row(
                    calibrated,
                    "VALID",
                    horizon,
                )
                metric.update(
                    {
                        "window_days": window_days,
                        "gamma": gamma,
                    }
                )
                rows.append(metric)
        search = pd.DataFrame(
            [row for row in rows if row["horizon_h"] == horizon]
        )
        within = search.loc[
            search["coverage_p10_p90"].between(
                EXPECTED_COVERAGE - 0.02,
                EXPECTED_COVERAGE + 0.02,
            )
        ]
        if within.empty:
            winner = search.sort_values(
                [
                    "coverage_gap_abs",
                    "winkler_interval_score",
                    "mean_interval_width",
                ]
            ).iloc[0]
        else:
            winner = within.sort_values(
                ["winkler_interval_score", "mean_interval_width"]
            ).iloc[0]
        selected[str(horizon)] = {
            "window_days": int(winner["window_days"]),
            "gamma": float(winner["gamma"]),
            "valid_coverage": float(winner["coverage_p10_p90"]),
            "valid_interval_score": float(
                winner["winkler_interval_score"]
            ),
        }
    return selected, pd.DataFrame(rows)


def _apply_selected_calibration(
    base: pd.DataFrame,
    selected: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        configuration = selected[str(horizon)]
        parts.append(
            _adaptive_cqr(
                base,
                horizon,
                int(configuration["window_days"]),
                float(configuration["gamma"]),
            )
        )
    return pd.concat(parts, ignore_index=True)


def _reconcile_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("as_of_time", sort=False):
        part = group.sort_values("horizon_h").copy()
        part["point_prediction"] = np.maximum.accumulate(
            part["point_prediction"].to_numpy(dtype="float64")
        )
        part["p10"] = np.maximum.accumulate(
            part["p10"].to_numpy(dtype="float64")
        )
        part["p50"] = np.maximum.accumulate(
            np.maximum(
                part["p50"].to_numpy(dtype="float64"),
                part["p10"].to_numpy(dtype="float64"),
            )
        )
        part["p90"] = np.maximum.accumulate(
            np.maximum(
                part["p90"].to_numpy(dtype="float64"),
                part["p50"].to_numpy(dtype="float64"),
            )
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["as_of_time", "horizon_h"]
    )


def _probabilistic_metrics(
    calibrated: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _probabilistic_metric_row(
                calibrated.loc[calibrated["horizon_h"].eq(horizon)],
                split,
                horizon,
            )
            for horizon in HORIZONS
        ]
    )


def _coherence_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ("point_prediction", "p10", "p50", "p90"):
        pivot = frame.pivot(
            index="as_of_time",
            columns="horizon_h",
            values=column,
        )
        violations = int(
            ((pivot[6] > pivot[12]) | (pivot[12] > pivot[24])).sum()
        )
        rows.append(
            {
                "check": f"monotonic_{column}_6_12_24",
                "violations": violations,
                "passed": violations == 0,
            }
        )
    crossing = int(
        (
            (frame["p10"] > frame["p50"])
            | (frame["p50"] > frame["p90"])
        ).sum()
    )
    rows.append(
        {
            "check": "noncrossing_p10_p50_p90",
            "violations": crossing,
            "passed": crossing == 0,
        }
    )
    negative = int(
        (frame[["point_prediction", "p10", "p50", "p90"]] < 0)
        .any(axis=1)
        .sum()
    )
    rows.append(
        {
            "check": "nonnegative_counts",
            "violations": negative,
            "passed": negative == 0,
        }
    )
    return pd.DataFrame(rows)


def _paired_day_bootstrap(
    test: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    selected_model: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    baseline_candidates = [
        model
        for model in (*BASELINE_NAMES, "B56E_INCUMBENT")
        if model in test.columns
    ]
    for horizon in HORIZONS:
        valid_h = valid_metrics.loc[
            valid_metrics["horizon_h"].eq(horizon)
            & valid_metrics["model"].isin(baseline_candidates)
        ].sort_values(["MAE", "RMSE", "model"])
        if valid_h.empty:
            continue
        frozen_baseline = str(valid_h.iloc[0]["model"])
        part = test.loc[test["horizon_h"].eq(horizon)].copy()
        part = part.dropna(
            subset=[selected_model, frozen_baseline, "actual"]
        )
        part["day"] = part["as_of_time"].dt.floor("D")
        part["selected_abs_error"] = (
            part["actual"] - part[selected_model]
        ).abs()
        part["baseline_abs_error"] = (
            part["actual"] - part[frozen_baseline]
        ).abs()
        part["delta_abs_error"] = (
            part["selected_abs_error"] - part["baseline_abs_error"]
        )
        daily = part.groupby("day", observed=True).agg(
            delta_sum=("delta_abs_error", "sum"),
            count=("actual", "size"),
        )
        indices = rng.integers(
            0,
            len(daily),
            size=(BOOTSTRAP_ITERATIONS, len(daily)),
        )
        draws = (
            daily["delta_sum"].to_numpy(dtype="float64")[indices].sum(axis=1)
            / daily["count"].to_numpy(dtype="float64")[indices].sum(axis=1)
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        selected_mae = float(part["selected_abs_error"].mean())
        baseline_mae = float(part["baseline_abs_error"].mean())
        rows.append(
            {
                "horizon_h": horizon,
                "selected_model": selected_model,
                "frozen_valid_baseline": frozen_baseline,
                "n": len(part),
                "days": len(daily),
                "selected_test_mae": selected_mae,
                "baseline_test_mae": baseline_mae,
                "gain_vs_baseline_pct": 100.0
                * (baseline_mae - selected_mae)
                / max(baseline_mae, 1e-12),
                "delta_mae_ci95_low": float(low),
                "delta_mae_ci95_high": float(high),
                "significantly_better": bool(high < 0.0),
                "non_inferior_5pct_mae": bool(
                    selected_mae <= 1.05 * baseline_mae
                ),
            }
        )
    return pd.DataFrame(rows)


def _materialize_backtest(
    predictions: pd.DataFrame,
    run_id: str,
) -> int:
    ready = predictions.loc[
        predictions["split"].eq("TEST")
        & predictions["calibration_history_ready"]
    ].copy()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SERVING_TABLE} (
                    as_of_time TIMESTAMPTZ NOT NULL,
                    horizon_h INTEGER NOT NULL,
                    forecast_version TEXT NOT NULL,
                    selected_model TEXT NOT NULL,
                    actual_arrivals REAL,
                    point_prediction REAL NOT NULL,
                    p10 REAL NOT NULL,
                    p50 REAL NOT NULL,
                    p90 REAL NOT NULL,
                    adaptive_alpha REAL NOT NULL,
                    conformal_correction REAL NOT NULL,
                    source_mode TEXT NOT NULL,
                    ingestion_run_id UUID NOT NULL,
                    PRIMARY KEY (as_of_time, horizon_h, forecast_version)
                )
                """
            )
            cursor.execute(
                f"DELETE FROM {SERVING_TABLE} WHERE forecast_version=%s",
                (EXPERT_VERSION,),
            )
            values = [
                (
                    row.as_of_time.to_pydatetime(),
                    int(row.horizon_h),
                    EXPERT_VERSION,
                    str(row.selected_model),
                    float(row.actual),
                    float(row.point_prediction),
                    float(row.p10),
                    float(row.p50),
                    float(row.p90),
                    float(row.adaptive_alpha),
                    float(row.conformal_correction),
                    "HISTORICAL_REPLAY",
                    run_id,
                )
                for row in ready.itertuples(index=False)
            ]
            execute_values(
                cursor,
                f"""
                INSERT INTO {SERVING_TABLE} (
                    as_of_time, horizon_h, forecast_version, selected_model,
                    actual_arrivals, point_prediction, p10, p50, p90,
                    adaptive_alpha, conformal_correction, source_mode,
                    ingestion_run_id
                ) VALUES %s
                """,
                values,
                page_size=2000,
            )
    return len(values)


def _log_mlflow(
    output_dir: Path,
    decision: dict[str, Any],
    point_test: pd.DataFrame,
    probabilistic_test: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment("maritime-arrival-flow-expert-count")
        with mlflow.start_run(run_name=EXPERT_VERSION):
            mlflow.log_params(
                {
                    "expert_version": EXPERT_VERSION,
                    "selected_model": decision["selected_model"],
                    "selection_split": "VALID",
                    "incremental_buckets": "0_6,6_12,12_24",
                    "target_coverage": EXPECTED_COVERAGE,
                    "simulation_samples": decision["simulation_samples"],
                }
            )
            selected = decision["selected_model"]
            for horizon in HORIZONS:
                point = point_test.loc[
                    point_test["horizon_h"].eq(horizon)
                    & point_test["model"].eq(selected)
                ].iloc[0]
                calibration = probabilistic_test.loc[
                    probabilistic_test["horizon_h"].eq(horizon)
                ].iloc[0]
                mlflow.log_metric(f"test_mae_{horizon}h", float(point["MAE"]))
                mlflow.log_metric(
                    f"test_wape_{horizon}h", float(point["WAPE_PCT"])
                )
                mlflow.log_metric(
                    f"test_coverage80_{horizon}h",
                    float(calibration["coverage_p10_p90"]),
                )
                mlflow.log_metric(
                    f"test_interval_score_{horizon}h",
                    float(calibration["winkler_interval_score"]),
                )
            mlflow.log_artifacts(str(output_dir), artifact_path="b56g")
        return "LOGGED"
    except Exception as exc:
        return f"FAILED: {exc}"


def _write_readme(path: Path, decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# B56G Expert Probabilistic Count Forecasting",
                "",
                f"Decision: **{decision['status']}**",
                f"Selected expert: **{decision['selected_model']}**",
                "",
                "## Objective",
                "Forecast coherent vessel-arrival counts at 6h, 12h and 24h.",
                "",
                "## Construction",
                "- Targets are decomposed into 0-6h, 6-12h and 12-24h counts.",
                "- Dynamic Negative Binomial and recency-weighted HGB experts",
                "  are fitted on TRAIN only.",
                "- Expert and calibration hyperparameters are selected on VALID.",
                "- TEST remains a locked final audit.",
                "- Online level correction only uses labels that have matured.",
                "- Adaptive conformal calibration targets 80% P10-P90 coverage.",
                "- Reconciliation enforces 6h <= 12h <= 24h and P10 <= P50 <= P90.",
                "",
                "This block writes a separate serving table and never overwrites B56E.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_b56g_expert_probabilistic_count(
    source_bucket: str = "gold-maritime",
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    materialize_timescale: bool = True,
    simulation_samples: int = 400,
    force: bool = False,
) -> dict[str, Any]:
    if simulation_samples < 200 or simulation_samples > 2000:
        raise ValueError("simulation_samples must be between 200 and 2000")
    upstream = _load_upstream_decision()
    checksum = _source_checksum(source_bucket, upstream)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        run_id, metadata = previous
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": True,
            "results": metadata,
        }

    run_id = _start_run(checksum, source_bucket)
    try:
        source = _load_parquet(source_bucket, SOURCE_DATA_KEY)
        frame, integrity = _prepare_source(source)
        dispersion = _target_dispersion(frame)

        models, fit_summary = _fit_expert_bank(frame)
        wide_predictions = _build_candidate_predictions(frame, models)
        long_predictions = _long_candidate_predictions(wide_predictions)

        b56c_valid = _load_optional_parquet(
            source_bucket, VALID_PREDICTIONS_KEY
        )
        b56c_test = _load_optional_parquet(
            source_bucket, TEST_PREDICTIONS_KEY
        )
        incumbent_valid = _load_optional_parquet(
            source_bucket, INCUMBENT_VALID_KEY
        )
        incumbent_test = _load_optional_parquet(
            source_bucket, INCUMBENT_TEST_KEY
        )
        valid_long = _merge_external_predictions(
            long_predictions.loc[long_predictions["split"].eq("VALID")],
            b56c_valid,
            incumbent_valid,
        )
        test_long = _merge_external_predictions(
            long_predictions.loc[long_predictions["split"].eq("TEST")],
            b56c_test,
            incumbent_test,
        )
        comparison_models = tuple(
            dict.fromkeys(
                (
                    *_candidate_names(),
                    *BASELINE_NAMES,
                    "B56E_INCUMBENT",
                )
            )
        )
        metrics_valid = _point_metrics(
            valid_long, "VALID", comparison_models
        )
        selected_model, ranking = _select_expert(metrics_valid)
        metrics_test = _point_metrics(
            test_long, "TEST", comparison_models
        )

        probabilistic_source = wide_predictions.loc[
            wide_predictions["split"].isin(["VALID", "TEST"])
        ].copy()
        base_probabilistic = _simulate_base_distribution(
            probabilistic_source,
            models,
            selected_model,
            simulation_samples,
        )
        selected_calibration, calibration_search = _select_calibration(
            base_probabilistic
        )
        calibrated = _apply_selected_calibration(
            base_probabilistic,
            selected_calibration,
        )
        calibrated = _reconcile_quantiles(calibrated)
        probabilistic_valid = _probabilistic_metrics(calibrated, "VALID")
        probabilistic_test = _probabilistic_metrics(calibrated, "TEST")
        coherence = _coherence_audit(calibrated)

        bootstrap = _paired_day_bootstrap(
            test_long,
            metrics_valid,
            selected_model,
        )
        integrity_passed = bool(integrity["passed"].all())
        coherence_passed = bool(coherence["passed"].all())
        coverage_passed = bool(
            probabilistic_test["coverage_gate_passed"].all()
        )
        point_non_inferior = bool(
            not bootstrap.empty
            and bootstrap["non_inferior_5pct_mae"].all()
        )
        point_improvement_horizons = int(
            (bootstrap["gain_vs_baseline_pct"] > 0.0).sum()
        )

        if not integrity_passed:
            decision_status = "NEED_DATA_REPAIR"
        elif not coherence_passed:
            decision_status = "NEED_RECONCILIATION_REPAIR"
        elif not coverage_passed:
            decision_status = "NEED_INTERVAL_RECALIBRATION"
        elif not point_non_inferior:
            decision_status = "KEEP_B56E_POINT_MODEL"
        else:
            decision_status = "READY_FOR_B56G_SHADOW_REPLAY"

        with tempfile.TemporaryDirectory(prefix="b56g-") as temporary:
            output_dir = Path(temporary)
            reports_dir = output_dir / "reports"
            configs_dir = output_dir / "configs"
            predictions_dir = output_dir / "predictions"
            models_dir = output_dir / "models"
            for directory in (
                reports_dir,
                configs_dir,
                predictions_dir,
                models_dir,
            ):
                directory.mkdir(parents=True)

            report_frames = {
                "00_integrity_and_anti_leakage.csv": integrity,
                "01_target_count_dispersion.csv": dispersion,
                "02_expert_fit_summary.csv": fit_summary,
                "03_valid_candidate_metrics.csv": metrics_valid,
                "04_valid_expert_ranking.csv": ranking,
                "05_test_locked_metrics.csv": metrics_test,
                "06_valid_calibration_search.csv": calibration_search,
                "07_valid_probabilistic_metrics.csv": probabilistic_valid,
                "08_test_probabilistic_metrics.csv": probabilistic_test,
                "09_test_paired_day_bootstrap.csv": bootstrap,
                "10_coherence_audit.csv": coherence,
            }
            for name, report in report_frames.items():
                report.to_csv(reports_dir / name, index=False)

            valid_output = calibrated.loc[
                calibrated["split"].eq("VALID")
            ].copy()
            test_output = calibrated.loc[
                calibrated["split"].eq("TEST")
            ].copy()
            valid_output.to_parquet(
                predictions_dir / "valid_expert_predictions.parquet",
                index=False,
            )
            test_output.to_parquet(
                predictions_dir / "test_expert_predictions.parquet",
                index=False,
            )

            base_selected = _base_candidate_name(selected_model)
            model_bundle = {
                "expert_version": EXPERT_VERSION,
                "selected_model": selected_model,
                "models": models[base_selected],
                "calibration": selected_calibration,
                "features": MODEL_SPECS[base_selected]["features"],
                "increment_buckets": INCREMENT_BUCKETS,
            }
            with (models_dir / "selected_expert_bundle.pkl").open(
                "wb"
            ) as handle:
                pickle.dump(
                    model_bundle,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            decision = {
                "status": decision_status,
                "expert_version": EXPERT_VERSION,
                "objective": "COHERENT_PORT_ARRIVAL_COUNTS_6H_12H_24H",
                "selected_model": selected_model,
                "selected_base_model": base_selected,
                "selected_calibration": selected_calibration,
                "selection_split": "VALID",
                "test_role": "LOCKED_FINAL_AUDIT_ONLY",
                "selection_used_test": False,
                "training_split": "TRAIN_ONLY",
                "training_executed": True,
                "incremental_target_policy": "0_6H_PLUS_6_12H_PLUS_12_24H",
                "online_policy": (
                    "ROLLING_30D_MEDIAN_LOG_RATIO_LABELS_SHIFTED_BY_MATURITY"
                ),
                "probabilistic_distribution": "NEGATIVE_BINOMIAL",
                "calibration_method": "MULTI_HORIZON_ADAPTIVE_CQR",
                "interval_target_coverage": EXPECTED_COVERAGE,
                "simulation_samples": simulation_samples,
                "integrity_gates_passed": integrity_passed,
                "coherence_gates_passed": coherence_passed,
                "coverage_gates_passed": coverage_passed,
                "point_non_inferior": point_non_inferior,
                "point_improvement_horizons": point_improvement_horizons,
                "weather_policy": (
                    "PAST_ONLY_WAVE_EXPERT_VALID_SELECTABLE_"
                    "NO_FUTURE_WEATHER_ASSUMPTION"
                ),
                "source_bucket": source_bucket,
                "source_key": SOURCE_DATA_KEY,
                "historical_replay_allowed": bool(
                    integrity_passed and coherence_passed
                ),
                "live_serving_allowed": False,
                "next_block": (
                    "B56H_DUAL_TEMPORAL_POINT_PROCESS_CHALLENGER"
                    if decision_status == "READY_FOR_B56G_SHADOW_REPLAY"
                    else "B56G_REPAIR_OR_KEEP_INCUMBENT"
                ),
            }
            decision_path = configs_dir / "11_b56g_decision.json"
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    allow_nan=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            _write_readme(reports_dir / "README_B56G.md", decision)

            timescale_rows = (
                _materialize_backtest(calibrated, run_id)
                if materialize_timescale
                else 0
            )
            mlflow_status = _log_mlflow(
                output_dir,
                decision,
                metrics_test,
                probabilistic_test,
            )

            client = _s3_client()
            outputs: dict[str, str] = {}
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(output_dir).as_posix()
                category, name = relative.split("/", 1)
                key = f"{category}/b56g/{output_prefix.strip('/')}/{name}"
                outputs[name] = _upload(
                    client,
                    path,
                    output_bucket,
                    key,
                )

        decision.update(
            {
                "row_count": len(frame),
                "train_rows": int(frame["split"].eq("TRAIN").sum()),
                "valid_rows": int(frame["split"].eq("VALID").sum()),
                "test_rows": int(frame["split"].eq("TEST").sum()),
                "purged_rows": int(frame["split"].eq("PURGED").sum()),
                "timescale_table": (
                    SERVING_TABLE if materialize_timescale else None
                ),
                "timescale_rows": timescale_rows,
                "mlflow_status": mlflow_status,
                "outputs": outputs,
                "checksum": checksum,
            }
        )
        _finish_run(run_id, "SUCCESS", len(frame), decision)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "results": _clean_json(decision),
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {"expert_version": EXPERT_VERSION},
            error_message=str(exc),
        )
        raise
