from __future__ import annotations

import hashlib
import io
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
from catboost import CatBoostRegressor
from psycopg2.extras import Json
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


TRAINING_VERSION = "b57c-event-aware-temporal-baselines-v1.1"
SOURCE_FEATURE_VERSION = "b57b-event-aware-daily-gold-v1"
SOURCE_BUCKET = "gold-maritime"
SOURCE_KEY = "datasets/b57b/version=1/tir_daily_predictive_gold_v1.parquet"
SOURCE_NAME = "b57c_event_aware_temporal_baselines"
DATASET_NAME = "tir_daily_event_aware_temporal_models"
PURGE_DAYS = 7
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 500

TARGETS = {
    "TIR_VOLUME": {
        "column": "target_tir_rows",
        "kind": "COUNT",
        "eligibility": "TARGET_AVAILABLE",
    },
    "DURATION_MEDIAN": {
        "column": "target_duration_median_h",
        "kind": "DURATION",
        "eligibility": "MODEL_READY",
    },
    "LONG_24H_RATE": {
        "column": "target_long_24h_rate",
        "kind": "RATE",
        "eligibility": "MODEL_READY",
    },
}

TRACKS = {
    "FULL_NO_PORT": {
        "cv_years": (2023, 2024, 2025),
        "test_start": "2026-01-01",
        "test_end": "2026-12-31",
        "allow_port": False,
        "official": True,
    },
    "PRE_BREAK_WITH_PORT": {
        "cv_years": (2023, 2024),
        "test_start": "2025-01-01",
        "test_end": "2025-04-30",
        "allow_port": True,
        "official": False,
    },
}

BASELINE_NAMES = {
    "TRAIN_MEAN",
    "TRAIN_MEDIAN",
    "SEASONAL_NAIVE_7D",
    "SEASONAL_NAIVE_28D",
    "SEASONAL_NAIVE_364D",
    "EWMA_28D",
}

BUNDLE_COMPLEXITY = {
    "HGB_CALENDAR": 10,
    "HGB_CALENDAR_TIR": 20,
    "HGB_CALENDAR_TIR_WEATHER": 30,
    "HGB_CALENDAR_TIR_WEATHER_EVENTS": 40,
    "HGB_CALENDAR_TIR_WEATHER_EVENTS_PORT": 50,
    "CATBOOST_QUANTILE_MAXIMAL": 60,
}

FORBIDDEN_EXACT = {
    "prediction_date",
    "prediction_at",
    "feature_version",
    "tir_source_day_observed_flag",
    "model_ready_flag",
}


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


def _load_upstream_decision() -> dict[str, Any]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, metadata
                FROM audit.ingestion_run
                WHERE source_name='b57b_event_aware_gold'
                  AND dataset_name='tir_daily_event_aware_gold'
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("B57B decision is missing")
    status, metadata = row
    metadata = dict(metadata or {})
    if status != "SUCCESS":
        raise RuntimeError(f"Latest B57B status is {status}")
    if metadata.get("feature_version") != SOURCE_FEATURE_VERSION:
        raise RuntimeError("Latest B57B feature version is not supported")
    if metadata.get("status") != "READY_FOR_EVENT_AWARE_BASELINES":
        raise RuntimeError("B57B did not approve event-aware baselines")
    if int(metadata.get("critical_leakage_violations", -1)) != 0:
        raise RuntimeError("B57B anti-leakage gate did not pass")
    return metadata


def _load_source(bucket: str, key: str) -> tuple[pd.DataFrame, str]:
    response = _s3_client().get_object(Bucket=bucket, Key=key)
    payload = response["Body"].read()
    checksum = hashlib.sha256(TRAINING_VERSION.encode("ascii") + payload).hexdigest()
    frame = pd.read_parquet(io.BytesIO(payload))
    if frame.empty:
        raise RuntimeError("B57B predictive Gold is empty")
    required = {
        "prediction_date",
        "prediction_at",
        "feature_version",
        "model_ready_flag",
        *(item["column"] for item in TARGETS.values()),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"B57B source misses required columns: {missing}")
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"], utc=True)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True)
    frame = frame.sort_values("prediction_date").reset_index(drop=True)
    if frame["prediction_date"].duplicated().any():
        raise RuntimeError("B57B source violates one-row-per-day grain")
    versions = set(frame["feature_version"].dropna().astype(str).unique())
    if versions != {SOURCE_FEATURE_VERSION}:
        raise RuntimeError(f"Unexpected B57B feature versions: {sorted(versions)}")
    for column in frame.columns:
        if column not in {"prediction_date", "prediction_at", "feature_version"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, checksum


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


def _start_run(checksum: str, bucket: str, key: str) -> str:
    metadata = {
        "training_version": TRAINING_VERSION,
        "source_feature_version": SOURCE_FEATURE_VERSION,
        "official_protocol": "FULL_NO_PORT_WALK_FORWARD_PURGED_7D",
        "selection_policy": "CV_ONLY_TEST_DIAGNOSTIC_ONLY",
        "source_object": f"s3://{bucket}/{key}",
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
                    f"s3://{bucket}/{key}",
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


def _update_progress(run_id: str, progress: dict[str, Any]) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET metadata = metadata || %s
                WHERE run_id=%s AND status='RUNNING'
                """,
                (
                    Json(
                        _clean_json({"progress": progress}),
                        dumps=lambda value: json.dumps(
                            value, default=_json_default, allow_nan=False
                        ),
                    ),
                    run_id,
                ),
            )


def _feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    all_columns = set(frame.columns)
    target_columns = {item["column"] for item in TARGETS.values()}
    forbidden = target_columns | FORBIDDEN_EXACT

    event = sorted(
        column
        for column in all_columns
        if column
        in {
            "calendar_known_event_flag",
            "calendar_aid_el_fitr_flag",
            "calendar_aid_al_adha_flag",
        }
    )
    calendar = sorted(
        column
        for column in all_columns
        if column.startswith("calendar_") and column not in set(event) | forbidden
    )
    weather = sorted(
        column
        for column in all_columns
        if column.startswith(
            (
                "hist_weather_",
                "hist_wave_",
                "hist_storm_",
            )
        )
        or column
        in {
            "weather_history_available_flag",
            "weather_history_stale_flag",
            "weather_forecast_available_flag",
        }
    )
    port = sorted(
        column
        for column in all_columns
        if column.startswith("hist_port_")
        or column in {"port_history_available_flag", "port_source_break_flag"}
    )
    tir = sorted(
        column
        for column in all_columns
        if (
            column.startswith("hist_tir_")
            or column in {"tir_history_available_flag", "cold_start_28d_flag"}
        )
        and column not in forbidden
    )
    groups = {
        "CALENDAR": calendar,
        "TIR_HISTORY": tir,
        "WEATHER_HISTORY": weather,
        "KNOWN_EVENTS": event,
        "PORT_HISTORY": port,
    }
    used = set().union(*map(set, groups.values()))
    unexpected = sorted(
        column
        for column in all_columns
        if column not in forbidden
        and column not in used
        and not column.startswith("target_")
    )
    allowed_quality = {
        "feature_version",
        "prediction_date",
        "prediction_at",
        "tir_source_day_observed_flag",
        "model_ready_flag",
    }
    unexpected = [column for column in unexpected if column not in allowed_quality]
    if unexpected:
        raise RuntimeError(f"Unclassified predictive columns: {unexpected}")
    return groups


def _feature_contract(frame: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    target_columns = {item["column"] for item in TARGETS.values()}
    family_by_feature = {
        feature: family for family, features in groups.items() for feature in features
    }
    for column in frame.columns:
        if column in target_columns:
            role = "TARGET_CURRENT_DAY"
            allowed = False
        elif column in family_by_feature:
            role = family_by_feature[column]
            allowed = True
        elif column in {"prediction_date", "prediction_at", "feature_version"}:
            role = "IDENTIFIER"
            allowed = False
        else:
            role = "QUALITY_OR_LABEL_AVAILABILITY"
            allowed = False
        rows.append(
            {
                "column": column,
                "role": role,
                "allowed_in_x": allowed,
                "null_pct": 100.0 * frame[column].isna().mean(),
                "unique_values": int(frame[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _anti_leakage_audit(
    frame: pd.DataFrame, groups: dict[str, list[str]]
) -> tuple[pd.DataFrame, int]:
    x_columns = set().union(*map(set, groups.values()))
    target_columns = {item["column"] for item in TARGETS.values()}
    checks = [
        {
            "check": "NO_TARGET_IN_X",
            "violations": len(x_columns & target_columns),
            "details": sorted(x_columns & target_columns),
        },
        {
            "check": "NO_IDENTIFIER_IN_X",
            "violations": len(x_columns & FORBIDDEN_EXACT),
            "details": sorted(x_columns & FORBIDDEN_EXACT),
        },
        {
            "check": "NO_REALIZED_OR_RETROSPECTIVE_IN_X",
            "violations": len(
                [
                    column
                    for column in x_columns
                    if column.startswith("realized_")
                    or "retrospective" in column.lower()
                ]
            ),
            "details": sorted(
                column
                for column in x_columns
                if column.startswith("realized_")
                or "retrospective" in column.lower()
            ),
        },
        {
            "check": "ONE_ROW_PER_PREDICTION_DATE",
            "violations": int(frame["prediction_date"].duplicated().sum()),
            "details": [],
        },
        {
            "check": "PREDICTION_AT_EQUALS_DAY_START",
            "violations": int(
                (~frame["prediction_at"].eq(frame["prediction_date"])).sum()
            ),
            "details": [],
        },
        {
            "check": "NO_PORT_FEATURES_AFTER_SOURCE_BREAK",
            "violations": int(
                frame.loc[
                    frame.get("port_source_break_flag", 0).eq(1),
                    [
                        column
                        for column in groups["PORT_HISTORY"]
                        if column.startswith("hist_port_")
                    ],
                ]
                .notna()
                .sum()
                .sum()
            ),
            "details": [],
        },
    ]
    report = pd.DataFrame(checks)
    return report, int(report["violations"].sum())


def _eligible_mask(frame: pd.DataFrame, target_spec: dict[str, str]) -> pd.Series:
    target = target_spec["column"]
    mask = frame[target].notna() & frame["cold_start_28d_flag"].eq(0)
    if target_spec["eligibility"] == "MODEL_READY":
        mask &= frame["model_ready_flag"].eq(1)
    return mask


def _date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _split_masks(
    frame: pd.DataFrame,
    eligible: pd.Series,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    purge_start = validation_start - pd.Timedelta(days=PURGE_DAYS)
    train = eligible & frame["prediction_date"].lt(purge_start)
    purged = eligible & frame["prediction_date"].ge(purge_start) & frame[
        "prediction_date"
    ].lt(validation_start)
    validation = eligible & frame["prediction_date"].between(
        validation_start, validation_end
    )
    return train, validation, purged


def _safe_features(
    frame: pd.DataFrame, train_mask: pd.Series, requested: list[str]
) -> tuple[list[str], list[str]]:
    kept = []
    dropped = []
    train = frame.loc[train_mask, requested]
    for feature in requested:
        values = train[feature]
        if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
            dropped.append(feature)
        else:
            kept.append(feature)
    return kept, dropped


def _bundles(groups: dict[str, list[str]], allow_port: bool) -> dict[str, list[str]]:
    calendar = groups["CALENDAR"]
    tir = groups["TIR_HISTORY"]
    weather = groups["WEATHER_HISTORY"]
    events = groups["KNOWN_EVENTS"]
    bundles = {
        "HGB_CALENDAR": [*calendar],
        "HGB_CALENDAR_TIR": [*calendar, *tir],
        "HGB_CALENDAR_TIR_WEATHER": [*calendar, *tir, *weather],
        "HGB_CALENDAR_TIR_WEATHER_EVENTS": [
            *calendar,
            *tir,
            *weather,
            *events,
        ],
    }
    if allow_port:
        bundles["HGB_CALENDAR_TIR_WEATHER_EVENTS_PORT"] = [
            *calendar,
            *tir,
            *weather,
            *events,
            *groups["PORT_HISTORY"],
        ]
    return bundles


def _build_hgb(kind: str) -> HistGradientBoostingRegressor:
    loss = {
        "COUNT": "poisson",
        "DURATION": "absolute_error",
        "RATE": "squared_error",
    }[kind]
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def _build_catboost_quantile() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="Quantile:alpha=0.5",
        iterations=160,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=5.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=int(os.getenv("B54D_THREAD_COUNT", "2")),
    )


def _clip(values: np.ndarray, kind: str) -> np.ndarray:
    result = np.asarray(values, dtype="float64")
    if kind == "COUNT":
        return np.clip(result, 0.0, None)
    if kind == "DURATION":
        return np.clip(result, 0.0, 720.0)
    return np.clip(result, 0.0, 1.0)


def _metrics(y_true: np.ndarray, prediction: np.ndarray, kind: str) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    prediction = _clip(prediction, kind)
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
    result = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2_score(y_true, prediction),
        "MEDAE": median_absolute_error(y_true, prediction),
        "BIAS": float(np.mean(prediction - y_true)),
        "WAPE_PCT": 100 * np.abs(prediction - y_true).sum() / max(denominator, 1e-12),
        "SMAPE_PCT": 100 * smape,
    }
    if kind == "COUNT":
        result["POISSON_DEVIANCE"] = mean_poisson_deviance(
            np.clip(y_true, 0.0, None), np.clip(prediction, 1e-6, None)
        )
    else:
        result["POISSON_DEVIANCE"] = np.nan
    return result


def _baseline_predictions(
    frame: pd.DataFrame,
    target: str,
    train_mask: pd.Series,
) -> dict[str, np.ndarray]:
    values = frame[target].astype("float64")
    train_values = values[train_mask]
    return {
        "TRAIN_MEAN": np.full(len(frame), float(train_values.mean())),
        "TRAIN_MEDIAN": np.full(len(frame), float(train_values.median())),
        "SEASONAL_NAIVE_7D": values.shift(7).to_numpy(),
        "SEASONAL_NAIVE_28D": values.shift(28).to_numpy(),
        "SEASONAL_NAIVE_364D": values.shift(364).to_numpy(),
        "EWMA_28D": values.shift(1).ewm(span=28, adjust=False).mean().to_numpy(),
    }


def _fit_predict_candidates(
    frame: pd.DataFrame,
    target_name: str,
    target_spec: dict[str, str],
    train_mask: pd.Series,
    predict_mask: pd.Series,
    groups: dict[str, list[str]],
    allow_port: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    target = target_spec["column"]
    kind = target_spec["kind"]
    predictions = _baseline_predictions(frame, target, train_mask)
    fitted: dict[str, Any] = {}
    feature_rows = []
    bundles = _bundles(groups, allow_port)
    for model_name, requested in bundles.items():
        features, dropped = _safe_features(frame, train_mask, requested)
        if not features:
            continue
        model = _build_hgb(kind)
        model.fit(frame.loc[train_mask, features], frame.loc[train_mask, target])
        values = np.full(len(frame), np.nan, dtype="float64")
        values[predict_mask.to_numpy()] = _clip(
            model.predict(frame.loc[predict_mask, features]), kind
        )
        predictions[model_name] = values
        fitted[model_name] = {
            "model": model,
            "features": features,
            "target": target,
            "target_name": target_name,
            "kind": kind,
        }
        feature_rows.append(
            {
                "target_name": target_name,
                "model": model_name,
                "requested_features": len(requested),
                "used_features": len(features),
                "dropped_zero_variance": json.dumps(dropped),
            }
        )

    if kind == "DURATION":
        maximal_name = list(bundles)[-1]
        requested = bundles[maximal_name]
        features, dropped = _safe_features(frame, train_mask, requested)
        if features:
            model = _build_catboost_quantile()
            model.fit(
                frame.loc[train_mask, features],
                frame.loc[train_mask, target],
            )
            values = np.full(len(frame), np.nan, dtype="float64")
            values[predict_mask.to_numpy()] = _clip(
                model.predict(frame.loc[predict_mask, features]), kind
            )
            name = "CATBOOST_QUANTILE_MAXIMAL"
            predictions[name] = values
            fitted[name] = {
                "model": model,
                "features": features,
                "target": target,
                "target_name": target_name,
                "kind": kind,
            }
            feature_rows.append(
                {
                    "target_name": target_name,
                    "model": name,
                    "requested_features": len(requested),
                    "used_features": len(features),
                    "dropped_zero_variance": json.dumps(dropped),
                }
            )
    return predictions, fitted, feature_rows


def _evaluate_predictions(
    frame: pd.DataFrame,
    prediction_mask: pd.Series,
    target_name: str,
    target_spec: dict[str, str],
    predictions: dict[str, np.ndarray],
    track: str,
    fold: str,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = target_spec["column"]
    kind = target_spec["kind"]
    actual_all = frame[target].to_numpy(dtype="float64")
    base_mask = prediction_mask.to_numpy() & np.isfinite(actual_all)
    metric_rows = []
    prediction_rows = []
    for model, values in predictions.items():
        values = np.asarray(values, dtype="float64")
        valid = base_mask & np.isfinite(values)
        if valid.sum() < 20:
            continue
        metric_rows.append(
            {
                "phase": phase,
                "track": track,
                "fold": fold,
                "target_name": target_name,
                "target": target,
                "kind": kind,
                "model": model,
                "n": int(valid.sum()),
                **_metrics(actual_all[valid], values[valid], kind),
            }
        )
        dates = frame.loc[valid, "prediction_date"].to_numpy()
        prediction_rows.extend(
            {
                "phase": phase,
                "track": track,
                "fold": fold,
                "prediction_date": date,
                "target_name": target_name,
                "target": target,
                "model": model,
                "actual": actual,
                "prediction": prediction,
            }
            for date, actual, prediction in zip(
                dates, actual_all[valid], _clip(values[valid], kind)
            )
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _complexity(model: str) -> int:
    if model in BASELINE_NAMES:
        return 0
    return BUNDLE_COMPLEXITY.get(model, 100)


def _select_models(cv_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    summary = (
        cv_metrics.groupby(["track", "target_name", "model"], observed=True)
        .agg(
            folds=("fold", "nunique"),
            n=("n", "sum"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            R2=("R2", "mean"),
            BIAS=("BIAS", "mean"),
            WAPE_PCT=("WAPE_PCT", "mean"),
        )
        .reset_index()
    )
    summary["complexity_rank"] = summary["model"].map(_complexity)
    selections: dict[str, str] = {}
    rows = []
    for (track, target), part in summary.groupby(
        ["track", "target_name"], observed=True
    ):
        expected_folds = len(TRACKS[str(track)]["cv_years"])
        complete = part[part["folds"] == expected_folds].copy()
        if complete.empty:
            raise RuntimeError(f"No complete CV candidate for {track}/{target}")
        complete = complete.sort_values(["MAE", "complexity_rank", "RMSE", "model"])
        best = complete.iloc[0]
        baseline = complete[complete["model"].isin(BASELINE_NAMES)].sort_values(
            ["MAE", "RMSE", "model"]
        ).iloc[0]
        uplift = 100 * (baseline["MAE"] - best["MAE"]) / max(
            float(baseline["MAE"]), 1e-12
        )
        key = f"{track}:{target}"
        selections[key] = str(best["model"])
        rows.append(
            {
                "track": track,
                "target_name": target,
                "selected_model": best["model"],
                "selected_cv_mae": best["MAE"],
                "best_baseline": baseline["model"],
                "best_baseline_cv_mae": baseline["MAE"],
                "cv_uplift_vs_best_baseline_pct": uplift,
                "selection_used_test": False,
            }
        )
    return pd.DataFrame(rows), selections


def _bootstrap_ablation(
    predictions: pd.DataFrame,
    track: str,
    target_name: str,
    base_model: str,
    enhanced_model: str,
    family: str,
    has_train_variation: bool,
) -> dict[str, Any]:
    if not has_train_variation:
        return {
            "track": track,
            "target_name": target_name,
            "family": family,
            "base_model": base_model,
            "enhanced_model": enhanced_model,
            "n": 0,
            "gain_pct": None,
            "delta_mae_ci95_low": None,
            "delta_mae_ci95_high": None,
            "decision": "NOT_TESTABLE_NO_TRAIN_VARIATION",
        }
    part = predictions[
        (predictions["track"] == track)
        & (predictions["target_name"] == target_name)
        & (predictions["model"].isin([base_model, enhanced_model]))
    ]
    if part.empty:
        return {
            "track": track,
            "target_name": target_name,
            "family": family,
            "base_model": base_model,
            "enhanced_model": enhanced_model,
            "n": 0,
            "gain_pct": None,
            "delta_mae_ci95_low": None,
            "delta_mae_ci95_high": None,
            "decision": "NOT_TESTABLE_MISSING_PREDICTIONS",
        }
    pivot = part.pivot_table(
        index=["fold", "prediction_date", "actual"],
        columns="model",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    if base_model not in pivot or enhanced_model not in pivot:
        has_both = pd.DataFrame()
    else:
        has_both = pivot.dropna(subset=[base_model, enhanced_model]).copy()
    if len(has_both) < 30:
        return {
            "track": track,
            "target_name": target_name,
            "family": family,
            "base_model": base_model,
            "enhanced_model": enhanced_model,
            "n": len(has_both),
            "gain_pct": None,
            "delta_mae_ci95_low": None,
            "delta_mae_ci95_high": None,
            "decision": "NOT_TESTABLE_INSUFFICIENT_ROWS",
        }
    has_both["delta_abs"] = (
        (has_both["actual"] - has_both[enhanced_model]).abs()
        - (has_both["actual"] - has_both[base_model]).abs()
    )
    dates = pd.to_datetime(has_both["prediction_date"], utc=True)
    has_both["block"] = (
        has_both["fold"].astype(str)
        + ":"
        + ((dates.astype("int64") // 86_400_000_000_000) // 7).astype(str)
    )
    blocks = has_both.groupby("block", observed=True)["delta_abs"].agg(["sum", "count"])
    rng = np.random.default_rng(RANDOM_SEED + len(target_name) + len(family))
    positions = np.arange(len(blocks))
    draws = np.empty(BOOTSTRAP_ITERATIONS, dtype="float64")
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sample = blocks.iloc[
            rng.choice(positions, size=len(positions), replace=True)
        ]
        draws[iteration] = sample["sum"].sum() / sample["count"].sum()
    base_mae = float((has_both["actual"] - has_both[base_model]).abs().mean())
    enhanced_mae = float(
        (has_both["actual"] - has_both[enhanced_model]).abs().mean()
    )
    gain = 100 * (base_mae - enhanced_mae) / max(base_mae, 1e-12)
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    keep = bool(gain >= 2.0 and ci_high < 0)
    return {
        "track": track,
        "target_name": target_name,
        "family": family,
        "base_model": base_model,
        "enhanced_model": enhanced_model,
        "n": len(has_both),
        "blocks": len(blocks),
        "base_mae": base_mae,
        "enhanced_mae": enhanced_mae,
        "gain_pct": gain,
        "delta_mae_ci95_low": ci_low,
        "delta_mae_ci95_high": ci_high,
        "decision": "KEEP" if keep else "UPLIFT_NOT_PROVEN",
    }


def _target_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name, spec in TARGETS.items():
        target = spec["column"]
        eligible = _eligible_mask(frame, spec)
        part = frame.loc[eligible, ["prediction_date", target]].copy()
        part["year"] = part["prediction_date"].dt.year
        for year, values in part.groupby("year", observed=True)[target]:
            rows.append(
                {
                    "target_name": target_name,
                    "target": target,
                    "year": int(year),
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "p10": values.quantile(0.10),
                    "p90": values.quantile(0.90),
                }
            )
    return pd.DataFrame(rows)


def _upload(client, path: Path, bucket: str, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/octet-stream",
        ".pkl": "application/octet-stream",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
    return f"s3://{bucket}/{key}"


def _log_mlflow(
    output_dir: Path,
    decision: dict[str, Any],
    selection: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("smart-port-tir-event-aware")
        with mlflow.start_run(run_name=TRAINING_VERSION):
            mlflow.log_params(
                {
                    "training_version": TRAINING_VERSION,
                    "official_track": "FULL_NO_PORT",
                    "protocol": "WALK_FORWARD_PURGED_7D",
                    "selection": "CV_ONLY_TEST_DIAGNOSTIC_ONLY",
                    "targets": ",".join(TARGETS),
                }
            )
            for row in selection.itertuples():
                if row.track != "FULL_NO_PORT":
                    continue
                mlflow.log_metric(
                    f"cv_mae_{row.target_name.lower()}", float(row.selected_cv_mae)
                )
                metric = test_metrics[
                    (test_metrics["track"] == row.track)
                    & (test_metrics["target_name"] == row.target_name)
                    & (test_metrics["model"] == row.selected_model)
                ]
                if not metric.empty:
                    mlflow.log_metric(
                        f"test_mae_{row.target_name.lower()}", float(metric.iloc[0]["MAE"])
                    )
            mlflow.log_artifacts(str(output_dir), artifact_path="b57c")
        return "LOGGED"
    except Exception as exc:
        return f"ERROR: {exc}"


def run_b57c_event_aware_baselines(
    source_bucket: str = SOURCE_BUCKET,
    source_key: str = SOURCE_KEY,
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    upstream = _load_upstream_decision()
    frame, checksum = _load_source(source_bucket, source_key)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "results": previous[1],
        }

    run_id = _start_run(checksum, source_bucket, source_key)
    try:
        groups = _feature_groups(frame)
        feature_contract = _feature_contract(frame, groups)
        leakage_report, leakage_violations = _anti_leakage_audit(frame, groups)
        if leakage_violations:
            raise RuntimeError(
                f"B57C anti-leakage gate failed with {leakage_violations} violations"
            )

        cv_metrics_parts = []
        cv_prediction_parts = []
        feature_usage_rows = []
        split_rows = []

        for track, track_spec in TRACKS.items():
            allow_port = bool(track_spec["allow_port"])
            for target_name, target_spec in TARGETS.items():
                eligible = _eligible_mask(frame, target_spec)
                for year in track_spec["cv_years"]:
                    validation_start = _date(f"{year}-01-01")
                    validation_end = _date(f"{year}-12-31")
                    train_mask, valid_mask, purge_mask = _split_masks(
                        frame, eligible, validation_start, validation_end
                    )
                    if train_mask.sum() < 700 or valid_mask.sum() < 90:
                        raise RuntimeError(
                            f"Insufficient rows for {track}/{target_name}/{year}: "
                            f"train={train_mask.sum()} valid={valid_mask.sum()}"
                        )
                    predictions, _, usage = _fit_predict_candidates(
                        frame,
                        target_name,
                        target_spec,
                        train_mask,
                        valid_mask,
                        groups,
                        allow_port,
                    )
                    metrics, prediction_rows = _evaluate_predictions(
                        frame,
                        valid_mask,
                        target_name,
                        target_spec,
                        predictions,
                        track,
                        str(year),
                        "CV",
                    )
                    cv_metrics_parts.append(metrics)
                    cv_prediction_parts.append(prediction_rows)
                    for row in usage:
                        feature_usage_rows.append(
                            {"phase": "CV", "track": track, "fold": year, **row}
                        )
                    split_rows.extend(
                        [
                            {
                                "track": track,
                                "target_name": target_name,
                                "fold": str(year),
                                "split": "TRAIN",
                                "rows": int(train_mask.sum()),
                                "first_date": frame.loc[train_mask, "prediction_date"].min(),
                                "last_date": frame.loc[train_mask, "prediction_date"].max(),
                            },
                            {
                                "track": track,
                                "target_name": target_name,
                                "fold": str(year),
                                "split": "PURGED",
                                "rows": int(purge_mask.sum()),
                                "first_date": frame.loc[purge_mask, "prediction_date"].min(),
                                "last_date": frame.loc[purge_mask, "prediction_date"].max(),
                            },
                            {
                                "track": track,
                                "target_name": target_name,
                                "fold": str(year),
                                "split": "VALID",
                                "rows": int(valid_mask.sum()),
                                "first_date": frame.loc[valid_mask, "prediction_date"].min(),
                                "last_date": frame.loc[valid_mask, "prediction_date"].max(),
                            },
                        ]
                    )
                    _update_progress(
                        run_id,
                        {
                            "stage": "CROSS_VALIDATION",
                            "track": track,
                            "target": target_name,
                            "fold": str(year),
                            "completed_models": int(len(metrics)),
                            "updated_at": pd.Timestamp.now(tz="UTC"),
                        },
                    )

        cv_metrics = pd.concat(cv_metrics_parts, ignore_index=True)
        cv_predictions = pd.concat(cv_prediction_parts, ignore_index=True)
        cv_summary, selections = _select_models(cv_metrics)

        final_metric_parts = []
        final_prediction_parts = []
        final_models: dict[str, Any] = {}
        for track, track_spec in TRACKS.items():
            allow_port = bool(track_spec["allow_port"])
            test_start = _date(str(track_spec["test_start"]))
            test_end = min(_date(str(track_spec["test_end"])), frame["prediction_date"].max())
            for target_name, target_spec in TARGETS.items():
                eligible = _eligible_mask(frame, target_spec)
                train_mask, test_mask, purge_mask = _split_masks(
                    frame, eligible, test_start, test_end
                )
                predictions, fitted, usage = _fit_predict_candidates(
                    frame,
                    target_name,
                    target_spec,
                    train_mask,
                    test_mask,
                    groups,
                    allow_port,
                )
                metrics, prediction_rows = _evaluate_predictions(
                    frame,
                    test_mask,
                    target_name,
                    target_spec,
                    predictions,
                    track,
                    "FINAL",
                    "TEST_DIAGNOSTIC",
                )
                final_metric_parts.append(metrics)
                final_prediction_parts.append(prediction_rows)
                selected = selections[f"{track}:{target_name}"]
                if selected in fitted:
                    final_models[f"{track}__{target_name}__{selected}"] = fitted[selected]
                for row in usage:
                    feature_usage_rows.append(
                        {"phase": "FINAL", "track": track, "fold": "FINAL", **row}
                    )
                split_rows.extend(
                    [
                        {
                            "track": track,
                            "target_name": target_name,
                            "fold": "FINAL",
                            "split": "TRAIN",
                            "rows": int(train_mask.sum()),
                            "first_date": frame.loc[train_mask, "prediction_date"].min(),
                            "last_date": frame.loc[train_mask, "prediction_date"].max(),
                        },
                        {
                            "track": track,
                            "target_name": target_name,
                            "fold": "FINAL",
                            "split": "PURGED",
                            "rows": int(purge_mask.sum()),
                            "first_date": frame.loc[purge_mask, "prediction_date"].min(),
                            "last_date": frame.loc[purge_mask, "prediction_date"].max(),
                        },
                        {
                            "track": track,
                            "target_name": target_name,
                            "fold": "FINAL",
                            "split": "TEST_DIAGNOSTIC",
                            "rows": int(test_mask.sum()),
                            "first_date": frame.loc[test_mask, "prediction_date"].min(),
                            "last_date": frame.loc[test_mask, "prediction_date"].max(),
                        },
                    ]
                )
                _update_progress(
                    run_id,
                    {
                        "stage": "FINAL_DIAGNOSTIC",
                        "track": track,
                        "target": target_name,
                        "completed_models": int(len(metrics)),
                        "updated_at": pd.Timestamp.now(tz="UTC"),
                    },
                )

        test_metrics = pd.concat(final_metric_parts, ignore_index=True)
        test_predictions = pd.concat(final_prediction_parts, ignore_index=True)
        split_audit = pd.DataFrame(split_rows)
        feature_usage = pd.DataFrame(feature_usage_rows)

        ablation_rows = []
        event_variation = any(
            frame.loc[
                frame["prediction_date"].lt(_date("2026-01-01")), feature
            ].nunique(dropna=True)
            > 1
            for feature in groups["KNOWN_EVENTS"]
        )
        weather_variation = any(
            frame[feature].nunique(dropna=True) > 1
            for feature in groups["WEATHER_HISTORY"]
        )
        port_variation = any(
            frame.loc[
                frame["prediction_date"].lt(_date("2025-01-01")), feature
            ].nunique(dropna=True)
            > 1
            for feature in groups["PORT_HISTORY"]
        )
        for track in TRACKS:
            for target_name in TARGETS:
                ablation_rows.append(
                    _bootstrap_ablation(
                        cv_predictions,
                        track,
                        target_name,
                        "HGB_CALENDAR_TIR",
                        "HGB_CALENDAR_TIR_WEATHER",
                        "WEATHER_HISTORY",
                        weather_variation,
                    )
                )
                ablation_rows.append(
                    _bootstrap_ablation(
                        cv_predictions,
                        track,
                        target_name,
                        "HGB_CALENDAR_TIR_WEATHER",
                        "HGB_CALENDAR_TIR_WEATHER_EVENTS",
                        "KNOWN_EVENTS",
                        event_variation,
                    )
                )
                if TRACKS[track]["allow_port"]:
                    ablation_rows.append(
                        _bootstrap_ablation(
                            cv_predictions,
                            track,
                            target_name,
                            "HGB_CALENDAR_TIR_WEATHER_EVENTS",
                            "HGB_CALENDAR_TIR_WEATHER_EVENTS_PORT",
                            "PORT_HISTORY",
                            port_variation,
                        )
                    )
        ablation = pd.DataFrame(ablation_rows)
        _update_progress(
            run_id,
            {
                "stage": "REPORTING_AND_UPLOAD",
                "completed_ablation_rows": len(ablation),
                "updated_at": pd.Timestamp.now(tz="UTC"),
            },
        )

        final_selection_rows = []
        stable_official_models = 0
        for selection in cv_summary.to_dict("records"):
            part = test_metrics[
                (test_metrics["track"] == selection["track"])
                & (test_metrics["target_name"] == selection["target_name"])
            ]
            selected_row = part[part["model"] == selection["selected_model"]]
            baseline_rows = part[part["model"].isin(BASELINE_NAMES)]
            if selected_row.empty or baseline_rows.empty:
                test_uplift = np.nan
                stable = False
            else:
                selected_test_mae = float(selected_row.iloc[0]["MAE"])
                baseline_test_mae = float(baseline_rows["MAE"].min())
                test_uplift = 100 * (baseline_test_mae - selected_test_mae) / max(
                    baseline_test_mae, 1e-12
                )
                stable = bool(
                    selection["selected_model"] not in BASELINE_NAMES
                    and selection["cv_uplift_vs_best_baseline_pct"] >= 2.0
                    and test_uplift >= -2.0
                )
            final_selection_rows.append(
                {
                    **selection,
                    "test_uplift_vs_best_baseline_pct": test_uplift,
                    "stable_ml_uplift": stable,
                }
            )
            if selection["track"] == "FULL_NO_PORT" and stable:
                stable_official_models += 1
        final_selection = pd.DataFrame(final_selection_rows)

        target_stability = _target_stability(frame)
        if leakage_violations:
            decision_status = "NEED_DATA_REPAIR"
        elif stable_official_models >= 2:
            decision_status = "READY_FOR_EVENT_AWARE_MVP"
        else:
            decision_status = "BASELINES_ONLY_NO_STABLE_ML_UPLIFT"
        official = final_selection[final_selection["track"] == "FULL_NO_PORT"]
        decision = {
            "status": decision_status,
            "training_version": TRAINING_VERSION,
            "source_rows": len(frame),
            "source_first_date": frame["prediction_date"].min(),
            "source_last_date": frame["prediction_date"].max(),
            "targets": list(TARGETS),
            "features": int(sum(map(len, groups.values()))),
            "official_track": "FULL_NO_PORT",
            "official_protocol": "WALK_FORWARD_2023_2024_2025_TEST_2026_PURGED_7D",
            "port_track_policy": "PRE_BREAK_HISTORICAL_DIAGNOSTIC_ONLY",
            "selection_used_test": False,
            "test_is_diagnostic_only": True,
            "daily_rolling_origin": True,
            "purge_days": PURGE_DAYS,
            "critical_leakage_violations": leakage_violations,
            "gates_passed": leakage_violations == 0,
            "stable_official_models": stable_official_models,
            "selected_models": {
                row.target_name: row.selected_model for row in official.itertuples()
            },
            "cv_uplift_vs_best_baseline_pct": {
                row.target_name: row.cv_uplift_vs_best_baseline_pct
                for row in official.itertuples()
            },
            "test_uplift_vs_best_baseline_pct": {
                row.target_name: row.test_uplift_vs_best_baseline_pct
                for row in official.itertuples()
            },
            "weather_ablation": {
                row.target_name: row.decision
                for row in ablation[
                    (ablation["track"] == "FULL_NO_PORT")
                    & (ablation["family"] == "WEATHER_HISTORY")
                ].itertuples()
            },
            "known_event_ablation": {
                row.target_name: row.decision
                for row in ablation[
                    (ablation["track"] == "FULL_NO_PORT")
                    & (ablation["family"] == "KNOWN_EVENTS")
                ].itertuples()
            },
            "upstream_status": upstream.get("status"),
            "training_executed": True,
            "bronze_modified": False,
            "next_block": (
                "B57D_PROBABILISTIC_OPERATIONAL_FORECAST_API"
                if decision_status == "READY_FOR_EVENT_AWARE_MVP"
                else (
                    "B57C_FEATURE_OR_TARGET_REPAIR"
                    if decision_status == "NEED_DATA_REPAIR"
                    else "B57C_BASELINE_OPERATIONALIZATION_AND_DATA_COLLECTION"
                )
            ),
        }

        with tempfile.TemporaryDirectory(prefix="b57c-") as temporary:
            output_dir = Path(temporary)
            feature_contract.to_csv(output_dir / "00_feature_contract.csv", index=False)
            leakage_report.to_csv(output_dir / "01_anti_leakage_audit.csv", index=False)
            split_audit.to_csv(output_dir / "02_walk_forward_split_audit.csv", index=False)
            cv_metrics.to_csv(output_dir / "03_cv_fold_metrics.csv", index=False)
            cv_summary.to_csv(output_dir / "04_cv_model_summary.csv", index=False)
            test_metrics.to_csv(output_dir / "05_final_test_metrics.csv", index=False)
            final_selection.to_csv(output_dir / "06_model_selection.csv", index=False)
            ablation.to_csv(output_dir / "07_feature_family_ablation_bootstrap.csv", index=False)
            target_stability.to_csv(output_dir / "08_target_stability_by_year.csv", index=False)
            feature_usage.to_csv(output_dir / "09_feature_usage_and_zero_variance.csv", index=False)
            cv_predictions.to_parquet(output_dir / "cv_predictions.parquet", index=False)
            test_predictions.to_parquet(output_dir / "test_predictions.parquet", index=False)
            split_audit.to_parquet(output_dir / "temporal_split_audit.parquet", index=False)
            decision_path = output_dir / "b57c_decision.json"
            decision_path.write_text(
                json.dumps(_clean_json(decision), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            for name, payload in final_models.items():
                safe_name = name.lower().replace("__", "-")
                with (output_dir / f"{safe_name}.pkl").open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            (output_dir / "README_B57C.md").write_text(
                "\n".join(
                    [
                        "# B57C Event-aware temporal baselines",
                        "",
                        f"Decision: **{decision_status}**",
                        "",
                        "- Official track: FULL_NO_PORT (usable after the May 2025 port-source break).",
                        "- Selection: walk-forward 2023/2024/2025 only; 2026 is diagnostic test only.",
                        "- Seven-day purge before every validation and test boundary.",
                        "- PRE_BREAK_WITH_PORT is historical diagnosis, never the current production model.",
                        "- Event flags with no training variation are reported as not testable.",
                        "- Seasonal baselines are daily rolling-origin forecasts using only prior outcomes.",
                    ]
                ),
                encoding="utf-8",
            )

            mlflow_status = _log_mlflow(
                output_dir, decision, final_selection, test_metrics
            )
            client = _s3_client()
            uploaded = {}
            for path in sorted(output_dir.iterdir()):
                if path.name in {
                    "cv_predictions.parquet",
                    "test_predictions.parquet",
                }:
                    key = f"predictions/b57c/{output_prefix}/{path.name}"
                elif path.name == "temporal_split_audit.parquet":
                    key = f"datasets/b57c/{output_prefix}/{path.name}"
                elif path.suffix == ".pkl":
                    key = f"models/b57c/{output_prefix}/{path.name}"
                elif path == decision_path:
                    key = f"configs/b57c/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b57c/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload(client, path, output_bucket, key)

        metadata = {
            **decision,
            "checksum": checksum,
            "mlflow_status": mlflow_status,
            "outputs": uploaded,
            "output_prefix": f"s3://{output_bucket}/reports/b57c/{output_prefix}/",
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
