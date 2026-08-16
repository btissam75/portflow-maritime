from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_values

from model_trainer.arrival_flow_baselines import (
    HORIZONS,
    _clean_json,
    _db_connection,
    _json_default,
    _metrics,
    _s3_client,
    _upload,
)


HYBRID_VERSION = "b56g-v2-b56e-adaptive-conformal-v1"
SOURCE_NAME = "b56g_v2_hybrid_calibration"
DATASET_NAME = "port_arrival_flow_b56e_adaptive_intervals"
UPSTREAM_VERSION = "b56e-arrival-flow-probabilistic-ensemble-v1"
UPSTREAM_SOURCE_NAME = "b56e_arrival_flow_probabilistic_ensemble"
UPSTREAM_DATASET_NAME = "port_arrival_flow_probabilistic_6h_12h_24h"

VALID_KEY = (
    "predictions/b56e/version=1/valid_probabilistic_predictions.parquet"
)
TEST_KEY = "predictions/b56e/version=1/test_probabilistic_predictions.parquet"
SERVING_TABLE = "serving.maritime_arrival_flow_hybrid_backtest_v2"

EXPECTED_COVERAGE = 0.80
COVERAGE_MIN = 0.77
COVERAGE_MAX = 0.83
CALIBRATION_WINDOWS_DAYS = (30, 60, 90)
CALIBRATION_GAMMAS = (0.001, 0.005, 0.01)
MIN_HISTORY_HOURS = 7 * 24
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
        raise RuntimeError("B56E result is missing")
    status, checksum, metadata = row
    metadata = dict(metadata or {})
    if status != "SUCCESS":
        raise RuntimeError(f"Latest B56E run is {status}")
    if metadata.get("ensemble_version") != UPSTREAM_VERSION:
        raise RuntimeError("Latest B56E result has an unexpected version")
    if metadata.get("selection_used_test") is not False:
        raise RuntimeError("B56E used TEST for model selection")
    metadata["upstream_checksum"] = checksum
    return metadata


def _load_parquet(bucket: str, key: str) -> pd.DataFrame:
    body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def _source_checksum(bucket: str, upstream: dict[str, Any]) -> str:
    client = _s3_client()
    digest = hashlib.sha256(HYBRID_VERSION.encode("ascii"))
    digest.update(str(upstream.get("upstream_checksum", "")).encode("ascii"))
    for key in (VALID_KEY, TEST_KEY):
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
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _start_run(checksum: str, bucket: str) -> str:
    metadata = {
        "hybrid_version": HYBRID_VERSION,
        "upstream_version": UPSTREAM_VERSION,
        "training_executed": False,
        "policy": (
            "PRESERVE_B56E_POINT_VALID_ONLY_CALIBRATION_"
            "MATURED_LABELS_TEST_PROSPECTIVE_SHADOW"
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
                    f"s3://{bucket}/{VALID_KEY}",
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


def _prepare(
    valid_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "as_of_time",
        "split",
        "horizon_h",
        "target",
        "actual",
        "selected_model",
        "point_prediction",
        "p10",
        "p50",
        "p90",
        "calibration_history_ready",
    }
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for expected_split, raw in (("VALID", valid_raw), ("TEST", test_raw)):
        missing = sorted(required - set(raw.columns))
        rows.append(
            {
                "check": f"{expected_split.lower()}_required_columns",
                "value": ",".join(missing) or "NONE",
                "passed": not missing,
            }
        )
        if missing:
            continue
        frame = raw.copy()
        frame["as_of_time"] = pd.to_datetime(frame["as_of_time"], utc=True)
        frame["horizon_h"] = pd.to_numeric(
            frame["horizon_h"], errors="coerce"
        ).astype("Int64")
        frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
        for column in ("point_prediction", "p10", "p50", "p90"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["split"] = frame["split"].astype(str)
        split_ok = set(frame["split"].unique()) == {expected_split}
        duplicate_rows = int(
            frame.duplicated(["as_of_time", "horizon_h"], keep=False).sum()
        )
        horizons = sorted(frame["horizon_h"].dropna().astype(int).unique())
        expected_rows = int(frame["as_of_time"].nunique()) * len(HORIZONS)
        point_finite = bool(np.isfinite(frame["point_prediction"]).all())
        target_finite = bool(np.isfinite(frame["actual"]).all())
        rows.extend(
            [
                {
                    "check": f"{expected_split.lower()}_split_label",
                    "value": split_ok,
                    "passed": split_ok,
                },
                {
                    "check": f"{expected_split.lower()}_unique_grain",
                    "value": duplicate_rows,
                    "passed": duplicate_rows == 0,
                },
                {
                    "check": f"{expected_split.lower()}_horizons",
                    "value": ",".join(map(str, horizons)),
                    "passed": horizons == list(HORIZONS),
                },
                {
                    "check": f"{expected_split.lower()}_complete_grid",
                    "value": f"{len(frame)}/{expected_rows}",
                    "passed": len(frame) == expected_rows,
                },
                {
                    "check": f"{expected_split.lower()}_finite_point",
                    "value": point_finite,
                    "passed": point_finite,
                },
                {
                    "check": f"{expected_split.lower()}_finite_target",
                    "value": target_finite,
                    "passed": target_finite,
                },
            ]
        )
        frames.append(frame)
    if len(frames) != 2:
        raise RuntimeError("B56E prediction schema is incomplete")
    combined = pd.concat(frames, ignore_index=True)
    temporal_order = bool(
        combined.loc[combined["split"].eq("VALID"), "as_of_time"].max()
        < combined.loc[combined["split"].eq("TEST"), "as_of_time"].min()
    )
    rows.extend(
        [
            {
                "check": "strict_valid_before_test",
                "value": temporal_order,
                "passed": temporal_order,
            },
            {
                "check": "point_source_frozen_b56e",
                "value": UPSTREAM_VERSION,
                "passed": True,
            },
            {
                "check": "test_not_used_for_hyperparameter_selection",
                "value": "VALID_ONLY",
                "passed": True,
            },
            {
                "check": "test_labels_used_only_after_maturity",
                "value": "SHIFT_BY_HORIZON_H",
                "passed": True,
            },
        ]
    )
    checks = pd.DataFrame(rows)
    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"B56G-v2 integrity checks failed: {failed}")
    combined = combined.rename(
        columns={
            "point_prediction": "source_point_prediction",
            "p10": "base_p10",
            "p50": "base_p50",
            "p90": "base_p90",
            "selected_model": "source_selected_model",
        }
    ).sort_values(["as_of_time", "horizon_h"])
    return combined.reset_index(drop=True), checks


def _pinball(
    actual: np.ndarray,
    prediction: np.ndarray,
    quantile: float,
) -> float:
    error = actual - prediction
    return float(
        np.mean(np.maximum(quantile * error, (quantile - 1.0) * error))
    )


def _interval_metric_row(
    frame: pd.DataFrame,
    split: str,
    horizon: int,
    prefix: str,
    ready_column: str,
) -> dict[str, Any]:
    part = frame.loc[
        frame["split"].eq(split)
        & frame["horizon_h"].eq(horizon)
        & frame[ready_column].astype(bool)
    ].copy()
    low_column = f"{prefix}p10"
    mid_column = f"{prefix}p50"
    high_column = f"{prefix}p90"
    part = part.dropna(
        subset=["actual", low_column, mid_column, high_column]
    )
    actual = part["actual"].to_numpy(dtype="float64")
    low = part[low_column].to_numpy(dtype="float64")
    mid = part[mid_column].to_numpy(dtype="float64")
    high = part[high_column].to_numpy(dtype="float64")
    covered = (actual >= low) & (actual <= high)
    alpha = 1.0 - EXPECTED_COVERAGE
    interval_score = (
        high
        - low
        + (2.0 / alpha) * (low - actual) * (actual < low)
        + (2.0 / alpha) * (actual - high) * (actual > high)
    )
    coverage = float(np.mean(covered))
    return {
        "split": split,
        "horizon_h": horizon,
        "n": len(part),
        "coverage_p10_p90": coverage,
        "coverage_target": EXPECTED_COVERAGE,
        "coverage_gap_abs": abs(coverage - EXPECTED_COVERAGE),
        "mean_interval_width": float(np.mean(high - low)),
        "median_interval_width": float(np.median(high - low)),
        "winkler_interval_score": float(np.mean(interval_score)),
        "pinball_p10": _pinball(actual, low, 0.10),
        "pinball_p50": _pinball(actual, mid, 0.50),
        "pinball_p90": _pinball(actual, high, 0.90),
        "below_p10_rate": float(np.mean(actual < low)),
        "above_p90_rate": float(np.mean(actual > high)),
        "coverage_gate_passed": COVERAGE_MIN <= coverage <= COVERAGE_MAX,
    }


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
    source_point = result["source_point_prediction"].to_numpy(dtype="float64")
    raw_score = np.maximum(base_low - actual, actual - base_high)

    low = np.full(len(result), np.nan, dtype="float64")
    mid = np.full(len(result), np.nan, dtype="float64")
    high = np.full(len(result), np.nan, dtype="float64")
    ready = np.zeros(len(result), dtype=bool)
    alpha_used = np.full(len(result), np.nan, dtype="float64")
    correction_used = np.full(len(result), np.nan, dtype="float64")
    target_alpha = 1.0 - EXPECTED_COVERAGE
    adaptive_alpha = target_alpha
    matured_scores: list[float] = []
    window_hours = window_days * 24

    for position in range(len(result)):
        matured_position = position - horizon
        if (
            matured_position >= 0
            and np.isfinite(raw_score[matured_position])
            and np.isfinite(low[matured_position])
            and np.isfinite(high[matured_position])
        ):
            matured_scores.append(float(raw_score[matured_position]))
            missed = float(
                actual[matured_position] < low[matured_position]
                or actual[matured_position] > high[matured_position]
            )
            adaptive_alpha = float(
                np.clip(
                    adaptive_alpha + gamma * (target_alpha - missed),
                    0.02,
                    0.40,
                )
            )
        finite_base = bool(
            np.isfinite(base_low[position])
            and np.isfinite(base_high[position])
        )
        history = matured_scores[-window_hours:]
        if finite_base and len(history) >= MIN_HISTORY_HOURS:
            correction = float(
                np.quantile(history, 1.0 - adaptive_alpha)
            )
            minimum = -0.45 * max(
                base_high[position] - base_low[position], 0.0
            )
            correction = max(correction, minimum)
            ready[position] = True
        elif finite_base:
            correction = 0.0
        else:
            continue
        low[position] = max(0.0, base_low[position] - correction)
        high[position] = max(low[position], base_high[position] + correction)
        candidate_mid = (
            base_mid[position]
            if np.isfinite(base_mid[position])
            else source_point[position]
        )
        mid[position] = float(
            np.clip(candidate_mid, low[position], high[position])
        )
        alpha_used[position] = adaptive_alpha
        correction_used[position] = correction

    result["point_prediction"] = source_point
    result["p10"] = low
    result["p50"] = mid
    result["p90"] = high
    result["calibration_history_ready"] = ready
    result["adaptive_alpha"] = alpha_used
    result["conformal_correction"] = correction_used
    result["calibration_window_days"] = window_days
    result["calibration_gamma"] = gamma
    result["selected_model"] = (
        "B56E_POINT_PLUS_ADAPTIVE_CONFORMAL_INTERVALS"
    )
    return result


def _select_calibration(
    source: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    valid = source.loc[source["split"].eq("VALID")].copy()
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        horizon_rows: list[dict[str, Any]] = []
        for window_days in CALIBRATION_WINDOWS_DAYS:
            for gamma in CALIBRATION_GAMMAS:
                calibrated = _adaptive_cqr(
                    valid, horizon, window_days, gamma
                )
                metric = _interval_metric_row(
                    calibrated,
                    "VALID",
                    horizon,
                    prefix="",
                    ready_column="calibration_history_ready",
                )
                metric.update({"window_days": window_days, "gamma": gamma})
                rows.append(metric)
                horizon_rows.append(metric)
        search = pd.DataFrame(horizon_rows)
        within = search.loc[
            search["coverage_p10_p90"].between(
                EXPECTED_COVERAGE - 0.02,
                EXPECTED_COVERAGE + 0.02,
            )
        ]
        candidates = within if not within.empty else search
        winner = candidates.sort_values(
            [
                "coverage_gap_abs",
                "winkler_interval_score",
                "mean_interval_width",
            ]
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


def _apply_calibration(
    source: pd.DataFrame,
    selected: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        config = selected[str(horizon)]
        parts.append(
            _adaptive_cqr(
                source,
                horizon,
                int(config["window_days"]),
                float(config["gamma"]),
            )
        )
    return pd.concat(parts, ignore_index=True)


def _monotone_finite(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    previous = np.nan
    for index, value in enumerate(output):
        if not np.isfinite(value):
            continue
        if np.isfinite(previous):
            output[index] = max(previous, value)
        previous = output[index]
    return output


def _reconcile(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("as_of_time", sort=False):
        part = group.sort_values("horizon_h").copy()
        part["point_prediction"] = _monotone_finite(
            part["point_prediction"].to_numpy(dtype="float64")
        )
        for column in ("p10", "p50", "p90"):
            part[column] = _monotone_finite(
                part[column].to_numpy(dtype="float64")
            )
        finite = part[["p10", "p50", "p90"]].notna().all(axis=1)
        part.loc[finite, "p50"] = np.maximum(
            part.loc[finite, "p50"], part.loc[finite, "p10"]
        )
        part.loc[finite, "p90"] = np.maximum(
            part.loc[finite, "p90"], part.loc[finite, "p50"]
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["as_of_time", "horizon_h"]
    )


def _point_fidelity(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("VALID", "TEST"):
        for horizon in HORIZONS:
            part = frame.loc[
                frame["split"].eq(split)
                & frame["horizon_h"].eq(horizon)
            ]
            source = part["source_point_prediction"].to_numpy(
                dtype="float64"
            )
            hybrid = part["point_prediction"].to_numpy(dtype="float64")
            actual = part["actual"].to_numpy(dtype="float64")
            source_metrics = _metrics(actual, source)
            hybrid_metrics = _metrics(actual, hybrid)
            rows.append(
                {
                    "split": split,
                    "horizon_h": horizon,
                    "n": len(part),
                    "source_b56e_mae": source_metrics["MAE"],
                    "hybrid_mae": hybrid_metrics["MAE"],
                    "mae_delta": (
                        hybrid_metrics["MAE"] - source_metrics["MAE"]
                    ),
                    "source_b56e_rmse": source_metrics["RMSE"],
                    "hybrid_rmse": hybrid_metrics["RMSE"],
                    "adjusted_point_rows": int(
                        (np.abs(hybrid - source) > 1e-12).sum()
                    ),
                    "max_point_adjustment": float(
                        np.max(np.abs(hybrid - source))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _interval_metrics(
    frame: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    if source == "BASE_B56E":
        prefix = "base_"
        ready_column = "calibration_history_ready_source"
    else:
        prefix = ""
        ready_column = "calibration_history_ready"
    return pd.DataFrame(
        [
            {
                "interval_source": source,
                **_interval_metric_row(
                    frame,
                    split,
                    horizon,
                    prefix,
                    ready_column,
                ),
            }
            for split in ("VALID", "TEST")
            for horizon in HORIZONS
        ]
    )


def _coherence_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ready = frame.loc[frame["calibration_history_ready"]].copy()
    for column in ("point_prediction", "p10", "p50", "p90"):
        pivot = ready.pivot(
            index="as_of_time",
            columns="horizon_h",
            values=column,
        ).dropna()
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
            (ready["p10"] > ready["p50"])
            | (ready["p50"] > ready["p90"])
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
        (ready[["point_prediction", "p10", "p50", "p90"]] < 0)
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


def _paired_interval_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    alpha = 1.0 - EXPECTED_COVERAGE
    for horizon in HORIZONS:
        part = frame.loc[
            frame["split"].eq("TEST")
            & frame["horizon_h"].eq(horizon)
            & frame["calibration_history_ready"]
        ].dropna(subset=["base_p10", "base_p90", "p10", "p90"])
        actual = part["actual"].to_numpy(dtype="float64")

        def score(low: np.ndarray, high: np.ndarray) -> np.ndarray:
            return (
                high
                - low
                + (2.0 / alpha) * (low - actual) * (actual < low)
                + (2.0 / alpha) * (actual - high) * (actual > high)
            )

        base_score = score(
            part["base_p10"].to_numpy(dtype="float64"),
            part["base_p90"].to_numpy(dtype="float64"),
        )
        hybrid_score = score(
            part["p10"].to_numpy(dtype="float64"),
            part["p90"].to_numpy(dtype="float64"),
        )
        daily = pd.DataFrame(
            {
                "day": part["as_of_time"].dt.floor("D"),
                "delta": hybrid_score - base_score,
            }
        ).groupby("day", observed=True).agg(
            delta_sum=("delta", "sum"),
            count=("delta", "size"),
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
        rows.append(
            {
                "horizon_h": horizon,
                "n": len(part),
                "days": len(daily),
                "base_mean_interval_score": float(np.mean(base_score)),
                "hybrid_mean_interval_score": float(
                    np.mean(hybrid_score)
                ),
                "interval_score_gain_pct": 100.0
                * (float(np.mean(base_score)) - float(np.mean(hybrid_score)))
                / max(float(np.mean(base_score)), 1e-12),
                "delta_score_ci95_low": float(low),
                "delta_score_ci95_high": float(high),
                "significantly_better": bool(high < 0.0),
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
                (HYBRID_VERSION,),
            )
            values = [
                (
                    row.as_of_time.to_pydatetime(),
                    int(row.horizon_h),
                    HYBRID_VERSION,
                    str(row.selected_model),
                    float(row.actual),
                    float(row.point_prediction),
                    float(row.p10),
                    float(row.p50),
                    float(row.p90),
                    float(row.adaptive_alpha),
                    float(row.conformal_correction),
                    "RETROSPECTIVE_SHADOW",
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
    point_fidelity: pd.DataFrame,
    interval_metrics: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment("maritime-arrival-flow-hybrid-calibration")
        with mlflow.start_run(run_name=HYBRID_VERSION):
            mlflow.log_params(
                {
                    "hybrid_version": HYBRID_VERSION,
                    "point_source": UPSTREAM_VERSION,
                    "target_coverage": EXPECTED_COVERAGE,
                    "selection_split": "VALID",
                    "test_role": "RETROSPECTIVE_SHADOW",
                }
            )
            metrics: dict[str, float] = {}
            for row in point_fidelity.loc[
                point_fidelity["split"].eq("TEST")
            ].itertuples(index=False):
                metrics[f"test_mae_{int(row.horizon_h)}h"] = float(
                    row.hybrid_mae
                )
            for row in interval_metrics.loc[
                interval_metrics["split"].eq("TEST")
                & interval_metrics["interval_source"].eq("HYBRID_V2")
            ].itertuples(index=False):
                metrics[f"test_coverage_{int(row.horizon_h)}h"] = float(
                    row.coverage_p10_p90
                )
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(output_dir))
        return "LOGGED"
    except Exception as exc:
        return f"SKIPPED:{type(exc).__name__}"


def _write_readme(path: Path, decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# B56G-v2 hybrid calibration",
                "",
                f"Decision: {decision['status']}",
                "",
                "- B56E remains the point-forecast source.",
                "- Hyperparameters are selected on VALID only.",
                "- TEST labels enter adaptation only after horizon maturity.",
                "- Existing TEST is retrospective evidence, not a new untouched holdout.",
                "- Formal promotion requires future prospective shadow data.",
                "- P10/P50/P90 and 6h/12h/24h are reconciled.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_b56g_v2_hybrid_calibration(
    source_bucket: str = "gold-maritime",
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=2",
    materialize_timescale: bool = True,
    force: bool = False,
) -> dict[str, Any]:
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
        valid_raw = _load_parquet(source_bucket, VALID_KEY)
        test_raw = _load_parquet(source_bucket, TEST_KEY)
        source, integrity = _prepare(valid_raw, test_raw)
        source["calibration_history_ready_source"] = (
            source["calibration_history_ready"].astype(bool)
        )

        selected, search = _select_calibration(source)
        calibrated = _apply_calibration(source, selected)
        calibrated = _reconcile(calibrated)

        point_fidelity = _point_fidelity(calibrated)
        base_intervals = _interval_metrics(calibrated, "BASE_B56E")
        hybrid_intervals = _interval_metrics(calibrated, "HYBRID_V2")
        interval_metrics = pd.concat(
            [base_intervals, hybrid_intervals], ignore_index=True
        )
        coherence = _coherence_audit(calibrated)
        bootstrap = _paired_interval_bootstrap(calibrated)

        test_hybrid = hybrid_intervals.loc[
            hybrid_intervals["split"].eq("TEST")
        ]
        test_fidelity = point_fidelity.loc[
            point_fidelity["split"].eq("TEST")
        ]
        integrity_passed = bool(integrity["passed"].all())
        coherence_passed = bool(coherence["passed"].all())
        coverage_passed = bool(
            test_hybrid["coverage_gate_passed"].all()
        )
        point_preserved = bool(
            (test_fidelity["adjusted_point_rows"] == 0).all()
            and (test_fidelity["mae_delta"].abs() <= 1e-12).all()
        )
        if not integrity_passed:
            decision_status = "NEED_DATA_REPAIR"
        elif not point_preserved:
            decision_status = "NEED_POINT_FIDELITY_REPAIR"
        elif not coherence_passed:
            decision_status = "NEED_RECONCILIATION_REPAIR"
        elif not coverage_passed:
            decision_status = "NEED_INTERVAL_RECALIBRATION"
        else:
            decision_status = "READY_FOR_PROSPECTIVE_SHADOW"

        with tempfile.TemporaryDirectory(prefix="b56g-v2-") as temporary:
            output_dir = Path(temporary)
            reports_dir = output_dir / "reports"
            configs_dir = output_dir / "configs"
            predictions_dir = output_dir / "predictions"
            for directory in (reports_dir, configs_dir, predictions_dir):
                directory.mkdir(parents=True)

            reports = {
                "00_integrity_and_temporal_contract.csv": integrity,
                "01_valid_calibration_search.csv": search,
                "02_point_fidelity.csv": point_fidelity,
                "03_interval_metrics_before_after.csv": interval_metrics,
                "04_interval_score_paired_bootstrap.csv": bootstrap,
                "05_coherence_audit.csv": coherence,
            }
            for name, report in reports.items():
                report.to_csv(reports_dir / name, index=False)

            valid_output = calibrated.loc[
                calibrated["split"].eq("VALID")
            ].copy()
            test_output = calibrated.loc[
                calibrated["split"].eq("TEST")
            ].copy()
            valid_output.to_parquet(
                predictions_dir / "valid_hybrid_predictions.parquet",
                index=False,
            )
            test_output.to_parquet(
                predictions_dir / "test_hybrid_predictions.parquet",
                index=False,
            )

            decision = {
                "status": decision_status,
                "hybrid_version": HYBRID_VERSION,
                "objective": "B56E_POINT_WITH_CALIBRATED_COHERENT_INTERVALS",
                "point_source": UPSTREAM_VERSION,
                "point_model_policy": "PRESERVE_B56E_POINT",
                "selected_calibration": selected,
                "selection_split": "VALID",
                "test_role": "RETROSPECTIVE_SHADOW_NOT_NEW_HOLDOUT",
                "selection_used_test": False,
                "test_adaptation_policy": "LABELS_SHIFTED_BY_HORIZON_MATURITY",
                "training_executed": False,
                "integrity_gates_passed": integrity_passed,
                "point_fidelity_passed": point_preserved,
                "coherence_gates_passed": coherence_passed,
                "coverage_gates_passed": coverage_passed,
                "historical_replay_allowed": bool(
                    integrity_passed
                    and point_preserved
                    and coherence_passed
                ),
                "live_serving_allowed": False,
                "formal_promotion_allowed": False,
                "formal_promotion_blocker": (
                    "EXISTING_TEST_ALREADY_INFORMED_V2_DESIGN_"
                    "REQUIRE_FUTURE_PROSPECTIVE_SHADOW"
                ),
                "next_block": (
                    "B56G_V2_PROSPECTIVE_SHADOW_MONITOR"
                    if decision_status == "READY_FOR_PROSPECTIVE_SHADOW"
                    else "B56G_V2_REPAIR"
                ),
            }
            decision_path = configs_dir / "06_b56g_v2_decision.json"
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    allow_nan=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            _write_readme(reports_dir / "README_B56G_V2.md", decision)

            timescale_rows = (
                _materialize_backtest(test_output, run_id)
                if materialize_timescale
                else 0
            )
            mlflow_status = _log_mlflow(
                output_dir,
                decision,
                point_fidelity,
                interval_metrics,
            )

            client = _s3_client()
            outputs: dict[str, str] = {}
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(output_dir).as_posix()
                category, name = relative.split("/", 1)
                key = (
                    f"{category}/b56gv2/{output_prefix.strip('/')}/{name}"
                )
                outputs[name] = _upload(
                    client, path, output_bucket, key
                )

        decision.update(
            {
                "row_count": len(calibrated),
                "valid_rows": len(valid_output),
                "test_rows": len(test_output),
                "timescale_table": (
                    SERVING_TABLE if materialize_timescale else None
                ),
                "timescale_rows": timescale_rows,
                "mlflow_status": mlflow_status,
                "outputs": outputs,
                "checksum": checksum,
            }
        )
        _finish_run(run_id, "SUCCESS", len(calibrated), decision)
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
            {"hybrid_version": HYBRID_VERSION},
            error_message=str(exc),
        )
        raise
