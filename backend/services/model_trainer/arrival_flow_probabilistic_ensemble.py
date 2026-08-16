from __future__ import annotations

import hashlib
import json
import math
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


ENSEMBLE_VERSION = "b56e-arrival-flow-probabilistic-ensemble-v1"
SOURCE_NAME = "b56e_arrival_flow_probabilistic_ensemble"
DATASET_NAME = "port_arrival_flow_probabilistic_6h_12h_24h"
UPSTREAM_VERSION = "b56c-arrival-flow-enrichment-v1"
UPSTREAM_SOURCE_NAME = "b56c_arrival_flow_enrichment"
UPSTREAM_DATASET_NAME = "port_arrival_flow_enriched_6h_12h_24h"
SERVING_TABLE = "serving.maritime_arrival_flow_backtest_v1"

VALID_KEY = "predictions/b56c/version=1/valid_predictions.parquet"
TEST_KEY = "predictions/b56c/version=1/test_predictions.parquet"

OFFICIAL_POINT_CANDIDATES = (
    "RECENT_24H_RATE",
    "SEASONAL_NAIVE_24H",
    "SEASONAL_NAIVE_168H",
    "HGB_LEGACY_CORE",
)
DIAGNOSTIC_CANDIDATES = ("HGB_ENRICHED_HISTORY_WAVE",)
ADAPTIVE_POLICIES = (
    "ONLINE_BEST_30D",
    "ONLINE_INVERSE_MAE_30D",
)

WEIGHT_WINDOW_HOURS = 30 * 24
INTERVAL_WINDOW_HOURS = 90 * 24
MIN_HISTORY_HOURS = 7 * 24
EXPECTED_COVERAGE = 0.80
COVERAGE_MIN = 0.74
COVERAGE_MAX = 0.92
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 42
SOURCE_FRESHNESS_LIMIT_HOURS = 48


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
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(body)
        handle.flush()
        return pd.read_parquet(handle.name)


def _source_checksum(bucket: str, upstream: dict[str, Any]) -> str:
    client = _s3_client()
    digest = hashlib.sha256(ENSEMBLE_VERSION.encode("ascii"))
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
        "ensemble_version": ENSEMBLE_VERSION,
        "upstream_version": UPSTREAM_VERSION,
        "policy": (
            "REUSE_B56C_PREDICTIONS_VALID_SELECTION_TEST_AUDIT_"
            "PAST_ONLY_ONLINE_ADAPTATION_NO_RETRAINING"
        ),
        "training_executed": False,
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


def _validate_predictions(
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "as_of_time",
        "split",
        "horizon_h",
        "target",
        "actual",
        *OFFICIAL_POINT_CANDIDATES,
        *DIAGNOSTIC_CANDIDATES,
    }
    rows: list[dict[str, Any]] = []
    prepared: list[pd.DataFrame] = []
    for expected_split, source in (("VALID", valid), ("TEST", test)):
        missing = sorted(required - set(source.columns))
        if missing:
            raise RuntimeError(
                f"{expected_split} predictions miss columns: {missing}"
            )
        frame = source.copy()
        frame["as_of_time"] = pd.to_datetime(frame["as_of_time"], utc=True)
        frame["horizon_h"] = pd.to_numeric(
            frame["horizon_h"], errors="raise"
        ).astype("int64")
        if set(frame["split"].dropna().unique()) != {expected_split}:
            raise RuntimeError(f"{expected_split} artifact has mixed split labels")
        if set(frame["horizon_h"].unique()) != set(HORIZONS):
            raise RuntimeError(f"{expected_split} artifact has unexpected horizons")
        duplicates = int(
            frame.duplicated(["as_of_time", "horizon_h"], keep=False).sum()
        )
        if duplicates:
            raise RuntimeError(
                f"{expected_split} has {duplicates} duplicate horizon rows"
            )
        for column in (
            "actual",
            *OFFICIAL_POINT_CANDIDATES,
            *DIAGNOSTIC_CANDIDATES,
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        null_official = int(
            frame[["actual", *OFFICIAL_POINT_CANDIDATES]].isna().sum().sum()
        )
        if null_official:
            raise RuntimeError(
                f"{expected_split} has {null_official} null official predictions"
            )
        negative = int(
            (
                frame[["actual", *OFFICIAL_POINT_CANDIDATES]] < 0
            ).sum().sum()
        )
        if negative:
            raise RuntimeError(
                f"{expected_split} has {negative} negative count values"
            )
        rows.append(
            {
                "check": f"{expected_split}_SCHEMA_GRAIN_VALUES",
                "passed": True,
                "rows": len(frame),
                "duplicates": duplicates,
                "null_official_values": null_official,
                "negative_values": negative,
                "first_time": frame["as_of_time"].min(),
                "last_time": frame["as_of_time"].max(),
            }
        )
        prepared.append(
            frame.sort_values(["horizon_h", "as_of_time"]).reset_index(drop=True)
        )

    valid_frame, test_frame = prepared
    valid_max = valid_frame["as_of_time"].max()
    test_min = test_frame["as_of_time"].min()
    separation_hours = (test_min - valid_max).total_seconds() / 3600
    temporal_passed = separation_hours > max(HORIZONS)
    rows.append(
        {
            "check": "VALID_TEST_PURGE_STRICTLY_GREATER_THAN_24H",
            "passed": temporal_passed,
            "rows": None,
            "duplicates": 0,
            "null_official_values": 0,
            "negative_values": 0,
            "first_time": valid_max,
            "last_time": test_min,
            "separation_hours": separation_hours,
        }
    )
    if not temporal_passed:
        raise RuntimeError("VALID and TEST do not respect the 24-hour purge")
    return valid_frame, test_frame, pd.DataFrame(rows)


def _online_predictions(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    result = frame.copy().sort_values("as_of_time").reset_index(drop=True)
    rolling_errors: dict[str, pd.Series] = {}
    for candidate in OFFICIAL_POINT_CANDIDATES:
        absolute_error = (result["actual"] - result[candidate]).abs()
        rolling_errors[candidate] = (
            absolute_error.shift(horizon)
            .rolling(
                WEIGHT_WINDOW_HOURS,
                min_periods=MIN_HISTORY_HOURS,
            )
            .mean()
        )

    error_frame = pd.DataFrame(rolling_errors, index=result.index)
    ready = error_frame.notna().all(axis=1)
    fallback = result[list(OFFICIAL_POINT_CANDIDATES)].mean(axis=1)

    best_name = error_frame.fillna(np.inf).idxmin(axis=1)
    best_prediction = np.empty(len(result), dtype="float64")
    candidate_values = result[list(OFFICIAL_POINT_CANDIDATES)]
    for position, name in enumerate(best_name):
        best_prediction[position] = (
            fallback.iloc[position]
            if not ready.iloc[position]
            else candidate_values.iloc[position][name]
        )

    inverse = 1.0 / error_frame.clip(lower=0.05)
    inverse = inverse.div(inverse.sum(axis=1), axis=0)
    inverse_prediction = (inverse * candidate_values).sum(axis=1)
    inverse_prediction = inverse_prediction.where(ready, fallback)

    result["ONLINE_BEST_30D"] = np.clip(best_prediction, 0.0, None)
    result["ONLINE_INVERSE_MAE_30D"] = np.clip(
        inverse_prediction.to_numpy(dtype="float64"),
        0.0,
        None,
    )
    result["adaptive_history_ready"] = ready
    for candidate in OFFICIAL_POINT_CANDIDATES:
        result[f"weight_{candidate}"] = inverse[candidate].where(ready)
    result["online_best_candidate"] = best_name.where(ready, "EQUAL_FALLBACK")
    return result


def _build_adaptive_frames(
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        valid_h = valid.loc[valid["horizon_h"] == horizon].copy()
        test_h = test.loc[test["horizon_h"] == horizon].copy()
        valid_online = _online_predictions(valid_h, horizon)

        combined = pd.concat([valid_h, test_h], ignore_index=True)
        combined_online = _online_predictions(combined, horizon)
        test_online = combined_online.loc[
            combined_online["split"].eq("TEST")
        ].copy()

        valid_parts.append(valid_online)
        test_parts.append(test_online)
    return (
        pd.concat(valid_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def _point_metrics(
    frame: pd.DataFrame,
    split: str,
    selection_only: bool,
) -> pd.DataFrame:
    candidates = (
        *OFFICIAL_POINT_CANDIDATES,
        *DIAGNOSTIC_CANDIDATES,
        *ADAPTIVE_POLICIES,
    )
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        part = frame.loc[frame["horizon_h"] == horizon].copy()
        if selection_only:
            part = part.loc[part["adaptive_history_ready"]]
        actual = part["actual"].to_numpy(dtype="float64")
        for model in candidates:
            prediction = part[model].to_numpy(dtype="float64")
            finite = np.isfinite(actual) & np.isfinite(prediction)
            result = _metrics(actual[finite], prediction[finite])
            rows.append(
                {
                    "split": split,
                    "horizon_h": horizon,
                    "model": model,
                    "selection_eligible": model
                    in (*OFFICIAL_POINT_CANDIDATES, *ADAPTIVE_POLICIES),
                    "n": int(finite.sum()),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def _select_on_valid(metrics_valid: pd.DataFrame) -> dict[str, str]:
    selected: dict[str, str] = {}
    for horizon in HORIZONS:
        eligible = metrics_valid.loc[
            (metrics_valid["horizon_h"] == horizon)
            & metrics_valid["selection_eligible"]
        ].sort_values(["MAE", "RMSE", "model"])
        if eligible.empty:
            raise RuntimeError(f"No eligible strategy at horizon {horizon}")
        selected[str(horizon)] = str(eligible.iloc[0]["model"])
    return selected


def _add_probabilistic_intervals(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    selected: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_outputs: list[pd.DataFrame] = []
    test_outputs: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        model = selected[str(horizon)]
        valid_h = valid.loc[valid["horizon_h"] == horizon].copy()
        test_h = test.loc[test["horizon_h"] == horizon].copy()
        combined = pd.concat([valid_h, test_h], ignore_index=True)
        combined = combined.sort_values("as_of_time").reset_index(drop=True)
        residual = combined["actual"] - combined[model]
        matured = residual.shift(horizon)
        roller = matured.rolling(
            INTERVAL_WINDOW_HOURS,
            min_periods=MIN_HISTORY_HOURS,
        )
        q10 = roller.quantile(0.10)
        q50 = roller.quantile(0.50)
        q90 = roller.quantile(0.90)

        output = combined[
            ["as_of_time", "split", "horizon_h", "target", "actual"]
        ].copy()
        output["selected_model"] = model
        output["point_prediction"] = combined[model]
        output["p10"] = np.clip(combined[model] + q10, 0.0, None)
        output["p50"] = np.clip(combined[model] + q50, 0.0, None)
        output["p90"] = np.clip(combined[model] + q90, 0.0, None)
        output["calibration_history_ready"] = q90.notna()
        valid_outputs.append(output.loc[output["split"].eq("VALID")])
        test_outputs.append(output.loc[output["split"].eq("TEST")])
    return (
        pd.concat(valid_outputs, ignore_index=True),
        pd.concat(test_outputs, ignore_index=True),
    )


def _pinball(y: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = y - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def _probabilistic_metrics(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        part = frame.loc[
            (frame["horizon_h"] == horizon)
            & frame["calibration_history_ready"]
        ].copy()
        y = part["actual"].to_numpy(dtype="float64")
        p10 = part["p10"].to_numpy(dtype="float64")
        p50 = part["p50"].to_numpy(dtype="float64")
        p90 = part["p90"].to_numpy(dtype="float64")
        coverage = float(np.mean((y >= p10) & (y <= p90)))
        rows.append(
            {
                "split": split,
                "horizon_h": horizon,
                "selected_model": part["selected_model"].iloc[0],
                "n": len(part),
                "coverage_p10_p90": coverage,
                "coverage_target": EXPECTED_COVERAGE,
                "coverage_gate_passed": COVERAGE_MIN <= coverage <= COVERAGE_MAX,
                "mean_interval_width": float(np.mean(p90 - p10)),
                "median_interval_width": float(np.median(p90 - p10)),
                "pinball_p10": _pinball(y, p10, 0.10),
                "pinball_p50": _pinball(y, p50, 0.50),
                "pinball_p90": _pinball(y, p90, 0.90),
                "under_p90_rate": float(np.mean(y > p90)),
                "over_p10_rate": float(np.mean(y < p10)),
            }
        )
    return pd.DataFrame(rows)


def _paired_day_bootstrap(
    test: pd.DataFrame,
    selected: dict[str, str],
    metrics_valid: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    baseline_names = (
        "RECENT_24H_RATE",
        "SEASONAL_NAIVE_24H",
        "SEASONAL_NAIVE_168H",
    )
    for horizon in HORIZONS:
        valid_baselines = metrics_valid.loc[
            (metrics_valid["horizon_h"] == horizon)
            & metrics_valid["model"].isin(baseline_names)
        ].sort_values(["MAE", "RMSE", "model"])
        frozen_baseline = str(valid_baselines.iloc[0]["model"])
        selected_model = selected[str(horizon)]
        part = test.loc[test["horizon_h"] == horizon].copy()
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
            selected_sum=("selected_abs_error", "sum"),
            baseline_sum=("baseline_abs_error", "sum"),
            count=("actual", "size"),
        )
        indices = rng.integers(
            0,
            len(daily),
            size=(BOOTSTRAP_ITERATIONS, len(daily)),
        )
        delta_values = daily["delta_sum"].to_numpy(dtype="float64")
        count_values = daily["count"].to_numpy(dtype="float64")
        draws = delta_values[indices].sum(axis=1) / count_values[indices].sum(
            axis=1
        )
        selected_mae = float(part["selected_abs_error"].mean())
        baseline_mae = float(part["baseline_abs_error"].mean())
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "horizon_h": horizon,
                "selected_model": selected_model,
                "frozen_valid_baseline": frozen_baseline,
                "n": len(part),
                "days": len(daily),
                "selected_test_mae": selected_mae,
                "baseline_test_mae": baseline_mae,
                "gain_vs_baseline_pct": (
                    100
                    * (baseline_mae - selected_mae)
                    / max(baseline_mae, 1e-12)
                ),
                "delta_mae_ci95_low": float(low),
                "delta_mae_ci95_high": float(high),
                "significantly_better": bool(high < 0),
                "non_inferior": bool(high <= 0.05),
            }
        )
    return pd.DataFrame(rows)


def _weight_summary(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        part = frame.loc[
            (frame["horizon_h"] == horizon) & frame["adaptive_history_ready"]
        ]
        for candidate in OFFICIAL_POINT_CANDIDATES:
            rows.append(
                {
                    "split": split,
                    "horizon_h": horizon,
                    "candidate": candidate,
                    "mean_inverse_mae_weight": float(
                        part[f"weight_{candidate}"].mean()
                    ),
                    "online_best_selection_rate": float(
                        part["online_best_candidate"].eq(candidate).mean()
                    ),
                    "history_ready_rows": len(part),
                }
            )
    return pd.DataFrame(rows)


def _source_freshness(
    upstream: dict[str, Any],
    test: pd.DataFrame,
) -> pd.DataFrame:
    latest_eligible = pd.Timestamp(test["as_of_time"].max())
    now = pd.Timestamp(datetime.now(timezone.utc))
    age_hours = (now - latest_eligible).total_seconds() / 3600
    completeness_break = pd.to_datetime(
        upstream.get("source_completeness_break_start"),
        utc=True,
        errors="coerce",
    )
    safe_end = pd.to_datetime(
        upstream.get("safe_label_end_exclusive"),
        utc=True,
        errors="coerce",
    )
    live_allowed = bool(
        age_hours <= SOURCE_FRESHNESS_LIMIT_HOURS
        and pd.isna(completeness_break)
        and upstream.get("operational_quality_approved") is True
    )
    return pd.DataFrame(
        [
            {
                "latest_model_eligible_time": latest_eligible,
                "safe_label_end_exclusive": safe_end,
                "source_completeness_break_start": completeness_break,
                "age_hours_at_audit": age_hours,
                "freshness_limit_hours": SOURCE_FRESHNESS_LIMIT_HOURS,
                "operational_quality_approved": bool(
                    upstream.get("operational_quality_approved", False)
                ),
                "live_serving_allowed": live_allowed,
                "historical_replay_allowed": True,
                "reason": (
                    "SOURCE_COMPLETE_AND_FRESH"
                    if live_allowed
                    else "SOURCE_BREAK_OR_STALE_DATA"
                ),
            }
        ]
    )


def _materialize_backtest(
    predictions: pd.DataFrame,
    run_id: str,
) -> int:
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
                    source_mode TEXT NOT NULL,
                    ingestion_run_id UUID NOT NULL,
                    PRIMARY KEY (as_of_time, horizon_h, forecast_version)
                )
                """
            )
            cursor.execute(
                f"DELETE FROM {SERVING_TABLE} WHERE forecast_version=%s",
                (ENSEMBLE_VERSION,),
            )
            ready = predictions.loc[
                predictions["calibration_history_ready"]
            ].copy()
            values = [
                (
                    row.as_of_time.to_pydatetime(),
                    int(row.horizon_h),
                    ENSEMBLE_VERSION,
                    str(row.selected_model),
                    float(row.actual),
                    float(row.point_prediction),
                    float(row.p10),
                    float(row.p50),
                    float(row.p90),
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
                    source_mode, ingestion_run_id
                ) VALUES %s
                """,
                values,
                page_size=2000,
            )
    return len(values)


def _log_mlflow(
    output_dir: Path,
    decision: dict[str, Any],
    metrics_test: pd.DataFrame,
    probabilistic_test: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("smart-port-maritime-arrival-flow")
        with mlflow.start_run(run_name=ENSEMBLE_VERSION):
            mlflow.log_params(
                {
                    "ensemble_version": ENSEMBLE_VERSION,
                    "upstream_version": UPSTREAM_VERSION,
                    "weight_window_h": WEIGHT_WINDOW_HOURS,
                    "interval_window_h": INTERVAL_WINDOW_HOURS,
                    "training_executed": False,
                    "selection_split": "VALID",
                    "test_role": "LOCKED_AUDIT_ONLY",
                }
            )
            for horizon in HORIZONS:
                selected = decision["selected_models"][str(horizon)]
                row = metrics_test.loc[
                    (metrics_test["horizon_h"] == horizon)
                    & metrics_test["model"].eq(selected)
                ].iloc[0]
                calibration = probabilistic_test.loc[
                    probabilistic_test["horizon_h"] == horizon
                ].iloc[0]
                mlflow.log_metric(f"test_mae_{horizon}h", float(row["MAE"]))
                mlflow.log_metric(
                    f"test_coverage80_{horizon}h",
                    float(calibration["coverage_p10_p90"]),
                )
            mlflow.log_artifacts(str(output_dir), artifact_path="b56e")
        return "LOGGED"
    except Exception as exc:
        return f"FAILED_NON_FATAL: {exc}"


def _write_readme(
    path: Path,
    decision: dict[str, Any],
) -> None:
    selected = "\n".join(
        f"- {horizon}h: {model}"
        for horizon, model in decision["selected_models"].items()
    )
    path.write_text(
        "\n".join(
            [
                "# B56E - Maritime arrival-flow probabilistic ensemble",
                "",
                "This block does not train a new model. It reuses frozen B56C",
                "VALID and TEST predictions to validate past-only online policies",
                "and rolling asymmetric P10/P50/P90 calibration.",
                "",
                "## Selected on VALID",
                selected,
                "",
                "## Serving decision",
                f"- Status: {decision['status']}",
                f"- Historical replay: {decision['historical_replay_allowed']}",
                f"- Live serving: {decision['live_serving_allowed']}",
                f"- Source status: {decision['source_status']}",
                "",
                "TEST is used only as a locked final audit. Weather-enriched",
                "predictions remain diagnostic because B56C did not approve",
                "past-wave observations as an operationally selectable model.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_b56e_arrival_flow_probabilistic_ensemble(
    source_bucket: str = "gold-maritime",
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
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
        valid, test, integrity = _validate_predictions(valid_raw, test_raw)
        valid_adaptive, test_adaptive = _build_adaptive_frames(valid, test)

        metrics_valid = _point_metrics(
            valid_adaptive,
            split="VALID",
            selection_only=True,
        )
        selected = _select_on_valid(metrics_valid)
        metrics_test = _point_metrics(
            test_adaptive,
            split="TEST",
            selection_only=False,
        )

        probabilistic_valid_predictions, probabilistic_test_predictions = (
            _add_probabilistic_intervals(
                valid_adaptive,
                test_adaptive,
                selected,
            )
        )
        probabilistic_valid = _probabilistic_metrics(
            probabilistic_valid_predictions,
            "VALID",
        )
        probabilistic_test = _probabilistic_metrics(
            probabilistic_test_predictions,
            "TEST",
        )
        bootstrap = _paired_day_bootstrap(
            test_adaptive,
            selected,
            metrics_valid,
        )
        weights = pd.concat(
            [
                _weight_summary(valid_adaptive, "VALID"),
                _weight_summary(test_adaptive, "TEST"),
            ],
            ignore_index=True,
        )
        freshness = _source_freshness(upstream, test_adaptive)

        coverage_passed = bool(
            probabilistic_test["coverage_gate_passed"].all()
        )
        integrity_passed = bool(integrity["passed"].all())
        live_allowed = bool(freshness.iloc[0]["live_serving_allowed"])
        replay_allowed = integrity_passed and coverage_passed

        if not integrity_passed:
            decision_status = "NEED_DATA_REPAIR"
        elif not coverage_passed:
            decision_status = "NEED_INTERVAL_RECALIBRATION"
        elif not live_allowed:
            decision_status = "READY_FOR_HISTORICAL_REPLAY_NOT_LIVE"
        else:
            decision_status = "READY_FOR_CONTROLLED_LIVE_SHADOW"

        with tempfile.TemporaryDirectory(prefix="b56e-") as temporary:
            output_dir = Path(temporary)
            reports_dir = output_dir / "reports"
            configs_dir = output_dir / "configs"
            predictions_dir = output_dir / "predictions"
            reports_dir.mkdir(parents=True)
            configs_dir.mkdir(parents=True)
            predictions_dir.mkdir(parents=True)

            report_frames = {
                "00_upstream_and_integrity_contract.csv": integrity,
                "01_metrics_valid_selection.csv": metrics_valid,
                "02_metrics_test_locked.csv": metrics_test,
                "03_online_weight_summary.csv": weights,
                "04_probabilistic_calibration_valid.csv": probabilistic_valid,
                "05_probabilistic_calibration_test.csv": probabilistic_test,
                "06_paired_day_bootstrap_test.csv": bootstrap,
                "07_source_freshness_and_serving_gate.csv": freshness,
            }
            for name, frame in report_frames.items():
                frame.to_csv(reports_dir / name, index=False)

            probabilistic_valid_predictions.to_parquet(
                predictions_dir / "valid_probabilistic_predictions.parquet",
                index=False,
            )
            probabilistic_test_predictions.to_parquet(
                predictions_dir / "test_probabilistic_predictions.parquet",
                index=False,
            )

            decision = {
                "status": decision_status,
                "ensemble_version": ENSEMBLE_VERSION,
                "objective": "PORT_ARRIVAL_COUNTS_NEXT_6H_12H_24H",
                "source_bucket": source_bucket,
                "source_valid_key": VALID_KEY,
                "source_test_key": TEST_KEY,
                "source_status": str(freshness.iloc[0]["reason"]),
                "source_completeness_break_start": upstream.get(
                    "source_completeness_break_start"
                ),
                "latest_model_eligible_time": freshness.iloc[0][
                    "latest_model_eligible_time"
                ],
                "selected_models": selected,
                "selection_split": "VALID",
                "test_role": "LOCKED_FINAL_AUDIT_ONLY",
                "selection_used_test": False,
                "training_executed": False,
                "adaptive_weight_window_hours": WEIGHT_WINDOW_HOURS,
                "interval_window_hours": INTERVAL_WINDOW_HOURS,
                "interval_target_coverage": EXPECTED_COVERAGE,
                "coverage_gates_passed": coverage_passed,
                "integrity_gates_passed": integrity_passed,
                "historical_replay_allowed": replay_allowed,
                "live_serving_allowed": live_allowed,
                "weather_policy": (
                    "PAST_WAVE_PREDICTIONS_DIAGNOSTIC_ONLY_"
                    "NO_FUTURE_WEATHER_FORECAST_AVAILABLE"
                ),
                "next_block": (
                    "B56F_MARITIME_HISTORICAL_REPLAY_SERVING"
                    if replay_allowed
                    else "B56E_REPAIR"
                ),
            }
            decision_path = configs_dir / "08_b56e_decision.json"
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    allow_nan=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            _write_readme(reports_dir / "README_B56E.md", decision)

            timescale_rows = (
                _materialize_backtest(
                    probabilistic_test_predictions,
                    run_id,
                )
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
                key = (
                    f"{category}/b56e/{output_prefix.strip('/')}/{name}"
                )
                outputs[name] = _upload(client, path, output_bucket, key)

        decision.update(
            {
                "row_count": len(valid_adaptive) + len(test_adaptive),
                "valid_rows": len(valid_adaptive),
                "test_rows": len(test_adaptive),
                "timescale_table": (
                    SERVING_TABLE if materialize_timescale else None
                ),
                "timescale_rows": timescale_rows,
                "mlflow_status": mlflow_status,
                "outputs": outputs,
                "checksum": checksum,
            }
        )
        _finish_run(
            run_id,
            "SUCCESS",
            len(valid_adaptive) + len(test_adaptive),
            decision,
        )
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
            {"ensemble_version": ENSEMBLE_VERSION},
            error_message=str(exc),
        )
        raise
