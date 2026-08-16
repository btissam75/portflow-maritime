from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
from model_trainer.arrival_flow_hybrid_calibration import (
    TEST_KEY,
    UPSTREAM_VERSION,
    VALID_KEY,
    _coherence_audit,
    _load_parquet,
    _load_upstream_decision,
    _paired_interval_bootstrap,
    _point_fidelity,
    _prepare,
)


ASYMMETRIC_VERSION = "b56g-v2.1-asymmetric-aci-v1"
SOURCE_NAME = "b56g_v21_asymmetric_calibration"
DATASET_NAME = "port_arrival_flow_asymmetric_intervals"
SERVING_TABLE = "serving.maritime_arrival_flow_asymmetric_backtest_v21"

EXPECTED_COVERAGE = 0.80
COVERAGE_MIN = 0.77
COVERAGE_MAX = 0.83
TAIL_MIN = 0.07
TAIL_MAX = 0.13
WINDOW_DAYS = (7, 14, 30)
GAMMAS = (0.005, 0.01, 0.02)
MIN_HISTORY_HOURS = 7 * 24
ROLLING_AUDIT_DAYS = (7, 30, 90)


def _source_checksum(bucket: str, upstream: dict[str, Any]) -> str:
    client = _s3_client()
    digest = hashlib.sha256(ASYMMETRIC_VERSION.encode("ascii"))
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
        "asymmetric_version": ASYMMETRIC_VERSION,
        "point_source": UPSTREAM_VERSION,
        "training_executed": False,
        "policy": (
            "B56E_POINT_IDENTITY_OR_ASYMMETRIC_ACI_VALID_SELECTION_"
            "MATURED_LABELS_ROLLING_GATES_SHADOW_ONLY"
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


def _pinball(
    actual: np.ndarray,
    prediction: np.ndarray,
    quantile: float,
) -> float:
    error = actual - prediction
    return float(
        np.mean(np.maximum(quantile * error, (quantile - 1.0) * error))
    )


def _interval_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    part = frame.loc[frame["calibration_history_ready"]].dropna(
        subset=["actual", "p10", "p50", "p90"]
    )
    if part.empty:
        return {
            "n": 0,
            "coverage_p10_p90": np.nan,
            "coverage_gap_abs": np.inf,
            "below_p10_rate": np.nan,
            "above_p90_rate": np.nan,
            "lower_tail_gap_abs": np.inf,
            "upper_tail_gap_abs": np.inf,
            "mean_interval_width": np.nan,
            "median_interval_width": np.nan,
            "winkler_interval_score": np.inf,
            "pinball_p10": np.nan,
            "pinball_p50": np.nan,
            "pinball_p90": np.nan,
            "coverage_gate_passed": False,
            "lower_tail_gate_passed": False,
            "upper_tail_gate_passed": False,
        }
    actual = part["actual"].to_numpy(dtype="float64")
    low = part["p10"].to_numpy(dtype="float64")
    mid = part["p50"].to_numpy(dtype="float64")
    high = part["p90"].to_numpy(dtype="float64")
    below = float(np.mean(actual < low))
    above = float(np.mean(actual > high))
    coverage = 1.0 - below - above
    alpha = 1.0 - EXPECTED_COVERAGE
    interval_score = (
        high
        - low
        + (2.0 / alpha) * (low - actual) * (actual < low)
        + (2.0 / alpha) * (actual - high) * (actual > high)
    )
    return {
        "n": len(part),
        "coverage_p10_p90": coverage,
        "coverage_gap_abs": abs(coverage - EXPECTED_COVERAGE),
        "below_p10_rate": below,
        "above_p90_rate": above,
        "lower_tail_gap_abs": abs(below - 0.10),
        "upper_tail_gap_abs": abs(above - 0.10),
        "mean_interval_width": float(np.mean(high - low)),
        "median_interval_width": float(np.median(high - low)),
        "winkler_interval_score": float(np.mean(interval_score)),
        "pinball_p10": _pinball(actual, low, 0.10),
        "pinball_p50": _pinball(actual, mid, 0.50),
        "pinball_p90": _pinball(actual, high, 0.90),
        "coverage_gate_passed": COVERAGE_MIN <= coverage <= COVERAGE_MAX,
        "lower_tail_gate_passed": TAIL_MIN <= below <= TAIL_MAX,
        "upper_tail_gate_passed": TAIL_MIN <= above <= TAIL_MAX,
    }


def _identity(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    result = frame.loc[frame["horizon_h"].eq(horizon)].copy()
    result = result.sort_values("as_of_time").reset_index(drop=True)
    result["point_prediction"] = result["source_point_prediction"]
    result["p10"] = result["base_p10"]
    result["p50"] = result["base_p50"]
    result["p90"] = result["base_p90"]
    result["calibration_history_ready"] = result[
        "calibration_history_ready_source"
    ].astype(bool)
    result["adaptive_alpha_lower"] = 0.10
    result["adaptive_alpha_upper"] = 0.10
    result["lower_correction"] = 0.0
    result["upper_correction"] = 0.0
    result["selected_policy"] = "IDENTITY_B56E"
    result["window_days"] = 0
    result["gamma"] = 0.0
    return result


def _asymmetric_aci(
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
    point = result["source_point_prediction"].to_numpy(dtype="float64")
    lower_score = base_low - actual
    upper_score = actual - base_high

    low = np.full(len(result), np.nan, dtype="float64")
    mid = np.full(len(result), np.nan, dtype="float64")
    high = np.full(len(result), np.nan, dtype="float64")
    ready = np.zeros(len(result), dtype=bool)
    alpha_lower_used = np.full(len(result), np.nan, dtype="float64")
    alpha_upper_used = np.full(len(result), np.nan, dtype="float64")
    lower_correction = np.full(len(result), np.nan, dtype="float64")
    upper_correction = np.full(len(result), np.nan, dtype="float64")

    target_tail = 0.10
    adaptive_lower = target_tail
    adaptive_upper = target_tail
    lower_history: list[float] = []
    upper_history: list[float] = []
    window_hours = window_days * 24

    for position in range(len(result)):
        matured_position = position - horizon
        if (
            matured_position >= 0
            and np.isfinite(lower_score[matured_position])
            and np.isfinite(upper_score[matured_position])
            and np.isfinite(low[matured_position])
            and np.isfinite(high[matured_position])
        ):
            lower_history.append(float(lower_score[matured_position]))
            upper_history.append(float(upper_score[matured_position]))
            lower_miss = float(
                actual[matured_position] < low[matured_position]
            )
            upper_miss = float(
                actual[matured_position] > high[matured_position]
            )
            adaptive_lower = float(
                np.clip(
                    adaptive_lower
                    + gamma * (target_tail - lower_miss),
                    0.01,
                    0.30,
                )
            )
            adaptive_upper = float(
                np.clip(
                    adaptive_upper
                    + gamma * (target_tail - upper_miss),
                    0.01,
                    0.30,
                )
            )

        finite_base = bool(
            np.isfinite(base_low[position])
            and np.isfinite(base_high[position])
        )
        lower_window = lower_history[-window_hours:]
        upper_window = upper_history[-window_hours:]
        if (
            finite_base
            and len(lower_window) >= MIN_HISTORY_HOURS
            and len(upper_window) >= MIN_HISTORY_HOURS
        ):
            lower_adjustment = float(
                np.quantile(lower_window, 1.0 - adaptive_lower)
            )
            upper_adjustment = float(
                np.quantile(upper_window, 1.0 - adaptive_upper)
            )
            width = max(base_high[position] - base_low[position], 0.0)
            lower_adjustment = max(lower_adjustment, -0.40 * width)
            upper_adjustment = max(upper_adjustment, -0.40 * width)
            ready[position] = True
        elif finite_base:
            lower_adjustment = 0.0
            upper_adjustment = 0.0
        else:
            continue

        low[position] = max(
            0.0, base_low[position] - lower_adjustment
        )
        high[position] = max(
            low[position], base_high[position] + upper_adjustment
        )
        candidate_mid = (
            base_mid[position]
            if np.isfinite(base_mid[position])
            else point[position]
        )
        mid[position] = float(
            np.clip(candidate_mid, low[position], high[position])
        )
        alpha_lower_used[position] = adaptive_lower
        alpha_upper_used[position] = adaptive_upper
        lower_correction[position] = lower_adjustment
        upper_correction[position] = upper_adjustment

    result["point_prediction"] = point
    result["p10"] = low
    result["p50"] = mid
    result["p90"] = high
    result["calibration_history_ready"] = ready
    result["adaptive_alpha_lower"] = alpha_lower_used
    result["adaptive_alpha_upper"] = alpha_upper_used
    result["lower_correction"] = lower_correction
    result["upper_correction"] = upper_correction
    result["selected_policy"] = (
        f"ASYMMETRIC_ACI_{window_days}D_GAMMA_{gamma:g}"
    )
    result["window_days"] = window_days
    result["gamma"] = gamma
    return result


def _recent_slice(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    end = frame["as_of_time"].max()
    return frame.loc[frame["as_of_time"] > end - pd.Timedelta(days=days)]


def _candidate_row(
    calibrated: pd.DataFrame,
    horizon: int,
    policy: str,
    window_days: int,
    gamma: float,
) -> dict[str, Any]:
    full = _interval_metrics(calibrated)
    recent = _interval_metrics(_recent_slice(calibrated, 30))
    all_recent_gates = bool(
        recent["coverage_gate_passed"]
        and recent["lower_tail_gate_passed"]
        and recent["upper_tail_gate_passed"]
    )
    score = (
        4.0 * recent["coverage_gap_abs"]
        + 2.0 * recent["lower_tail_gap_abs"]
        + 2.0 * recent["upper_tail_gap_abs"]
        + full["coverage_gap_abs"]
        + 0.01 * full["winkler_interval_score"]
    )
    return {
        "horizon_h": horizon,
        "policy": policy,
        "window_days": window_days,
        "gamma": gamma,
        **{f"full_{key}": value for key, value in full.items()},
        **{f"recent30_{key}": value for key, value in recent.items()},
        "recent30_all_gates_passed": all_recent_gates,
        "selection_score": float(score),
    }


def _select_on_valid(
    source: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    valid = source.loc[source["split"].eq("VALID")].copy()
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        identity = _identity(valid, horizon)
        rows.append(
            _candidate_row(identity, horizon, "IDENTITY_B56E", 0, 0.0)
        )
        for window_days in WINDOW_DAYS:
            for gamma in GAMMAS:
                calibrated = _asymmetric_aci(
                    valid, horizon, window_days, gamma
                )
                rows.append(
                    _candidate_row(
                        calibrated,
                        horizon,
                        str(calibrated["selected_policy"].iloc[-1]),
                        window_days,
                        gamma,
                    )
                )
        ranking = pd.DataFrame(
            [row for row in rows if row["horizon_h"] == horizon]
        )
        ranking = ranking.loc[np.isfinite(ranking["selection_score"])].copy()
        if ranking.empty:
            raise RuntimeError(
                f"No evaluable VALID calibration policy for {horizon}h"
            )
        gated = ranking.loc[ranking["recent30_all_gates_passed"]]
        candidates = gated if not gated.empty else ranking
        winner = candidates.sort_values(
            [
                "selection_score",
                "full_winkler_interval_score",
                "policy",
            ]
        ).iloc[0]
        selected[str(horizon)] = {
            "policy": str(winner["policy"]),
            "window_days": int(winner["window_days"]),
            "gamma": float(winner["gamma"]),
            "valid_full_coverage": float(
                winner["full_coverage_p10_p90"]
            ),
            "valid_recent30_coverage": float(
                winner["recent30_coverage_p10_p90"]
            ),
            "valid_recent30_lower_tail": float(
                winner["recent30_below_p10_rate"]
            ),
            "valid_recent30_upper_tail": float(
                winner["recent30_above_p90_rate"]
            ),
        }
    return selected, pd.DataFrame(rows)


def _apply_selected(
    source: pd.DataFrame,
    selected: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        config = selected[str(horizon)]
        if config["policy"] == "IDENTITY_B56E":
            part = _identity(source, horizon)
        else:
            part = _asymmetric_aci(
                source,
                horizon,
                int(config["window_days"]),
                float(config["gamma"]),
            )
        parts.append(part)
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


def _audit_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("VALID", "TEST"):
        for horizon in HORIZONS:
            part = frame.loc[
                frame["split"].eq(split)
                & frame["horizon_h"].eq(horizon)
            ].copy()
            for window_days in (0, *ROLLING_AUDIT_DAYS):
                audit = (
                    part if window_days == 0 else _recent_slice(part, window_days)
                )
                metrics = _interval_metrics(audit)
                rows.append(
                    {
                        "split": split,
                        "horizon_h": horizon,
                        "audit_window_days": window_days,
                        "audit_window": (
                            "FULL" if window_days == 0 else f"LAST_{window_days}D"
                        ),
                        "selected_policy": str(
                            part["selected_policy"].iloc[-1]
                        ),
                        **metrics,
                        "all_gates_passed": bool(
                            metrics["coverage_gate_passed"]
                            and metrics["lower_tail_gate_passed"]
                            and metrics["upper_tail_gate_passed"]
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
                    selected_policy TEXT NOT NULL,
                    actual_arrivals REAL,
                    point_prediction REAL NOT NULL,
                    p10 REAL NOT NULL,
                    p50 REAL NOT NULL,
                    p90 REAL NOT NULL,
                    lower_correction REAL NOT NULL,
                    upper_correction REAL NOT NULL,
                    source_mode TEXT NOT NULL,
                    ingestion_run_id UUID NOT NULL,
                    PRIMARY KEY (as_of_time, horizon_h, forecast_version)
                )
                """
            )
            cursor.execute(
                f"DELETE FROM {SERVING_TABLE} WHERE forecast_version=%s",
                (ASYMMETRIC_VERSION,),
            )
            values = [
                (
                    row.as_of_time.to_pydatetime(),
                    int(row.horizon_h),
                    ASYMMETRIC_VERSION,
                    str(row.selected_policy),
                    float(row.actual),
                    float(row.point_prediction),
                    float(row.p10),
                    float(row.p50),
                    float(row.p90),
                    float(row.lower_correction),
                    float(row.upper_correction),
                    "RETROSPECTIVE_SHADOW",
                    run_id,
                )
                for row in ready.itertuples(index=False)
            ]
            execute_values(
                cursor,
                f"""
                INSERT INTO {SERVING_TABLE} (
                    as_of_time, horizon_h, forecast_version, selected_policy,
                    actual_arrivals, point_prediction, p10, p50, p90,
                    lower_correction, upper_correction, source_mode,
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
    audit: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment(
            "maritime-arrival-flow-asymmetric-calibration"
        )
        with mlflow.start_run(run_name=ASYMMETRIC_VERSION):
            mlflow.log_params(
                {
                    "version": ASYMMETRIC_VERSION,
                    "point_source": UPSTREAM_VERSION,
                    "selection_split": "VALID",
                    "windows": ",".join(map(str, WINDOW_DAYS)),
                    "gammas": ",".join(map(str, GAMMAS)),
                }
            )
            metrics: dict[str, float] = {}
            test30 = audit.loc[
                audit["split"].eq("TEST")
                & audit["audit_window_days"].eq(30)
            ]
            for row in test30.itertuples(index=False):
                horizon = int(row.horizon_h)
                metrics[f"test30_coverage_{horizon}h"] = float(
                    row.coverage_p10_p90
                )
                metrics[f"test30_below_p10_{horizon}h"] = float(
                    row.below_p10_rate
                )
                metrics[f"test30_above_p90_{horizon}h"] = float(
                    row.above_p90_rate
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
                "# B56G-v2.1 asymmetric calibration",
                "",
                f"Decision: {decision['status']}",
                "",
                "- B56E point forecasts are preserved.",
                "- Identity competes with asymmetric 7/14/30-day ACI.",
                "- Lower and upper tails are calibrated independently.",
                "- Selection uses VALID only, including its latest 30 days.",
                "- TEST labels enter adaptation only after maturity.",
                "- Formal promotion requires future prospective shadow data.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_b56g_v21_asymmetric_calibration(
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
        source, integrity = _prepare(valid_raw, test_raw)
        source["calibration_history_ready_source"] = (
            source["calibration_history_ready"].astype(bool)
        )

        selected, ranking = _select_on_valid(source)
        calibrated = _reconcile(_apply_selected(source, selected))
        point_fidelity = _point_fidelity(calibrated)
        audit = _audit_metrics(calibrated)
        coherence = _coherence_audit(calibrated)
        bootstrap = _paired_interval_bootstrap(calibrated)

        test_fidelity = point_fidelity.loc[
            point_fidelity["split"].eq("TEST")
        ]
        test30 = audit.loc[
            audit["split"].eq("TEST")
            & audit["audit_window_days"].eq(30)
        ]
        integrity_passed = bool(integrity["passed"].all())
        point_preserved = bool(
            (test_fidelity["adjusted_point_rows"] == 0).all()
            and (test_fidelity["mae_delta"].abs() <= 1e-12).all()
        )
        coherence_passed = bool(coherence["passed"].all())
        recent30_gates_passed = bool(test30["all_gates_passed"].all())

        if not integrity_passed:
            decision_status = "NEED_DATA_REPAIR"
        elif not point_preserved:
            decision_status = "NEED_POINT_FIDELITY_REPAIR"
        elif not coherence_passed:
            decision_status = "NEED_RECONCILIATION_REPAIR"
        elif not recent30_gates_passed:
            decision_status = "SHADOW_WITH_RECENT_CALIBRATION_WARNING"
        else:
            decision_status = "READY_FOR_PROSPECTIVE_SHADOW"

        with tempfile.TemporaryDirectory(prefix="b56g-v21-") as temporary:
            output_dir = Path(temporary)
            reports_dir = output_dir / "reports"
            configs_dir = output_dir / "configs"
            predictions_dir = output_dir / "predictions"
            for directory in (reports_dir, configs_dir, predictions_dir):
                directory.mkdir(parents=True)

            reports = {
                "00_integrity_and_temporal_contract.csv": integrity,
                "01_valid_policy_ranking.csv": ranking,
                "02_point_fidelity.csv": point_fidelity,
                "03_rolling_interval_audit.csv": audit,
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
                predictions_dir / "valid_asymmetric_predictions.parquet",
                index=False,
            )
            test_output.to_parquet(
                predictions_dir / "test_asymmetric_predictions.parquet",
                index=False,
            )

            decision = {
                "status": decision_status,
                "asymmetric_version": ASYMMETRIC_VERSION,
                "objective": "B56E_POINT_WITH_ASYMMETRIC_ADAPTIVE_INTERVALS",
                "point_source": UPSTREAM_VERSION,
                "selected_policies": selected,
                "selection_split": "VALID",
                "selection_recent_window": "LAST_30D_OF_VALID",
                "test_role": "RETROSPECTIVE_SHADOW_NOT_NEW_HOLDOUT",
                "selection_used_test": False,
                "training_executed": False,
                "test_adaptation_policy": "LABELS_SHIFTED_BY_HORIZON_MATURITY",
                "integrity_gates_passed": integrity_passed,
                "point_fidelity_passed": point_preserved,
                "coherence_gates_passed": coherence_passed,
                "recent30_gates_passed": recent30_gates_passed,
                "historical_replay_allowed": bool(
                    integrity_passed
                    and point_preserved
                    and coherence_passed
                ),
                "live_serving_allowed": False,
                "formal_promotion_allowed": False,
                "formal_promotion_blocker": (
                    "REQUIRE_FUTURE_PROSPECTIVE_SHADOW_"
                    "WITH_7D_30D_90D_ROLLING_GATES"
                ),
                "next_block": "B56G_V21_PROSPECTIVE_SHADOW_MONITOR",
            }
            decision_path = configs_dir / "06_b56g_v21_decision.json"
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    allow_nan=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            _write_readme(reports_dir / "README_B56G_V21.md", decision)

            timescale_rows = (
                _materialize_backtest(test_output, run_id)
                if materialize_timescale
                else 0
            )
            mlflow_status = _log_mlflow(output_dir, decision, audit)

            client = _s3_client()
            outputs: dict[str, str] = {}
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(output_dir).as_posix()
                category, name = relative.split("/", 1)
                key = (
                    f"{category}/b56gv21/{output_prefix.strip('/')}/{name}"
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
            {"asymmetric_version": ASYMMETRIC_VERSION},
            error_message=str(exc),
        )
        raise
