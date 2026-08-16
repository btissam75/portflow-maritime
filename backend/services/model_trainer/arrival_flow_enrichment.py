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

import mlflow
import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_values

from model_trainer.arrival_flow_baselines import (
    BASELINE_NAMES,
    HORIZONS,
    PURGE_HOURS,
    SOURCE_FEATURE_VERSION,
    SOURCE_TABLE,
    TARGET_BY_HORIZON,
    _baseline_predictions,
    _build_model,
    _clean_json,
    _db_connection,
    _detect_source_completeness,
    _evaluate_part,
    _json_default,
    _metrics,
    _query_frame,
    _s3_client,
    _target_stability,
    _temporal_split,
    _upload,
)


ENRICHMENT_VERSION = "b56c-arrival-flow-enrichment-v1"
SOURCE_NAME = "b56c_arrival_flow_enrichment"
DATASET_NAME = "port_arrival_flow_enriched_6h_12h_24h"
TIMESCALE_TABLE = "features.port_arrival_flow_enriched_v1"
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 1000
MIN_PROMOTION_UPLIFT_PCT = 5.0

SOURCE_COLUMNS = (
    "as_of_time",
    "arrivals_prev_1h",
    "arrivals_last_6h",
    "arrivals_last_24h",
    "arrivals_last_168h",
    "departures_prev_1h",
    "departures_last_6h",
    "departures_last_24h",
    "vessels_in_port_observed",
    "wave_height_lag_1h_m",
    "wave_period_lag_1h_s",
    "wave_direction_lag_1h_deg",
    "weather_available_flag",
    "hour_of_day",
    "day_of_week",
    "month",
    "weekend_flag",
    *TARGET_BY_HORIZON.values(),
)

LEGACY_FEATURES = (
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

ARRIVAL_HISTORY_FEATURES = (
    *LEGACY_FEATURES,
    "arrivals_lag_2h",
    "arrivals_lag_3h",
    "arrivals_lag_6h",
    "arrivals_lag_12h",
    "arrivals_lag_24h",
    "arrivals_lag_48h",
    "arrivals_lag_168h",
    "arrival_rate_3h",
    "arrival_rate_12h",
    "arrival_rate_48h",
    "arrival_rate_72h",
    "arrival_rate_168h",
    "arrival_std_6h",
    "arrival_std_24h",
    "arrival_trend_6h_vs_24h",
    "arrival_trend_24h_vs_168h",
    "hour_of_year_sin",
    "hour_of_year_cos",
    "month_start_flag",
    "month_end_flag",
    "history_168h_available_flag",
)

OPERATIONAL_FEATURES = (
    "departures_prev_1h",
    "departures_last_6h",
    "departures_last_24h",
    "departures_lag_2h",
    "departures_lag_6h",
    "departures_lag_24h",
    "departure_rate_12h",
    "departure_rate_72h",
    "net_flow_last_6h",
    "net_flow_last_24h",
    "vessels_in_port_observed",
    "occupancy_lag_1h",
    "occupancy_mean_6h",
    "occupancy_mean_24h",
    "occupancy_max_24h",
    "occupancy_change_1h",
    "occupancy_change_6h",
    "operational_quality_approved_flag",
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

MODEL_FEATURES = {
    "HGB_LEGACY_CORE": LEGACY_FEATURES,
    "HGB_ENRICHED_HISTORY": ARRIVAL_HISTORY_FEATURES,
    "HGB_ENRICHED_OPERATIONAL": (*ARRIVAL_HISTORY_FEATURES, *OPERATIONAL_FEATURES),
    "HGB_ENRICHED_HISTORY_WAVE": (*ARRIVAL_HISTORY_FEATURES, *WAVE_FEATURES),
}

FORBIDDEN_FEATURE_TOKENS = (
    "target_",
    "actual_ata",
    "actual_atd",
    "planned_eta",
    "future_",
    "arrival_delay",
)


def _load_b56b_decision() -> dict[str, Any]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, metadata
                FROM audit.ingestion_run
                WHERE source_name='b56b_arrival_flow_temporal_baselines'
                  AND dataset_name='port_arrival_flow_6h_12h_24h'
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("B56B v1.1 result is missing")
    status, metadata = row
    metadata = dict(metadata or {})
    if status != "SUCCESS":
        raise RuntimeError(f"Latest B56B run is {status}")
    if metadata.get("training_version") != "b56b-arrival-flow-temporal-v1.1":
        raise RuntimeError("Latest B56B result is not v1.1")
    if metadata.get("target_stability_passed") is not True:
        raise RuntimeError("B56B target stability gate did not pass")
    if int(metadata.get("temporal_leakage_violations", -1)) != 0:
        raise RuntimeError("B56B leakage gate did not pass")
    return metadata


def _load_b56a_readiness() -> dict[str, Any]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT metadata
                FROM audit.ingestion_run
                WHERE source_name='b56a_operational_feasibility'
                  AND dataset_name='port_hourly_state_feasibility'
                  AND status='SUCCESS'
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("B56A readiness metadata is missing")
    return dict(row[0] or {})


def _load_source() -> pd.DataFrame:
    frame = _query_frame(
        f"""
        SELECT {', '.join(SOURCE_COLUMNS)}
        FROM {SOURCE_TABLE}
        WHERE feature_version=%s
        ORDER BY as_of_time
        """,
        (SOURCE_FEATURE_VERSION,),
    )
    if frame.empty:
        raise RuntimeError("B56A hourly source table is empty")
    frame["as_of_time"] = pd.to_datetime(frame["as_of_time"], utc=True)
    if frame["as_of_time"].duplicated().any():
        raise RuntimeError("B56A source contains duplicate hourly keys")
    gaps = frame["as_of_time"].diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise RuntimeError("B56A source is not a contiguous hourly grid")
    for column in SOURCE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _rolling(series: pd.Series, window: int, operation: str) -> pd.Series:
    rolling = series.rolling(window, min_periods=window)
    if operation == "sum":
        return rolling.sum()
    if operation == "mean":
        return rolling.mean()
    if operation == "max":
        return rolling.max()
    if operation == "std":
        return rolling.std(ddof=0)
    raise ValueError(f"Unsupported rolling operation: {operation}")


def _build_enriched_frame(
    source: pd.DataFrame, operational_quality_approved: bool
) -> pd.DataFrame:
    frame = source.copy()
    timestamp = frame["as_of_time"]
    hour = frame["hour_of_day"]
    dow = frame["day_of_week"]
    month = frame["month"]
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    frame["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    hour_of_year = (timestamp.dt.dayofyear - 1) * 24 + timestamp.dt.hour
    frame["hour_of_year_sin"] = np.sin(2 * np.pi * hour_of_year / (365.25 * 24))
    frame["hour_of_year_cos"] = np.cos(2 * np.pi * hour_of_year / (365.25 * 24))
    frame["month_start_flag"] = timestamp.dt.is_month_start.astype("int8")
    frame["month_end_flag"] = timestamp.dt.is_month_end.astype("int8")

    arrivals = frame["arrivals_prev_1h"]
    for lag in (2, 3, 6, 12, 24, 48, 168):
        frame[f"arrivals_lag_{lag}h"] = arrivals.shift(lag - 1)
    for window in (3, 6, 12, 24, 48, 72, 168):
        frame[f"arrival_rate_{window}h"] = _rolling(
            arrivals, window, "sum"
        ) / float(window)
    frame["arrival_std_6h"] = _rolling(arrivals, 6, "std")
    frame["arrival_std_24h"] = _rolling(arrivals, 24, "std")
    frame["arrival_trend_6h_vs_24h"] = (
        frame["arrival_rate_6h"] - frame["arrival_rate_24h"]
    )
    frame["arrival_trend_24h_vs_168h"] = (
        frame["arrival_rate_24h"] - frame["arrival_rate_168h"]
    )
    frame["history_168h_available_flag"] = frame[
        "arrival_rate_168h"
    ].notna().astype("int8")

    departures = frame["departures_prev_1h"]
    for lag in (2, 6, 24):
        frame[f"departures_lag_{lag}h"] = departures.shift(lag - 1)
    for window in (6, 12, 24, 72):
        frame[f"departure_rate_{window}h"] = _rolling(
            departures, window, "sum"
        ) / float(window)
    frame["net_flow_last_6h"] = (
        frame["arrivals_last_6h"] - frame["departures_last_6h"]
    )
    frame["net_flow_last_24h"] = (
        frame["arrivals_last_24h"] - frame["departures_last_24h"]
    )
    occupancy = frame["vessels_in_port_observed"]
    frame["occupancy_lag_1h"] = occupancy.shift(1)
    frame["occupancy_mean_6h"] = _rolling(occupancy, 6, "mean")
    frame["occupancy_mean_24h"] = _rolling(occupancy, 24, "mean")
    frame["occupancy_max_24h"] = _rolling(occupancy, 24, "max")
    frame["occupancy_change_1h"] = occupancy - occupancy.shift(1)
    frame["occupancy_change_6h"] = occupancy - occupancy.shift(6)
    frame["operational_quality_approved_flag"] = int(
        operational_quality_approved
    )

    radians = np.deg2rad(frame["wave_direction_lag_1h_deg"])
    frame["wave_direction_sin"] = np.sin(radians)
    frame["wave_direction_cos"] = np.cos(radians)
    wave_height = frame["wave_height_lag_1h_m"]
    wave_period = frame["wave_period_lag_1h_s"]
    for window in (3, 6, 12, 24, 72):
        frame[f"wave_height_mean_{window}h_m"] = _rolling(
            wave_height, window, "mean"
        )
        frame[f"wave_coverage_{window}h_pct"] = (
            wave_height.notna().rolling(window, min_periods=window).mean() * 100
        )
    for window in (6, 12, 24, 72):
        frame[f"wave_height_max_{window}h_m"] = _rolling(
            wave_height, window, "max"
        )
    frame["wave_height_std_24h_m"] = _rolling(wave_height, 24, "std")
    frame["wave_period_mean_6h_s"] = _rolling(wave_period, 6, "mean")
    frame["wave_period_mean_24h_s"] = _rolling(wave_period, 24, "mean")
    frame["wave_height_trend_6h_vs_24h_m"] = (
        frame["wave_height_mean_6h_m"] - frame["wave_height_mean_24h_m"]
    )
    return frame


def _all_feature_names() -> tuple[str, ...]:
    ordered: list[str] = []
    for features in MODEL_FEATURES.values():
        for feature in features:
            if feature not in ordered:
                ordered.append(feature)
    return tuple(ordered)


def _checksum(frame: pd.DataFrame) -> str:
    columns = [
        "as_of_time",
        *_all_feature_names(),
        *TARGET_BY_HORIZON.values(),
    ]
    digest = hashlib.sha256(ENRICHMENT_VERSION.encode("ascii"))
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
        "enrichment_version": ENRICHMENT_VERSION,
        "source_feature_version": SOURCE_FEATURE_VERSION,
        "split_policy": "TEMPORAL_70_15_15_PURGED_24H_FROZEN_FROM_B56B",
        "eta_policy": "EXCLUDED_NO_REVISION_HISTORY",
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


def _feature_contract(operational_quality_approved: bool) -> pd.DataFrame:
    family_by_feature: dict[str, str] = {}
    for feature in LEGACY_FEATURES:
        family_by_feature[feature] = (
            "ARRIVAL_HISTORY" if feature.startswith("arrivals_") else "CALENDAR"
        )
    for feature in ARRIVAL_HISTORY_FEATURES:
        family_by_feature.setdefault(
            feature,
            "CALENDAR" if any(token in feature for token in ("hour_", "month_"))
            else "ARRIVAL_HISTORY",
        )
    for feature in OPERATIONAL_FEATURES:
        family_by_feature[feature] = "OPERATIONAL_DIAGNOSTIC"
    for feature in WAVE_FEATURES:
        family_by_feature[feature] = "PAST_WAVE"

    rows = []
    for feature in _all_feature_names():
        family = family_by_feature[feature]
        quality_approved = family != "PAST_WAVE" and not (
            family == "OPERATIONAL_DIAGNOSTIC"
            and not operational_quality_approved
        )
        rows.append(
            {
                "feature": feature,
                "family": family,
                "available_at_as_of": True,
                "target_derived": False,
                "quality_approved_for_selection": quality_approved,
                "source_max_time_policy": (
                    "AS_OF_TIME"
                    if feature.startswith("occupancy_")
                    or feature == "vessels_in_port_observed"
                    else "AS_OF_TIME_MINUS_1H_OR_EARLIER"
                ),
            }
        )
    for horizon, target in TARGET_BY_HORIZON.items():
        rows.append(
            {
                "feature": target,
                "family": f"FUTURE_{horizon}H_TARGET",
                "available_at_as_of": False,
                "target_derived": True,
                "quality_approved_for_selection": False,
                "source_max_time_policy": f"AS_OF_TIME_TO_PLUS_{horizon - 1}H",
            }
        )
    return pd.DataFrame(rows)


def _equivalent(left: pd.Series, right: pd.Series) -> bool:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype="float64")
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype="float64")
    return bool(
        np.allclose(
            left_values,
            right_values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
    )


def _leakage_audit(
    source: pd.DataFrame,
    enriched: pd.DataFrame,
    split_audit: pd.DataFrame,
) -> pd.DataFrame:
    arrivals = source["arrivals_prev_1h"]
    departures = source["departures_prev_1h"]
    wave = source["wave_height_lag_1h_m"]
    feature_names = _all_feature_names()
    forbidden = sorted(
        feature
        for feature in feature_names
        if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    recalculations = {
        "arrivals_lag_24h": arrivals.shift(23),
        "arrival_rate_24h": arrivals.rolling(24, min_periods=24).sum() / 24.0,
        "arrival_rate_168h": arrivals.rolling(168, min_periods=168).sum() / 168.0,
        "departures_lag_24h": departures.shift(23),
        "departure_rate_72h": departures.rolling(72, min_periods=72).sum() / 72.0,
        "occupancy_change_6h": (
            source["vessels_in_port_observed"]
            - source["vessels_in_port_observed"].shift(6)
        ),
        "wave_height_mean_24h_m": wave.rolling(24, min_periods=24).mean(),
        "wave_height_max_72h_m": wave.rolling(72, min_periods=72).max(),
    }
    recalculation_failures = [
        feature
        for feature, expected in recalculations.items()
        if not _equivalent(enriched[feature], expected)
    ]
    split_lookup = split_audit.set_index("split")
    purge_ok = True
    if {"TRAIN", "VALID", "TEST"}.issubset(split_lookup.index):
        train_max = pd.Timestamp(split_lookup.loc["TRAIN", "last_time"])
        valid_min = pd.Timestamp(split_lookup.loc["VALID", "first_time"])
        valid_max = pd.Timestamp(split_lookup.loc["VALID", "last_time"])
        test_min = pd.Timestamp(split_lookup.loc["TEST", "first_time"])
        purge = pd.Timedelta(hours=PURGE_HOURS)
        purge_ok = (valid_min - train_max > purge) and (test_min - valid_max > purge)

    checks = [
        {
            "check": "FEATURE_TARGET_CONTRACT_DISJOINT",
            "severity": "CRITICAL",
            "passed": not set(feature_names).intersection(TARGET_BY_HORIZON.values()),
            "evidence": "Model features and future targets are disjoint.",
        },
        {
            "check": "FORBIDDEN_FINAL_EVENT_AND_ETA_COLUMNS_ABSENT",
            "severity": "CRITICAL",
            "passed": not forbidden,
            "evidence": json.dumps(forbidden),
        },
        {
            "check": "INDEPENDENT_FEATURE_RECALCULATION",
            "severity": "CRITICAL",
            "passed": not recalculation_failures,
            "evidence": json.dumps(recalculation_failures),
        },
        {
            "check": "STRICT_TEMPORAL_PURGE",
            "severity": "CRITICAL",
            "passed": purge_ok,
            "evidence": f"Required purge is greater than {PURGE_HOURS} hours.",
        },
        {
            "check": "HOURLY_GRAIN_UNIQUE",
            "severity": "CRITICAL",
            "passed": not enriched["as_of_time"].duplicated().any(),
            "evidence": "Exactly one row exists per as_of_time.",
        },
        {
            "check": "ETA_REVISION_HISTORY_AVAILABLE",
            "severity": "WARNING",
            "passed": False,
            "evidence": "ETA excluded because no recorded_at revision history exists.",
        },
        {
            "check": "WEATHER_PUBLICATION_TIME_AVAILABLE",
            "severity": "WARNING",
            "passed": False,
            "evidence": "Weather valid time exists; publication time is unavailable.",
        },
    ]
    return pd.DataFrame(checks)


def _missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in _all_feature_names():
        missing = int(frame[feature].isna().sum())
        rows.append(
            {
                "feature": feature,
                "rows": len(frame),
                "missing_rows": missing,
                "missing_pct": 100 * missing / max(len(frame), 1),
                "n_unique_non_null": int(frame[feature].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_pct", "feature"], ascending=[False, True]
    )


def _redundancy_report(frame: pd.DataFrame) -> pd.DataFrame:
    features = list(_all_feature_names())
    correlation = frame[features].corr(method="spearman", min_periods=500)
    rows = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= 0.95:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "spearman": float(value),
                        "abs_spearman": abs(float(value)),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["feature_left", "feature_right", "spearman", "abs_spearman"],
    ).sort_values("abs_spearman", ascending=False)


def _bootstrap_comparison(
    predictions: pd.DataFrame,
    horizon: int,
    reference: str,
    challenger: str,
    comparison: str,
) -> dict[str, Any]:
    frame = predictions.dropna(subset=["actual", reference, challenger]).copy()
    frame["day"] = pd.to_datetime(frame["as_of_time"], utc=True).dt.floor("D")
    frame["delta_abs"] = (
        (frame["actual"] - frame[challenger]).abs()
        - (frame["actual"] - frame[reference]).abs()
    )
    daily = frame.groupby("day", observed=True)["delta_abs"].agg(["sum", "count"])
    if daily.empty:
        raise RuntimeError(f"No rows for bootstrap comparison {comparison}")
    rng = np.random.default_rng(RANDOM_SEED + horizon + len(comparison))
    positions = np.arange(len(daily))
    draws = np.empty(BOOTSTRAP_ITERATIONS, dtype="float64")
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sample = daily.iloc[rng.choice(positions, size=len(positions), replace=True)]
        draws[iteration] = sample["sum"].sum() / sample["count"].sum()
    reference_mae = float((frame["actual"] - frame[reference]).abs().mean())
    challenger_mae = float((frame["actual"] - frame[challenger]).abs().mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "comparison": comparison,
        "horizon_h": horizon,
        "reference": reference,
        "challenger": challenger,
        "n": len(frame),
        "days": len(daily),
        "reference_mae": reference_mae,
        "challenger_mae": challenger_mae,
        "gain_pct": 100 * (reference_mae - challenger_mae) / max(reference_mae, 1e-12),
        "delta_mae_ci95_low": float(low),
        "delta_mae_ci95_high": float(high),
        "challenger_better_significantly": bool(high < 0),
    }


def _materialize_timescale(
    full_frame: pd.DataFrame,
    run_id: str,
) -> int:
    feature_names = _all_feature_names()
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS features")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TIMESCALE_TABLE} (
                    as_of_time TIMESTAMPTZ NOT NULL,
                    feature_version TEXT NOT NULL,
                    model_eligible_flag SMALLINT NOT NULL,
                    exclusion_reason TEXT,
                    split TEXT,
                    target_arrivals_next_6h REAL,
                    target_arrivals_next_12h REAL,
                    target_arrivals_next_24h REAL,
                    feature_payload JSONB NOT NULL,
                    ingestion_run_id UUID NOT NULL,
                    PRIMARY KEY (as_of_time, feature_version)
                )
                """
            )
            cursor.execute(
                f"DELETE FROM {TIMESCALE_TABLE} WHERE feature_version=%s",
                (ENRICHMENT_VERSION,),
            )
            rows = []
            for row in full_frame.itertuples(index=False):
                payload = {
                    feature: _clean_json(getattr(row, feature))
                    for feature in feature_names
                }
                rows.append(
                    (
                        row.as_of_time.to_pydatetime(),
                        ENRICHMENT_VERSION,
                        int(row.model_eligible_flag),
                        row.exclusion_reason,
                        None if pd.isna(row.split) else str(row.split),
                        _clean_json(row.target_arrivals_next_6h),
                        _clean_json(row.target_arrivals_next_12h),
                        _clean_json(row.target_arrivals_next_24h),
                        Json(payload),
                        run_id,
                    )
                )
            execute_values(
                cursor,
                f"""
                INSERT INTO {TIMESCALE_TABLE} (
                    as_of_time, feature_version, model_eligible_flag,
                    exclusion_reason, split, target_arrivals_next_6h,
                    target_arrivals_next_12h, target_arrivals_next_24h,
                    feature_payload, ingestion_run_id
                ) VALUES %s
                """,
                rows,
                page_size=500,
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS ix_port_arrival_flow_enriched_v1_eligible
                ON {TIMESCALE_TABLE} (feature_version, model_eligible_flag, as_of_time)
                """
            )
    return len(rows)


def _log_mlflow(
    output_dir: Path,
    decision: dict[str, Any],
    metrics_test: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("smart-port-arrival-flow-enrichment")
        with mlflow.start_run(run_name=ENRICHMENT_VERSION):
            mlflow.log_params(
                {
                    "enrichment_version": ENRICHMENT_VERSION,
                    "source_feature_version": SOURCE_FEATURE_VERSION,
                    "split_policy": "TEMPORAL_70_15_15_PURGED_24H",
                    "eta_policy": "EXCLUDED_NO_REVISION_HISTORY",
                    "feature_count": len(_all_feature_names()),
                }
            )
            for horizon in HORIZONS:
                selected = decision["selected_models"][str(horizon)]
                row = metrics_test[
                    (metrics_test["horizon_h"] == horizon)
                    & (metrics_test["model"] == selected)
                ].iloc[0]
                mlflow.log_metric(f"test_mae_{horizon}h", float(row["MAE"]))
                mlflow.log_metric(f"test_wape_{horizon}h", float(row["WAPE_PCT"]))
            mlflow.log_artifacts(str(output_dir), artifact_path="b56c")
        return "LOGGED"
    except Exception as exc:
        return f"ERROR: {exc}"


def _metric_value(
    metrics: pd.DataFrame,
    horizon: int,
    model: str,
    metric: str = "MAE",
) -> float:
    row = metrics[
        (metrics["horizon_h"] == horizon) & (metrics["model"] == model)
    ]
    if row.empty:
        raise RuntimeError(f"Missing metric for {horizon}h/{model}")
    return float(row.iloc[0][metric])


def run_b56c_arrival_flow_enrichment(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    materialize_timescale: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    upstream_b56b = _load_b56b_decision()
    b56a_readiness = _load_b56a_readiness()
    operational_quality_approved = bool(
        b56a_readiness.get("readiness", {}).get("occupancy")
    )
    source = _load_source()
    enriched = _build_enriched_frame(source, operational_quality_approved)
    checksum = _checksum(enriched)
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
        completeness, break_start = _detect_source_completeness(enriched)
        if break_start is None:
            safe_label_end = enriched["as_of_time"].max() + pd.Timedelta(hours=1)
        else:
            safe_label_end = break_start - pd.Timedelta(hours=23)
        pre_break = enriched[enriched["as_of_time"] < safe_label_end].copy()
        frame, split_audit = _temporal_split(pre_break)
        train = frame[frame["split"] == "TRAIN"]
        valid = frame[frame["split"] == "VALID"]
        test = frame[frame["split"] == "TEST"]
        if min(len(train), len(valid), len(test)) < 1000:
            raise RuntimeError("At least one temporal split is too small")

        leakage = _leakage_audit(source, enriched, split_audit)
        critical_failures = leakage[
            leakage["severity"].eq("CRITICAL") & ~leakage["passed"]
        ]
        if not critical_failures.empty:
            failed = critical_failures["check"].tolist()
            raise RuntimeError(f"Critical anti-leakage checks failed: {failed}")

        selection_eligible_models = {
            "HGB_LEGACY_CORE",
            "HGB_ENRICHED_HISTORY",
        }
        if operational_quality_approved:
            selection_eligible_models.add("HGB_ENRICHED_OPERATIONAL")

        all_metrics_valid = []
        all_metrics_test = []
        all_valid_predictions = []
        all_test_predictions = []
        models: dict[str, Any] = {}
        selected_models: dict[str, str] = {}
        selected_enriched_models: dict[str, str] = {}
        validation_uplift_vs_baseline: dict[str, float] = {}
        incremental_valid_gain: dict[str, float] = {}
        incremental_test_gain: dict[str, float] = {}

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
                    "selection_eligible": model_name in selection_eligible_models,
                }

            valid_metrics, valid_predictions = _evaluate_part(
                frame, "VALID", horizon, predictions
            )
            test_metrics, test_predictions = _evaluate_part(
                frame, "TEST", horizon, predictions
            )
            for metric_frame in (valid_metrics, test_metrics):
                metric_frame["selection_eligible"] = metric_frame["model"].isin(
                    set(BASELINE_NAMES) | selection_eligible_models
                )
                metric_frame["feature_count"] = metric_frame["model"].map(
                    {name: len(features) for name, features in MODEL_FEATURES.items()}
                ).fillna(0).astype("int64")

            selectable = valid_metrics[valid_metrics["selection_eligible"]]
            selected = str(selectable.sort_values(["MAE", "RMSE"]).iloc[0]["model"])
            enriched_candidates = valid_metrics[
                valid_metrics["model"].isin(
                    selection_eligible_models - {"HGB_LEGACY_CORE"}
                )
            ]
            selected_enriched = str(
                enriched_candidates.sort_values(["MAE", "RMSE"]).iloc[0]["model"]
            )
            best_baseline = valid_metrics[
                valid_metrics["model"].isin(BASELINE_NAMES)
            ].sort_values(["MAE", "RMSE"]).iloc[0]
            selected_row = valid_metrics[valid_metrics["model"].eq(selected)].iloc[0]
            selected_models[str(horizon)] = selected
            selected_enriched_models[str(horizon)] = selected_enriched
            validation_uplift_vs_baseline[str(horizon)] = float(
                100
                * (float(best_baseline["MAE"]) - float(selected_row["MAE"]))
                / max(float(best_baseline["MAE"]), 1e-12)
            )

            legacy_valid = _metric_value(valid_metrics, horizon, "HGB_LEGACY_CORE")
            enriched_valid = _metric_value(valid_metrics, horizon, selected_enriched)
            legacy_test = _metric_value(test_metrics, horizon, "HGB_LEGACY_CORE")
            enriched_test = _metric_value(test_metrics, horizon, selected_enriched)
            incremental_valid_gain[str(horizon)] = float(
                100 * (legacy_valid - enriched_valid) / max(legacy_valid, 1e-12)
            )
            incremental_test_gain[str(horizon)] = float(
                100 * (legacy_test - enriched_test) / max(legacy_test, 1e-12)
            )

            valid_predictions["selected_model"] = selected
            valid_predictions["selected_enriched_model"] = selected_enriched
            test_predictions["selected_model"] = selected
            test_predictions["selected_enriched_model"] = selected_enriched
            all_metrics_valid.append(valid_metrics)
            all_metrics_test.append(test_metrics)
            all_valid_predictions.append(valid_predictions)
            all_test_predictions.append(test_predictions)

        metrics_valid = pd.concat(all_metrics_valid, ignore_index=True)
        metrics_test = pd.concat(all_metrics_test, ignore_index=True)
        valid_predictions = pd.concat(all_valid_predictions, ignore_index=True)
        test_predictions = pd.concat(all_test_predictions, ignore_index=True)

        comparison_rows = []
        significant_incremental_gain: dict[str, bool] = {}
        for horizon in HORIZONS:
            part = test_predictions[test_predictions["horizon_h"] == horizon]
            selected_enriched = selected_enriched_models[str(horizon)]
            incremental = _bootstrap_comparison(
                part,
                horizon,
                "HGB_LEGACY_CORE",
                selected_enriched,
                "BEST_ENRICHED_VS_LEGACY_CORE",
            )
            comparison_rows.append(incremental)
            significant_incremental_gain[str(horizon)] = bool(
                incremental["challenger_better_significantly"]
            )
            comparison_rows.append(
                _bootstrap_comparison(
                    part,
                    horizon,
                    "HGB_ENRICHED_HISTORY",
                    "HGB_ENRICHED_HISTORY_WAVE",
                    "PAST_WAVE_ABLATION",
                )
            )
            comparison_rows.append(
                _bootstrap_comparison(
                    part,
                    horizon,
                    "HGB_ENRICHED_HISTORY",
                    "HGB_ENRICHED_OPERATIONAL",
                    "OPERATIONAL_DIAGNOSTIC_ABLATION",
                )
            )
        comparisons = pd.DataFrame(comparison_rows)

        target_stability = _target_stability(frame)
        stability_passed = bool(target_stability["stable_70_to_150_pct"].all())
        promotion_horizons = [
            horizon
            for horizon in HORIZONS
            if incremental_valid_gain[str(horizon)] >= MIN_PROMOTION_UPLIFT_PCT
            and incremental_test_gain[str(horizon)] > 0
            and significant_incremental_gain[str(horizon)]
        ]
        if not stability_passed:
            decision_status = "NEED_DATA_REPAIR"
        elif promotion_horizons:
            decision_status = "READY_FOR_ENRICHED_FLOW_MVP"
        else:
            decision_status = "NO_STABLE_ENRICHMENT_UPLIFT"

        full_output = enriched.copy()
        full_output["model_eligible_flag"] = 0
        full_output["exclusion_reason"] = "INCOMPLETE_SOURCE_PERIOD_OR_TARGET_WINDOW"
        full_output["split"] = "NOT_MODEL_ELIGIBLE"
        full_output.loc[frame.index, "model_eligible_flag"] = 1
        full_output.loc[frame.index, "exclusion_reason"] = None
        full_output.loc[frame.index, "split"] = frame["split"].astype("string")
        full_output["feature_version"] = ENRICHMENT_VERSION

        feature_contract = _feature_contract(operational_quality_approved)
        missingness = _missingness_report(frame)
        redundancy = _redundancy_report(train)
        timescale_rows = (
            _materialize_timescale(full_output, run_id)
            if materialize_timescale
            else 0
        )

        decision = {
            "status": decision_status,
            "enrichment_version": ENRICHMENT_VERSION,
            "objective": "PORT_ARRIVAL_COUNTS_NEXT_6H_12H_24H",
            "full_rows": len(full_output),
            "model_rows": len(frame),
            "train_rows": len(train),
            "valid_rows": len(valid),
            "test_rows": len(test),
            "purged_rows": int((frame["split"] == "PURGED").sum()),
            "excluded_incomplete_rows": len(full_output) - len(frame),
            "source_completeness_break_start": break_start,
            "safe_label_end_exclusive": safe_label_end,
            "feature_count": len(_all_feature_names()),
            "selected_models": selected_models,
            "selected_enriched_models": selected_enriched_models,
            "validation_uplift_vs_best_baseline_pct": validation_uplift_vs_baseline,
            "incremental_valid_gain_vs_legacy_core_pct": incremental_valid_gain,
            "incremental_test_gain_vs_legacy_core_pct": incremental_test_gain,
            "significant_incremental_gain": significant_incremental_gain,
            "promotion_horizons": promotion_horizons,
            "minimum_promotion_uplift_pct": MIN_PROMOTION_UPLIFT_PCT,
            "operational_quality_approved": operational_quality_approved,
            "operational_model_selection_eligible": operational_quality_approved,
            "target_stability_passed": stability_passed,
            "critical_leakage_violations": 0,
            "eta_features_used": False,
            "eta_exclusion_reason": "NO_AS_OF_REVISION_HISTORY",
            "weather_policy": "PAST_OBSERVATIONS_ONLY_DIAGNOSTIC_ABLATION",
            "weather_model_selection_eligible": False,
            "official_protocol": "TEMPORAL_70_15_15_PURGED_24H_VALID_SELECTION_TEST_FINAL",
            "selection_used_test": False,
            "timescale_table": TIMESCALE_TABLE if materialize_timescale else None,
            "timescale_rows": timescale_rows,
            "upstream_b56b_status": upstream_b56b.get("status"),
            "next_block": (
                "B56D_PROBABILISTIC_ARRIVAL_FLOW"
                if decision_status == "READY_FOR_ENRICHED_FLOW_MVP"
                else (
                    "B56C_DATA_REPAIR"
                    if decision_status == "NEED_DATA_REPAIR"
                    else "B56C_EXTERNAL_OPERATIONAL_DATA_ACQUISITION"
                )
            ),
        }

        with tempfile.TemporaryDirectory(prefix="b56c-") as temporary:
            output_dir = Path(temporary)
            completeness.to_csv(output_dir / "00_source_completeness_by_month.csv", index=False)
            split_audit.to_csv(output_dir / "01_temporal_split_audit.csv", index=False)
            feature_contract.to_csv(output_dir / "02_feature_contract.csv", index=False)
            missingness.to_csv(output_dir / "03_feature_missingness.csv", index=False)
            leakage.to_csv(output_dir / "04_anti_leakage_audit.csv", index=False)
            redundancy.to_csv(output_dir / "05_feature_redundancy.csv", index=False)
            metrics_valid.to_csv(output_dir / "06_metrics_valid.csv", index=False)
            metrics_test.to_csv(output_dir / "07_metrics_test.csv", index=False)
            comparisons.to_csv(output_dir / "08_incremental_ablation_bootstrap.csv", index=False)
            target_stability.to_csv(output_dir / "09_target_stability.csv", index=False)
            decision_path = output_dir / "10_b56c_decision.json"
            decision_path.write_text(
                json.dumps(_clean_json(decision), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            full_output.to_parquet(
                output_dir / "arrival_flow_enriched_full.parquet", index=False
            )
            frame.to_parquet(
                output_dir / "arrival_flow_enriched_model_ready.parquet", index=False
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

            (output_dir / "README_B56C.md").write_text(
                "\n".join(
                    [
                        "# B56C Arrival Flow Feature Enrichment",
                        "",
                        f"Decision: **{decision_status}**",
                        "",
                        "Targets: arrival counts over the next 6h, 12h and 24h.",
                        "Features: calendar, strictly past arrival history, diagnostic",
                        "departure/occupancy state, and strictly past wave observations.",
                        "Final ETA values are excluded because revision timestamps are absent.",
                        "Model selection uses validation only; test remains final evaluation.",
                    ]
                ),
                encoding="utf-8",
            )

            mlflow_status = _log_mlflow(output_dir, decision, metrics_test)
            client = _s3_client()
            uploaded = {}
            for path in sorted(output_dir.iterdir()):
                if path.name == "arrival_flow_enriched_full.parquet":
                    key = f"datasets/b56c/{output_prefix}/{path.name}"
                elif path.name == "arrival_flow_enriched_model_ready.parquet":
                    key = f"datasets/b56c/{output_prefix}/{path.name}"
                elif path.name in {"valid_predictions.parquet", "test_predictions.parquet"}:
                    key = f"predictions/b56c/{output_prefix}/{path.name}"
                elif path.suffix == ".pkl":
                    key = f"models/b56c/{output_prefix}/{path.name}"
                elif path == decision_path:
                    key = f"configs/b56c/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b56c/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload(client, path, output_bucket, key)

        metadata = {
            **decision,
            "checksum": checksum,
            "mlflow_status": mlflow_status,
            "outputs": uploaded,
            "output_prefix": f"s3://{output_bucket}/reports/b56c/{output_prefix}/",
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
            {"enrichment_version": ENRICHMENT_VERSION},
            error_message=str(exc),
        )
        raise
