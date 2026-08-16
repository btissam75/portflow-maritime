from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values
from sklearn.ensemble import HistGradientBoostingRegressor

from prefect_flows.b58cc_core import (
    assign_purged_split,
    block_bootstrap_error_delta,
    circular_absolute_error,
    decide_marginal_family,
)


MODEL_VERSION = "b58cc-observed-masking-feature-ablation-v3-calendar-audited"
DATASET_VERSION = "b58cc-wave-external-weather-ablation-v2"
SOURCE_NAME = "b58cc_weather_feature_ablation"
DATASET_NAME = "maritime_wave_external_weather_ablation"
SOURCE_BUCKET = "gold-maritime"
SOURCE_KEY = (
    "datasets/b58a/version=1/"
    "maritime_weather_hourly_past_only_v1.parquet"
)
EXTERNAL_TABLE = "features.maritime_external_weather_hourly_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=3"
HORIZONS_H = (6, 12, 24, 48, 72)
OFFICIAL_LATENCY_H = 3
PURGE_H = 72
RANDOM_SEED = 20260802
BOOTSTRAP_ITERATIONS = 500

TARGET_COMPONENTS = {
    "wave_height_m": "target_wave_height_m",
    "wave_period_s": "target_wave_period_s",
    "direction_sin": "target_direction_sin",
    "direction_cos": "target_direction_cos",
}
TARGETS = ("wave_height_m", "wave_period_s", "wave_direction_deg")
TRACKS = (
    "WAVE_ONLY",
    "ATMOSPHERE",
    "ATMOSPHERE_VISIBILITY",
    "FULL_WEATHER",
)
MARGINAL_COMPARISONS = (
    ("ATMOSPHERE", "WAVE_ONLY", "ATMOSPHERE"),
    ("ATMOSPHERE_VISIBILITY", "ATMOSPHERE", "VISIBILITY"),
    ("FULL_WEATHER", "ATMOSPHERE_VISIBILITY", "MARINE_CURRENT"),
)
KNOWN_FUTURE_CALENDAR_FEATURES = frozenset(
    {
        "target_hour_sin",
        "target_hour_cos",
        "target_day_sin",
        "target_day_cos",
        "target_weekend",
    }
)
FORBIDDEN_MODEL_FEATURE_TOKENS = (
    "target_wave_",
    "target_direction_",
    "future_",
    "actual_",
    "pred__",
    "available_at",
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


def _query_frame(query: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _load_upstream_contracts() -> dict[str, Any]:
    frame = _query_frame(
        """
        SELECT DISTINCT ON (source_name)
            source_name, status, checksum, metadata
        FROM audit.ingestion_run
        WHERE source_name IN (
            'b58a_weather_timeseries_audit',
            'b58cb_prefect_external_weather_enrichment'
        )
          AND status='SUCCESS'
        ORDER BY source_name, started_at DESC
        """
    )
    if set(frame["source_name"]) != {
        "b58a_weather_timeseries_audit",
        "b58cb_prefect_external_weather_enrichment",
    }:
        raise RuntimeError("B58A and B58C-B SUCCESS contracts are required")
    contracts: dict[str, Any] = {}
    for row in frame.itertuples(index=False):
        metadata = row.metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        contracts[row.source_name] = {
            "status": row.status,
            "checksum": row.checksum,
            "metadata": metadata,
        }
    b58a = contracts["b58a_weather_timeseries_audit"]["metadata"]
    b58cb = contracts["b58cb_prefect_external_weather_enrichment"]["metadata"]
    b58a_decision = b58a.get("decision") or b58a.get("status")
    if b58a_decision != "READY_FOR_WAVE_ONLY_TEMPORAL_BASELINES":
        raise RuntimeError(f"B58A contract is not ready: {b58a_decision}")
    if not bool(b58cb.get("integrity_passed")):
        raise RuntimeError("B58C-B integrity gate is not passed")
    if not bool(b58cb.get("atmospheric_track_ready")):
        raise RuntimeError("B58C-B atmospheric track is not ready")
    if b58cb.get("reanalysis_status") != "RESEARCH_ONLY_NOT_HISTORICALLY_AVAILABLE":
        raise RuntimeError("B58C-B retrospective availability contract changed")
    return contracts


def _load_wave_source(client) -> pd.DataFrame:
    payload = client.get_object(Bucket=SOURCE_BUCKET, Key=SOURCE_KEY)["Body"].read()
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
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    frame = frame.sort_values("observed_at").reset_index(drop=True)
    if frame["observed_at"].isna().any() or frame["observed_at"].duplicated().any():
        raise RuntimeError("B58A observed_at is invalid or duplicated")
    expected = pd.date_range(
        frame["observed_at"].min(), frame["observed_at"].max(), freq="h", tz="UTC"
    )
    if len(frame) != len(expected):
        raise RuntimeError("B58A is not an uninterrupted hourly grid")
    if frame[["wave_height_m", "wave_period_s", "wave_direction_deg"]].isna().any().any():
        raise RuntimeError("B58A wave targets contain missing values")
    return frame


def _load_external_weather() -> pd.DataFrame:
    frame = _query_frame(
        f"""
        SELECT observed_at, wind_speed_ms, wind_direction_deg,
               surface_current_ms, visibility_m, pressure_hpa,
               sea_surface_temperature, availability_semantics
        FROM {EXTERNAL_TABLE}
        ORDER BY observed_at
        """
    )
    if frame.empty:
        raise RuntimeError("B58C-B external weather table is empty")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    if frame["observed_at"].isna().any() or frame["observed_at"].duplicated().any():
        raise RuntimeError("External weather timestamps are invalid or duplicated")
    semantics = set(frame["availability_semantics"].dropna().astype(str))
    if semantics != {"RETROSPECTIVE_REANALYSIS_RESEARCH_ONLY"}:
        raise RuntimeError(f"Unexpected external availability semantics: {semantics}")
    return frame


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


def _external_time_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    result = frame.copy()
    numeric = (
        "wind_speed_ms",
        "pressure_hpa",
        "visibility_m",
        "surface_current_ms",
        "sea_surface_temperature",
    )
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    direction = np.deg2rad(pd.to_numeric(result["wind_direction_deg"], errors="coerce"))
    result["wind_direction_sin"] = np.sin(direction)
    result["wind_direction_cos"] = np.cos(direction)

    groups = {"atmosphere": [], "visibility": [], "marine": []}
    specifications = {
        "atmosphere": (
            "wind_speed_ms",
            "pressure_hpa",
            "wind_direction_sin",
            "wind_direction_cos",
        ),
        "visibility": ("visibility_m",),
        "marine": ("surface_current_ms", "sea_surface_temperature"),
    }
    for group, columns in specifications.items():
        for column in columns:
            for lag_h in (0, 3, 6, 12, 24):
                feature = f"ext_{column}_lag_{lag_h}h"
                result[feature] = result[column].shift(lag_h)
                groups[group].append(feature)
            if column not in {"wind_direction_sin", "wind_direction_cos"}:
                for window_h in (6, 24):
                    feature = f"ext_{column}_mean_{window_h}h"
                    result[feature] = result[column].rolling(
                        window_h, min_periods=max(1, window_h // 2)
                    ).mean()
                    groups[group].append(feature)
        availability = f"ext_{group}_available_flag"
        result[availability] = result[list(columns)].notna().all(axis=1).astype("float32")
        groups[group].append(availability)

    result["ext_wind_speed_trend_6h"] = (
        result["wind_speed_ms"] - result["wind_speed_ms"].shift(6)
    )
    result["ext_pressure_trend_6h"] = (
        result["pressure_hpa"] - result["pressure_hpa"].shift(6)
    )
    groups["atmosphere"].extend(
        ["ext_wind_speed_trend_6h", "ext_pressure_trend_6h"]
    )
    keep = ["observed_at", *sum(groups.values(), [])]
    return result[keep], groups


def _wave_feature_columns(frame: pd.DataFrame) -> list[str]:
    direct = [
        "observation_count",
        "source_count",
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
    past = [column for column in frame.columns if column.startswith("past_wave_")]
    features = [column for column in [*direct, *past] if column in frame.columns]
    if not past:
        raise RuntimeError("Past-only wave features were not found")
    forbidden = [
        column
        for column in features
        if any(token in column.lower() for token in ("target_", "future_", "actual_"))
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden wave features: {forbidden}")
    return features


def _build_examples(
    wave: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    engineered_external, external_groups = _external_time_features(external)
    merged = wave.merge(
        engineered_external,
        on="observed_at",
        how="left",
        validate="one_to_one",
    )
    wave_features = _wave_feature_columns(merged)
    all_external = sum(external_groups.values(), [])
    source_features = [*wave_features, *all_external]
    boundaries = {
        "train_boundary": merged.loc[int(len(merged) * 0.70), "observed_at"],
        "valid_boundary": merged.loc[int(len(merged) * 0.85), "observed_at"],
    }
    blocks: list[pd.DataFrame] = []
    for horizon_h in HORIZONS_H:
        issue_indices = np.arange(OFFICIAL_LATENCY_H, len(merged) - horizon_h)
        source_indices = issue_indices - OFFICIAL_LATENCY_H
        target_indices = issue_indices + horizon_h
        source = merged.iloc[source_indices][source_features].reset_index(drop=True)
        source = source.apply(pd.to_numeric, errors="coerce")
        issue_at = merged.iloc[issue_indices]["observed_at"].reset_index(drop=True)
        target_at = merged.iloc[target_indices]["observed_at"].reset_index(drop=True)
        block = source.copy()
        block.insert(0, "issue_at", issue_at)
        block.insert(1, "source_at", merged.iloc[source_indices]["observed_at"].to_numpy())
        block.insert(2, "target_at", target_at)
        block["horizon_h"] = float(horizon_h)
        block["latency_h"] = float(OFFICIAL_LATENCY_H)
        for calendar in (
            _calendar_features(issue_at, "issue"),
            _calendar_features(target_at, "target"),
        ):
            for column in calendar.columns:
                block[column] = calendar[column].to_numpy()
        block["target_wave_height_m"] = merged.iloc[target_indices]["wave_height_m"].to_numpy()
        block["target_wave_period_s"] = merged.iloc[target_indices]["wave_period_s"].to_numpy()
        block["target_wave_direction_deg"] = merged.iloc[target_indices]["wave_direction_deg"].to_numpy()
        radians = np.deg2rad(block["target_wave_direction_deg"].to_numpy(dtype="float64"))
        block["target_direction_sin"] = np.sin(radians)
        block["target_direction_cos"] = np.cos(radians)
        block["full_external_source_available"] = (
            block[
                [
                    "ext_atmosphere_available_flag",
                    "ext_visibility_available_flag",
                    "ext_marine_available_flag",
                ]
            ]
            .eq(1.0)
            .all(axis=1)
        )
        block["split"] = assign_purged_split(
            block["issue_at"],
            block["target_at"],
            boundaries["train_boundary"],
            boundaries["valid_boundary"],
            PURGE_H,
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
    track_features = {
        "WAVE_ONLY": [*wave_features, *engineered],
        "ATMOSPHERE": [
            *wave_features,
            *engineered,
            *external_groups["atmosphere"],
        ],
        "ATMOSPHERE_VISIBILITY": [
            *wave_features,
            *engineered,
            *external_groups["atmosphere"],
            *external_groups["visibility"],
        ],
        "FULL_WEATHER": [
            *wave_features,
            *engineered,
            *external_groups["atmosphere"],
            *external_groups["visibility"],
            *external_groups["marine"],
        ],
    }
    all_model_features = sorted(set(sum(track_features.values(), [])))
    examples[all_model_features] = examples[all_model_features].replace(
        [np.inf, -np.inf], np.nan
    ).astype("float32")
    split_audit = (
        examples.groupby("split", as_index=False)
        .agg(
            rows=("issue_at", "size"),
            first_issue_at=("issue_at", "min"),
            last_issue_at=("issue_at", "max"),
            last_target_at=("target_at", "max"),
            full_external_rows=("full_external_source_available", "sum"),
        )
    )
    return examples, track_features, split_audit


def _fit_tracks(
    examples: pd.DataFrame,
    track_features: dict[str, list[str]],
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, HistGradientBoostingRegressor]], pd.DataFrame]:
    active = examples.loc[examples["split"].isin(["TRAIN", "VALID", "TEST"])].copy()
    train = active.loc[active["split"].eq("TRAIN")]
    if train.empty:
        raise RuntimeError("Temporal split produced an empty train partition")
    retained_models: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
    inventory: list[dict[str, Any]] = []
    max_iter = int(os.getenv("B58CC_MAX_ITER", "160"))
    completed_models = 0
    for track in TRACKS:
        features = track_features[track]
        x_train = train[features].to_numpy(dtype="float32", copy=True)
        x_active = active[features].to_numpy(dtype="float32", copy=True)
        track_models: dict[str, HistGradientBoostingRegressor] = {}
        track_predictions: dict[str, np.ndarray] = {}
        for component, target_column in TARGET_COMPONENTS.items():
            _update_progress(
                run_id,
                "FITTING_TRACKS",
                track=track,
                component=component,
                feature_count=len(features),
                completed_models=completed_models,
                total_models=len(TRACKS) * len(TARGET_COMPONENTS),
            )
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.055,
                max_iter=max_iter,
                max_leaf_nodes=25,
                min_samples_leaf=64,
                l2_regularization=2.0,
                max_bins=127,
                early_stopping=False,
                random_state=RANDOM_SEED,
            )
            model.fit(
                x_train,
                train[target_column].to_numpy(dtype="float32", copy=False),
            )
            track_models[component] = model
            track_predictions[component] = model.predict(x_active).astype(
                "float32", copy=False
            )
            completed_models += 1
            inventory.append(
                {
                    "track": track,
                    "component": component,
                    "family": "HIST_GRADIENT_BOOSTING",
                    "features": len(features),
                    "train_rows": len(train),
                    "iterations": int(model.n_iter_),
                    "selection_split": "VALID",
                    "test_role": "DIAGNOSTIC_ONLY",
                }
            )
        active[f"pred__{track}__wave_height_m"] = np.clip(
            track_predictions["wave_height_m"], 0.0, None
        )
        active[f"pred__{track}__wave_period_s"] = np.clip(
            track_predictions["wave_period_s"], 0.0, None
        )
        active[f"pred__{track}__wave_direction_deg"] = (
            np.degrees(
                np.arctan2(
                    track_predictions["direction_sin"],
                    track_predictions["direction_cos"],
                )
            )
            + 360.0
        ) % 360.0
        if track == "FULL_WEATHER":
            retained_models[track] = track_models
        del x_train, x_active, track_predictions
        if track != "FULL_WEATHER":
            del track_models
        gc.collect()
    return active, retained_models, pd.DataFrame(inventory)


def _target_errors(
    actual: pd.Series, predicted: pd.Series, target: str
) -> tuple[np.ndarray, np.ndarray]:
    if target == "wave_direction_deg":
        signed = (predicted.to_numpy() - actual.to_numpy() + 180.0) % 360.0 - 180.0
        return np.abs(signed), signed
    signed = predicted.to_numpy(dtype="float64") - actual.to_numpy(dtype="float64")
    return np.abs(signed), signed


def _metric_rows(
    predictions: pd.DataFrame,
    tracks: tuple[str, ...] = TRACKS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cohorts = {
        "ALL_OBSERVED_MASKED": pd.Series(True, index=predictions.index),
        "COMMON_FULL_EXTERNAL": predictions["full_external_source_available"].astype(bool),
    }
    for split in ("VALID", "TEST"):
        for cohort, cohort_mask in cohorts.items():
            base = predictions.loc[predictions["split"].eq(split) & cohort_mask]
            for horizon_h in HORIZONS_H:
                subset = base.loc[base["horizon_h"].eq(float(horizon_h))]
                for track in tracks:
                    for target in TARGETS:
                        actual = subset[f"target_{target}"]
                        predicted = subset[f"pred__{track}__{target}"]
                        absolute, signed = _target_errors(actual, predicted, target)
                        rows.append(
                            {
                                "split": split,
                                "cohort": cohort,
                                "track": track,
                                "horizon_h": horizon_h,
                                "target": target,
                                "rows": len(subset),
                                "mae": float(np.mean(absolute)) if len(subset) else np.nan,
                                "rmse": float(np.sqrt(np.mean(np.square(signed)))) if len(subset) else np.nan,
                                "bias": float(np.mean(signed)) if len(subset) else np.nan,
                            }
                        )
    return pd.DataFrame(rows)


def _bootstrap_ablation(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions.loc[
        predictions["split"].eq("VALID")
        & predictions["full_external_source_available"].astype(bool)
    ].copy()
    rows: list[dict[str, Any]] = []
    for comparison_index, (candidate, baseline, family) in enumerate(MARGINAL_COMPARISONS):
        for horizon_index, horizon_h in enumerate(HORIZONS_H):
            subset = valid.loc[valid["horizon_h"].eq(float(horizon_h))].copy()
            for target_index, target in enumerate(TARGETS):
                baseline_error, _ = _target_errors(
                    subset[f"target_{target}"], subset[f"pred__{baseline}__{target}"], target
                )
                candidate_error, _ = _target_errors(
                    subset[f"target_{target}"], subset[f"pred__{candidate}__{target}"], target
                )
                working = pd.DataFrame(
                    {
                        "target_at": subset["target_at"].to_numpy(),
                        "baseline_error": baseline_error,
                        "candidate_error": candidate_error,
                    }
                )
                result = block_bootstrap_error_delta(
                    working,
                    "baseline_error",
                    "candidate_error",
                    iterations=BOOTSTRAP_ITERATIONS,
                    seed=RANDOM_SEED + comparison_index * 100 + horizon_index * 10 + target_index,
                )
                rows.append(
                    {
                        "selection_split": "VALID",
                        "cohort": "COMMON_FULL_EXTERNAL",
                        "family": family,
                        "baseline_track": baseline,
                        "candidate_track": candidate,
                        "horizon_h": horizon_h,
                        "target": target,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _family_decisions(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in bootstrap.groupby("family", sort=False):
        decision = decide_marginal_family(group)
        rows.append({"family": family, **decision})
    return pd.DataFrame(rows)


def _outage_stress(
    predictions: pd.DataFrame,
    models: dict[str, dict[str, HistGradientBoostingRegressor]],
    features: list[str],
) -> pd.DataFrame:
    test = predictions.loc[predictions["split"].eq("TEST")].copy()
    scenarios = {
        "NORMAL": (),
        "VISIBILITY_OUTAGE": ("ext_visibility_",),
        "MARINE_OUTAGE": ("ext_surface_current_", "ext_sea_surface_", "ext_marine_"),
        "VISIBILITY_AND_MARINE_OUTAGE": (
            "ext_visibility_",
            "ext_surface_current_",
            "ext_sea_surface_",
            "ext_marine_",
        ),
    }
    rows: list[dict[str, Any]] = []
    for scenario, prefixes in scenarios.items():
        x_test = test[features].copy()
        masked = [column for column in features if column.startswith(prefixes)] if prefixes else []
        if masked:
            x_test[masked] = np.nan
            for column in masked:
                if column.endswith("available_flag"):
                    x_test[column] = 0.0
        raw = {
            component: model.predict(x_test)
            for component, model in models["FULL_WEATHER"].items()
        }
        predicted_values = {
            "wave_height_m": np.clip(raw["wave_height_m"], 0.0, None),
            "wave_period_s": np.clip(raw["wave_period_s"], 0.0, None),
            "wave_direction_deg": (
                np.degrees(np.arctan2(raw["direction_sin"], raw["direction_cos"])) + 360.0
            )
            % 360.0,
        }
        for horizon_h in HORIZONS_H:
            mask = test["horizon_h"].eq(float(horizon_h)).to_numpy()
            for target in TARGETS:
                actual = test.loc[mask, f"target_{target}"]
                predicted = pd.Series(predicted_values[target][mask], index=actual.index)
                absolute, signed = _target_errors(actual, predicted, target)
                rows.append(
                    {
                        "split": "TEST_DIAGNOSTIC_ONLY",
                        "scenario": scenario,
                        "horizon_h": horizon_h,
                        "target": target,
                        "rows": int(mask.sum()),
                        "mae": float(np.mean(absolute)),
                        "rmse": float(np.sqrt(np.mean(np.square(signed)))),
                        "masked_features": len(masked),
                    }
                )
    result = pd.DataFrame(rows)
    normal = result.loc[result["scenario"].eq("NORMAL"), ["horizon_h", "target", "mae"]].rename(
        columns={"mae": "normal_mae"}
    )
    result = result.merge(normal, on=["horizon_h", "target"], how="left")
    result["degradation_pct"] = 100.0 * (result["mae"] - result["normal_mae"]) / result[
        "normal_mae"
    ]
    return result


def _leakage_audit(
    examples: pd.DataFrame,
    track_features: dict[str, list[str]],
) -> pd.DataFrame:
    active = examples.loc[examples["split"].isin(["TRAIN", "VALID", "TEST"])]
    train = active.loc[active["split"].eq("TRAIN")]
    valid = active.loc[active["split"].eq("VALID")]
    test = active.loc[active["split"].eq("TEST")]
    feature_names = set(sum(track_features.values(), []))
    forbidden = sorted(
        column
        for column in feature_names
        if column not in KNOWN_FUTURE_CALENDAR_FEATURES
        and any(
            token in column.lower()
            for token in FORBIDDEN_MODEL_FEATURE_TOKENS
        )
    )
    issue_at = active["issue_at"].reset_index(drop=True)
    target_at = active["target_at"].reset_index(drop=True)
    horizons = active["horizon_h"].reset_index(drop=True)
    expected_target_at = issue_at + pd.to_timedelta(horizons, unit="h")
    expected_calendar = _calendar_features(expected_target_at, "target")
    calendar_violation = target_at.ne(expected_target_at).to_numpy()
    missing_calendar = sorted(
        KNOWN_FUTURE_CALENDAR_FEATURES.difference(active.columns)
    )
    if missing_calendar:
        calendar_violation[:] = True
    else:
        for column in sorted(KNOWN_FUTURE_CALENDAR_FEATURES):
            observed = active[column].reset_index(drop=True).to_numpy(dtype="float64")
            expected = expected_calendar[column].to_numpy(dtype="float64")
            calendar_violation |= ~np.isclose(
                observed,
                expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=False,
            )
    calendar_violation_rows = int(calendar_violation.sum())
    checks = [
        (
            "SOURCE_STRICTLY_BEFORE_ISSUE",
            bool((active["source_at"] < active["issue_at"]).all()),
            "CRITICAL",
            int((active["source_at"] >= active["issue_at"]).sum()),
        ),
        (
            "TARGET_STRICTLY_AFTER_ISSUE",
            bool((active["target_at"] > active["issue_at"]).all()),
            "CRITICAL",
            int((active["target_at"] <= active["issue_at"]).sum()),
        ),
        (
            "TRAIN_TARGET_BEFORE_VALID_ISSUE",
            bool(train["target_at"].max() < valid["issue_at"].min()),
            "CRITICAL",
            0,
        ),
        (
            "VALID_TARGET_BEFORE_TEST_ISSUE",
            bool(valid["target_at"].max() < test["issue_at"].min()),
            "CRITICAL",
            0,
        ),
        (
            "NO_FORBIDDEN_FEATURE_COLUMNS",
            not forbidden,
            "CRITICAL",
            len(forbidden),
            ",".join(forbidden),
        ),
        (
            "KNOWN_FUTURE_CALENDAR_DERIVED_FROM_ISSUE_AND_HORIZON",
            calendar_violation_rows == 0,
            "CRITICAL",
            calendar_violation_rows,
            (
                "target_at=issue_at+horizon; deterministic calendar only"
                if not missing_calendar
                else f"missing={','.join(missing_calendar)}"
            ),
        ),
        (
            "SELECTION_USES_VALID_ONLY",
            True,
            "CRITICAL",
            0,
            "model and family selection use VALID; TEST is diagnostic only",
        ),
        (
            "RETROSPECTIVE_REANALYSIS_NOT_OPERATIONALLY_AVAILABLE",
            False,
            "EXPECTED_RESEARCH_LIMITATION",
            len(active),
            "ERA5 retrospective values were not available at historical issue time",
        ),
    ]
    checks = [
        (*check, "") if len(check) == 4 else check
        for check in checks
    ]
    return pd.DataFrame(
        checks,
        columns=["check", "passed", "severity", "violation_rows", "details"],
    )


def _checksum(contracts: dict[str, Any]) -> str:
    payload = {
        "model_version": MODEL_VERSION,
        "source_key": SOURCE_KEY,
        "upstream": {
            name: contract.get("checksum") for name, contract in contracts.items()
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _start_run(checksum: str) -> str:
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
                    Json(
                        {
                            "model_version": MODEL_VERSION,
                            "dataset_version": DATASET_VERSION,
                            "orchestrator": "PREFECT",
                            "training_executed": False,
                            "selection_used_test": False,
                            "production_promotion_allowed": False,
                        }
                    ),
                ),
            )
            return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    progress = _clean_json(
        {"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details}
    )
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
                (Json({"progress": progress}), run_id),
            )


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


def _materialize_metrics(run_id: str, metrics: pd.DataFrame) -> int:
    columns = [
        "run_id",
        "split",
        "cohort",
        "track",
        "horizon_h",
        "target",
        "rows",
        "mae",
        "rmse",
        "bias",
    ]
    payload = metrics.copy()
    payload.insert(0, "run_id", run_id)
    payload = payload[columns].replace({np.nan: None})
    values = [tuple(row) for row in payload.itertuples(index=False, name=None)]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit.weather_feature_ablation_metric_v1 (
                    run_id UUID NOT NULL REFERENCES audit.ingestion_run(run_id),
                    split TEXT NOT NULL,
                    cohort TEXT NOT NULL,
                    track TEXT NOT NULL,
                    horizon_h INTEGER NOT NULL,
                    target TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    mae DOUBLE PRECISION,
                    rmse DOUBLE PRECISION,
                    bias DOUBLE PRECISION,
                    PRIMARY KEY (run_id, split, cohort, track, horizon_h, target)
                )
                """
            )
            execute_values(
                cursor,
                f"INSERT INTO audit.weather_feature_ablation_metric_v1 ({', '.join(columns)}) VALUES %s",
                values,
                page_size=1_000,
            )
    return len(values)


def _upload(client, path: Path, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/octet-stream",
    }.get(path.suffix, "application/octet-stream")
    client.upload_file(
        str(path), OUTPUT_BUCKET, key, ExtraArgs={"ContentType": content_type}
    )
    client.head_object(Bucket=OUTPUT_BUCKET, Key=key)
    return f"s3://{OUTPUT_BUCKET}/{key}"


def run_b58cc_weather_ablation(force: bool = False) -> dict[str, Any]:
    contracts = _load_upstream_contracts()
    checksum = _checksum(contracts)
    run_id = _start_run(checksum)
    client = _s3_client()
    try:
        _update_progress(run_id, "LOADING_SOURCES")
        wave = _load_wave_source(client)
        external = _load_external_weather()
        examples, track_features, split_audit = _build_examples(wave, external)
        _update_progress(run_id, "EXAMPLES_READY", rows=len(examples))

        predictions, models, inventory = _fit_tracks(
            examples, track_features, run_id
        )
        _update_progress(run_id, "EVALUATING_ABLATIONS")
        metrics = _metric_rows(predictions)
        bootstrap = _bootstrap_ablation(predictions)
        family_decisions = _family_decisions(bootstrap)
        outage = _outage_stress(
            predictions, models, track_features["FULL_WEATHER"]
        )
        leakage = _leakage_audit(examples, track_features)
        critical_leakage = int(
            (
                leakage["severity"].eq("CRITICAL")
                & ~leakage["passed"].astype(bool)
            ).sum()
        )
        research_signal = bool(family_decisions["keep"].astype(bool).any())
        decision_name = (
            "NEED_TEMPORAL_OR_FEATURE_REPAIR"
            if critical_leakage
            else "RESEARCH_SIGNAL_FOUND_NEED_ISSUE_TIME_FORECASTS"
            if research_signal
            else "NO_STABLE_EXTERNAL_WEATHER_UPLIFT"
        )
        next_block = (
            "B58C_C2_TEMPORAL_OR_FEATURE_REPAIR"
            if critical_leakage
            else "B58C_D_ISSUE_TIME_WEATHER_FORECAST_COLLECTION"
            if research_signal
            else "B58B_WAVE_ONLY_REMAINS_REFERENCE"
        )
        valid_complete = metrics.loc[
            metrics["split"].eq("VALID")
            & metrics["cohort"].eq("COMMON_FULL_EXTERNAL")
        ]
        baseline = valid_complete.loc[valid_complete["track"].eq("WAVE_ONLY"), [
            "horizon_h", "target", "mae"
        ]].rename(columns={"mae": "baseline_mae"})
        track_scores = valid_complete.merge(
            baseline, on=["horizon_h", "target"], how="left"
        )
        track_scores["gain_pct_vs_wave_only"] = 100.0 * (
            track_scores["baseline_mae"] - track_scores["mae"]
        ) / track_scores["baseline_mae"]
        score_summary = (
            track_scores.groupby("track", as_index=False)
            .agg(
                median_gain_pct_vs_wave_only=("gain_pct_vs_wave_only", "median"),
                mean_gain_pct_vs_wave_only=("gain_pct_vs_wave_only", "mean"),
            )
            .sort_values("median_gain_pct_vs_wave_only", ascending=False)
        )
        selected_research_track = str(score_summary.iloc[0]["track"])

        decision = {
            "status": "SUCCESS",
            "decision": decision_name,
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "row_count": len(examples),
            "active_rows": len(predictions),
            "train_rows": int(predictions["split"].eq("TRAIN").sum()),
            "valid_rows": int(predictions["split"].eq("VALID").sum()),
            "test_rows": int(predictions["split"].eq("TEST").sum()),
            "purged_rows": int(examples["split"].eq("EXCLUDED_PURGE").sum()),
            "horizons_h": list(HORIZONS_H),
            "latency_h": OFFICIAL_LATENCY_H,
            "purge_h": PURGE_H,
            "selected_research_track": selected_research_track,
            "research_signal_found": research_signal,
            "critical_temporal_leakage_checks_failed": critical_leakage,
            "selection_split": "VALID",
            "test_role": "DIAGNOSTIC_ONLY",
            "selection_used_test": False,
            "training_executed": True,
            "source_modified": False,
            "synthetic_rows_created": 0,
            "availability_semantics": "RETROSPECTIVE_REANALYSIS_RESEARCH_ONLY",
            "historical_replay_allowed": False,
            "production_promotion_allowed": False,
            "navigation_use_allowed": False,
            "next_block": next_block,
        }

        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b58cc-") as temporary:
            directory = Path(temporary)
            files: dict[str, pd.DataFrame | dict[str, Any] | str] = {
                "01_temporal_split_audit.csv": split_audit,
                "02_feature_inventory.csv": pd.DataFrame(
                    [
                        {
                            "track": track,
                            "feature": feature,
                            "feature_role": (
                                "PAST_WAVE_OR_CALENDAR"
                                if not feature.startswith("ext_")
                                else "RETROSPECTIVE_EXTERNAL_RESEARCH_ONLY"
                            ),
                        }
                        for track, features in track_features.items()
                        for feature in features
                    ]
                ),
                "03_model_inventory.csv": inventory,
                "04_all_metrics.csv": metrics,
                "05_valid_marginal_bootstrap.csv": bootstrap,
                "06_family_decisions.csv": family_decisions,
                "07_valid_track_scores.csv": score_summary,
                "08_test_outage_stress.csv": outage,
                "09_anti_leakage_audit.csv": leakage,
                "10_b58cc_final_decision.json": decision,
                "README_B58CC.md": "\n".join(
                    [
                        "# B58C-C observed masking and feature ablation",
                        "",
                        f"Decision: {decision_name}",
                        "",
                        "Selection uses VALID only; TEST is diagnostic only.",
                        "External ERA5 features are retrospective research features.",
                        "A positive result requires issue-time forecast collection before production.",
                        "No Bronze/Core row was modified and no synthetic row was created.",
                    ]
                ),
            }
            test_predictions = predictions.loc[predictions["split"].eq("TEST"), [
                "issue_at",
                "source_at",
                "target_at",
                "horizon_h",
                "full_external_source_available",
                *[f"target_{target}" for target in TARGETS],
                *[
                    f"pred__{track}__{target}"
                    for track in TRACKS
                    for target in TARGETS
                ],
            ]]
            prediction_path = directory / "test_predictions_diagnostic.parquet"
            test_predictions.to_parquet(prediction_path, index=False)
            outputs[prediction_path.name] = _upload(
                client,
                prediction_path,
                f"predictions/b58cc/{OUTPUT_PREFIX}/{prediction_path.name}",
            )
            for name, content in files.items():
                path = directory / name
                if isinstance(content, pd.DataFrame):
                    content.to_csv(path, index=False)
                elif isinstance(content, dict):
                    path.write_text(
                        json.dumps(_clean_json(content), indent=2, ensure_ascii=True),
                        encoding="utf-8",
                    )
                else:
                    path.write_text(content, encoding="utf-8")
                category = "configs" if path.suffix == ".json" else "reports"
                outputs[name] = _upload(
                    client,
                    path,
                    f"{category}/b58cc/{OUTPUT_PREFIX}/{name}",
                )

        materialized_metrics = _materialize_metrics(run_id, metrics)
        metadata = {
            **decision,
            "checksum": checksum,
            "family_decisions": family_decisions.to_dict(orient="records"),
            "track_scores": score_summary.to_dict(orient="records"),
            "materialized_metric_rows": materialized_metrics,
            "metric_table": "audit.weather_feature_ablation_metric_v1",
            "outputs": outputs,
        }
        _finish_run(run_id, "SUCCESS", len(examples), metadata)
        return {"status": "SUCCESS", "run_id": run_id, "results": _clean_json(metadata)}
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "model_version": MODEL_VERSION,
                "dataset_version": DATASET_VERSION,
                "orchestrator": "PREFECT",
            },
            str(exc),
        )
        raise
