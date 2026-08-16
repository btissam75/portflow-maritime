from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd


MODEL_VERSION = "b62a-governed-metocean-tail-challenger-v1"
DATASET_VERSION = "b62a-metocean-tail-augmented-train-v1"
SOURCE_B62_VERSION = "b62-weather-wave-vessel-autogluon-v1"
HORIZONS_H = (6, 12, 24, 48, 72)
CHALLENGER_TARGETS = ("wave_height_m", "wave_period_s")
LAGS_H = (0, 1, 3, 6, 12, 24, 48, 72, 168)
ROLLING_WINDOWS_H = (6, 24, 72, 168)
CALIBRATION_DAYS = 180
EXPECTED_COVERAGE = 0.80

FEATURE_SIGNALS = (
    "wave_height_m",
    "wave_period_s",
    "wave_direction_sin",
    "wave_direction_cos",
    "wind_speed_ms",
    "wind_direction_sin",
    "wind_direction_cos",
    "pressure_hpa",
    "visibility_m",
    "precipitation",
)
ROLLING_SIGNALS = (
    "wave_height_m",
    "wave_period_s",
    "wind_speed_ms",
    "pressure_hpa",
    "visibility_m",
    "precipitation",
)
CALENDAR_FEATURES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
)


def target_column(variable: str, horizon_h: int) -> str:
    return f"target__{variable}__h{int(horizon_h)}"


def target_role_column(horizon_h: int) -> str:
    return f"target_role__h{int(horizon_h)}"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("x__")]


def target_columns() -> list[str]:
    return [
        target_column(variable, horizon_h)
        for variable in CHALLENGER_TARGETS
        for horizon_h in HORIZONS_H
    ]


def build_supervised_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    required = {
        "observed_at",
        "evaluation_role",
        *FEATURE_SIGNALS,
        *CALENDAR_FEATURES,
    }
    missing = sorted(required.difference(hourly.columns))
    if missing:
        raise ValueError(f"B62A hourly source is missing columns: {missing}")
    source = hourly.copy().sort_values("observed_at").reset_index(drop=True)
    columns: dict[str, Any] = {
        "issue_at": pd.to_datetime(source["observed_at"], errors="raise", utc=True),
        "evaluation_role": source["evaluation_role"].astype(str),
        "data_origin": "REAL_METOCEAN",
        "sample_weight": 1.0,
    }
    for signal in FEATURE_SIGNALS:
        values = pd.to_numeric(source[signal], errors="coerce").astype("float64")
        for lag_h in LAGS_H:
            columns[f"x__{signal}__lag{lag_h}"] = values.shift(lag_h)
    for signal in ROLLING_SIGNALS:
        history = pd.to_numeric(source[signal], errors="coerce").shift(1)
        for window_h in ROLLING_WINDOWS_H:
            rolling = history.rolling(window_h, min_periods=max(3, window_h // 4))
            columns[f"x__{signal}__mean{window_h}"] = rolling.mean()
            columns[f"x__{signal}__std{window_h}"] = rolling.std()
            columns[f"x__{signal}__max{window_h}"] = rolling.max()
    for column in CALENDAR_FEATURES:
        columns[f"x__{column}"] = pd.to_numeric(source[column], errors="coerce")
    for horizon_h in HORIZONS_H:
        columns[target_role_column(horizon_h)] = source["evaluation_role"].shift(
            -horizon_h
        )
        for variable in CHALLENGER_TARGETS:
            columns[target_column(variable, horizon_h)] = pd.to_numeric(
                source[variable], errors="coerce"
            ).shift(-horizon_h)
    output = pd.DataFrame(columns)
    output["dataset_version"] = DATASET_VERSION
    return output


def split_train_calibration(
    supervised: pd.DataFrame,
    calibration_days: int = CALIBRATION_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    train = supervised.loc[supervised["evaluation_role"].eq("TRAIN")].copy()
    if train.empty:
        raise ValueError("B62A has no TRAIN rows")
    cutoff = train["issue_at"].max() - pd.Timedelta(days=calibration_days)
    model_train_end = cutoff - pd.Timedelta(hours=max(HORIZONS_H))
    model_train = train.loc[train["issue_at"].lt(model_train_end)].copy()
    calibration = train.loc[train["issue_at"].ge(cutoff)].copy()
    if len(model_train) < 24 * 365 * 2:
        raise ValueError("B62A model TRAIN contains less than two years")
    if len(calibration) < 24 * 90:
        raise ValueError("B62A calibration contains less than 90 days")
    model_train["training_role"] = "MODEL_TRAIN_REAL"
    calibration["training_role"] = "TRAIN_CALIBRATION_REAL_ONLY"
    return model_train, calibration, cutoff


def tail_thresholds(model_train: pd.DataFrame, quantile: float = 0.90) -> dict[str, float]:
    if not 0.80 <= quantile <= 0.99:
        raise ValueError("Tail quantile must be between 0.80 and 0.99")
    thresholds: dict[str, float] = {}
    for variable in CHALLENGER_TARGETS:
        values = []
        for horizon_h in HORIZONS_H:
            column = target_column(variable, horizon_h)
            target = pd.to_numeric(model_train[column], errors="coerce")
            same_role = model_train[target_role_column(horizon_h)].eq("TRAIN")
            values.append(target.loc[same_role])
        pooled = pd.concat(values, ignore_index=True).dropna()
        if pooled.empty:
            raise ValueError(f"No TRAIN target values for {variable}")
        thresholds[variable] = float(pooled.quantile(quantile))
    return thresholds


def _apply_group_scale(frame: pd.DataFrame, signal: str, scale: np.ndarray) -> None:
    columns = [column for column in frame if column.startswith(f"x__{signal}__")]
    if columns:
        frame.loc[:, columns] = frame[columns].mul(scale, axis=0)


def _apply_group_offset(frame: pd.DataFrame, signal: str, offset: np.ndarray) -> None:
    columns = [column for column in frame if column.startswith(f"x__{signal}__")]
    if columns:
        frame.loc[:, columns] = frame[columns].add(offset, axis=0)


def generate_tail_augmentation(
    model_train: pd.DataFrame,
    synthetic_rows: int = 8_000,
    synthetic_weight: float = 0.10,
    tail_quantile: float = 0.90,
    seed: int = 20260811,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if synthetic_rows < 100:
        raise ValueError("At least 100 synthetic rows are required")
    if not 0.0 < synthetic_weight <= 0.25:
        raise ValueError("Synthetic sample weight must be in (0, 0.25]")
    thresholds = tail_thresholds(model_train, tail_quantile)
    tail_mask = pd.Series(False, index=model_train.index)
    for variable in CHALLENGER_TARGETS:
        for horizon_h in HORIZONS_H:
            column = target_column(variable, horizon_h)
            tail_mask |= pd.to_numeric(model_train[column], errors="coerce").ge(
                thresholds[variable]
            ) & model_train[target_role_column(horizon_h)].eq("TRAIN")
    parents = model_train.loc[tail_mask].copy()
    if len(parents) < 100:
        raise ValueError(f"Only {len(parents)} rare TRAIN parents were found")

    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(parents), size=synthetic_rows)
    synthetic = parents.iloc[take].copy().reset_index(drop=True)
    synthetic["source_parent_at"] = synthetic["issue_at"]
    synthetic["scenario_id"] = [f"b62a-{seed}-{index:06d}" for index in range(len(synthetic))]
    synthetic["issue_at"] = pd.NaT
    synthetic["evaluation_role"] = "TRAIN_SYNTHETIC_SUPPLEMENT"
    synthetic["training_role"] = "MODEL_TRAIN_SYNTHETIC_LOW_WEIGHT"
    synthetic["data_origin"] = "PHYSICS_CONSTRAINED_COUNTERFACTUAL_FROM_REAL_TRAIN"
    synthetic["sample_weight"] = float(synthetic_weight)

    wave_height_scale = rng.lognormal(mean=0.03, sigma=0.05, size=synthetic_rows)
    wave_period_scale = rng.lognormal(mean=0.01, sigma=0.03, size=synthetic_rows)
    wind_scale = rng.lognormal(mean=0.02, sigma=0.06, size=synthetic_rows)
    precipitation_scale = rng.lognormal(mean=0.00, sigma=0.10, size=synthetic_rows)
    visibility_scale = rng.lognormal(mean=-0.01, sigma=0.05, size=synthetic_rows)
    pressure_offset = rng.normal(loc=-0.25, scale=0.75, size=synthetic_rows)

    _apply_group_scale(synthetic, "wave_height_m", wave_height_scale)
    _apply_group_scale(synthetic, "wave_period_s", wave_period_scale)
    _apply_group_scale(synthetic, "wind_speed_ms", wind_scale)
    _apply_group_scale(synthetic, "precipitation", precipitation_scale)
    _apply_group_scale(synthetic, "visibility_m", visibility_scale)
    _apply_group_offset(synthetic, "pressure_hpa", pressure_offset)

    for horizon_h in HORIZONS_H:
        height = target_column("wave_height_m", horizon_h)
        period = target_column("wave_period_s", horizon_h)
        synthetic[height] = pd.to_numeric(synthetic[height], errors="coerce").mul(
            wave_height_scale
        ).clip(0.0, 30.0)
        synthetic[period] = pd.to_numeric(synthetic[period], errors="coerce").mul(
            wave_period_scale
        ).clip(0.5, 40.0)
    return synthetic, thresholds


def make_stress_features(
    model_train: pd.DataFrame,
    thresholds: dict[str, float],
    scenarios: int = 500,
    seed: int = 20260812,
) -> pd.DataFrame:
    if scenarios < 10:
        raise ValueError("At least ten stress scenarios are required")
    mask = pd.Series(False, index=model_train.index)
    for variable, threshold in thresholds.items():
        for horizon_h in HORIZONS_H:
            mask |= pd.to_numeric(
                model_train[target_column(variable, horizon_h)], errors="coerce"
            ).ge(threshold)
    parents = model_train.loc[mask].copy()
    if parents.empty:
        raise ValueError("No real tail parent is available for stress testing")
    rng = np.random.default_rng(seed)
    source = parents.iloc[rng.integers(0, len(parents), size=scenarios)].copy()
    source = source.reset_index(drop=True)
    source["source_parent_at"] = source["issue_at"]
    source["stress_scenario_id"] = [f"b62a-stress-{index:05d}" for index in range(scenarios)]
    source["stress_role"] = "SYNTHETIC_STRESS_NO_PERFORMANCE_CLAIM"
    _apply_group_scale(source, "wind_speed_ms", np.full(scenarios, 1.35))
    _apply_group_offset(source, "pressure_hpa", np.full(scenarios, -8.0))
    _apply_group_scale(source, "visibility_m", np.full(scenarios, 0.50))
    _apply_group_scale(source, "precipitation", np.full(scenarios, 1.50))
    return source


def finite_frame_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(selected, index=True).values.tobytes())
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
