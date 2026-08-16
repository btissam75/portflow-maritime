from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


RANDOM_SEED = 20260805
SOURCE_DATASET_VERSION = "b61a-governed-port-call-enrichment-v1"
MODEL_VERSION = "b61b-multitask-temporal-survival-moe-v1"
RISK_NAMES = ("gt1", "gt3", "gt6")
RISK_TARGETS = (
    "target_delay_gt_1h",
    "target_delay_gt_3h",
    "target_delay_gt_6h",
)
HAZARD_HORIZONS = (6, 12, 24)
HAZARD_TARGETS = tuple(
    f"target_gt3_breach_within_{horizon}h" for horizon in HAZARD_HORIZONS
)
QUANTILES = (0.1, 0.5, 0.9)


IDENTIFIER_COLUMNS = [
    "port_call_id",
    "landmark_at",
    "split",
    "early_warning_eligible",
    "pre_breach_eligible",
    "per_call_sample_weight",
    "training_allowed",
    "validation_allowed",
    "test_allowed",
    "production_claim_allowed",
]

CATEGORICAL_FEATURES = [
    "port_code",
    "terminal_code",
    "vessel_type",
    "cargo_group",
    "current_plan_state",
]

# Compact, stable feature contract. The B61A registry remains authoritative; this
# list prevents a 395-column materialization from exhausting the 3 GB worker.
CORE_NUMERIC_FEATURES = [
    "arrival_delay_h",
    "planned_stay_h",
    "planned_berth_offset_h",
    "elapsed_since_arrival_h",
    "time_to_planned_departure_h",
    "overdue_h",
    "plan_progress_ratio",
    "berth_event_observed",
    "hours_since_berth_observed",
    "landmark_h",
    "vessel_history_prior_calls",
    "vessel_history_prior_mean_stay_h",
    "vessel_history_prior_std_stay_h",
    "vessel_history_prior_mean_delay_h",
    "vessel_history_prior_late_gt3_rate",
    "vessel_history_prior_late_gt6_rate",
    "vessel_history_last_stay_h",
    "vessel_history_last_delay_h",
    "vessel_history_recent5_mean_stay_h",
    "vessel_history_recent5_late_gt3_rate",
    "vessel_history_days_since_last_departure",
    "terminal_history_prior_calls",
    "terminal_history_prior_mean_stay_h",
    "terminal_history_prior_mean_delay_h",
    "terminal_history_prior_late_gt3_rate",
    "terminal_history_prior_late_gt6_rate",
    "cargo_history_prior_calls",
    "cargo_history_prior_mean_stay_h",
    "cargo_history_prior_mean_delay_h",
    "cargo_history_prior_late_gt3_rate",
    "port_history_prior_calls",
    "port_history_prior_mean_stay_h",
    "port_history_prior_mean_delay_h",
    "port_history_prior_late_gt3_rate",
    "port_history_prior_late_gt6_rate",
    "port_arrivals_last_6h",
    "port_departures_last_6h",
    "port_arrivals_last_24h",
    "port_departures_last_24h",
    "port_arrivals_last_168h",
    "port_departures_last_168h",
    "port_active_calls_observed",
    "port_flow_imbalance_6h",
    "port_flow_imbalance_24h",
    "port_completed_mean_delay_last_24h",
    "port_completed_late_gt3_rate_last_24h",
    "terminal_arrivals_last_6h",
    "terminal_departures_last_6h",
    "terminal_arrivals_last_24h",
    "terminal_departures_last_24h",
    "terminal_active_calls_observed",
    "terminal_flow_imbalance_6h",
    "terminal_flow_imbalance_24h",
    "terminal_completed_mean_delay_last_24h",
    "terminal_completed_late_gt3_rate_last_24h",
    "landmark_hour_sin",
    "landmark_hour_cos",
    "landmark_dow_sin",
    "landmark_dow_cos",
    "landmark_doy_sin",
    "landmark_doy_cos",
    "landmark_month_sin",
    "landmark_month_cos",
    "landmark_weekend",
    "known_event_any_now",
    "known_event_any_6h",
    "known_event_any_12h",
    "known_event_any_24h",
    "known_holiday_any_now",
    "known_holiday_any_24h",
    "known_national_holiday_now",
    "known_ramadan_now",
    "known_eid_fitr_now",
    "known_eid_adha_now",
    "known_hijri_new_year_now",
    "known_mawlid_now",
    "known_year_end_now",
    "known_event_count_now",
    "known_event_pre_7d",
    "known_event_post_7d",
    "known_days_to_next_event_90d",
    "known_days_since_event_90d",
    "known_ramadan_day",
    "known_ramadan_progress",
    "known_ramadan_last10",
    "calendar_month_end_last3d",
    "calendar_quarter_end_last7d",
    "calendar_year_end_last21d",
    "calendar_week_of_year_sin",
    "calendar_week_of_year_cos",
    "port_occupancy_proxy",
    "terminal_occupancy_proxy",
    "port_queue_pressure_proxy",
    "terminal_queue_pressure_proxy",
    "operational_pressure_index",
    "vessel_delay_susceptibility_bayes",
    "vessel_history_evidence_strength",
    "known_event_operational_pressure_interaction",
    "operational_pressure_ge_train_q90",
]

RESEARCH_NUMERIC_FEATURES = [
    "research_wave_height_m_lag3h",
    "research_wave_period_s_lag3h",
    "research_wind_speed_ms_lag3h",
    "research_surface_current_ms_lag3h",
    "research_visibility_m_lag3h",
    "research_pressure_hpa_lag3h",
    "research_precipitation_lag3h",
    "research_wave_height_m_roll_24h_mean",
    "research_wave_height_m_roll_24h_max",
    "research_wind_speed_ms_roll_24h_mean",
    "research_wind_speed_ms_roll_24h_max",
    "research_visibility_m_roll_24h_mean",
    "research_weather_age_h",
    "research_wave_energy_flux_kw_m",
    "research_wave_steepness",
    "research_wind_wave_alignment_cos",
    "research_gust_factor",
    "research_local_wave_height_m_ge_q90",
    "research_local_wind_speed_ms_ge_q90",
    "research_reference_low_visibility_1km_flag",
    "research_pressure_delta_3h",
    "research_pressure_delta_24h",
    "research_compound_marine_operational_pressure",
    "research_weather_source_completeness",
]

TARGET_COLUMNS = [
    "target_departure_delay_h",
    "target_departure_delay_class",
    *RISK_TARGETS,
    "target_remaining_h",
    "target_breach_gt3_observed",
    "target_breach_or_censor_h",
    *HAZARD_TARGETS,
]


@dataclass(frozen=True)
class FeatureContract:
    core_numeric: tuple[str, ...]
    research_numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def core(self) -> list[str]:
        return [*self.core_numeric, *self.categorical]

    @property
    def research(self) -> list[str]:
        return [*self.core_numeric, *self.research_numeric, *self.categorical]


def available_contract(columns: Iterable[str]) -> FeatureContract:
    present = set(columns)
    return FeatureContract(
        tuple(column for column in CORE_NUMERIC_FEATURES if column in present),
        tuple(column for column in RESEARCH_NUMERIC_FEATURES if column in present),
        tuple(column for column in CATEGORICAL_FEATURES if column in present),
    )


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def split_validation_roles(frame: pd.DataFrame) -> pd.Series:
    """Split VALID chronologically into model selection and calibration halves."""
    roles = pd.Series("UNUSED", index=frame.index, dtype="object")
    roles.loc[frame["split"].eq("TRAIN")] = "TRAIN_FIT"
    valid = frame.loc[frame["split"].eq("VALID"), ["landmark_at"]].copy()
    if valid.empty:
        raise ValueError("VALID split is empty")
    times = pd.to_datetime(valid["landmark_at"], utc=True)
    unique_times = np.sort(times.unique())
    boundary = pd.Timestamp(unique_times[max(1, len(unique_times) // 2) - 1])
    roles.loc[valid.index[times.le(boundary)]] = "VALID_SELECT"
    roles.loc[valid.index[times.gt(boundary)]] = "VALID_CALIBRATE"
    roles.loc[frame["split"].eq("TEST")] = "TEST_DIAGNOSTIC_ONLY"
    if not roles.eq("VALID_CALIBRATE").any():
        ordered = valid.assign(_time=times).sort_values("_time").index
        cut = max(1, len(ordered) // 2)
        roles.loc[ordered[:cut]] = "VALID_SELECT"
        roles.loc[ordered[cut:]] = "VALID_CALIBRATE"
    return roles


def delay_class_index(delay_hours: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(delay_hours, dtype="float64")
    result = np.zeros(len(values), dtype="int64")
    result[values > 1.0] = 1
    result[values > 3.0] = 2
    result[values > 6.0] = 3
    return result


def enforce_risk_order(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype="float64"), 0.0, 1.0)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected risk probabilities with shape (n, 3)")
    values[:, 1] = np.minimum(values[:, 0], values[:, 1])
    values[:, 2] = np.minimum(values[:, 1], values[:, 2])
    return values


def enforce_hazard_order(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype="float64"), 0.0, 1.0)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected cumulative hazard probabilities with shape (n, 3)")
    return np.maximum.accumulate(values, axis=1)


def enforce_quantile_order(predictions: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(predictions, dtype="float64"), 0.0, None)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected quantiles with shape (n, 3)")
    return np.sort(values, axis=1)


def risk_from_class_probabilities(class_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(class_probabilities, dtype="float64")
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("Expected four ordinal class probabilities")
    result = np.column_stack(
        [
            probabilities[:, 1:].sum(axis=1),
            probabilities[:, 2:].sum(axis=1),
            probabilities[:, 3],
        ]
    )
    return enforce_risk_order(result)


def expected_calibration_error(
    actual: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    actual = np.asarray(actual, dtype="float64")
    probability = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        upper = probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        mask = (probability >= edges[index]) & upper
        if mask.any():
            score += mask.mean() * abs(actual[mask].mean() - probability[mask].mean())
    return float(score)


def binary_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    actual = np.asarray(actual, dtype="int64")
    probability = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    predicted = probability >= threshold
    tp = int(np.sum(predicted & (actual == 1)))
    fp = int(np.sum(predicted & (actual == 0)))
    fn = int(np.sum(~predicted & (actual == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    auc = float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None
    ap = float(average_precision_score(actual, probability)) if len(np.unique(actual)) == 2 else None
    return {
        "rows": len(actual),
        "positive_rate": float(actual.mean()),
        "roc_auc": auc,
        "average_precision": ap,
        "brier": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "ece_10": expected_calibration_error(actual, probability),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    residual = np.asarray(actual, dtype="float64") - np.asarray(predicted, dtype="float64")
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def regression_metrics(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype="float64")
    predicted = enforce_quantile_order(quantiles)
    return {
        "rows": len(actual),
        "mae_p50": float(mean_absolute_error(actual, predicted[:, 1])),
        "rmse_p50": float(mean_squared_error(actual, predicted[:, 1]) ** 0.5),
        "bias_p50": float(np.mean(predicted[:, 1] - actual)),
        "pinball_p10": pinball_loss(actual, predicted[:, 0], 0.1),
        "pinball_p50": pinball_loss(actual, predicted[:, 1], 0.5),
        "pinball_p90": pinball_loss(actual, predicted[:, 2], 0.9),
        "coverage_p10_p90": float(np.mean((actual >= predicted[:, 0]) & (actual <= predicted[:, 2]))),
        "mean_interval_width": float(np.mean(predicted[:, 2] - predicted[:, 0])),
    }


def select_binary_threshold(actual: np.ndarray, probability: np.ndarray) -> float:
    best = (float("inf"), 0.5)
    actual = np.asarray(actual, dtype="int64")
    probability = np.asarray(probability, dtype="float64")
    for threshold in np.linspace(0.1, 0.9, 33):
        predicted = probability >= threshold
        fp = np.mean(predicted & (actual == 0))
        fn = np.mean(~predicted & (actual == 1))
        cost = fp + 1.5 * fn
        if cost < best[0]:
            best = (float(cost), float(threshold))
    return best[1]


def select_blend_weight(
    actual: np.ndarray,
    tabular: np.ndarray,
    sequence: np.ndarray,
    task: str,
) -> tuple[float, float]:
    """Return the tabular weight and VALID_SELECT objective."""
    best = (float("inf"), 1.0)
    for weight in np.linspace(0.0, 1.0, 9):
        prediction = weight * tabular + (1.0 - weight) * sequence
        if task == "BINARY":
            prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
            objective = brier_score_loss(actual, prediction) + 0.2 * log_loss(
                actual, prediction, labels=[0, 1]
            )
        elif task == "QUANTILE":
            prediction = enforce_quantile_order(prediction)
            objective = (
                pinball_loss(actual, prediction[:, 0], 0.1)
                + pinball_loss(actual, prediction[:, 1], 0.5)
                + pinball_loss(actual, prediction[:, 2], 0.9)
            ) / 3.0
        else:
            raise ValueError(f"Unknown blend task: {task}")
        if objective < best[0]:
            best = (float(objective), float(weight))
    return best[1], best[0]


def conformalize_interval(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.2,
) -> tuple[float, float]:
    actual = np.asarray(actual, dtype="float64")
    lower = np.asarray(lower, dtype="float64")
    upper = np.asarray(upper, dtype="float64")
    if len(actual) < 20:
        return 0.0, 0.0
    lower_error = np.maximum(lower - actual, 0.0)
    upper_error = np.maximum(actual - upper, 0.0)
    level = min(1.0, (1.0 - alpha / 2.0) * (len(actual) + 1) / len(actual))
    return (
        float(np.quantile(lower_error, level, method="higher")),
        float(np.quantile(upper_error, level, method="higher")),
    )


def apply_conformal(
    quantiles: np.ndarray, lower_correction: float, upper_correction: float
) -> np.ndarray:
    result = enforce_quantile_order(quantiles).copy()
    result[:, 0] = np.clip(result[:, 0] - lower_correction, 0.0, None)
    result[:, 2] = np.maximum(result[:, 2] + upper_correction, result[:, 1])
    return result


def regime_labels(frame: pd.DataFrame) -> np.ndarray:
    state = frame.get("current_plan_state", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str)
    pressure = pd.to_numeric(
        frame.get("operational_pressure_ge_train_q90", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    event = pd.to_numeric(
        frame.get("known_event_any_24h", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    return np.where(
        pressure.ge(0.5),
        "HIGH_PRESSURE",
        np.where(event.ge(0.5), "KNOWN_EVENT", np.where(state.str.contains("BERTH", case=False), "BERTHED", "NORMAL")),
    )


def stable_hash_sample(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame.copy()
    keys = frame["port_call_id"].astype(str) + "|" + frame["landmark_at"].astype(str)
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype="uint64")
    positions = np.argpartition(hashes, maximum_rows)[:maximum_rows]
    return frame.iloc[np.sort(positions)].copy()


def to_json_text(value: Any) -> str:
    return json.dumps(clean_json(value), sort_keys=True, ensure_ascii=True)
