from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


AUDIT_VERSION = "b60a2-predictive-signal-v2"
SOURCE_DATASET_VERSION = "b60a-maritime-multitask-hourly-v1"
CV_FOLDS = 4
CV_VALIDATION_HOURS = 720
LINEAR_ALPHA = 10.0
NONLINEAR_TREES = 48
PRIMARY_SCENARIOS = ("LINEAR_FULL", "NONLINEAR_FULL")


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purge_h: int


@dataclass
class SignalAuditResult:
    reports: dict[str, pd.DataFrame]
    decision: dict[str, Any]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [clean_json(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def feature_block(feature: str) -> str:
    if feature.startswith("research_ext_"):
        return "external_weather"
    if feature.startswith("cargo_"):
        return "cargo_mix"
    if feature.startswith(("arrivals_", "departures_", "vessels_in_port_")):
        return "arrival_history"
    if feature.startswith("wave_"):
        return "wave_history"
    if feature.startswith(("issue_", "target_time_", "known_event_")):
        return "calendar"
    if feature.startswith(("delayed_", "mean_arrival_", "weather_available_")):
        return "port_state"
    return "other"


def target_horizon(target: str) -> int:
    match = re.search(r"_(\d+)h$", target)
    if match is not None:
        return int(match.group(1))
    interval = re.search(r"target_arrivals_(\d+)_(\d+)h$", target)
    if interval is not None:
        return int(interval.group(2))
    if target in {
        "target_next_arrival_wait_h",
        "target_next_arrival_observed_24h",
    }:
        return 24
    raise ValueError(f"Target has no operational horizon mapping: {target}")


def is_classification_target(target: str) -> bool:
    return target == "target_next_arrival_observed_24h"


def target_groups(targets: list[str]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for target in targets:
        grouped.setdefault(target_horizon(target), []).append(target)
    return dict(sorted(grouped.items()))


def horizon_specific_features(features: list[str], horizon_h: int) -> list[str]:
    selected = []
    for feature in dict.fromkeys(features):
        target_time = re.search(r"target_time_(\d+)h_", feature)
        event_time = re.search(r"_at_(\d+)h(?:_|$)", feature)
        if target_time is not None and int(target_time.group(1)) != horizon_h:
            continue
        if event_time is not None and int(event_time.group(1)) != horizon_h:
            continue
        selected.append(feature)
    return selected


def make_rolling_origin_folds(
    frame: pd.DataFrame,
    split_column: str,
    purge_h: int,
    folds: int = CV_FOLDS,
    validation_hours: int = CV_VALIDATION_HOURS,
) -> list[TemporalFold]:
    train_positions = np.flatnonzero(frame[split_column].eq("TRAIN").to_numpy())
    if len(train_positions) < 5_000:
        raise ValueError(f"Not enough TRAIN rows for rolling CV: {len(train_positions)}")
    if not np.array_equal(train_positions, np.arange(train_positions[0], train_positions[-1] + 1)):
        raise ValueError("TRAIN rows must be contiguous for temporal rolling CV")
    validation_hours = min(validation_hours, max(336, len(train_positions) // 12))
    first_start = max(2_000 + purge_h, int(len(train_positions) * 0.55))
    last_start = len(train_positions) - validation_hours
    if first_start >= last_start:
        raise ValueError("TRAIN interval cannot support the requested rolling folds")
    starts = np.unique(np.linspace(first_start, last_start, folds, dtype=int))
    if len(starts) != folds:
        raise ValueError("Could not construct distinct rolling-origin folds")
    result = []
    base = int(train_positions[0])
    for fold_number, relative_start in enumerate(starts, start=1):
        validation_start = base + int(relative_start)
        validation_end = validation_start + validation_hours
        train_end_exclusive = validation_start - purge_h
        fold_train = np.arange(base, train_end_exclusive, dtype=int)
        fold_valid = np.arange(validation_start, validation_end, dtype=int)
        if fold_train[-1] + purge_h >= fold_valid[0]:
            raise AssertionError("Temporal purge invariant failed")
        result.append(
            TemporalFold(
                fold=fold_number,
                train_positions=fold_train,
                validation_positions=fold_valid,
                train_end=pd.Timestamp(frame.iloc[fold_train[-1]]["as_of_time"]),
                validation_start=pd.Timestamp(frame.iloc[fold_valid[0]]["as_of_time"]),
                validation_end=pd.Timestamp(frame.iloc[fold_valid[-1]]["as_of_time"]),
                purge_h=purge_h,
            )
        )
    return result


def safe_baseline_lags(horizon_h: int) -> list[int]:
    return sorted(set([horizon_h, max(horizon_h, 24), max(horizon_h, 168)]))


def _baseline_predictions(
    values: np.ndarray,
    train_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    horizon_h: int,
) -> tuple[np.ndarray, int, float]:
    candidates = safe_baseline_lags(horizon_h)
    calibration_length = min(1_440, max(336, len(train_positions) // 5))
    calibration_positions = train_positions[-calibration_length:]
    best_lag = candidates[0]
    best_mae = math.inf
    for lag in candidates:
        valid = calibration_positions >= lag
        positions = calibration_positions[valid]
        actual = values[positions]
        predicted = values[positions - lag]
        finite = np.isfinite(actual) & np.isfinite(predicted)
        if int(finite.sum()) < 100:
            continue
        mae = float(np.mean(np.abs(actual[finite] - predicted[finite])))
        if mae < best_mae:
            best_mae = mae
            best_lag = lag
    predictions = values[evaluation_positions - best_lag]
    return predictions, best_lag, best_mae


def _fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evaluation: np.ndarray,
    targets: list[str],
) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    evaluation_scaled = scaler.transform(x_evaluation)
    prediction = np.empty((len(x_evaluation), len(targets)), dtype="float64")
    regression_indices = [
        index for index, target in enumerate(targets) if not is_classification_target(target)
    ]
    if regression_indices:
        model = Ridge(alpha=LINEAR_ALPHA)
        model.fit(train_scaled, y_train[:, regression_indices])
        values = np.asarray(model.predict(evaluation_scaled)).reshape(
            len(x_evaluation), -1
        )
        prediction[:, regression_indices] = values
    for index, target in enumerate(targets):
        if not is_classification_target(target):
            continue
        classes = np.unique(y_train[:, index])
        if len(classes) < 2:
            prediction[:, index] = float(classes[0])
            continue
        classifier = LogisticRegression(C=1.0, max_iter=300, random_state=2026)
        classifier.fit(train_scaled, y_train[:, index].astype("int64"))
        prediction[:, index] = classifier.predict_proba(evaluation_scaled)[:, 1]
    return prediction


def _fit_extra_trees(
    x_train: np.ndarray,
    y_train: np.ndarray,
    targets: list[str],
) -> dict[str, Any]:
    specification: dict[str, Any] = {"regression": None, "classification": {}}
    regression_indices = [
        index for index, target in enumerate(targets) if not is_classification_target(target)
    ]
    if regression_indices:
        regression_target = y_train[:, regression_indices]
        center = np.mean(regression_target, axis=0)
        scale = np.std(regression_target, axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        model = ExtraTreesRegressor(
            n_estimators=NONLINEAR_TREES,
            max_depth=10,
            min_samples_leaf=20,
            max_features=0.70,
            max_samples=0.70,
            bootstrap=True,
            n_jobs=2,
            random_state=2026,
        )
        scaled_target = (regression_target - center) / scale
        model.fit(
            x_train,
            scaled_target.ravel() if scaled_target.shape[1] == 1 else scaled_target,
        )
        specification["regression"] = (model, center, scale, regression_indices)
    for index, target in enumerate(targets):
        if not is_classification_target(target):
            continue
        classes = np.unique(y_train[:, index])
        if len(classes) < 2:
            specification["classification"][index] = float(classes[0])
            continue
        classifier = ExtraTreesClassifier(
            n_estimators=NONLINEAR_TREES,
            max_depth=10,
            min_samples_leaf=20,
            max_features=0.70,
            max_samples=0.70,
            bootstrap=True,
            n_jobs=2,
            random_state=2026,
        )
        classifier.fit(x_train, y_train[:, index].astype("int64"))
        specification["classification"][index] = classifier
    return specification


def _extra_trees_predict(
    specification: dict[str, Any],
    x: np.ndarray,
    targets: list[str],
) -> np.ndarray:
    prediction = np.empty((len(x), len(targets)), dtype="float64")
    if specification["regression"] is not None:
        model, center, scale, indices = specification["regression"]
        values = np.asarray(model.predict(x)).reshape(len(x), -1)
        prediction[:, indices] = values * scale + center
    for index, model in specification["classification"].items():
        if isinstance(model, float):
            prediction[:, index] = model
        else:
            prediction[:, index] = model.predict_proba(x)[:, 1]
    return prediction


def _clip_predictions(task: str, targets: list[str], prediction: np.ndarray) -> np.ndarray:
    result = np.asarray(prediction, dtype="float64").copy()
    for index, target in enumerate(targets):
        if is_classification_target(target):
            result[:, index] = np.clip(result[:, index], 0.0, 1.0)
        elif task == "ARRIVAL" or "height" in target or "period" in target:
            result[:, index] = np.maximum(result[:, index], 0.0)
        elif "direction_sin" in target or "direction_cos" in target:
            result[:, index] = np.clip(result[:, index], -1.0, 1.0)
    return result


def _metric_records(
    task: str,
    targets: list[str],
    actual: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index, target in enumerate(targets):
        observed = actual[:, index]
        predicted = prediction[:, index]
        errors = predicted - observed
        denominator = float(np.abs(observed).sum())
        classification = is_classification_target(target)
        auc = (
            float(roc_auc_score(observed.astype("int64"), predicted))
            if classification and len(np.unique(observed)) == 2
            else np.nan
        )
        primary_loss = (
            float(np.mean(np.square(errors)))
            if classification
            else float(np.mean(np.abs(errors)))
        )
        rows.append(
            {
                "target": target,
                "metric_role": "COMPONENT" if "direction_" in target else "PRIMARY",
                "primary_metric": "BRIER" if classification else "MAE",
                "primary_loss": primary_loss,
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "wape_pct": float(100.0 * np.abs(errors).sum() / denominator)
                if denominator > 1e-12
                else np.nan,
                "bias": float(np.mean(errors)),
                "roc_auc": auc,
            }
        )
    if task == "WAVE":
        by_name = {target: index for index, target in enumerate(targets)}
        horizons = sorted(
            {
                target_horizon(target)
                for target in targets
                if "direction_sin" in target
            }
        )
        for horizon_h in horizons:
            sin_target = f"target_wave_direction_sin_{horizon_h}h"
            cos_target = f"target_wave_direction_cos_{horizon_h}h"
            if sin_target not in by_name or cos_target not in by_name:
                continue
            sin_index, cos_index = by_name[sin_target], by_name[cos_target]
            observed_angle = np.degrees(
                np.arctan2(actual[:, sin_index], actual[:, cos_index])
            ) % 360.0
            predicted_angle = np.degrees(
                np.arctan2(prediction[:, sin_index], prediction[:, cos_index])
            ) % 360.0
            difference = np.abs(predicted_angle - observed_angle)
            circular_error = np.minimum(difference, 360.0 - difference)
            rows.append(
                {
                    "target": f"target_wave_direction_deg_{horizon_h}h",
                    "metric_role": "PRIMARY",
                    "primary_metric": "CIRCULAR_MAE_DEG",
                    "primary_loss": float(np.mean(circular_error)),
                    "mae": float(np.mean(circular_error)),
                    "rmse": float(np.sqrt(np.mean(np.square(circular_error)))),
                    "wape_pct": np.nan,
                    "bias": np.nan,
                    "roc_auc": np.nan,
                }
            )
    return rows


def _append_metrics(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    task: str,
    targets: list[str],
    actual: np.ndarray,
    prediction: np.ndarray,
    baseline_metrics: dict[str, float] | None,
) -> dict[str, float]:
    metrics = _metric_records(task, targets, actual, prediction)
    result: dict[str, float] = {}
    for values in metrics:
        target = values["target"]
        baseline_loss = (
            values["primary_loss"]
            if baseline_metrics is None
            else baseline_metrics[target]
        )
        lift = (
            100.0 * (baseline_loss - values["primary_loss"]) / baseline_loss
            if baseline_loss > 1e-12
            else 0.0
        )
        rows.append(
            {
                **context,
                **values,
                "baseline_primary_loss": baseline_loss,
                "lift_vs_naive_pct": lift,
            }
        )
        result[target] = float(values["primary_loss"])
    return result


def _scenario_feature_sets(
    features: list[str], task: str
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for feature in features:
        grouped.setdefault(feature_block(feature), []).append(feature)
    own_history = "arrival_history" if task == "ARRIVAL" else "wave_history"
    scenarios = {
        "LINEAR_FULL": features,
        "LINEAR_HISTORY": grouped.get(own_history, []),
    }
    for block, block_features in grouped.items():
        scenarios[f"LINEAR_BLOCK_ONLY::{block}"] = block_features
        remaining = [feature for feature in features if feature not in set(block_features)]
        if remaining:
            scenarios[f"LINEAR_DROP::{block}"] = remaining
    scenarios = {name: value for name, value in scenarios.items() if value}
    return scenarios, grouped


def _evaluate_one_window(
    frame: pd.DataFrame,
    task: str,
    track: str,
    targets: list[str],
    horizon_h: int,
    features: list[str],
    compact_features: list[str],
    train_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    split_role: str,
    fold: int,
    fold_times: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_values = frame[targets].to_numpy(dtype="float64")
    actual = target_values[evaluation_positions]
    baseline = np.empty_like(actual)
    baseline_lags: dict[str, int] = {}
    baseline_calibration: dict[str, float] = {}
    for index, target in enumerate(targets):
        predicted, lag, calibration_mae = _baseline_predictions(
            target_values[:, index], train_positions, evaluation_positions, horizon_h
        )
        baseline[:, index] = predicted
        baseline_lags[target] = lag
        baseline_calibration[target] = calibration_mae

    finite = np.isfinite(actual).all(axis=1) & np.isfinite(baseline).all(axis=1)
    if int(finite.sum()) < 200:
        raise ValueError(
            f"Insufficient evaluation rows for {task}/{track}/{horizon_h}h: {finite.sum()}"
        )
    eval_positions = evaluation_positions[finite]
    actual = actual[finite]
    baseline = baseline[finite]
    y_train = target_values[train_positions]
    train_finite = np.isfinite(y_train).all(axis=1)
    train_positions = train_positions[train_finite]
    y_train = y_train[train_finite]

    context = {
        "task": task,
        "track": track,
        "horizon_h": horizon_h,
        "split_role": split_role,
        "fold": fold,
        "train_rows": len(train_positions),
        "evaluation_rows": len(eval_positions),
        **fold_times,
    }
    metric_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    baseline_rows = []
    baseline_metrics = _append_metrics(
        metric_rows,
        {**context, "scenario": "SAFE_NAIVE"},
        task,
        targets,
        actual,
        baseline,
        None,
    )
    train_mean_prediction = np.tile(np.mean(y_train, axis=0), (len(actual), 1))
    train_mean_prediction = _clip_predictions(
        task, targets, train_mean_prediction
    )
    _append_metrics(
        metric_rows,
        {**context, "scenario": "TRAIN_MEAN"},
        task,
        targets,
        actual,
        train_mean_prediction,
        baseline_metrics,
    )
    for target in targets:
        baseline_rows.append(
            {
                **context,
                "target": target,
                "selected_lag_h": baseline_lags[target],
                "calibration_mae": baseline_calibration[target],
                "availability_rule": "TARGET_SHIFT_LAG_GE_HORIZON",
            }
        )

    scenarios, _ = _scenario_feature_sets(features, task)
    for scenario, scenario_features in scenarios.items():
        prediction = _fit_linear_probe(
            frame.iloc[train_positions][scenario_features].to_numpy(dtype="float64"),
            y_train,
            frame.iloc[eval_positions][scenario_features].to_numpy(dtype="float64"),
            targets,
        )
        prediction = _clip_predictions(task, targets, prediction)
        _append_metrics(
            metric_rows,
            {**context, "scenario": scenario},
            task,
            targets,
            actual,
            prediction,
            baseline_metrics,
        )

    rng = np.random.default_rng(10_000 + horizon_h + fold)
    shuffled_y = y_train[rng.permutation(len(y_train))]
    placebo_prediction = _fit_linear_probe(
        frame.iloc[train_positions][features].to_numpy(dtype="float64"),
        shuffled_y,
        frame.iloc[eval_positions][features].to_numpy(dtype="float64"),
        targets,
    )
    placebo_prediction = _clip_predictions(task, targets, placebo_prediction)
    _append_metrics(
        metric_rows,
        {**context, "scenario": "LINEAR_PLACEBO"},
        task,
        targets,
        actual,
        placebo_prediction,
        baseline_metrics,
    )

    compact_features = [feature for feature in compact_features if feature in features]
    x_train = frame.iloc[train_positions][compact_features].to_numpy(dtype="float32")
    x_evaluation = frame.iloc[eval_positions][compact_features].to_numpy(dtype="float32")
    nonlinear_model = _fit_extra_trees(x_train, y_train, targets)
    nonlinear_prediction = _clip_predictions(
        task,
        targets,
        _extra_trees_predict(nonlinear_model, x_evaluation, targets),
    )
    nonlinear_mae = _append_metrics(
        metric_rows,
        {**context, "scenario": "NONLINEAR_FULL"},
        task,
        targets,
        actual,
        nonlinear_prediction,
        baseline_metrics,
    )

    compact_blocks: dict[str, list[int]] = {}
    for index, feature in enumerate(compact_features):
        compact_blocks.setdefault(feature_block(feature), []).append(index)
    for block, columns in compact_blocks.items():
        permuted = x_evaluation.copy()
        order = rng.permutation(len(permuted))
        permuted[:, columns] = permuted[order][:, columns]
        permuted_prediction = _clip_predictions(
            task,
            targets,
            _extra_trees_predict(nonlinear_model, permuted, targets),
        )
        permuted_metrics = _metric_records(task, targets, actual, permuted_prediction)
        for values in permuted_metrics:
            target = values["target"]
            full_loss = nonlinear_mae[target]
            permutation_rows.append(
                {
                    **context,
                    "target": target,
                    "metric_role": values["metric_role"],
                    "primary_metric": values["primary_metric"],
                    "block": block,
                    "full_primary_loss": full_loss,
                    "permuted_primary_loss": values["primary_loss"],
                    "conditional_loss_pct": 100.0 * (values["primary_loss"] - full_loss) / full_loss
                    if full_loss > 1e-12
                    else 0.0,
                    "features_permuted": len(columns),
                }
            )
    return metric_rows, permutation_rows, baseline_rows


def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    cv = metrics.loc[metrics["split_role"].eq("TRAIN_ROLLING_CV")].copy()
    keys = [
        "task",
        "track",
        "horizon_h",
        "target",
        "metric_role",
        "primary_metric",
        "scenario",
    ]
    rows = []
    for key, group in cv.groupby(keys, sort=False):
        lifts = group["lift_vs_naive_pct"].to_numpy(dtype="float64")
        mean_lift = float(np.mean(lifts))
        standard_error = float(np.std(lifts, ddof=1) / np.sqrt(len(lifts))) if len(lifts) > 1 else 0.0
        critical = 3.182 if len(lifts) == 4 else 4.303 if len(lifts) == 3 else 1.96
        ci_low = mean_lift - critical * standard_error
        ci_high = mean_lift + critical * standard_error
        positive_fraction = float(np.mean(lifts > 0.0))
        rows.append(
            {
                **dict(zip(keys, key)),
                "folds": len(group),
                "mean_primary_loss": float(group["primary_loss"].mean()),
                "mean_mae": float(group["mae"].mean()),
                "mean_rmse": float(group["rmse"].mean()),
                "mean_baseline_primary_loss": float(
                    group["baseline_primary_loss"].mean()
                ),
                "mean_lift_pct": mean_lift,
                "median_lift_pct": float(np.median(lifts)),
                "minimum_lift_pct": float(np.min(lifts)),
                "positive_fold_fraction": positive_fraction,
                "lift_ci95_low": ci_low,
                "lift_ci95_high": ci_high,
                "signal_label": classify_signal(
                    mean_lift,
                    ci_low,
                    positive_fraction,
                    float(np.min(lifts)),
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_signal(
    mean_lift: float,
    ci_low: float,
    positive_fraction: float,
    minimum_lift: float | None = None,
) -> str:
    # Four temporal folds give a very wide Student interval. A probe is also
    # considered stable when every fold improves and the worst fold stays positive.
    all_folds_positive = (
        minimum_lift is not None
        and positive_fraction == 1.0
        and minimum_lift > 0.0
    )
    stable = positive_fraction >= 0.75 and (ci_low > 0.0 or all_folds_positive)
    if stable and mean_lift >= 10.0:
        return "STRONG_SIGNAL"
    if stable and mean_lift >= 5.0:
        return "USEFUL_SIGNAL"
    if stable and mean_lift >= 1.0:
        return "WEAK_SIGNAL"
    if mean_lift > 1.0:
        return "UNSTABLE_SIGNAL"
    return "NO_SIGNAL"


def _placebo_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    cv = metrics.loc[
        metrics["split_role"].eq("TRAIN_ROLLING_CV")
        & metrics["metric_role"].eq("PRIMARY")
    ].copy()
    keys = ["task", "track", "horizon_h", "target", "fold"]
    placebo = cv.loc[
        cv["scenario"].eq("LINEAR_PLACEBO"), keys + ["primary_loss"]
    ].rename(columns={"primary_loss": "placebo_primary_loss"})
    train_mean = cv.loc[
        cv["scenario"].eq("TRAIN_MEAN"), keys + ["primary_loss"]
    ].rename(columns={"primary_loss": "train_mean_primary_loss"})
    paired = placebo.merge(train_mean, on=keys, how="inner", validate="one_to_one")
    paired["placebo_gain_vs_train_mean_pct"] = np.where(
        paired["train_mean_primary_loss"] > 1e-12,
        100.0
        * (paired["train_mean_primary_loss"] - paired["placebo_primary_loss"])
        / paired["train_mean_primary_loss"],
        0.0,
    )
    rows = []
    target_keys = ["task", "track", "horizon_h", "target"]
    for key, group in paired.groupby(target_keys, sort=False):
        gains = group["placebo_gain_vs_train_mean_pct"].to_numpy(dtype="float64")
        mean_gain = float(np.mean(gains))
        positive_fraction = float(np.mean(gains > 0.0))
        suspicious = mean_gain >= 5.0 and positive_fraction >= 0.75
        rows.append(
            {
                **dict(zip(target_keys, key)),
                "folds": len(group),
                "mean_train_mean_primary_loss": float(
                    group["train_mean_primary_loss"].mean()
                ),
                "mean_placebo_primary_loss": float(
                    group["placebo_primary_loss"].mean()
                ),
                "mean_placebo_gain_vs_train_mean_pct": mean_gain,
                "max_placebo_gain_vs_train_mean_pct": float(np.max(gains)),
                "positive_fold_fraction": positive_fraction,
                "suspicious_placebo": suspicious,
                "diagnosis": (
                    "SUSPICIOUS_PLACEBO_SIGNAL"
                    if suspicious
                    else "NEGATIVE_CONTROL_PASSED"
                ),
            }
        )
    return pd.DataFrame(rows)


def _block_signal_summary(
    metrics: pd.DataFrame, permutations: pd.DataFrame
) -> pd.DataFrame:
    cv = metrics.loc[metrics["split_role"].eq("TRAIN_ROLLING_CV")].copy()
    perm = permutations.loc[permutations["split_role"].eq("TRAIN_ROLLING_CV")].copy()
    keys = [
        "task", "track", "horizon_h", "target", "metric_role", "primary_metric"
    ]
    blocks = sorted(
        {
            scenario.split("::", 1)[1]
            for scenario in cv["scenario"]
            if "::" in scenario
        }
        | set(perm["block"].unique())
    )
    rows = []
    for key, target_rows in cv.groupby(keys, sort=False):
        target_context = dict(zip(keys, key))
        full = target_rows.loc[target_rows["scenario"].eq("LINEAR_FULL")].set_index("fold")
        for block in blocks:
            only = target_rows.loc[
                target_rows["scenario"].eq(f"LINEAR_BLOCK_ONLY::{block}")
            ]
            dropped = target_rows.loc[
                target_rows["scenario"].eq(f"LINEAR_DROP::{block}")
            ].set_index("fold")
            conditional_values = []
            for fold in full.index.intersection(dropped.index):
                full_loss = float(full.loc[fold, "primary_loss"])
                drop_loss = float(dropped.loc[fold, "primary_loss"])
                conditional_values.append(
                    100.0 * (drop_loss - full_loss) / full_loss
                    if full_loss > 1e-12
                    else 0.0
                )
            perm_values = perm.loc[
                (perm["task"] == key[0])
                & (perm["track"] == key[1])
                & (perm["horizon_h"] == key[2])
                & (perm["target"] == key[3])
                & (perm["block"] == block),
                "conditional_loss_pct",
            ].to_numpy(dtype="float64")
            marginal = only["lift_vs_naive_pct"].to_numpy(dtype="float64")
            conditional_mean = float(np.mean(conditional_values)) if conditional_values else np.nan
            nonlinear_mean = float(np.mean(perm_values)) if len(perm_values) else np.nan
            marginal_mean = float(np.mean(marginal)) if len(marginal) else np.nan
            conditional_positive = (
                float(np.mean(np.asarray(conditional_values) > 0.0))
                if conditional_values
                else 0.0
            )
            nonlinear_positive = float(np.mean(perm_values > 0.0)) if len(perm_values) else 0.0
            evidence_values = np.asarray(
                [conditional_mean, nonlinear_mean], dtype="float64"
            )
            evidence_values = evidence_values[np.isfinite(evidence_values)]
            evidence = (
                float(evidence_values.max()) if len(evidence_values) else np.nan
            )
            stable_fraction = max(conditional_positive, nonlinear_positive)
            if np.isfinite(evidence) and evidence >= 5.0 and stable_fraction >= 0.75:
                label = "STRONG_CONDITIONAL_SIGNAL"
            elif np.isfinite(evidence) and evidence >= 2.0 and stable_fraction >= 0.75:
                label = "USEFUL_CONDITIONAL_SIGNAL"
            elif np.isfinite(marginal_mean) and marginal_mean >= 5.0:
                label = "REDUNDANT_OR_MARGINAL_SIGNAL"
            else:
                label = "NO_STABLE_INCREMENTAL_SIGNAL"
            rows.append(
                {
                    **target_context,
                    "block": block,
                    "linear_block_only_lift_pct": marginal_mean,
                    "linear_drop_block_loss_pct": conditional_mean,
                    "linear_positive_fold_fraction": conditional_positive,
                    "nonlinear_permutation_loss_pct": nonlinear_mean,
                    "nonlinear_positive_fold_fraction": nonlinear_positive,
                    "block_signal_label": label,
                }
            )
    return pd.DataFrame(rows)


def _target_signal_summary(
    aggregate: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    primary = aggregate.loc[
        aggregate["metric_role"].eq("PRIMARY")
        & aggregate["scenario"].isin(PRIMARY_SCENARIOS)
    ].copy()
    primary["scenario_order"] = primary["scenario"].map(
        {"LINEAR_FULL": 0, "NONLINEAR_FULL": 1}
    )
    primary = primary.sort_values(
        ["task", "track", "target", "mean_lift_pct", "scenario_order"],
        ascending=[True, True, True, False, True],
    )
    selected = primary.groupby(
        ["task", "track", "horizon_h", "target"], as_index=False
    ).first()
    confirmation = metrics.loc[
        metrics["split_role"].eq("VALID_CONFIRMATION")
        & metrics["metric_role"].eq("PRIMARY")
    ][
        [
            "task",
            "track",
            "horizon_h",
            "target",
            "scenario",
            "primary_loss",
            "baseline_primary_loss",
            "mae",
            "roc_auc",
            "lift_vs_naive_pct",
        ]
    ]
    confirmation = confirmation.rename(
        columns={
            "primary_loss": "valid_primary_loss",
            "baseline_primary_loss": "valid_baseline_primary_loss",
            "mae": "valid_mae",
            "roc_auc": "valid_roc_auc",
            "lift_vs_naive_pct": "valid_lift_pct",
        }
    )
    selected = selected.merge(
        confirmation,
        on=["task", "track", "horizon_h", "target", "scenario"],
        how="left",
    )
    selected["valid_confirmation"] = np.select(
        [
            selected["valid_lift_pct"] >= 5.0,
            selected["valid_lift_pct"] > 0.0,
        ],
        ["CONFIRMED", "POSITIVE_BUT_WEAK"],
        default="NOT_CONFIRMED_UNDER_REGIME_SHIFT",
    )
    return selected.drop(columns=["scenario_order"], errors="ignore")


def audit_predictive_signal(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    representation_sets: dict[str, list[str]],
) -> SignalAuditResult:
    frame = dataset.copy().sort_values("as_of_time").reset_index(drop=True)
    frame["as_of_time"] = pd.to_datetime(frame["as_of_time"], utc=True)
    if not frame["as_of_time"].is_monotonic_increasing:
        raise ValueError("B60A dataset is not time ordered")
    if not frame["dataset_version"].eq(SOURCE_DATASET_VERSION).all():
        raise ValueError("Unexpected B60A dataset version")

    metric_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    feature_inventory_rows = []
    forbidden_features = []

    for task in ("ARRIVAL", "WAVE"):
        task_key = task.lower()
        split_column = f"split_{task_key}"
        targets = feature_sets[f"{task_key}_targets"]
        for horizon_h, horizon_targets in target_groups(targets).items():
            folds = make_rolling_origin_folds(frame, split_column, horizon_h)
            for fold in folds:
                fold_rows.append(
                    {
                        "task": task,
                        "horizon_h": horizon_h,
                        "fold": fold.fold,
                        "train_rows": len(fold.train_positions),
                        "validation_rows": len(fold.validation_positions),
                        "train_end": fold.train_end,
                        "validation_start": fold.validation_start,
                        "validation_end": fold.validation_end,
                        "purge_h": fold.purge_h,
                        "purge_valid": bool(
                            fold.train_end
                            + pd.Timedelta(hours=int(horizon_h))
                            < fold.validation_start
                        ),
                    }
                )
            for track in ("CORE", "RESEARCH"):
                track_key = f"{task_key}_{track.lower()}"
                pruned = horizon_specific_features(
                    representation_sets[f"{track_key}_pruned"], horizon_h
                )
                compact = horizon_specific_features(
                    representation_sets[f"{track_key}_compact"], horizon_h
                )
                missing = sorted(set(pruned + compact + horizon_targets) - set(frame.columns))
                if missing:
                    raise ValueError(f"Missing B60A columns for {track_key}: {missing}")
                forbidden_features.extend(
                    feature
                    for feature in pruned + compact
                    if feature.startswith("target_")
                    and not feature.startswith("target_time_")
                )
                feature_inventory_rows.append(
                    {
                        "task": task,
                        "track": track,
                        "horizon_h": horizon_h,
                        "targets": len(horizon_targets),
                        "pruned_features": len(pruned),
                        "compact_features": len(compact),
                        "external_weather_features": sum(
                            feature_block(feature) == "external_weather"
                            for feature in pruned
                        ),
                    }
                )
                for fold in folds:
                    rows, permutations, baselines = _evaluate_one_window(
                        frame=frame,
                        task=task,
                        track=track,
                        targets=horizon_targets,
                        horizon_h=horizon_h,
                        features=pruned,
                        compact_features=compact,
                        train_positions=fold.train_positions,
                        evaluation_positions=fold.validation_positions,
                        split_role="TRAIN_ROLLING_CV",
                        fold=fold.fold,
                        fold_times={
                            "train_end": fold.train_end,
                            "evaluation_start": fold.validation_start,
                            "evaluation_end": fold.validation_end,
                            "purge_h": fold.purge_h,
                        },
                    )
                    metric_rows.extend(rows)
                    permutation_rows.extend(permutations)
                    baseline_rows.extend(baselines)

                train_positions = np.flatnonzero(frame[split_column].eq("TRAIN").to_numpy())
                valid_positions = np.flatnonzero(frame[split_column].eq("VALID").to_numpy())
                train_positions = train_positions[
                    train_positions <= valid_positions[0] - horizon_h - 1
                ]
                rows, permutations, baselines = _evaluate_one_window(
                    frame=frame,
                    task=task,
                    track=track,
                    targets=horizon_targets,
                    horizon_h=horizon_h,
                    features=pruned,
                    compact_features=compact,
                    train_positions=train_positions,
                    evaluation_positions=valid_positions,
                    split_role="VALID_CONFIRMATION",
                    fold=0,
                    fold_times={
                        "train_end": frame.iloc[train_positions[-1]]["as_of_time"],
                        "evaluation_start": frame.iloc[valid_positions[0]]["as_of_time"],
                        "evaluation_end": frame.iloc[valid_positions[-1]]["as_of_time"],
                        "purge_h": horizon_h,
                    },
                )
                metric_rows.extend(rows)
                permutation_rows.extend(permutations)
                baseline_rows.extend(baselines)

    metrics = pd.DataFrame(metric_rows)
    permutations = pd.DataFrame(permutation_rows)
    aggregate = _aggregate_metrics(metrics)
    blocks = _block_signal_summary(metrics, permutations)
    targets = _target_signal_summary(aggregate, metrics)
    folds = pd.DataFrame(fold_rows).drop_duplicates()
    features = pd.DataFrame(feature_inventory_rows)
    baselines = pd.DataFrame(baseline_rows)

    core_targets = targets.loc[targets["track"].eq("CORE")]
    positive_labels = {"STRONG_SIGNAL", "USEFUL_SIGNAL", "WEAK_SIGNAL"}
    stable_targets = int(core_targets["signal_label"].isin(positive_labels).sum())
    target_count = len(core_targets)
    stable_fraction = stable_targets / target_count if target_count else 0.0
    stable_tasks = sorted(
        core_targets.loc[
            core_targets["signal_label"].isin(positive_labels), "task"
        ].unique()
    )
    required_tasks = ["ARRIVAL", "WAVE"]
    confirmed = int(core_targets["valid_confirmation"].eq("CONFIRMED").sum())
    placebo_check = _placebo_diagnostics(metrics)
    suspicious_placebos = int(placebo_check["suspicious_placebo"].sum())
    placebo_max_gain = (
        float(placebo_check["mean_placebo_gain_vs_train_mean_pct"].max())
        if len(placebo_check)
        else np.nan
    )
    finite_metrics = bool(
        np.isfinite(
            metrics[
                [
                    "primary_loss",
                    "mae",
                    "rmse",
                    "baseline_primary_loss",
                    "lift_vs_naive_pct",
                ]
            ].to_numpy()
        ).all()
    )
    gates = pd.DataFrame(
        [
            ("SOURCE_B60A_DATASET_VERSION", True, 0),
            ("NO_TARGET_IN_FEATURES", len(forbidden_features) == 0, len(forbidden_features)),
            ("ALL_ROLLING_FOLDS_PURGED", bool(folds["purge_valid"].all()), int((~folds["purge_valid"]).sum())),
            ("SIGNAL_SELECTION_USES_TRAIN_CV_ONLY", True, 0),
            ("VALID_CONFIRMATION_NOT_SELECTION", True, 0),
            ("TEST_NOT_READ_FOR_SCORING", True, 0),
            ("SAFE_BASELINE_LAG_GE_HORIZON", bool((baselines["selected_lag_h"] >= baselines["horizon_h"]).all()), int((baselines["selected_lag_h"] < baselines["horizon_h"]).sum())),
            ("PLACEBO_DOES_NOT_BEAT_TRAIN_MEAN", suspicious_placebos == 0, suspicious_placebos),
            ("FINITE_PRIMARY_METRICS", finite_metrics, 0 if finite_metrics else 1),
            ("RESEARCH_TRACK_NOT_PROMOTABLE", True, 0),
        ],
        columns=["gate", "passed", "observed"],
    )
    gates["severity"] = "CRITICAL"
    gates_passed = bool(gates["passed"].all())
    if not gates_passed:
        decision_name = "BLOCKED_PREDICTIVE_SIGNAL_AUDIT_REPAIR_REQUIRED"
    elif set(required_tasks).issubset(stable_tasks):
        decision_name = "PREDICTIVE_SIGNAL_CONFIRMED_FOR_B60B"
    elif stable_tasks:
        decision_name = "PARTIAL_PREDICTIVE_SIGNAL_PROCEED_WITH_LIMITS"
    else:
        decision_name = "INSUFFICIENT_SIGNAL_REDEFINE_TARGETS"

    reports = {
        "01_temporal_folds.csv": folds,
        "02_feature_inventory.csv": features,
        "03_safe_baselines.csv": baselines,
        "04_fold_metrics.csv": metrics,
        "05_model_signal_summary.csv": aggregate,
        "06_nonlinear_block_permutation.csv": permutations,
        "07_block_signal_summary.csv": blocks,
        "08_target_signal_summary.csv": targets,
        "09_placebo_diagnostics.csv": placebo_check,
        "10_quality_gates.csv": gates,
    }
    decision = {
        "status": "SUCCESS",
        "decision": decision_name,
        "audit_version": AUDIT_VERSION,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "source_rows": len(frame),
        "rolling_folds": CV_FOLDS,
        "target_track_rows": len(targets),
        "core_primary_targets": target_count,
        "core_targets_with_stable_signal": stable_targets,
        "core_stable_signal_pct": 100.0 * stable_fraction,
        "tasks_with_stable_core_signal": stable_tasks,
        "required_signal_tasks": required_tasks,
        "core_targets_confirmed_on_valid": confirmed,
        "placebo_max_gain_vs_train_mean_pct": placebo_max_gain,
        "suspicious_placebo_targets": suspicious_placebos,
        "quality_gates_passed": gates_passed,
        "selection_split": "TRAIN_ROLLING_CV_ONLY",
        "validation_role": "REGIME_SHIFT_CONFIRMATION_ONLY_NOT_SELECTION",
        "test_role": "UNTOUCHED",
        "selection_used_valid": False,
        "selection_used_test": False,
        "predictive_probe_training_executed": True,
        "final_model_training_executed": False,
        "research_weather_role": "RETROSPECTIVE_RESEARCH_ONLY",
        "production_promotion_allowed": False,
        "source_modified": False,
        "next_block": "B60B_ADVANCED_TIME_SERIES_ROLLING_ORIGIN_BENCHMARK",
        "target_signals": targets.to_dict(orient="records"),
    }
    return SignalAuditResult(reports=reports, decision=clean_json(decision))


def content_checksum(
    dataset_bytes: bytes,
    feature_sets_bytes: bytes,
    representation_sets_bytes: bytes,
) -> str:
    digest = hashlib.sha256(AUDIT_VERSION.encode("ascii"))
    for value in (dataset_bytes, feature_sets_bytes, representation_sets_bytes):
        digest.update(hashlib.sha256(value).digest())
    return digest.hexdigest()


def decision_json(value: dict[str, Any]) -> str:
    return json.dumps(clean_json(value), indent=2, ensure_ascii=True)
